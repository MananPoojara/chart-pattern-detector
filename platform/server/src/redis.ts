import { RedisClient } from "bun";

export const REDIS_URL = process.env.REDIS_URL ?? "redis://127.0.0.1:6380";
export const P = process.env.KEY_PREFIX ?? "cpd:";
export const INTERVALS = ["1D", "FIVE_MINUTE", "FIFTEEN_MINUTE"] as const;

// Command client: fails fast while Redis is down (no offline queue) so the API can answer 503.
// Bun 1.3.8's client stays broken after a failed connect, so the watchdog replaces it instead of reconnecting.
// `redis` is a live binding: importers always see the current client.
const newCmd = () => new RedisClient(REDIS_URL, { enableOfflineQueue: false, autoReconnect: false, connectionTimeout: 2000 });
export let redis = newCmd();

let connecting: Promise<unknown> | null = null;
function ensureCmd() {
  if (redis.connected || connecting) return;
  redis.close();
  redis = newCmd();
  connecting = redis.connect().catch(() => {}).finally(() => (connecting = null));
}

export async function getJSON<T = any>(key: string): Promise<T | null> {
  const s = await redis.get(P + key);
  return s ? JSON.parse(s) : null;
}

// Subscriber: a subscribed connection can't run commands, so it gets its own client.
// The watchdog recreates it (and re-subscribes) whenever it drops.
type Handler = (message: string, channel: string) => void;
const handlers = new Map<string, Handler[]>(); // full channel name -> handlers
let sub: RedisClient | null = null;
let subscribing = false;

export function onChannel(channel: string, h: Handler) {
  const hs = handlers.get(P + channel);
  if (hs) hs.push(h);
  else handlers.set(P + channel, [h]);
}

async function ensureSub() {
  if ((sub && sub.connected) || subscribing) return;
  subscribing = true;
  try {
    sub?.close();
    sub = new RedisClient(REDIS_URL, { autoReconnect: false, connectionTimeout: 2000 });
    await sub.connect();
    for (const [ch, hs] of handlers) await sub.subscribe(ch, (m, c) => hs.forEach((h) => h(m, c)));
  } catch {
    sub?.close();
    sub = null;
  } finally {
    subscribing = false;
  }
}

export async function startRedis() {
  ensureCmd();
  await connecting;
  await ensureSub();
  setInterval(() => (ensureCmd(), ensureSub()), 3000).unref?.();
}

export const subConnected = () => !!sub?.connected;
