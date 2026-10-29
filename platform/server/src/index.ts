/**
 * PatternEdge Bun server — relays Redis (written by the Python services) to HTTP + WebSocket.
 * Binding spec: platform/CONTRACT.md.
 *
 *   GET /api/health     {ok, redis, scanner_age_s, serve_age_s}
 *   GET /api/models     Model[]                (cpd:models)
 *   GET /api/universe   {[basket]: string[]}   (cpd:universe)
 *   GET /api/board      Board                  (cpd:board)
 *   GET /api/signals?limit=50   Signal[] newest first (cpd:signals)
 *   GET /api/run?model=&ticker=&interval=&bars=&days=   RunResult
 *        RPUSH cpd:jobs -> wait cpd:done (or poll) -> GET cpd:result:<id>; 120 s -> 504.
 *        Identical in-flight runs share one job; finished results cached 10 min (the Python worker is serial,
 *        so a cached view must not queue behind a slow first run); intraday entries drop when a bar for
 *        that ticker closes.
 *   GET /api/candles?ticker=&interval=   candles only, fetched with the Yd Angel One account (Yd page). Any F&O
 *        or NSE500 ticker; interval 1D | ONE_MINUTE | THREE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | ONE_HOUR. Cached 30 s.
 *   WS  /ws             {"op":"sub"|"unsub","topics":[...]} -> {"topic","data"}
 *   GET /*              static platform/web/dist with SPA fallback
 *
 * Env: PORT (8787), REDIS_URL (redis://127.0.0.1:6380), KEY_PREFIX (cpd:), WEB_DIST (../web/dist).
 */
import { redis, getJSON, P, INTERVALS, onChannel, startRedis } from "./redis";
import { websocket, startRelay, type WsData } from "./ws";
import { serveStatic } from "./static";

const PORT = Number(process.env.PORT ?? 8787);
const DEV_ORIGIN = "http://localhost:5174";
const RUN_TIMEOUT_MS = Number(process.env.RUN_TIMEOUT_MS ?? 120_000);
const CACHE_MS = Number(process.env.RUN_CACHE_MS ?? 600_000); // = cpd:result TTL

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const json = (body: unknown, status = 200) => Response.json(body, { status });
const rawJson = (s: string) => new Response(s, { headers: { "Content-Type": "application/json" } });

function intParam(q: URLSearchParams, name: string, def: number, min: number, max: number) {
  const s = q.get(name);
  if (s === null || s === "") return def;
  if (!/^\d+$/.test(s) || +s < min || +s > max) throw new HttpError(400, `${name} must be an integer ${min}..${max}`);
  return +s;
}

async function requireJSON(key: string) {
  const s = await redis.get(P + key);
  if (!s) throw new HttpError(503, `${key} not available yet (model service not started?)`);
  return s;
}

// ---- /api/run -----------------------------------------------------------------
const doneWaiters = new Map<string, () => void>();
onChannel("done", (id) => doneWaiters.get(id)?.());

const inflight = new Map<string, Promise<string>>();
const cache = new Map<string, { at: number; body: string }>();

// A closed intraday bar makes cached runs for that ticker+interval stale. The scanner writes the bar to the
// candle store some seconds after publishing it, so a run that still ends before it must not be cached either.
const lastFinal = new Map<string, number>(); // "ticker|interval" -> time of the newest closed bar
for (const iv of INTERVALS) {
  if (iv === "1D") continue;
  onChannel(`bar:${iv}`, (raw) => {
    let b: { ticker?: string; final?: boolean; time?: number };
    try { b = JSON.parse(raw); } catch { return; }
    if (!b.final || !b.ticker) return;
    lastFinal.set(`${b.ticker}|${iv}`, Number(b.time));
    const mid = `|${b.ticker}|${iv}|`;
    for (const k of cache.keys()) if (k.includes(mid)) cache.delete(k);
  });
}
const cacheable = (body: string, ticker: string, interval: string) => {
  try {
    const r = JSON.parse(body);
    if (r.error) return false; // don't cache failed runs
    const fin = lastFinal.get(`${ticker}|${interval}`);
    const last = r.candles?.[r.candles.length - 1]?.time;
    return !(fin && typeof last === "number" && last < fin);
  } catch {
    return false;
  }
};

async function runJob(job: Record<string, unknown> & { id: string }): Promise<string> {
  const resultKey = `${P}result:${job.id}`;
  let wake = () => {};
  doneWaiters.set(job.id, () => wake());
  try {
    await redis.rpush(P + "jobs", JSON.stringify(job));
    const deadline = Date.now() + RUN_TIMEOUT_MS;
    // Wait for cpd:done, but also poll every 2 s in case the subscriber missed it (reconnect).
    while (Date.now() < deadline) {
      await new Promise<void>((r) => {
        wake = r;
        setTimeout(r, Math.min(2000, Math.max(0, deadline - Date.now())));
      });
      const res = await redis.get(resultKey);
      if (res) return res;
    }
    throw new HttpError(504, "model run timed out");
  } finally {
    doneWaiters.delete(job.id);
  }
}

async function handleRun(q: URLSearchParams): Promise<Response> {
  const model = q.get("model") ?? "";
  const ticker = q.get("ticker") ?? "";
  const interval = q.get("interval") ?? "1D";
  if (!(INTERVALS as readonly string[]).includes(interval)) throw new HttpError(400, `interval must be one of ${INTERVALS.join(", ")}`);
  const bars = intParam(q, "bars", 400, 20, 5000);
  const days = intParam(q, "days", 10, 1, 250);

  const [models, universe] = await Promise.all([getJSON<{ id: string }[]>("models"), getJSON<Record<string, string[]>>("universe")]);
  if (!models || !universe) throw new HttpError(503, "models/universe not available yet");
  if (!models.some((m) => m.id === model)) throw new HttpError(400, "unknown model");
  if (!Object.values(universe).some((l) => l.includes(ticker))) throw new HttpError(400, "unknown ticker");

  const key = [model, ticker, interval, bars, days].join("|");
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS) return rawJson(hit.body);

  let p = inflight.get(key);
  if (!p) {
    p = runJob({ id: crypto.randomUUID(), type: "run", model, ticker, interval, bars, days });
    inflight.set(key, p);
    p.then(
      (body) => {
        if (cacheable(body, ticker, interval)) cache.set(key, { at: Date.now(), body });
      },
      () => {},
    ).finally(() => inflight.delete(key));
  }
  return rawJson(await p);
}

// ---- /api/candles (Yd page) ----------------------------------------------------
const CANDLE_INTERVALS = ["1D", "ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR"];
const candleCache = new Map<string, { at: number; body: string }>();

async function handleCandles(q: URLSearchParams): Promise<Response> {
  const ticker = q.get("ticker") ?? "";
  const interval = q.get("interval") ?? "1D";
  if (!CANDLE_INTERVALS.includes(interval)) throw new HttpError(400, `interval must be one of ${CANDLE_INTERVALS.join(", ")}`);
  const universe = await getJSON<Record<string, string[]>>("universe");
  if (!universe) throw new HttpError(503, "universe not available yet");
  if (!Object.values(universe).some((l) => l.includes(ticker))) throw new HttpError(400, "unknown ticker");
  const key = `${ticker}|${interval}`;
  const hit = candleCache.get(key);
  if (hit && Date.now() - hit.at < 30_000) return rawJson(hit.body);
  let p = inflight.get("c|" + key);
  if (!p) {
    p = runJob({ id: crypto.randomUUID(), type: "candles", ticker, interval, profile: "yd" });
    inflight.set("c|" + key, p);
    p.then((body) => { if (!body.includes('"error"')) candleCache.set(key, { at: Date.now(), body }); }, () => {})
      .finally(() => inflight.delete("c|" + key));
  }
  return rawJson(await p);
}

setInterval(() => {
  const now = Date.now();
  for (const [k, v] of candleCache) if (now - v.at >= 30_000) candleCache.delete(k);
  for (const [k, v] of cache) if (now - v.at >= CACHE_MS) cache.delete(k);
}, CACHE_MS).unref?.();

// ---- routing ------------------------------------------------------------------
async function api(path: string, q: URLSearchParams): Promise<Response> {
  switch (path) {
    case "/api/health": {
      const age = (v: string | null | undefined) => (v ? Math.round(Date.now() / 1000 - Number(v)) : null);
      try {
        const st = await redis.hgetall(P + "status");
        return json({ ok: true, redis: true, scanner_age_s: age(st?.scanner_heartbeat), serve_age_s: age(st?.serve_heartbeat) });
      } catch {
        return json({ ok: false, redis: false, scanner_age_s: null, serve_age_s: null });
      }
    }
    case "/api/models":
      return rawJson(await requireJSON("models"));
    case "/api/universe":
      return rawJson(await requireJSON("universe"));
    case "/api/board":
      return rawJson(await requireJSON("board"));
    case "/api/signals": {
      const limit = intParam(q, "limit", 50, 1, 300);
      const items = await redis.lrange(P + "signals", 0, limit - 1);
      return rawJson(`[${items.join(",")}]`);
    }
    case "/api/run":
      return handleRun(q);
    case "/api/candles":
      return handleCandles(q);
  }
  throw new HttpError(404, "not found");
}

function cors(req: Request, res: Response) {
  if (req.headers.get("origin") === DEV_ORIGIN) {
    res.headers.set("Access-Control-Allow-Origin", DEV_ORIGIN);
    res.headers.set("Vary", "Origin");
  }
  return res;
}

const server = Bun.serve<WsData, {}>({
  port: PORT,
  hostname: "0.0.0.0",
  idleTimeout: 0, // /api/run may wait up to 120 s
  websocket,
  async fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname === "/ws") {
      if (server.upgrade(req, { data: { topics: new Set<string>() } })) return;
      return new Response("websocket upgrade expected", { status: 400 });
    }
    if (req.method === "OPTIONS")
      return cors(req, new Response(null, { status: 204, headers: { "Access-Control-Allow-Methods": "GET, OPTIONS", "Access-Control-Allow-Headers": "Content-Type" } }));
    if (req.method !== "GET" && req.method !== "HEAD") return json({ error: "method not allowed" }, 405);

    if (!url.pathname.startsWith("/api/")) return serveStatic(url.pathname);

    const t0 = performance.now();
    let res: Response;
    try {
      res = await api(url.pathname, url.searchParams);
    } catch (e: any) {
      // Anything that isn't ours is a Redis/connection failure -> 503
      res = e instanceof HttpError ? json({ error: e.message }, e.status) : json({ error: "redis unavailable" }, 503);
    }
    if (url.pathname !== "/api/health") console.log(`${req.method} ${url.pathname}${url.search} ${res.status} ${Math.round(performance.now() - t0)}ms`);
    return cors(req, res);
  },
});

startRelay(server);
await startRedis();
console.log(`cpd server on http://0.0.0.0:${server.port} (redis ${redis.connected ? "up" : "DOWN"}, prefix ${P})`);
