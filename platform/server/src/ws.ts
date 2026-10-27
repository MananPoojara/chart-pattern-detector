import type { Server, ServerWebSocket, WebSocketHandler } from "bun";
import { redis, P, INTERVALS, onChannel } from "./redis";

export type WsData = { topics: Set<string> };

const MAX_TOPICS = 200;
const IV = INTERVALS.join("|");
const TOPIC_RE = new RegExp(`^(bar:(${IV}):[A-Z0-9&._-]{1,40}|bars:(${IV})|signals|board)$`);
const sockets = new Set<ServerWebSocket<WsData>>();

// Raw JSON from Redis is spliced in as-is (no re-serialisation).
const frame = (topic: string, raw: string) => `{"topic":${JSON.stringify(topic)},"data":${raw}}`;

export const websocket: WebSocketHandler<WsData> = {
  maxPayloadLength: 16 * 1024,
  idleTimeout: 120,
  open(ws) {
    sockets.add(ws);
  },
  close(ws) {
    sockets.delete(ws);
  },
  async message(ws, msg) {
    let m: any;
    try {
      m = JSON.parse(String(msg));
    } catch {
      return;
    }
    if (!m || !Array.isArray(m.topics) || (m.op !== "sub" && m.op !== "unsub")) return;
    for (const t of m.topics) {
      if (typeof t !== "string" || !TOPIC_RE.test(t)) continue;
      if (m.op === "unsub") {
        ws.unsubscribe(t);
        ws.data.topics.delete(t);
        continue;
      }
      if (ws.data.topics.has(t) || ws.data.topics.size >= MAX_TOPICS) continue;
      ws.subscribe(t);
      ws.data.topics.add(t);
      if (t.startsWith("bars:")) {
        // snapshot of every ticker's latest bar, in the topic's usual {ticker, interval, ...Candle} shape,
        // so quiet stocks show a live price before their next tick
        const interval = t.slice(5);
        try {
          const all = (await redis.hgetall(`${P}last:${interval}`)) ?? {};
          for (const [tk, raw] of Object.entries(all)) {
            if (ws.readyState !== 1 || !raw.startsWith("{")) continue;
            ws.send(frame(t, `{"ticker":${JSON.stringify(tk)},"interval":${JSON.stringify(interval)},"final":false,${raw.slice(1)}`));
          }
        } catch {}
      } else if (t.startsWith("bar:")) {
        const [, interval, ticker] = t.split(":");
        try {
          const raw = await redis.hget(`${P}last:${interval}`, ticker!);
          if (raw && ws.readyState === 1) ws.send(frame(t, raw));
        } catch {} // Redis down: live bars will still arrive after reconnect
      }
    }
  },
};

export function startRelay(server: Server<WsData>) {
  for (const iv of INTERVALS) {
    onChannel(`bar:${iv}`, (raw) => {
      let ticker: unknown;
      try {
        ticker = JSON.parse(raw).ticker;
      } catch {
        return;
      }
      if (typeof ticker !== "string") return;
      server.publish(`bar:${iv}:${ticker}`, frame(`bar:${iv}:${ticker}`, raw));
      server.publish(`bars:${iv}`, frame(`bars:${iv}`, raw));
    });
  }
  onChannel("signal", (raw) => void server.publish("signals", frame("signals", raw)));

  // ponytail: board change detection by polling every 30 s; switch to a cpd:board pub/sub channel if latency matters.
  let lastBoard: string | null = null;
  const pollBoard = async () => {
    try {
      const b = await redis.get(P + "board");
      if (b && lastBoard !== null && b !== lastBoard) server.publish("board", frame("board", b));
      if (b) lastBoard = b;
    } catch {}
  };
  pollBoard();
  setInterval(pollBoard, Number(process.env.BOARD_POLL_MS ?? 30000));
  setInterval(() => sockets.forEach((ws) => ws.ping()), 30000);
}
