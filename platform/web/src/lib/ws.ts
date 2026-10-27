// One shared WebSocket for the whole app: ref-counted topic subscriptions,
// reconnect with backoff + resubscribe, and a per-ticker "last bar" store
// that components read via useSyncExternalStore (no React re-render fan-out per tick).
import { useSyncExternalStore } from "react";
import type { LiveBar } from "./types";

type Listener = (data: unknown) => void;
export type WsStatus = "connecting" | "open" | "closed";

const topics = new Map<string, Set<Listener>>();
const lastBars = new Map<string, LiveBar>(); // ticker -> latest 5m bar
const barListeners = new Map<string, Set<() => void>>();
const statusListeners = new Set<() => void>();

let sock: WebSocket | null = null;
let status: WsStatus = "closed";
let retry = 0;
let retryTimer: ReturnType<typeof setTimeout> | undefined;

function setStatus(s: WsStatus) {
  status = s;
  statusListeners.forEach((l) => l());
}

function send(op: "sub" | "unsub", list: string[]) {
  if (list.length && sock?.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ op, topics: list }));
}

function connect() {
  clearTimeout(retryTimer);
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  setStatus("connecting");
  const ws = new WebSocket(url);
  sock = ws;
  ws.onopen = () => {
    retry = 0;
    setStatus("open");
    send("sub", [...topics.keys()]);
  };
  ws.onmessage = (ev) => {
    let msg: { topic?: string; data?: unknown };
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg.topic) return;
    if (msg.topic.startsWith("bar")) {
      // the snapshot sent on sub (cpd:last) is a bare Candle: take ticker/interval from the topic
      const [, iv, tk] = msg.topic.split(":");
      if (tk && msg.data && typeof msg.data === "object") msg.data = { ticker: tk, interval: iv, final: false, ...msg.data };
      const bar = msg.data as LiveBar;
      if (bar?.ticker && bar.interval === "FIVE_MINUTE") {
        lastBars.set(bar.ticker, bar);
        barListeners.get(bar.ticker)?.forEach((l) => l());
      }
    }
    topics.get(msg.topic)?.forEach((l) => l(msg.data));
  };
  ws.onclose = () => {
    if (sock !== ws) return;
    sock = null;
    setStatus("closed");
    if (!topics.size) return; // nobody listening: reconnect lazily on next subscribe
    const delay = Math.min(30_000, 1000 * 2 ** retry++) * (0.75 + Math.random() / 2);
    retryTimer = setTimeout(connect, delay);
  };
}

/** Subscribe to a topic; returns an unsubscribe function. */
export function subscribe(topic: string, fn: Listener): () => void {
  let set = topics.get(topic);
  if (!set) {
    topics.set(topic, (set = new Set()));
    send("sub", [topic]);
  }
  set.add(fn);
  if (!sock) connect();
  return () => {
    set!.delete(fn);
    if (set!.size) return;
    topics.delete(topic);
    send("unsub", [topic]);
  };
}

export function useLastBar(ticker: string): LiveBar | undefined {
  return useSyncExternalStore(
    (cb) => {
      let set = barListeners.get(ticker);
      if (!set) barListeners.set(ticker, (set = new Set()));
      set.add(cb);
      return () => { set!.delete(cb); };
    },
    () => lastBars.get(ticker),
  );
}

export function useWsStatus(): WsStatus {
  return useSyncExternalStore(
    (cb) => { statusListeners.add(cb); return () => { statusListeners.delete(cb); }; },
    () => status,
  );
}
