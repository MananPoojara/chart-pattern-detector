import type { Board, Health, Interval, Model, RunResult, Signal, Universe, YdScreen } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { signal, headers: { accept: "application/json" } });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Can't reach the PatternEdge server.", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(body?.error ?? `Server answered ${res.status}.`, res.status);
  return body as T;
}

// Static-ish lists: fetch once per page load and share the promise.
const once = new Map<string, Promise<unknown>>();
function cached<T>(path: string): Promise<T> {
  let p = once.get(path) as Promise<T> | undefined;
  if (!p) {
    p = get<T>(path);
    p.catch(() => once.delete(path)); // allow retry after failure
    once.set(path, p);
  }
  return p;
}

export const api = {
  health: () => get<Health>("/api/health"),
  models: () => cached<Model[]>("/api/models"),
  universe: () => cached<Universe>("/api/universe"),
  board: () => get<Board>("/api/board"),
  signals: (limit = 30) => get<Signal[]>(`/api/signals?limit=${limit}`),
  run: (model: string, ticker: string, interval: Interval, signal?: AbortSignal) =>
    get<RunResult>(`/api/run?${new URLSearchParams({ model, ticker, interval })}`, signal),
  candles: (ticker: string, interval: string, signal?: AbortSignal) =>
    get<RunResult>(`/api/candles?${new URLSearchParams({ ticker, interval })}`, signal),
  ydScreen: () => get<YdScreen>("/api/yd/screen"),
};
