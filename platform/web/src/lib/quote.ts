import { useLastBar } from "./ws";

const barDate = (t: unknown) => (typeof t === "number" ? new Date(t * 1000).toISOString().slice(0, 10) : String(t).slice(0, 10));

/**
 * Last price + day change for a ticker. `close`/`chg` are the last daily close and its % change;
 * once a 5-min bar from a later session arrives, change is measured against that close.
 * Only the calling component re-renders per tick.
 */
export function useLiveQuote(ticker: string, close: number, chg: number, asof: string) {
  const bar = useLastBar(ticker);
  if (bar && barDate(bar.time) > asof) return { last: bar.close, chg: (bar.close / close - 1) * 100, live: true };
  return { last: close, chg, live: false };
}
