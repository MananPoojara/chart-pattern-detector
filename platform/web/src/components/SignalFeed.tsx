import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { subscribe } from "../lib/ws";
import { fmtTime, price } from "../lib/format";
import { linkProps } from "../lib/router";
import type { Signal } from "../lib/types";

const MAX = 30;

/** Recent scanner signals; new ones arrive over WS and are announced politely. */
export function SignalFeed({ limit = 12, ticker }: { limit?: number; ticker?: string }) {
  const [items, setItems] = useState<Signal[] | null>(null);
  const [fresh, setFresh] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    api.signals(MAX).then((s) => alive && setItems(s), () => alive && (setItems([]), setFailed(true)));
    const off = subscribe("signals", (d) => {
      const s = d as Signal;
      setItems((prev) => (prev?.some((x) => x.id === s.id) ? prev : [s, ...(prev ?? [])].slice(0, MAX)));
      setFresh(s.id);
    });
    return () => { alive = false; off(); };
  }, []);

  const shown = (items ?? []).filter((s) => !ticker || s.ticker === ticker).slice(0, limit);
  return (
    <div>
      <ul className="feed" aria-live="polite" aria-relevant="additions" aria-busy={items === null}>
        {shown.map((s) => (
          <li key={s.id} className={s.id === fresh ? "fresh" : undefined}>
            <time>{fmtTime(s.time).slice(-5)}</time>
            <span>
              <a className="tk" {...linkProps(`/t/${s.ticker}?model=${encodeURIComponent(s.model)}&interval=5m`)}>{s.ticker}</a>{" "}
              <span className={s.dir > 0 ? "up" : "down"}>{s.dir > 0 ? "buy" : "sell"}</span>
            </span>
            <span className="lvl">{price(s.close)}</span>
            <span className="sig">{s.signal} · {s.model} · stop {price(s.stop)} · target {price(s.target)}</span>
          </li>
        ))}
      </ul>
      {items !== null && !shown.length && (
        <p className="empty">
          {failed ? "Signals are unavailable while the server is unreachable." : "No signals yet today. The scanner posts here as 5-minute bars close during market hours."}
        </p>
      )}
    </div>
  );
}
