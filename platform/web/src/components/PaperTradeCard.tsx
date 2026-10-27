import type { PaperTrade } from "../lib/types";
import { dirClass, fmtTime, price, signed } from "../lib/format";

export function PaperTradeCard({ trade, days = 10 }: { trade: PaperTrade | null; days?: number }) {
  if (!trade) {
    return (
      <section className="card" aria-labelledby="pt-title">
        <h3 id="pt-title">Paper trade</h3>
        <p className="muted small">No buy signal from this model in the last {days} bars, so there is no trade to track.</p>
      </section>
    );
  }
  const risk = trade.entry - trade.stop;
  const rr = risk > 0 ? (trade.target - trade.entry) / risk : null;
  return (
    <section className="card" aria-labelledby="pt-title">
      <div className="card-head">
        <h3 id="pt-title">Paper trade</h3>
        <span className="pt-status">{trade.status}</span>
      </div>
      <dl className="pt-grid">
        <div><dt>Entry</dt><dd>{price(trade.entry)}</dd></div>
        <div><dt>Stop</dt><dd className="down">{price(trade.stop)}</dd></div>
        <div><dt>Target</dt><dd className="up">{price(trade.target)}</dd></div>
        <div><dt>R so far</dt><dd className={trade.R == null ? "muted" : dirClass(trade.R)}>{trade.R == null ? "—" : `${signed(trade.R)} R`}</dd></div>
      </dl>
      <p className="small muted">
        Signal {fmtTime(trade.signal)}{trade.entryTime != null && <>, filled {fmtTime(trade.entryTime)}</>}
        {rr != null && <>, reward/risk {rr.toFixed(1)}</>}
      </p>
      {trade.why && <p className="small">{trade.why}</p>}
      {trade.detail && <p className="small muted">{trade.detail}</p>}
    </section>
  );
}
