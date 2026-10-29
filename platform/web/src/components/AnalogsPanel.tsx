import { useState } from "react";
import { Icon } from "@iconify/react";
import { dirClass, fmtTime, pct } from "../lib/format";
import type { Analog, RunResult, Time } from "../lib/types";

const num = (v: unknown) => (typeof v === "number" && isFinite(v) ? v : null);
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const yoy = (v: number | null) => (v == null ? "—" : pct(v * 100, 0));
const OUTCOME = { target: "Target", stop: "Stop", time: "Time" } as const;
const range = (a: Analog) => {
  const s = fmtTime(a.start), e = fmtTime(a.end);
  if (typeof a.start === "number") return `${s} → ${e.slice(0, 10) === s.slice(0, 10) ? e.slice(11) : e.slice(5)}`;  // intraday
  return `${s} → ${e.slice(0, 4) === s.slice(0, 4) ? e.slice(5) : e}`;
};

/** PatternEdge analogs: past look-alikes of today's shape, what followed, and where today's context differs. */
export function AnalogsPanel({ result, onFocus, onToday }: {
  result: RunResult;
  onFocus: (start: Time, end: Time) => void;
  onToday: () => void;
}) {
  const [active, setActive] = useState<number | null>(null);
  const analogs = result.analogs ?? [];
  if (!analogs.length) return null;
  const st = result.stats ?? {};
  const H = Math.max(1, (result.forecast?.median.length ?? 21) - 1);
  const up = analogs.filter((a) => a.fwd > 0).length;
  const med = num(st.analog_median_fwd) ?? median(analogs.map((a) => a.fwd));
  const wsum = analogs.reduce((s, a) => s + a.ctx, 0);
  const ctxUp = num(st.analog_ctx_p_up) ?? (wsum ? analogs.reduce((s, a) => s + (a.fwd > 0 ? a.ctx : 0), 0) / wsum : null);
  const slopeNow = num(st.ma50_slope_now);
  const times = result.candles.map((c) => c.time);

  const focus = (i: number) => {
    const a = analogs[i];
    // the window plus the H bars that followed it (clamped to loaded bars)
    const at = times.findIndex((t) => (typeof t === "number" ? t >= Number(a.end) : String(t) >= String(a.end)));
    setActive(i);
    onFocus(a.start, at < 0 ? a.end : times[Math.min(times.length - 1, at + H)]);
  };

  return (
    <section className="card analogs" aria-labelledby="analogs-title">
      <div className="card-head">
        <h3 id="analogs-title">Analogs: this shape before</h3>
        <button className="btn btn-sm" onClick={() => { setActive(null); onToday(); }}>Back to today</button>
      </div>
      <p className="an-sum">
        <strong>{up} of {analogs.length}</strong> past look-alikes rose · median <span className={dirClass(med)}>{pct(med * 100, 1)}</span> in {H} bars
        {ctxUp != null && <> · context-weighted <strong>{(ctxUp * 100).toFixed(0)}%</strong></>}
        {slopeNow != null && <> · 50-DMA slope now {pct(slopeNow, 1)}/10 bars</>}
      </p>
      {!!result.flags?.length && (
        <ul className="an-flags" aria-label="Context differences from the analogs">
          {result.flags.map((f) => <li key={f}><Icon icon="solar:danger-triangle-linear" width={16} aria-hidden="true" /><span>{f}</span></li>)}
        </ul>
      )}
      <ol className="an-list" aria-label="Analogs, most similar first. Select one to show it on the chart.">
        {analogs.map((a, i) => (
          <li key={String(a.start)}>
            <button type="button" aria-pressed={active === i} onClick={() => focus(i)}>
              <span className="an-date">{range(a)}</span>
              <span className={`an-fwd ${dirClass(a.fwd)}`}>{pct(a.fwd * 100, 1)}</span>
              <span className="an-out" data-o={a.outcome}>{OUTCOME[a.outcome] ?? a.outcome}</span>
              <span className="an-meta">
                sim {a.sim.toFixed(2)} · ctx {a.ctx.toFixed(2)} · MA50 {pct(a.ma50Slope, 1)}
                {a.results && <> · {a.results.q} sales {yoy(a.results.salesYoY)} / profit {yoy(a.results.profitYoY)}</>}
              </span>
              {a.patterns.length > 0 && <span className="an-pat">{a.patterns.join(", ")}</span>}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
