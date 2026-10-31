import type { RunResult } from "../lib/types";

const LABELS: Record<string, string> = {
  action: "Action",
  p_target_first: "P(target first)",
  expected_R: "Expected R",
  rank: "Rank",
  rank_bucket: "Rank group",
  time_exit_bars: "Time exit",
  plan_asof: "Plan from close",
};
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
const fmt = (k: string, v: number | string) =>
  typeof v !== "number" ? (k === "action" ? sentence(v) : v)
    : k.startsWith("p_") ? `${(v * 100).toFixed(0)}%` : k === "time_exit_bars" ? `${v} bars`
    : Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 10 ? 2 : 0);

/** TradePlan stats + "why" drivers (share bars are a data graphic with text values). */
export function PlanDetails({ result }: { result: RunResult }) {
  const stats = Object.entries(result.stats ?? {});
  const drivers = [...(result.drivers ?? [])].sort((a, b) => Math.abs(b.share) - Math.abs(a.share)).slice(0, 8);
  const max = Math.max(...drivers.map((d) => Math.abs(d.share)), 1e-9);
  if (!stats.length && !drivers.length) return null;
  return (
    <section className="card" aria-labelledby="plan-title">
      <h3 id="plan-title">What drives TradePlan's score</h3>
      {stats.length > 0 && (
        <dl className="stats">
          {stats.map(([k, v]) => <div key={k}><dt>{LABELS[k] ?? sentence(k.replace(/_/g, " "))}</dt><dd>{fmt(k, v)}</dd></div>)}
        </dl>
      )}
      {drivers.length > 0 && (
        <ul className="drivers" aria-label="Main drivers, by share of the score">
          {drivers.map((d) => (
            <li key={d.name}>
              <span className="d-name">{d.name}</span>
              <span className="d-val">{typeof d.value === "number" ? d.value.toFixed(2) : d.value}</span>
              <span className="d-bar" aria-hidden="true"><span className={d.share >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(d.share) / max) * 100}%` }} /></span>
              <span className="sr-only">{(d.share * 100).toFixed(0)}% of score</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
