import { pct } from "../lib/format";

/** 30-close sparkline as a data graphic with a text alternative. */
export function Sparkline({ values, width = 96, height = 28 }: { values: number[]; width?: number; height?: number }) {
  if (values.length < 2) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values
    .map((v, i) => `${((i / (values.length - 1)) * width).toFixed(1)},${(height - ((v - min) / span) * height).toFixed(1)}`)
    .join(" ");
  const first = values[0], last = values[values.length - 1];
  const change = ((last - first) / first) * 100;
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={`Last ${values.length} closes: ${first.toFixed(2)} to ${last.toFixed(2)}, ${pct(change, 1)}`}>
      <polyline className={last >= first ? "up" : "down"} points={pts} />
    </svg>
  );
}
