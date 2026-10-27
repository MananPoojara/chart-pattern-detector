// Shared bits for the canvas primitives: time -> logical index, and a one-view primitive base.
import type {
  IChartApi, IPrimitivePaneRenderer, IPrimitivePaneView, ISeriesApi, ISeriesPrimitive,
  Logical, PrimitivePaneViewZOrder, SeriesAttachedParameter, SeriesType, Time as LwTime,
} from "lightweight-charts";
import type { Time } from "./types";
import type { Palette } from "./theme";

type Target = Parameters<IPrimitivePaneRenderer["draw"]>[0];
export type Scope = Parameters<Parameters<Target["useBitmapCoordinateSpace"]>[0]>[0];

export const tkey = (t: Time) => (typeof t === "number" ? t : Date.parse(t + "T00:00:00Z") / 1000);

/** Maps any contract Time to a (fractional) logical bar index, extrapolating past the last bar. */
export class TimeIndex {
  keys: number[] = [];
  daily = true;
  reset(times: Time[]) {
    this.keys = times.map(tkey);
    this.daily = typeof times[0] !== "number";
  }
  push(t: Time) { this.keys.push(tkey(t)); }
  get last() { return this.keys.length ? this.keys[this.keys.length - 1] : 0; }
  /** Exact index if t is a bar, floor index if between bars, extrapolated if after the last bar. */
  idx(t: Time): number {
    const k = tkey(t), a = this.keys, n = a.length;
    if (!n) return 0;
    if (k > a[n - 1]) return n - 1 + this.stepsAfter(a[n - 1], k);
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (a[m] <= k) lo = m; else hi = m - 1; }
    return lo;
  }
  /** Bar-count between two keys beyond the data. Daily: weekdays (ignores holidays).
   *  ponytail: intraday uses the smallest recent bar gap, so a future point past session close lands too far right. */
  private stepsAfter(from: number, to: number): number {
    if (this.daily) {
      let n = 0;
      for (let d = from + 86400; d <= to; d += 86400) { const w = new Date(d * 1000).getUTCDay(); if (w && w < 6) n++; }
      return n;
    }
    const a = this.keys;
    let step = Infinity;
    for (let i = Math.max(1, a.length - 10); i < a.length; i++) step = Math.min(step, a[i] - a[i - 1]);
    return (to - from) / (isFinite(step) && step > 0 ? step : 60);
  }
}

/** One pane view, drawn in bitmap space. Subclasses implement draw(); coordinates are media px * ratio. */
export abstract class Prim implements ISeriesPrimitive<LwTime> {
  protected chart?: IChartApi;
  protected series?: ISeriesApi<SeriesType>;
  private request?: () => void;
  private view: IPrimitivePaneView;
  constructor(protected ti: TimeIndex, protected pal: Palette, z: PrimitivePaneViewZOrder = "normal") {
    const renderer = { draw: (t: Target) => t.useBitmapCoordinateSpace((s) => this.chart && this.series && this.draw(s)) };
    this.view = { zOrder: () => z, renderer: () => renderer };
  }
  attached(p: SeriesAttachedParameter<LwTime>) {
    this.chart = p.chart as IChartApi; this.series = p.series as ISeriesApi<SeriesType>; this.request = p.requestUpdate;
  }
  detached() { this.chart = this.series = this.request = undefined; }
  paneViews() { return [this.view]; }
  setPalette(p: Palette) { this.pal = p; this.update(); }
  protected update() { this.request?.(); }
  protected x(t: Time) { return this.chart!.timeScale().logicalToCoordinate(this.ti.idx(t) as Logical); }
  protected y(price: number) { return this.series!.priceToCoordinate(price); }
  protected halfBar() { return this.chart!.timeScale().options().barSpacing / 2; }
  protected abstract draw(s: Scope): void;
}

export function label(s: Scope, pal: Palette, text: string, x: number, y: number, bg: string, fg: string) {
  const c = s.context, r = s.verticalPixelRatio, pad = 4 * r, h = 15 * r;
  c.font = `${11 * r}px ${pal.font}`;
  const w = c.measureText(text).width + pad * 2;
  c.fillStyle = bg; c.fillRect(x, y, w, h);
  c.fillStyle = fg; c.textBaseline = "middle"; c.fillText(text, x + pad, y + h / 2 + 0.5 * r);
  return { x, y, w, h };
}
