import {
  CandlestickSeries, ColorType, CrosshairMode, HistogramSeries, LineStyle, createChart, createSeriesMarkers,
  type IPriceLine, type ISeriesApi, type Logical, type SeriesMarker, type Time as LwTime,
} from "lightweight-charts";
import { BoxesPrimitive } from "./boxes";
import { ConePrimitive } from "./cone";
import { LinesPrimitive } from "./lines";
import { TimeIndex, tkey } from "./prim";
import { alpha, readPalette, type Palette, type ThemeName } from "./theme";
import { ZonesPrimitive } from "./zones";
import type { Candle, RunResult, Time } from "./types";

export * from "./types";
export type { Palette, ThemeName } from "./theme";

export interface PatternChart {
  setResult(r: RunResult): void;
  pushBar(c: Candle): void;
  setTheme(t: ThemeName): void;
  fit(): void;
  destroy(): void;
}

const MARKER_TEXT_LAST = 6; // text only on the newest markers so busy detectors don't bury candles
const MARKER_TEXT_GAP = 5; // bars between texted markers on the same side
const MARKER_TEXT_MAX = 18; // chars; long detector names are cut (full text stays in the API result)
const clip = (s: string) => (s.length > MARKER_TEXT_MAX ? s.slice(0, MARKER_TEXT_MAX - 1) + "…" : s);
const lw = (t: Time) => t as LwTime;

export function createPatternChart(el: HTMLElement, opts: { theme?: ThemeName } = {}): PatternChart {
  let theme: ThemeName = opts.theme ?? "dark";
  let pal: Palette = readPalette(el, theme);
  const chart = createChart(el, { autoSize: true });
  const candles = chart.addSeries(CandlestickSeries, { priceLineVisible: false });
  const volume = chart.addSeries(HistogramSeries, { priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
  candles.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.2 } });
  volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  // The markers plugin re-indexes every marker on any data change of its series, which would make pushBar O(markers).
  // So markers live on an invisible copy of the candles that only changes in setResult.
  const clear = "rgba(0,0,0,0)";
  const anchor = chart.addSeries(CandlestickSeries, { upColor: clear, downColor: clear, borderVisible: false, wickVisible: false,
    priceLineVisible: false, lastValueVisible: false });
  const markers = createSeriesMarkers(anchor, []);

  const ti = new TimeIndex();
  const zones = new ZonesPrimitive(ti, pal, "bottom");
  const boxes = new BoxesPrimitive(ti, pal);
  const lines = new LinesPrimitive(ti, pal);
  const cone = new ConePrimitive(ti, pal);
  const prims = [zones, boxes, lines, cone];
  prims.forEach((p) => candles.attachPrimitive(p));

  let bars: Candle[] = [];
  let times: Time[] = [];
  let result: RunResult | null = null;
  let priceLines: IPriceLine[] = [];
  let score: ISeriesApi<"Histogram"> | null = null;
  let rightPad = 8;
  let dead = false;

  const volBar = (c: Candle) => ({ time: lw(c.time), value: c.volume, color: alpha(c.close >= c.open ? pal.up : pal.down, 0.28) });
  /** candle time at or before t, or null if t is outside the loaded bars */
  const snap = (t: Time): Time | null => {
    const k = tkey(t);
    if (!times.length || k < ti.keys[0] || k > ti.last) return null;
    return times[ti.idx(t)];
  };

  function applyTheme() {
    chart.applyOptions({
      layout: { background: { type: ColorType.Solid, color: pal.bg }, textColor: pal.text, fontFamily: pal.font, fontSize: 11, attributionLogo: false,
        panes: { separatorColor: pal.border, separatorHoverColor: alpha(pal.text, 0.15) } },
      grid: { vertLines: { color: pal.grid }, horzLines: { color: pal.grid } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: pal.crosshair, width: 1, style: LineStyle.Dashed, labelBackgroundColor: pal.border },
        horzLine: { color: pal.crosshair, width: 1, style: LineStyle.Dashed, labelBackgroundColor: pal.border },
      },
      rightPriceScale: { borderColor: pal.border },
      timeScale: { borderColor: pal.border, rightOffset: rightPad },
    });
    candles.applyOptions({ upColor: pal.up, downColor: pal.down, borderVisible: false, wickUpColor: pal.up, wickDownColor: pal.down });
    prims.forEach((p) => p.setPalette(pal));
  }

  function renderOverlays(r: RunResult) {
    // markers: snapped onto loaded bars, sorted by time (required by the markers plugin)
    const ms = r.markers.flatMap((m) => { const t = snap(m.time); return t == null ? [] : [{ m, t, k: tkey(t) }]; }).sort((a, b) => a.k - b.k);
    // text on the newest few only: one per bar, and per side at most one every MARKER_TEXT_GAP bars,
    // so labels of stacked markers don't overprint each other
    const texted = new Set<number>(), lastAt = new Map<number, number>();
    let lastBar = -1;
    for (let i = ms.length - 1; i >= 0 && texted.size < MARKER_TEXT_LAST; i--) {
      const at = ti.idx(ms[i].t), prev = lastAt.get(ms[i].m.dir);
      if (at === lastBar || (prev != null && prev - at < MARKER_TEXT_GAP)) continue;
      texted.add(i); lastAt.set(ms[i].m.dir, at); lastBar = at;
    }
    markers.setMarkers(ms.map(({ m, t }, i): SeriesMarker<LwTime> => ({
      time: lw(t),
      position: m.dir > 0 ? "belowBar" : m.dir < 0 ? "aboveBar" : "inBar",
      shape: m.dir > 0 ? "arrowUp" : m.dir < 0 ? "arrowDown" : "circle",
      color: m.dir > 0 ? pal.up : m.dir < 0 ? pal.down : pal.accent,
      size: 0.6,
      text: texted.has(i) ? clip(m.text) : undefined,
    })));

    lines.set(r.lines, r.kind === "lines" ? 1.5 : 1);
    boxes.set(r.boxes);
    cone.set(r.forecast?.median.length ? r.forecast : null);

    // levels: plan's own levels, else the paper trade's
    const pt = r.paperTrade;
    const lv = (n: "entry" | "stop" | "target") => r.levels.find((l) => l.title === n)?.price ?? pt?.[n];
    zones.set(r.kind === "plan" || pt
      ? { signal: snap(pt?.signal ?? times[times.length - 1]) ?? times[times.length - 1], entry: lv("entry"), stop: lv("stop"), target: lv("target"), shade: r.kind === "plan", vline: !!pt }
      : null);

    priceLines.forEach((l) => candles.removePriceLine(l));
    const seen = new Set<string>();
    const levelList = [...r.levels, ...(pt ? [{ title: "entry", price: pt.entry }, { title: "stop", price: pt.stop }, { title: "target", price: pt.target }] : [])];
    priceLines = levelList.filter((l) => Number.isFinite(l.price) && !seen.has(l.title + l.price) && seen.add(l.title + l.price)).map((l) => candles.createPriceLine({
      price: l.price, title: l.title, lineWidth: 1, axisLabelVisible: true,
      lineStyle: l.title === "entry" ? LineStyle.Solid : LineStyle.Dashed,
      color: l.title === "stop" ? pal.down : l.title === "target" ? pal.up : l.title === "entry" ? pal.accent : pal.muted,
    }));

    const sc = (r.score ?? []).filter((p) => { const t = snap(p.time); return t != null && tkey(t) === tkey(p.time); });
    if (sc.length) {
      if (!score) {
        score = chart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: true, priceFormat: { type: "price", precision: 2, minMove: 0.01 } }, 1);
        chart.panes()[0].setStretchFactor(3);
        chart.panes()[1]?.setStretchFactor(1);
      }
      score.setData(sc.map((p) => ({ time: lw(p.time), value: p.value, color: alpha(p.value >= 0 ? pal.up : pal.down, 0.75) })));
    } else if (score) {
      chart.removeSeries(score);
      score = null;
    }
  }

  const api: PatternChart = {
    setResult(r) {
      // a re-run of the same view (live bar closed) keeps the user's zoom/scroll
      const same = !!result && result.model === r.model && result.ticker === r.ticker && result.interval === r.interval;
      result = r;
      bars = r.candles.slice();
      times = bars.map((c) => c.time);
      ti.reset(times);
      chart.applyOptions({ timeScale: { timeVisible: typeof times[0] === "number", secondsVisible: false } });
      const ohlc = bars.map((c) => ({ time: lw(c.time), open: c.open, high: c.high, low: c.low, close: c.close }));
      candles.setData(ohlc);
      anchor.setData(ohlc);
      volume.setData(bars.map(volBar));
      const ahead = r.forecast?.median.length ? ti.idx(r.forecast.median[r.forecast.median.length - 1].time) - (times.length - 1) : 0;
      rightPad = Math.max(8, Math.ceil(ahead) + 4);
      chart.applyOptions({ timeScale: { rightOffset: rightPad } });
      renderOverlays(r);
      if (!same) api.fit();
    },
    pushBar(c) {
      if (!times.length || dead) return;
      const k = tkey(c.time);
      if (k < ti.last) return; // stale tick; series.update would throw
      if (k > ti.last) { bars.push(c); times.push(c.time); ti.push(c.time); } else bars[bars.length - 1] = c;
      candles.update({ time: lw(c.time), open: c.open, high: c.high, low: c.low, close: c.close });
      volume.update(volBar(c));
    },
    setTheme(t) {
      theme = t;
      pal = readPalette(el, theme);
      applyTheme();
      if (result) { volume.setData(bars.map(volBar)); renderOverlays(result); }
    },
    fit() {
      const n = times.length;
      if (n) chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 120) as Logical, to: (n - 1 + rightPad) as Logical });
    },
    destroy() {
      if (dead) return;
      dead = true;
      prims.forEach((p) => candles.detachPrimitive(p));
      markers.detach();
      chart.remove();
    },
  };
  applyTheme();
  return api;
}
