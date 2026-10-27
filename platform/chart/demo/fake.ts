// Deterministic synthetic RunResults for every kind (daily + intraday).
import type { Candle, Kind, RunResult, Time } from "../src";

export function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

export function candles(n: number, intraday: boolean, seed = 7): Candle[] {
  const r = rng(seed), out: Candle[] = [];
  let px = 1500, t = intraday ? Date.UTC(2026, 9, 1, 9, 15) / 1000 : Date.UTC(2019, 0, 1) / 1000;
  while (out.length < n) {
    if (intraday) {
      const d = new Date(t * 1000), mins = d.getUTCHours() * 60 + d.getUTCMinutes();
      if (d.getUTCDay() % 6 === 0 || mins < 555 || mins > 925) { t += 300; if (mins > 925) t += 17 * 3600; continue; }
    } else if (new Date(t * 1000).getUTCDay() % 6 === 0) { t += 86400; continue; }
    const vol = intraday ? 0.002 : 0.014, drift = Math.sin(out.length / 90) * vol * 0.15;
    const o = px, c = o * (1 + drift + (r() - 0.5) * 2 * vol);
    const h = Math.max(o, c) * (1 + r() * vol * 0.6), l = Math.min(o, c) * (1 - r() * vol * 0.6);
    const time: Time = intraday ? t : new Date(t * 1000).toISOString().slice(0, 10);
    out.push({ time, open: o, high: h, low: l, close: c, volume: Math.round(1e5 * (0.5 + r() * 1.5)) });
    px = c; t += intraday ? 300 : 86400;
  }
  return out;
}

const fut = (cs: Candle[], k: number): Time => {
  const last = cs[cs.length - 1].time;
  if (typeof last === "number") return last + k * 300;
  const d = new Date(last + "T00:00:00Z");
  for (let i = 0; i < k; ) { d.setUTCDate(d.getUTCDate() + 1); if (d.getUTCDay() % 6) i++; }
  return d.toISOString().slice(0, 10);
};

export function fake(kind: Kind, intraday: boolean): RunResult {
  const cs = candles(2000, intraday), n = cs.length, r = rng(42), last = cs[n - 1];
  const res: RunResult = {
    id: "demo", model: `demo-${kind}`, kind, ticker: "DEMO", interval: intraday ? "FIVE_MINUTE" : "1D", asof: last.time,
    candles: cs, markers: [], boxes: [], lines: [], levels: [], forecast: null, score: null, paperTrade: null, note: "",
  };
  const pick = () => Math.floor(r() * n);
  if (kind === "rule") {
    res.note = "Arrows: rule-based pattern hits (3,000). Thin polylines: pattern pivots.";
    for (let i = 0; i < 3000; i++) { const b = pick(), d = r() < 0.5 ? 1 : -1; res.markers.push({ time: cs[b].time, dir: d, text: d > 0 ? "Hammer" : "Engulf" }); }
    for (const s of [n - 60, n - 30]) {
      const pts = [0, 6, 12, 18, 24].map((o, j) => ({ time: cs[s + o].time, value: j % 2 ? cs[s + o].high : cs[s + o].low }));
      res.lines.push({ points: pts, dir: 1, label: "W" });
    }
  }
  if (kind === "lines") {
    res.note = "Support (teal) and resistance (orange) trendlines, extended 10 bars ahead.";
    const a = n - 90;
    const lows = cs.slice(a), lo = Math.min(...lows.map((c) => c.low)), hi = Math.max(...lows.map((c) => c.high));
    res.lines.push({ dir: 1, label: "S", points: [{ time: cs[a].time, value: lo }, { time: fut(cs, 10), value: lo + (last.close - lo) * 0.6 }] });
    res.lines.push({ dir: -1, label: "R", points: [{ time: cs[a].time, value: hi }, { time: fut(cs, 10), value: hi - (hi - last.close) * 0.5 }] });
  }
  if (kind === "vision") {
    res.note = "YOLO boxes: bright = live (right edge), muted = historical detections.";
    for (let k = 0; k < 10; k++) {
      const s = n - 115 + k * 9 + Math.floor(r() * 4), e = Math.min(n - 1, s + 6 + Math.floor(r() * 8)), live = e === n - 1;
      const seg = cs.slice(s, e + 1), d = r() < 0.6 ? 1 : -1;
      res.boxes.push({ start: cs[s].time, end: cs[e].time, top: Math.max(...seg.map((c) => c.high)), bottom: Math.min(...seg.map((c) => c.low)), label: d > 0 ? "Buy" : "Sell", dir: d, conf: 0.35 + r() * 0.6, live });
    }
    const seg = cs.slice(n - 12);
    res.boxes.push({ start: cs[n - 12].time, end: last.time, top: Math.max(...seg.map((c) => c.high)), bottom: Math.min(...seg.map((c) => c.low)), label: "Buy", dir: 1, conf: 0.62, live: true });
  }
  if (kind === "forecast") {
    res.note = "Median path with 10–90% cone for the next 20 bars.";
    const f = { median: [] as { time: Time; value: number }[], lo: [] as { time: Time; value: number }[], hi: [] as { time: Time; value: number }[] };
    for (let k = 0; k <= 20; k++) {
      const t = k ? fut(cs, k) : last.time, m = last.close * (1 + 0.002 * k), w = last.close * 0.012 * Math.sqrt(k);
      f.median.push({ time: t, value: m }); f.lo.push({ time: t, value: m - w }); f.hi.push({ time: t, value: m + w });
    }
    res.forecast = f;
  }
  if (kind === "edge") {
    res.note = "Edge score per bar (pane below). Arrows on top/bottom 5% scores.";
    res.score = cs.map((c, i) => ({ time: c.time, value: Math.sin(i / 7) * 0.6 + (r() - 0.5) * 0.8 }));
    const sorted = res.score.map((p) => p.value).sort((a, b) => a - b), q5 = sorted[Math.floor(n * 0.05)], q95 = sorted[Math.floor(n * 0.95)];
    for (const p of res.score) if (p.value >= q95 || p.value <= q5) res.markers.push({ time: p.time, dir: p.value > 0 ? 1 : -1, text: p.value.toFixed(2) });
  }
  if (kind === "plan" || kind === "vision" || kind === "rule") {
    const s = cs[n - 6], entry = s.close, w = intraday ? 0.004 : 0.03, stop = entry * (1 - w), target = entry * (1 + 2 * w);
    res.paperTrade = { signal: s.time, why: "demo", entry, stop, target, status: "open", detail: "", R: null, entryTime: cs[n - 5].time };
    if (kind === "plan") {
      res.note = "TradePlan: entry→target zone (teal), entry→stop zone (orange) from the signal bar.";
      res.levels = [{ title: "entry", price: entry }, { title: "stop", price: stop }, { title: "target", price: target }];
    }
  }
  return res;
}
