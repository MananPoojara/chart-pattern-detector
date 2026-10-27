import { createPatternChart, type Kind } from "../src";
import { fake, rng } from "./fake";

const chart = createPatternChart(document.getElementById("chart")!, { theme: "dark" });
const state = { kind: "vision" as Kind, intraday: false, theme: "dark" as "dark" | "light", live: true };
let res = fake(state.kind, state.intraday);
const bar = document.getElementById("bar")!;
(window as unknown as { chart: unknown }).chart = chart;

function load() {
  const t0 = performance.now();
  res = fake(state.kind, state.intraday);
  chart.setResult(res);
  document.getElementById("note")!.textContent = `${res.note}  ·  setResult ${(performance.now() - t0).toFixed(0)} ms (incl. data gen)`;
  draw();
}

function btn(label: string, on: boolean, fn: () => void) {
  const b = document.createElement("button");
  b.textContent = label; b.setAttribute("aria-pressed", String(on)); b.onclick = fn;
  bar.append(b);
}
function sep() { const s = document.createElement("span"); s.className = "sep"; bar.append(s); }
function draw() {
  bar.replaceChildren();
  for (const k of ["rule", "lines", "vision", "forecast", "edge", "plan"] as Kind[]) btn(k, state.kind === k, () => { state.kind = k; load(); });
  sep();
  btn("daily", !state.intraday, () => { state.intraday = false; load(); });
  btn("5-min", state.intraday, () => { state.intraday = true; load(); });
  sep();
  btn(state.theme, false, () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    document.body.classList.toggle("light", state.theme === "light");
    chart.setTheme(state.theme); draw();
  });
  btn("live", state.live, () => { state.live = !state.live; draw(); });
  btn("fit", false, () => chart.fit());
}

// live ticks: wiggle the forming bar; every 20 ticks start a new one
const r = rng(99);
let ticks = 0;
setInterval(() => {
  if (!state.live) return;
  const cs = res.candles, last = { ...cs[cs.length - 1] };
  if (++ticks % 20 === 0) {
    const t = last.time;
    const next = typeof t === "number" ? t + 300 : (() => { const d = new Date(t + "T00:00:00Z"); do d.setUTCDate(d.getUTCDate() + 1); while (d.getUTCDay() % 6 === 0); return d.toISOString().slice(0, 10); })();
    const c = { time: next, open: last.close, high: last.close, low: last.close, close: last.close, volume: 0 };
    cs.push(c); chart.pushBar(c);
    return;
  }
  last.close *= 1 + (r() - 0.5) * 0.004;
  last.high = Math.max(last.high, last.close); last.low = Math.min(last.low, last.close); last.volume += 2000;
  cs[cs.length - 1] = last; chart.pushBar(last);
}, 250);

load();
