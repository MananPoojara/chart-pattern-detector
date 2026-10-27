// Dev-only fake backend, enabled with ?mock=1 (sticky for the tab). Replaces fetch for /api/*
// and WebSocket for /ws with deterministic synthetic data so the UI renders without the server.
import type { Board, Candle, Kind, Model, RunResult, Signal } from "./lib/types";

try { sessionStorage.setItem("cpd-mock", "1"); } catch { /* ignore */ }

let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 17);

const MODELS: Model[] = ([
  ["tradeplan", "TradePlan", "plan", "Ranks every stock nightly; entry, stop and target from a chart-image CNN + gradient boosting.", true],
  ["PatternEdge (ours)", "PatternEdge (ours)", "edge", "Scores each day's pattern quality; marks the top and bottom 5% of days.", false],
  ["stock-pattern", "stock-pattern", "rule", "Pivot-based detector for head-and-shoulders, double tops, triangles.", true],
  ["candlestick", "Candlestick rules", "rule", "TA-Lib style candlestick patterns: engulfing, hammer, doji.", true],
  ["harmonic", "Harmonic patterns", "rule", "XABCD Fibonacci ratios for Gartley, Bat and Crab.", true],
  ["trendlines", "Trendln", "lines", "Fits support and resistance through swing points.", true],
  ["sr-levels", "S/R levels", "lines", "Clusters swing highs and lows into horizontal levels.", true],
  ["yolo-foduucom", "YOLOv8 stock patterns", "vision", "YOLOv8 trained on chart screenshots: six pattern classes.", true],
  ["yolo-chartscan", "ChartScanAI", "vision", "YOLO detector for buy/sell setups on candlestick images.", true],
  ["chronos", "Chronos", "forecast", "Amazon's pretrained time-series transformer; median and 80% range.", false],
  ["timesfm", "TimesFM", "forecast", "Google's decoder-only forecaster, 20 bars ahead.", false],
  ["prophet", "Prophet", "forecast", "Additive trend + seasonality model.", false],
  ["lstm", "LSTM baseline", "forecast", "Small recurrent net trained per stock.", true],
  ["kronos", "Kronos", "forecast", "Foundation model for candlestick sequences.", false],
  ["vbt-patterns", "vectorbt patterns", "rule", "Rolling pattern-similarity search.", true],
] as const).map(([id, name, kind, how, intraday]) => ({ id, name, kind: kind as Kind, how, intraday, url: `https://github.com/example/${id}` }));

const UNIVERSE = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK", "KOTAKBANK",
  "HINDUNILVR", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC", "POWERGRID", "TATASTEEL", "M&M", "ADANIPORTS",
  "COALINDIA", "WIPRO", "HCLTECH", "ONGC", "JSWSTEEL", "TATAMOTORS", "DRREDDY", "CIPLA", "GRASIM", "BPCL", "EICHERMOT", "TRENT",
  "DIXON", "POLYCAB", "PERSISTENT", "COFORGE", "MPHASIS", "DMART", "IRCTC", "HAL", "BEL", "PFC", "RECLTD", "CANBK", "IDFCFIRSTB"];
const GROUPS = ["Banks", "IT", "Energy", "Metals", "Auto", "Pharma", "Capital goods", "FMCG"];

const today = new Date(Date.now() + 5.5 * 3600_000);
const iso = (d: Date) => d.toISOString().slice(0, 10);
function prevSessions(n: number): string[] {
  const out: string[] = [];
  const d = new Date(today);
  while (out.length < n) { d.setUTCDate(d.getUTCDate() - 1); if (d.getUTCDay() % 6) out.unshift(iso(d)); }
  return out;
}

function candles(ticker: string, interval: string, n = 300): Candle[] {
  seed = hash(ticker + interval) || 1;
  let p = 200 + (hash(ticker) % 3000);
  const vol = interval === "1D" ? 0.017 : 0.0025;
  const times: (string | number)[] = interval === "1D"
    ? prevSessions(n)
    : (() => { // 75 bars per session, IST wall clock as UTC epoch
        const days = prevSessions(Math.ceil(n / 75)); const t: number[] = [];
        for (const d of days) for (let i = 0; i < 75; i++) t.push(Date.parse(`${d}T09:15:00Z`) / 1000 + i * 300);
        return t.slice(-n);
      })();
  return times.map((time) => {
    const o = p, c = o * (1 + (rnd() - 0.48) * vol * 2);
    const h = Math.max(o, c) * (1 + rnd() * vol * 0.6), l = Math.min(o, c) * (1 - rnd() * vol * 0.6);
    p = c;
    return { time: time as Candle["time"], open: +o.toFixed(2), high: +h.toFixed(2), low: +l.toFixed(2), close: +c.toFixed(2), volume: Math.round(1e5 + rnd() * 9e5) };
  });
}

function board(): Board {
  const asof = prevSessions(1)[0];
  const rows = UNIVERSE.slice(0, 15).map((ticker, i) => {
    const cs = candles(ticker, "1D", 31);
    const close = cs[30].close, chg = (close / cs[29].close - 1) * 100;
    const atr = close * 0.022;
    return { ticker, rank: i + 1, group: GROUPS[hash(ticker) % GROUPS.length], p_win: 0.62 - i * 0.012, eR: 0.21 - i * 0.017,
      action: (i < 9 ? "BUY" : "NO TRADE") as "BUY" | "NO TRADE", close, chg, entry: +(close * 1.002).toFixed(2),
      stop: +(close - 1.5 * atr).toFixed(2), target: +(close + 2.5 * atr).toFixed(2), spark: cs.slice(1).map((c) => c.close) };
  });
  return { asof, rows, universe: 500 };
}

function run(modelId: string, ticker: string, interval: string): RunResult {
  const m = MODELS.find((x) => x.id === modelId) ?? MODELS[0];
  const cs = candles(ticker, interval, m.kind === "edge" ? 3000 : 300); // edge: full daily history, like the contract
  const last = cs[cs.length - 1];
  const at = (i: number) => cs[Math.max(0, cs.length - i)];
  const r: RunResult = { id: `${modelId}-${ticker}`, model: m.id, kind: m.kind, ticker, interval, asof: last.time, candles: cs,
    markers: [], boxes: [], lines: [], levels: [], forecast: null, score: null, paperTrade: null,
    note: `Mock ${m.name} output: shapes are synthetic.` } as RunResult;
  if (m.kind === "rule" || m.kind === "edge") r.markers = [40, 90, 150, 220].map((i, k) => ({ time: at(i).time, dir: (k % 2 ? -1 : 1) as 1 | -1, text: k % 2 ? "H&S" : "DB" }));
  if (m.kind === "lines") r.lines = [{ points: [{ time: at(120).time, value: at(120).low }, { time: last.time, value: at(120).low * 1.04 }], dir: 1, label: "support" }];
  if (m.kind === "vision") r.boxes = [{ start: at(60).time, end: at(30).time, top: at(45).high * 1.02, bottom: at(45).low * 0.98, label: "Head and shoulders", dir: -1, conf: 0.71, live: false },
    { start: at(18).time, end: last.time, top: last.high * 1.02, bottom: last.low * 0.97, label: "W_Bottom", dir: 1, conf: 0.64, live: true }];
  if (m.kind === "edge") r.score = cs.map((c, i) => ({ time: c.time, value: Math.sin(i / 9) * 0.5 + (rnd() - 0.5) * 0.3 }));
  if (m.kind === "edge") Object.assign(r, edgeAnalogs(cs));
  if (m.kind === "forecast") {
    const step = interval === "1D" ? 86400 : 300;
    const t0 = typeof last.time === "number" ? last.time : Date.parse(String(last.time)) / 1000;
    const pts = (f: (k: number) => number) => Array.from({ length: 20 }, (_, k) => ({ time: (interval === "1D" ? iso(new Date((t0 + k * step) * 1000)) : t0 + k * step) as Candle["time"], value: +f(k).toFixed(2) }));
    r.forecast = { median: pts((k) => last.close * (1 + k * 0.001)), lo: pts((k) => last.close * (1 - k * 0.004)), hi: pts((k) => last.close * (1 + k * 0.006)) };
  }
  const entry = at(6).close, risk = entry * 0.03;
  r.levels = [{ title: "entry", price: entry }, { title: "stop", price: entry - risk }, { title: "target", price: entry + risk * 1.8 }];
  r.paperTrade = { signal: at(7).time, why: "Bullish engulfing above the 50-day average with rising volume.", entry, stop: +(entry - risk).toFixed(2),
    target: +(entry + risk * 1.8).toFixed(2), status: "open", detail: "Filled next bar at the open; neither stop nor target touched yet.",
    R: +((last.close - entry) / risk).toFixed(2), entryTime: at(6).time };
  if (m.kind === "plan") {
    r.stats = { p_target_first: 0.58, expected_R: 0.14, rank_bucket: "top 5" };
    r.drivers = [{ name: "Chart-image CNN score", share: 0.34, value: 0.81 }, { name: "20-day momentum", share: 0.21, value: 0.062 },
      { name: "Volume vs 50-day", share: 0.15, value: 1.7 }, { name: "Distance to 52w high", share: -0.09, value: -0.04 }, { name: "Sector breadth", share: 0.07, value: 0.58 }];
  }
  return r;
}

/** CONTRACT "PatternEdge analogs": query box, past windows, cone from their paths, stats, flags. Synthetic numbers. */
function edgeAnalogs(cs: Candle[]): Partial<RunResult> {
  const W = 60, H = 20, n = cs.length, last = cs[n - 1];
  const box = (a: number, b: number) => { const w = cs.slice(a, b + 1); return { top: Math.max(...w.map((c) => c.high)), bottom: Math.min(...w.map((c) => c.low)) }; };
  const ends = [2280, 1510, 2790, 640, 1990, 1120, 2560, 870]; // non-overlapping, all <= n - 1 - H
  const PAT = [["Double bottom"], [], ["Ascending triangle", "Bull flag"], ["Head and shoulders"], [], ["Cup and handle"], ["Bull flag"], []];
  const analogs = ends.map((e, k) => {
    const fwdPath = cs.slice(e, e + H + 1).map((c) => c.close / cs[e].close - 1);
    const fwd = fwdPath[H];
    return { start: cs[e - W + 1].time, end: cs[e].time, sim: +(0.93 - k * 0.035).toFixed(2), ctx: +(0.35 + ((k * 37) % 55) / 100).toFixed(2),
      fwd: +fwd.toFixed(4), mfe: +Math.max(...fwdPath).toFixed(4), mae: +Math.min(...fwdPath).toFixed(4),
      outcome: (fwd > 0.05 ? "target" : fwd < -0.03 ? "stop" : "time") as "target" | "stop" | "time",
      ma50Slope: +(fwd > 0 ? 0.5 + k * 0.1 : -0.2 + k * 0.05).toFixed(2),
      results: k % 3 === 2 ? null : { q: `Q${(k % 4) + 1}`, salesYoY: +(0.04 + k * 0.02).toFixed(2), profitYoY: k === 4 ? null : +(0.12 - k * 0.03).toFixed(2) },
      patterns: PAT[k], path: fwdPath };
  });
  const q = (p: number) => Array.from({ length: H + 1 }, (_, i) => { const v = analogs.map((a) => a.path[i]).sort((x, y) => x - y); return v[Math.round(p * (v.length - 1))]; });
  const fut: string[] = []; for (let d = new Date(String(last.time) + "T00:00:00Z"); fut.length < H; ) { d.setUTCDate(d.getUTCDate() + 1); if (d.getUTCDay() % 6) fut.push(iso(d)); }
  const ts = [String(last.time), ...fut];
  const pts = (p: number) => q(p).map((v, i) => ({ time: ts[i], value: +(last.close * (1 + v)).toFixed(2) }));
  const up = analogs.filter((a) => a.fwd > 0).length, med = q(0.5)[H];
  const pc = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(1)}%`;
  return {
    analogs: analogs.map(({ path: _p, ...a }) => a),
    boxes: [{ start: cs[n - W].time, end: last.time, ...box(n - W, n - 1), label: `${up}/${analogs.length} analogs up · ${pc(med)} / ${H}d`, dir: med >= 0 ? 1 : -1, conf: 1, live: true },
      ...analogs.map((a, k) => ({ start: a.start, end: a.end, ...box(ends[k] - W + 1, ends[k]), label: `${String(a.end).slice(0, 7)} · ${pc(a.fwd)}`, dir: (a.fwd >= 0 ? 1 : -1) as 1 | -1, conf: a.sim, live: false }))],
    forecast: { median: pts(0.5), lo: pts(0.1), hi: pts(0.9) },
    stats: { analog_n: analogs.length, analog_p_up: up / analogs.length, analog_median_fwd: +med.toFixed(4), analog_ctx_p_up: 0.62, ma50_slope_now: 0.0, ma50_slope_analogs: 0.8 },
    flags: [`50-DMA flat now (+0.0%); in the ${up} analogs that went up it was rising (median +0.8%/10 bars)`, "NIFTY below its 50-DMA now; above it in 6 of 8 analogs"],
  };
}

const SIGNAL_MODELS = ["yolo-chartscan", "stock-pattern", "candlestick"];
const mkSignal = (i: number): Signal => {
  const ticker = UNIVERSE[(i * 7) % UNIVERSE.length], close = 200 + (hash(ticker) % 3000);
  const t = Date.parse(`${iso(today)}T09:15:00Z`) / 1000 + (i % 70) * 300;
  return { id: `s${i}-${Date.now()}`, time: t as Signal["time"], ticker, interval: "FIVE_MINUTE", model: SIGNAL_MODELS[i % 3],
    signal: ["Bull flag breakout", "Double bottom", "Bullish engulfing", "Bearish engulfing"][i % 4], dir: i % 4 === 3 ? -1 : 1,
    close, stop: +(close * 0.985).toFixed(2), target: +(close * 1.03).toFixed(2) };
};

const json = (body: unknown, delay = 120) => new Promise<Response>((r) => setTimeout(() => r(new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } })), delay));
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, location.origin);
  if (!url.pathname.startsWith("/api/")) return realFetch(input, init);
  const q = url.searchParams;
  switch (url.pathname) {
    case "/api/health": return json({ ok: true, redis: true, scanner_age_s: 2, serve_age_s: 1 });
    case "/api/models": return json(MODELS);
    case "/api/universe": return json({ nse500: UNIVERSE });
    case "/api/board": return json(board());
    case "/api/signals": return json(Array.from({ length: 14 }, (_, i) => mkSignal(40 - i)));
    case "/api/run": {
      const m = MODELS.find((x) => x.id === q.get("model"));
      if (q.get("interval") !== "1D" && m && !m.intraday) return json({ error: `${m.name} is daily-only` });
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => json(run(q.get("model")!, q.get("ticker")!, q.get("interval")!), 0).then(resolve), 900);
        init?.signal?.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("aborted", "AbortError")); });
      });
    }
  }
  return json({ error: "not mocked" });
};

class MockSocket extends EventTarget {
  static readonly OPEN = 1;
  readyState = 0;
  onopen: ((e: Event) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: ((e: Event) => void) | null = null;
  private subs = new Set<string>();
  private timer: ReturnType<typeof setInterval>;
  private live = new Map<string, Candle & { n: number }>();
  private n = 0;
  constructor(_url: string) {
    super();
    setTimeout(() => { this.readyState = 1; this.onopen?.(new Event("open")); }, 50);
    this.timer = setInterval(() => this.tick(), 1000);
  }
  send(raw: string) {
    const { op, topics } = JSON.parse(raw) as { op: string; topics: string[] };
    topics.forEach((t) => (op === "sub" ? this.subs.add(t) : this.subs.delete(t)));
  }
  close() { clearInterval(this.timer); }
  private emit(topic: string, data: unknown) { this.onmessage?.(new MessageEvent("message", { data: JSON.stringify({ topic, data }) })); }
  private tick() {
    this.n++;
    const tickers = new Set<string>();
    for (const t of this.subs) {
      if (t === "bars:FIVE_MINUTE") UNIVERSE.slice(0, 15).forEach((x) => tickers.add(x));
      else if (t.startsWith("bar:FIVE_MINUTE:")) tickers.add(t.split(":")[2]);
    }
    for (const ticker of tickers) {
      let b = this.live.get(ticker);
      if (!b) {
        const cs = candles(ticker, "FIVE_MINUTE", 300); const last = cs[cs.length - 1];
        b = { ...last, time: (last.time as number) + 300, open: last.close, high: last.close, low: last.close, n: 0 };
      }
      const c = +(b.close * (1 + (Math.random() - 0.5) * 0.002)).toFixed(2);
      b = { ...b, close: c, high: Math.max(b.high, c), low: Math.min(b.low, c), n: b.n + 1 };
      const final = b.n >= 20;
      const msg = { ticker, interval: "FIVE_MINUTE", final, time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume };
      if (this.subs.has("bars:FIVE_MINUTE")) this.emit("bars:FIVE_MINUTE", msg);
      if (this.subs.has(`bar:FIVE_MINUTE:${ticker}`)) this.emit(`bar:FIVE_MINUTE:${ticker}`, msg);
      this.live.set(ticker, final ? { ...b, time: (b.time as number) + 300, open: c, high: c, low: c, n: 0 } : b);
    }
    if (this.subs.has("signals") && this.n % 8 === 0) this.emit("signals", mkSignal(this.n));
  }
}
window.WebSocket = MockSocket as unknown as typeof WebSocket;
