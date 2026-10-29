import { useEffect, useMemo, useRef, useState } from "react";
import type { PatternChart } from "@cpd/chart";
import { api } from "../lib/api";
import { linkProps, navigate, useLocation } from "../lib/router";
import { dirClass, fmtTime, nseSessionOpen, pct, price, signed } from "../lib/format";
import type { RunResult, YdScreen, YdTrade } from "../lib/types";
import { Chart } from "../components/Chart";
import { PaperTradeCard } from "../components/PaperTradeCard";
import { SymbolSearch } from "../components/SymbolSearch";
import { ThemeToggle } from "../components/ThemeToggle";
import "../styles/terminal.css";

// URL value -> Angel One interval. The strategy runs on every timeframe.
const TF = { "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "1h": "ONE_HOUR", "1D": "1D" } as const;
type Tf = keyof typeof TF;
type Tab = "chart" | "model" | "symbols";

/** Mirror of cpd/yd.py costs(): ₹ fees for one intraday round trip (Angel One brokerage, NSE/SEBI/stamp/GST, Oct 2026). */
function fees(q: number, buy: number, sell: number) {
  const bv = q * buy, sv = q * sell;
  const brk = Math.min(20, 0.001 * bv) + Math.min(20, 0.001 * sv), reg = (0.0000297 + 0.000001) * (bv + sv);
  return brk + 0.00025 * sv + reg + 0.00003 * bv + 0.18 * (brk + reg);
}
const xv = (n: unknown) => (typeof n === "number" ? `${n.toFixed(2)}x` : "—");
const num = (n: unknown, d = 2) => (typeof n === "number" ? n.toFixed(d) : n == null ? "—" : String(n));

/** Yd: F&O stocks on every timeframe, data from the Yd Angel One account, with the Yd strategy (cpd/yd.py). */
export default function Yd({ ticker }: { ticker: string }) {
  const { query } = useLocation();
  const tf: Tf = (query.get("tf") as Tf) in TF ? (query.get("tf") as Tf) : "15m";
  const [fo, setFo] = useState<string[]>([]);
  const [res, setRes] = useState<RunResult | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("chart");
  const [filter, setFilter] = useState("");
  const [side, setSide] = useState<"screen" | "fo">("screen");
  const [screen, setScreen] = useState<YdScreen | null>(null);
  const chartRef = useRef<PatternChart | null>(null);

  // Screen: load once, then 20 s after every 5-minute bar close while the session is open (the Yd worker's live bars).
  useEffect(() => {
    let t: ReturnType<typeof setTimeout>;
    const load = () => api.ydScreen().then((s) => { if (!s.error) setScreen(s); }, () => {});
    const next = () => { t = setTimeout(() => { if (nseSessionOpen(new Date(Date.now() - 60_000))) load(); next(); }, 300_000 - (Date.now() % 300_000) + 20_000); };
    load(); next();
    return () => clearTimeout(t);
  }, []);

  useEffect(() => { api.universe().then((u) => setFo(u.fo ?? []), () => setErr("Can't load the F&O list from the server.")); }, []);
  useEffect(() => { document.title = `${ticker} · ${tf} · Yd`; setTab("chart"); }, [ticker, tf]);

  // Load, then refresh once a minute while the market is open (the server tops up stale candles from Angel One).
  useEffect(() => {
    const ac = new AbortController();
    const load = () => {
      setLoading(true);
      api.candles(ticker, TF[tf], ac.signal).then(
        (r) => { setLoading(false); if (r.error) setErr(r.error.replace(/^\w+Error: /, "")); else { setErr(""); setRes(r); } },
        (e: Error) => { if (e.name !== "AbortError") { setLoading(false); setErr(e.message); } },
      );
    };
    load();
    const id = setInterval(() => { if (nseSessionOpen() && tf !== "1D") load(); }, 60_000);
    return () => { ac.abort(); clearInterval(id); };
  }, [ticker, tf]);

  const go = (t: string, f: Tf = tf) => navigate(`/yd/${t}${f === "15m" ? "" : `?tf=${f}`}`);
  const shown = res && res.ticker === ticker && res.interval === TF[tf] ? res : null;
  const list = useMemo(() => fo.filter((t) => t.includes(filter.trim().toUpperCase())), [fo, filter]);
  const last = shown?.candles.at(-1);
  const prev = shown?.candles.at(-2);

  return (
    <div className="term" data-tab={tab}>
      <header className="term-bar">
        <a className="brand" {...linkProps("/")} aria-label="PatternEdge home">PatternEdge</a>
        <div className="quote">
          <h1 className="q-tk">{ticker}</h1>
          <span className="q-px">{price(last?.close)}</span>
          {last && prev && <span className={`q-chg ${dirClass(last.close - prev.close)}`}>{pct((last.close / prev.close - 1) * 100)}</span>}
          <span className="live-dot" data-live={nseSessionOpen() && tf !== "1D"}>Yd · {tf}{nseSessionOpen() && tf !== "1D" ? " · refresh 1 min" : ""}</span>
        </div>
        <div className="term-bar-end"><a className="btn btn-sm" {...linkProps(`/t/${ticker}`)}>Terminal</a> <ThemeToggle /></div>
      </header>

      <div className="tabs" role="tablist" aria-label="Yd panels">
        {(["symbols", "chart", "model"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} aria-controls={`panel-${t}`} onClick={() => { setTab(t); if (t === "chart") requestAnimationFrame(() => chartRef.current?.fit()); }}>
            {t === "symbols" ? "F&O" : t === "chart" ? "Chart" : "Strategy"}
          </button>
        ))}
      </div>

      <main id="main" className="term-grid">
        <aside id="panel-symbols" className="panel left" aria-label="F&O stocks">
          <SymbolSearch symbols={fo} onPick={(t) => go(t)} />
          <div className="seg yd-side" role="radiogroup" aria-label="List">
            {(["screen", "fo"] as const).map((k) => (
              <label key={k} data-checked={side === k}><input type="radio" name="yd-side" checked={side === k} onChange={() => setSide(k)} />{k === "screen" ? "Screen" : "F&O"}</label>
            ))}
          </div>
          {side === "screen" ? (
            <>
              <h2 className="panel-title">By VS_time {screen && <span className="muted">· {screen.session.slice(5)} {screen.slot}</span>}</h2>
              <ul className="watch">
                {screen?.rows.slice(0, 25).map((r) => (
                  <li key={r.ticker}>
                    <a {...linkProps(`/yd/${r.ticker}${location.search}`)} aria-current={r.ticker === ticker ? "page" : undefined}>
                      <span className="w-tk">{r.rank && r.rank <= 5 ? <span className="yd-rank">{r.rank}</span> : null}{r.ticker}</span>
                      <span className={r.vs_time != null && r.vs_time >= 2 ? "up" : "muted"}>{xv(r.vs_time)}</span>
                      <span className="yd-why">{r.ok ? (r.rank && r.rank <= 5 ? "tradeable" : "passes filters") : r.why}{r.results ? " · results day" : ""}</span>
                    </a>
                  </li>
                ))}
                {!screen && <li className="muted small">Loading screen…</li>}
              </ul>
              {screen && (
                <p className="small muted">
                  {screen.covered}/{screen.universe} F&amp;O stocks with 5m bars. Top 5 passing price ₹100–5,000, 20-day turnover ≥ ₹50 cr,
                  ASM/GSM {screen.lists.asm && screen.lists.gsm ? "(NSE lists)" : "(NSE lists unavailable: unknown)"} and 2% price-band
                  filters ({screen.lists.band}). Refreshes each 5-minute bar while the market is open.
                </p>
              )}
            </>
          ) : (
            <>
              <h2 className="panel-title">F&amp;O stocks <span className="muted">({fo.length})</span></h2>
              <label className="sr-only" htmlFor="yd-filter">Filter the list</label>
              <input id="yd-filter" className="yd-filter" placeholder="Filter list" value={filter} onChange={(e) => setFilter(e.target.value)} spellCheck={false} autoComplete="off" />
              <ul className="watch">
                {list.map((t) => (
                  <li key={t}><a {...linkProps(`/yd/${t}${location.search}`)} aria-current={t === ticker ? "page" : undefined}><span className="w-tk">{t}</span></a></li>
                ))}
                {!fo.length && <li className="muted small">Loading F&amp;O list…</li>}
              </ul>
            </>
          )}
        </aside>

        <section id="panel-chart" className="panel center" aria-label={`${ticker} chart`}>
          <div className="chart-wrap" data-stale={!shown && !!res}>
            <Chart result={shown ?? res} chartRef={chartRef} />
            {loading && !shown && <span className="refreshing" role="status"><span className="spinner" aria-hidden="true" /> Loading {ticker} {tf}…</span>}
            {err && !shown && (
              <div className="chart-msg" role="alert"><p><strong>No candles.</strong> {err}</p></div>
            )}
          </div>
        </section>

        <aside id="panel-model" className="panel right" aria-label="Timeframe and strategy">
          <h2 className="panel-title">Timeframe</h2>
          <div className="seg seg-6" role="radiogroup" aria-label="Timeframe">
            {(Object.keys(TF) as Tf[]).map((f) => (
              <label key={f} data-checked={tf === f}>
                <input type="radio" name="tf" checked={tf === f} onChange={() => go(ticker, f)} />
                {f}
              </label>
            ))}
          </div>
          {err && shown && <p className="err small" role="alert">{err}</p>}

          <h2 className="panel-title">Strategy · volume shock + opening range</h2>
          {shown && <Strategy r={shown} />}
          <p className="note small">Intraday candles come from the Yd Angel One account (finished bars; topped up each minute while the market is open). 1D is the NSE daily store. Rules are the user's design, not backtested.</p>
        </aside>
      </main>
    </div>
  );
}

function Strategy({ r }: { r: RunResult }) {
  const st = r.stats ?? {};
  const trades = r.trades ?? [];
  const latest = [...trades].reverse().find((t) => t.status !== "skip");
  return (
    <>
      <section className="card" aria-label="Volume shock">
        <dl className="stats">
          <div><dt>VS_time</dt><dd className={typeof st.VS_time === "number" && st.VS_time >= 2 ? "up" : ""}>{xv(st.VS_time)}</dd></div>
          <div><dt>VS_raw (log)</dt><dd>{xv(st.VS_raw)}</dd></div>
          <div><dt>VS_EOD {st.VS_EOD_day ? String(st.VS_EOD_day).slice(5) : ""}</dt><dd>{xv(st.VS_EOD)}</dd></div>
          {r.interval !== "1D" && <>
            <div><dt>Screen rank {st.screen_slot ? `· ${st.screen_slot}` : ""}</dt><dd>{num(st.rank, 0)}{typeof st.rank === "number" && st.rank <= 5 ? " · top 5" : ""}</dd></div>
            <div><dt>Turnover 20d</dt><dd>{typeof st.turnover20_cr === "number" ? `₹${st.turnover20_cr.toFixed(0)} cr` : "—"}</dd></div>
            <div><dt>Results day</dt><dd>{num(st.results_day)}</dd></div>
            <div><dt>Cost in R</dt><dd>{num(st.cost_R)}</dd></div>
            <div><dt>Break-even win</dt><dd>{typeof st.breakeven === "number" ? `${(st.breakeven * 100).toFixed(0)}%` : "—"}</dd></div>
            <div><dt>+ slippage</dt><dd>{typeof st.breakeven_slip === "number" ? `${(st.breakeven_slip * 100).toFixed(0)}%` : "—"}</dd></div>
          </>}
        </dl>
        {st.vs_note && <p className="small err">{st.vs_note}</p>}
        {r.interval !== "1D" && <p className="small muted">Universe: {num(st.universe)}. Costs: {num(st.cost_basis)}; break-even at 1.5R.</p>}
        {r.note && <p className="small muted">{r.note}</p>}
      </section>
      {r.interval !== "1D" && (r.paperTrade ? <PaperTradeCard trade={r.paperTrade} /> : <p className="small muted">No setup on the latest session.</p>)}
      {trades.length > 0 && (
        <section className="card" aria-label="Setups">
          <h3>Setups <span className="muted">({trades.length}, newest first)</span></h3>
          <ul className="yd-trades">
            {[...trades].reverse().slice(0, 12).map((t, i) => <TradeRow key={i} t={t} />)}
          </ul>
        </section>
      )}
      <SizeCalc seed={latest ?? (r.candles.length ? { entry: r.candles.at(-1)!.close, stop: r.candles.at(-1)!.close * 0.99 } : undefined)} />
      <section className="card" aria-label="Account rules">
        <h3>Account rules</h3>
        <ul className="small yd-rules">
          <li>Only the top 5 by VS_time; entries 09:30–11:00 (A), B by 13:30.</li>
          <li>Max 2 trades a day; stop for the day after 2 consecutive losses or a −1% day.</li>
          <li>Risk 0.5% of equity per trade; position ≤ 80% of equity (no MIS leverage).</li>
          <li>No averaging down, no widening stops, flat by 15:15.</li>
        </ul>
      </section>
    </>
  );
}

function TradeRow({ t }: { t: YdTrade }) {
  return (
    <li data-skip={t.status === "skip"}>
      <span>{fmtTime(t.signal).slice(5)}</span>
      <span>{t.setup} {t.side > 0 ? "long" : "short"} · {t.vs_time.toFixed(1)}x · {t.status === "closed" ? t.reason : t.status === "skip" ? t.reason : t.status}</span>
      <span className={t.R == null ? "muted" : dirClass(t.R)}>{t.R == null ? "—" : `${signed(t.R, 1)}R`}</span>
    </li>
  );
}

function SizeCalc({ seed }: { seed?: { entry: number; stop: number } }) {
  const [eq, setEq] = useState("50000");
  const [risk, setRisk] = useState("0.5");
  const [entry, setEntry] = useState("");
  const [stop, setStop] = useState("");
  useEffect(() => { if (seed) { setEntry(seed.entry.toFixed(2)); setStop(seed.stop.toFixed(2)); } }, [seed?.entry, seed?.stop]); // eslint-disable-line react-hooks/exhaustive-deps
  const E = +eq, O = +entry, S = +stop, R = Math.abs(O - S), dir = O >= S ? 1 : -1;
  const ok = E > 0 && O > 0 && S > 0 && R > 0 && +risk > 0;
  const qr = ok ? Math.floor((E * +risk) / 100 / R) : 0, qc = ok ? Math.floor((0.8 * E) / O) : 0, Q = Math.min(qr, qc);
  const T = O + dir * 1.5 * R;
  const f = Q ? fees(Q, dir > 0 ? O : T, dir > 0 ? T : O) : 0, slip = 0.0003 * Q * (O + T);
  const cR = Q ? f / (Q * R) : 0, cRs = Q ? (f + slip) / (Q * R) : 0;
  const field = (id: string, lab: string, v: string, set: (s: string) => void) => (
    <label htmlFor={id}>{lab}<input id={id} inputMode="decimal" value={v} onChange={(e) => set(e.target.value)} /></label>
  );
  return (
    <section className="card" aria-label="Position size">
      <h3>Position size</h3>
      <div className="yd-calc">
        {field("yd-eq", "Equity ₹", eq, setEq)}{field("yd-risk", "Risk %", risk, setRisk)}
        {field("yd-entry", "Entry", entry, setEntry)}{field("yd-stop", "Stop", stop, setStop)}
      </div>
      {ok ? (
        <dl className="stats">
          <div><dt>Qty ({dir > 0 ? "long" : "short"})</dt><dd>{Q}</dd></div>
          <div><dt>Binding</dt><dd>{qr <= qc ? "risk" : "80% cap"}</dd></div>
          <div><dt>Target 1.5R</dt><dd>{price(T)}</dd></div>
          <div><dt>Risk ₹</dt><dd>{(Q * R).toFixed(0)}</dd></div>
          <div><dt>Fees ₹</dt><dd>{f.toFixed(2)}</dd></div>
          <div><dt>Stop %</dt><dd>{((R / O) * 100).toFixed(2)}%</dd></div>
          <div><dt>Cost in R</dt><dd>{cR.toFixed(2)}</dd></div>
          <div><dt>Break-even</dt><dd>{(((1 + cR) / 2.5) * 100).toFixed(0)}%</dd></div>
          <div><dt>+ slip 0.03%</dt><dd>{(((1 + cRs) / 2.5) * 100).toFixed(0)}%</dd></div>
        </dl>
      ) : <p className="small muted">Enter equity, risk, entry and stop.</p>}
      {ok && (R / O < 0.006 || R / O > 0.015) && <p className="small err">Stop is outside the 0.6–1.5% band: the strategy skips this setup.</p>}
    </section>
  );
}
