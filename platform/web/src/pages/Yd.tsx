import { useEffect, useMemo, useRef, useState } from "react";
import type { PatternChart } from "@cpd/chart";
import { api } from "../lib/api";
import { linkProps, navigate, useLocation } from "../lib/router";
import { dirClass, nseSessionOpen, pct, price } from "../lib/format";
import type { RunResult } from "../lib/types";
import { Chart } from "../components/Chart";
import { SymbolSearch } from "../components/SymbolSearch";
import { ThemeToggle } from "../components/ThemeToggle";
import "../styles/terminal.css";

// URL value -> Angel One interval. The strategy runs on every timeframe.
const TF = { "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "1h": "ONE_HOUR", "1D": "1D" } as const;
type Tf = keyof typeof TF;
type Tab = "chart" | "model" | "symbols";

/** Yd: F&O stocks on every timeframe, data from the Yd Angel One account. Strategy panel comes later. */
export default function Yd({ ticker }: { ticker: string }) {
  const { query } = useLocation();
  const tf: Tf = (query.get("tf") as Tf) in TF ? (query.get("tf") as Tf) : "15m";
  const [fo, setFo] = useState<string[]>([]);
  const [res, setRes] = useState<RunResult | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("chart");
  const [filter, setFilter] = useState("");
  const chartRef = useRef<PatternChart | null>(null);

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
          <h2 className="panel-title">F&amp;O stocks <span className="muted">({fo.length})</span></h2>
          <label className="sr-only" htmlFor="yd-filter">Filter the list</label>
          <input id="yd-filter" className="yd-filter" placeholder="Filter list" value={filter} onChange={(e) => setFilter(e.target.value)} spellCheck={false} autoComplete="off" />
          <ul className="watch">
            {list.map((t) => (
              <li key={t}><a {...linkProps(`/yd/${t}${location.search}`)} aria-current={t === ticker ? "page" : undefined}><span className="w-tk">{t}</span></a></li>
            ))}
            {!fo.length && <li className="muted small">Loading F&amp;O list…</li>}
          </ul>
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

          <h2 className="panel-title">Strategy</h2>
          <div className="card yd-strategy">
            <p><strong>Not connected yet.</strong></p>
            <p className="muted small">The Yd strategy will run here on every timeframe and draw its entries, stops and targets on this chart.</p>
          </div>
          <p className="note small">Intraday candles come from the Yd Angel One account (finished bars; topped up each minute while the market is open). 1D is the NSE daily store.</p>
        </aside>
      </main>
    </div>
  );
}
