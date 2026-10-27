import { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import type { PatternChart } from "@cpd/chart";
import { api } from "../lib/api";
import { subscribe, useLastBar, useWsStatus } from "../lib/ws";
import { linkProps, navigate, useLocation } from "../lib/router";
import { defaultModel } from "../lib/models";
import { dirClass, nseSessionOpen, pct, price } from "../lib/format";
import { useLiveQuote } from "../lib/quote";
import type { Board, BoardRow, Candle, Interval, LiveBar, Model, RunResult } from "../lib/types";
import { Chart } from "../components/Chart";
import { ModelPicker } from "../components/ModelPicker";
import { SymbolSearch } from "../components/SymbolSearch";
import { PaperTradeCard } from "../components/PaperTradeCard";
import { PlanDetails } from "../components/PlanDetails";
import { RunLoading } from "../components/RunLoading";
import { SignalFeed } from "../components/SignalFeed";
import { ThemeToggle } from "../components/ThemeToggle";
import "../styles/terminal.css";

type Run =
  | { status: "loading"; result: RunResult | null }
  | { status: "ok"; result: RunResult }
  | { status: "error"; result: RunResult | null; error: string };

type Tab = "chart" | "model" | "symbols";
const runKey = (m: string, t: string, i: string) => `${m}|${t}|${i}`;
const dateOf = (t: unknown) => (typeof t === "number" ? new Date(t * 1000).toISOString().slice(0, 10) : String(t).slice(0, 10));

export default function Terminal({ ticker }: { ticker: string }) {
  const { query } = useLocation();
  const interval: Interval = query.get("interval") === "5m" ? "FIVE_MINUTE" : "1D";
  const [models, setModels] = useState<Model[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [board, setBoard] = useState<Board | null>(null);
  const [run, setRun] = useState<Run>({ status: "loading", result: null });
  const [nonce, setNonce] = useState(0);
  const [tab, setTab] = useState<Tab>("chart");
  const chartRef = useRef<PatternChart | null>(null);
  const lastFinal = useRef(0); // time of the newest closed live bar for this ticker
  const staleRetries = useRef(0);

  const model = models.find((m) => m.id === query.get("model")) ?? defaultModel(models, interval === "FIVE_MINUTE");
  const unsupported = !!model && interval === "FIVE_MINUTE" && !model.intraday;

  useEffect(() => {
    api.models().then(setModels, () => setRun({ status: "error", result: null, error: "Can't load the model list from the server." }));
    api.universe().then((u) => setSymbols(u.nse500 ?? Object.values(u).flat()), () => {});
    api.board().then(setBoard, () => {});
    return subscribe("bars:FIVE_MINUTE", () => {}); // feeds live prices for the watchlist (read via useLiveQuote)
  }, []);

  useEffect(() => setTab("chart"), [ticker]); // picking a symbol on mobile jumps back to the chart
  useEffect(() => { document.title = `${ticker}${model ? ` · ${model.name}` : ""} · PatternEdge`; }, [ticker, model]);

  const setUrl = (t: string, m: string | undefined, i: Interval) => {
    const q = new URLSearchParams();
    if (i === "FIVE_MINUTE" && !models.find((x) => x.id === m)?.intraday) m = defaultModel(models, true)?.id;  // live needs an intraday model
    if (m) q.set("model", m);
    if (i === "FIVE_MINUTE") q.set("interval", "5m");
    navigate(`/t/${t}${q.size ? `?${q}` : ""}`);
  };

  // Model run: cancel stale requests; keep the previous chart visible while loading.
  useEffect(() => {
    if (!model || unsupported) return;
    const ac = new AbortController();
    setRun((r) => ({ status: "loading", result: r.result }));
    let retry: ReturnType<typeof setTimeout> | undefined;
    api.run(model.id, ticker, interval, ac.signal).then(
      (res) => {
        setRun(res.error ? { status: "error", result: res.candles?.length ? res : null, error: res.error } : { status: "ok", result: res });
        // The scanner stores a closed bar some seconds after announcing it: if this run predates it, ask again.
        const last = res.candles?.[res.candles.length - 1]?.time;
        if (typeof last === "number" && lastFinal.current > last && staleRetries.current < 6) {
          staleRetries.current++;
          retry = setTimeout(() => setNonce((n) => n + 1), 10_000);
        } else staleRetries.current = 0;
      },
      (e: Error) => { if (e.name !== "AbortError") setRun((r) => ({ status: "error", result: r.result, error: e.message })); },
    );
    return () => { ac.abort(); clearTimeout(retry); };
  }, [model?.id, ticker, interval, unsupported, nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const currentKey = model ? runKey(model.id, ticker, interval) : "";
  const result = run.result && runKey(run.result.model, run.result.ticker, run.result.interval) === currentKey ? run.result : null;
  const shownResult = unsupported ? null : (result ?? run.result);
  const stale = !result; // chart shows another stock/model while the new run loads

  // Live bars: always subscribed for the header price; pushed into the chart only in 5m mode
  // and only when the chart holds this ticker's intraday result.
  const liveKey = useRef("");
  liveKey.current = result && interval === "FIVE_MINUTE" ? currentKey : "";
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const off = subscribe(`bar:FIVE_MINUTE:${ticker}`, (d) => {
      const b = d as LiveBar;
      if (b.ticker !== ticker) return;
      if (b.final) lastFinal.current = Math.max(lastFinal.current, Number(b.time));
      if (!liveKey.current) return;
      const c: Candle = { time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume };
      chartRef.current?.pushBar(c);
      if (b.final) { clearTimeout(timer); timer = setTimeout(() => setNonce((n) => n + 1), 1500); }
    });
    return () => { off(); clearTimeout(timer); lastFinal.current = 0; staleRetries.current = 0; };
  }, [ticker]);

  const boardRow = board?.rows.find((r) => r.ticker === ticker);
  const watch = board?.rows ?? [];
  const loading = run.status === "loading" && !unsupported;

  return (
    <div className="term" data-tab={tab}>
      <header className="term-bar">
        <a className="brand" {...linkProps("/")} aria-label="PatternEdge home">PatternEdge</a>
        <Quote ticker={ticker} result={shownResult && !stale ? shownResult : null} boardRow={boardRow} asof={board?.asof} />
        <div className="term-bar-end"><ThemeToggle /></div>
      </header>

      <div className="tabs" role="tablist" aria-label="Terminal panels">
        {(["symbols", "chart", "model"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} aria-controls={`panel-${t}`} onClick={() => { setTab(t); if (t === "chart") requestAnimationFrame(() => chartRef.current?.fit()); }}>
            {t === "symbols" ? "Symbols" : t === "chart" ? "Chart" : "Model"}
          </button>
        ))}
      </div>

      <main id="main" className="term-grid">
        <aside id="panel-symbols" className="panel left" aria-label="Symbols">
          <SymbolSearch symbols={symbols} onPick={(t) => setUrl(t, model?.id, interval)} />
          <h2 className="panel-title">Today's board</h2>
          <ul className="watch">
            {watch.map((r) => (
              <li key={r.ticker}>
                <a {...linkProps(`/t/${r.ticker}${location.search}`)} aria-current={r.ticker === ticker ? "page" : undefined}>
                  <span className="w-tk">{r.ticker}</span>
                  <WatchQuote row={r} asof={board!.asof} />
                </a>
              </li>
            ))}
            {!watch.length && <li className="muted small">Board unavailable.</li>}
          </ul>
        </aside>

        <section id="panel-chart" className="panel center" aria-label={`${ticker} chart`}>
          <div className="chart-wrap" data-stale={(stale && !!shownResult) || unsupported}>
            <Chart result={shownResult} chartRef={chartRef} />
            {loading && !result && <RunLoading key={currentKey} model={model?.name ?? "model"} slowKind={model?.kind === "vision" || model?.kind === "forecast"} />}
            {loading && result && <span className="refreshing" role="status"><span className="spinner" aria-hidden="true" /> Updating with the new bar…</span>}
            {unsupported && (
              <div className="chart-msg" role="alert">
                <p><strong>{model!.name} is daily-only.</strong> It needs end-of-day bars, so it has no 5-minute view.</p>
                <button className="btn btn-sm btn-primary" onClick={() => setUrl(ticker, model!.id, "1D")}>Show daily chart</button>
              </div>
            )}
            {run.status === "error" && !run.result && !unsupported && (
              <div className="chart-msg" role="alert">
                <p><strong>This run failed.</strong> {run.error.replace(/^\w+Error: /, "")}</p>
                <button className="btn btn-sm" onClick={() => setNonce((n) => n + 1)}>Run again</button>
              </div>
            )}
          </div>
        </section>

        <aside id="panel-model" className="panel right" aria-label="Model and trade">
          <h2 className="panel-title">Model</h2>
          <ModelPicker models={models} value={model?.id} onChange={(id) => setUrl(ticker, id, interval)} />

          <div className="seg" role="radiogroup" aria-label="Interval">
            {(["1D", "FIVE_MINUTE"] as Interval[]).map((i) => (
              <label key={i} data-checked={interval === i}>
                <input type="radio" name="interval" checked={interval === i} onChange={() => setUrl(ticker, model?.id, i)} />
                {i === "1D" ? "1D" : "5m live"}
              </label>
            ))}
          </div>

          {unsupported && <p className="err small">{model!.name} is daily-only. Pick another model or switch to 1D.</p>}
          {run.status === "error" && run.result && <p className="err small" role="alert">{run.error}</p>}
          {shownResult && !stale && !unsupported && (
            <>
              {shownResult.note && <p className="note small"><Icon icon="solar:info-circle-linear" width={15} aria-hidden="true" /> {shownResult.note}</p>}
              <PaperTradeCard trade={shownResult.paperTrade} />
              {shownResult.kind === "plan" && <PlanDetails result={shownResult} />}
            </>
          )}

          <h2 className="panel-title">Live signals on {ticker}</h2>
          <SignalFeed ticker={ticker} limit={6} />
        </aside>
      </main>
    </div>
  );
}

/** Header quote: live 5m price when streaming, else last close from the run or board. */
function Quote({ ticker, result, boardRow, asof }: { ticker: string; result: RunResult | null; boardRow?: BoardRow; asof?: string }) {
  const bar = useLastBar(ticker);
  const wsOpen = useWsStatus() === "open";
  const session = nseSessionOpen();
  const { last, chg } = useMemo(() => {
    const candles = result?.candles ?? [];
    const lastC = candles[candles.length - 1];
    const lp = bar?.close ?? lastC?.close ?? boardRow?.close;
    const ref = dateOf(bar?.time ?? lastC?.time);
    let prev: number | undefined;
    for (let i = candles.length - 1; i >= 0; i--) if (dateOf(candles[i].time) < ref) { prev = candles[i].close; break; }
    if (prev == null && boardRow && asof && ref > asof) prev = boardRow.close;
    if (lp != null && prev) return { last: lp, chg: (lp / prev - 1) * 100 };
    return { last: lp, chg: boardRow && !bar ? boardRow.chg : null };
  }, [bar, result, boardRow, asof]);
  const today = new Date(Date.now() + 5.5 * 3600_000).toISOString().slice(0, 10);
  const live = wsOpen && session && !!bar && dateOf(bar.time) === today;

  return (
    <div className="quote">
      <h1 className="q-tk">{ticker}</h1>
      <span className="q-px">{price(last)}</span>
      {chg != null && <span className={`q-chg ${dirClass(chg)}`}>{pct(chg)}</span>}
      <span className="live-dot" data-live={live}>{live ? "Live" : session ? (wsOpen ? "Waiting for ticks" : "Offline") : "Market closed"}</span>
    </div>
  );
}

function WatchQuote({ row, asof }: { row: BoardRow; asof: string }) {
  const q = useLiveQuote(row.ticker, row.close, row.chg, asof);
  return (
    <>
      <span className="w-px">{price(q.last)}</span>
      <span className={`w-chg ${dirClass(q.chg)}`}>{pct(q.chg)}</span>
    </>
  );
}
