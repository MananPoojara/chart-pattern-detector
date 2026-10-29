import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import { api } from "../lib/api";
import { subscribe, useWsStatus } from "../lib/ws";
import { linkProps, navigate } from "../lib/router";
import { defaultModel, groupModels } from "../lib/models";
import { dirClass, nseSessionOpen, pct, price, signed } from "../lib/format";
import { boardIntro, heroIntro, initHomeScroll } from "../lib/motion";
import type { Board, Model } from "../lib/types";
import { Sparkline } from "../components/Sparkline";
import { useLiveQuote } from "../lib/quote";
import { SignalFeed } from "../components/SignalFeed";
import { SplitHeading } from "../components/SplitHeading";
import { ThemeToggle } from "../components/ThemeToggle";
import "../styles/home.css";

const KIND_COPY = [
  { title: "Our models", body: "TradePlan ranks every NSE 500 stock each evening and writes an entry, stop and target. PatternEdge scores each day for pattern quality. These are the only models we measured as having an edge." },
  { title: "Rule detectors", body: "Classic chart-pattern code from open-source repos: pivots, head-and-shoulders, triangles, trendlines. Useful for seeing structure. On NSE their raw signals did not beat chance." },
  { title: "Vision (YOLO)", body: "Object detectors trained on chart images draw labelled boxes around patterns they recognise. Boxes on the right edge use only past bars; older boxes are hindsight." },
  { title: "Forecasters", body: "Time-series models project a median path with a shaded range. Read the width of the range as the model's uncertainty, not as a target." },
];

export function Home() {
  const root = useRef<HTMLDivElement>(null);
  const hero = useRef<HTMLElement>(null);
  const tbody = useRef<HTMLTableSectionElement>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [boardErr, setBoardErr] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const wsOpen = useWsStatus() === "open";
  const session = nseSessionOpen();

  useEffect(() => {
    api.board().then(setBoard, () => setBoardErr(true));
    api.models().then(setModels, () => {});
    const offBars = subscribe("bars:FIVE_MINUTE", () => {}); // feeds the per-ticker store read by <LivePrice>
    const offBoard = subscribe("board", (d) => setBoard(d as Board));
    return () => { offBars(); offBoard(); };
  }, []);

  useLayoutEffect(() => heroIntro(hero.current!), []);
  useEffect(() => initHomeScroll(root.current!), []);
  const animated = useRef(false);
  useLayoutEffect(() => {
    if (!board || animated.current || !tbody.current) return;
    animated.current = true;
    boardIntro([...tbody.current.rows]);
  }, [board]);

  const plan = defaultModel(models);
  const terminalHref = (t: string) => `/t/${t}${plan ? `?model=${plan.id}` : ""}`;
  const buys = board?.rows.filter((r) => r.action === "BUY").length ?? 0;

  return (
    <div ref={root} className="home">
      <header className="nav">
        <a className="brand" {...linkProps("/")}>PatternEdge</a>
        <nav aria-label="Sections">
          <a href="#board">Board</a>
          <a href="#record">Record</a>
          <a href="#models">Models</a>
          <a {...linkProps("/yd")}>Yd</a>
        </nav>
        <div className="nav-end">
          <ThemeToggle />
          <a className="btn btn-sm btn-primary" {...linkProps(terminalHref(board?.rows[0]?.ticker ?? "RELIANCE"))}>Terminal</a>
        </div>
      </header>

      <main id="main">
        <section ref={hero} className="hero" aria-labelledby="hero-title">
          <SplitHeading as="h1" id="hero-title" text="Chart patterns, tested against the NSE." className="hero-title" />
          <div className="hero-side">
            <p className="lede" data-intro>
              Thirteen open-source pattern detectors were no better than chance on NSE 500 history. One ranking held up
              out of sample: TradePlan's top five setups a day averaged <strong>+0.07 R per trade, 2018 to 2026</strong>.
            </p>
            <div className="hero-actions" data-intro>
              <a className="btn btn-primary" {...linkProps(terminalHref(board?.rows[0]?.ticker ?? "RELIANCE"))}>
                <Icon icon="solar:chart-square-linear" width={18} aria-hidden="true" /> Open the terminal
              </a>
              <a className="btn" href="#record">How it's measured</a>
            </div>
          </div>
        </section>

        <section id="board" className="board" aria-labelledby="board-title">
          <div className="board-head">
            <h2 id="board-title">Today's board</h2>
            <p className="board-meta">
              {board ? <>Ranked from the {board.asof} close across {board.universe} stocks; {buys} of {board.rows.length} pass the trade rules.</> : boardErr ? "" : "Loading the board…"}
            </p>
            <span className="live-dot" data-live={wsOpen && session}>{session ? (wsOpen ? "Live 5-min prices" : "Connecting…") : "Market closed, last close shown"}</span>
          </div>

          {boardErr && <p className="empty">The board is unavailable because the server can't be reached. It refreshes on its own once the server is back.</p>}

          <table className="board-table">
            <caption className="sr-only">Top TradePlan setups. Select a stock to open it in the terminal.</caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Stock</th>
                <th scope="col">30 days</th>
                <th scope="col" className="num">Last</th>
                <th scope="col" className="num">Day</th>
                <th scope="col" className="num" title="Model probability the target is hit before the stop">P(target first)</th>
                <th scope="col" className="num" title="Expected return per trade in units of risk">Exp. R</th>
                <th scope="col" className="num">Entry / stop / target</th>
                <th scope="col">Plan</th>
              </tr>
            </thead>
            <tbody ref={tbody}>
              {(board?.rows ?? []).map((r) => (
                <tr key={r.ticker} onClick={() => navigate(terminalHref(r.ticker))}>
                  <td className="rank">{r.rank}</td>
                  <td className="stock">
                    <a {...linkProps(terminalHref(r.ticker))}>{r.ticker}</a>
                    <span className="group">{r.group}</span>
                  </td>
                  <td className="sparkcell"><Sparkline values={r.spark} /></td>
                  <QuoteCells ticker={r.ticker} close={r.close} chg={r.chg} asof={board!.asof} />
                  <td className="num prob" data-label="P(target first)">{(r.p_win * 100).toFixed(0)}%</td>
                  <td className="num" data-label="Exp. R">{signed(r.eR)}</td>
                  <td className="num levels" data-label="Entry / stop / target">
                    {price(r.entry)} <span className="down">{price(r.stop)}</span> <span className="up">{price(r.target)}</span>
                  </td>
                  <td><span className="action" data-buy={r.action === "BUY"}>{r.action === "BUY" ? "Buy" : "No trade"}</span></td>
                </tr>
              ))}
              {!board && !boardErr && Array.from({ length: 8 }, (_, i) => (
                <tr key={i} className="skeleton" aria-hidden="true"><td colSpan={9} /></tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="split" aria-label="Live signals and model overview">
          <div>
            <SplitHeading text="Signals as they happen" />
            <p className="section-lede">Every closed 5-minute bar is scanned across the live watchlist. Buys and sells land here within seconds.</p>
            <SignalFeed limit={10} />
          </div>
          <div>
            <SplitHeading text="What the models are" />
            <dl className="kinds">
              {KIND_COPY.map((k) => (
                <div key={k.title}><dt>{k.title}</dt><dd>{k.body}</dd></div>
              ))}
            </dl>
          </div>
        </section>

        <section id="record" className="record" aria-labelledby="record-title">
          <SplitHeading id="record-title" text="What the record says" />
          <p className="record-figure"><span className="up">+0.07 R</span> per trade</p>
          <p className="record-line">TradePlan, top five setups per day, out of sample, 2018 to 2026.</p>
          <dl className="defs">
            <div><dt>What R means</dt><dd>One R is the distance from entry to stop. Risking ₹1,000 a trade, +0.07 R is about ₹70 per trade on average, before costs.</dd></div>
            <div><dt>Out of sample</dt><dd>Measured on trades from years the model was not fitted on, using the same entry, stop and target rules the board shows.</dd></div>
            <div><dt>The detectors</dt><dd>The open-source detectors are here to look at, not to trade. Their raw signals were noise on NSE stocks.</dd></div>
            <div><dt>The limits</dt><dd>An edge this small is easily erased by slippage, brokerage or skipped trades. Past results are not a promise.</dd></div>
          </dl>
        </section>

        <section id="models" className="catalog" aria-labelledby="models-title">
          <SplitHeading id="models-title" text="Every model draws in its own style" />
          {groupModels(models).map((g) => (
            <div className="catalog-group" key={g.label}>
              <h3>{g.label}</h3>
              <ul>
                {g.models.map((m) => (
                  <li key={m.id}>
                    <a className="m-name" {...linkProps(`/t/${board?.rows[0]?.ticker ?? "RELIANCE"}?model=${m.id}`)}>{m.name}</a>
                    <span className="m-how">{m.how}</span>
                    <span className="m-meta">
                      {m.intraday ? "Daily and 5-min" : "Daily only"}
                      {m.url && <a href={m.url} target="_blank" rel="noreferrer" aria-label={`${m.name} source code`}><Icon icon="solar:code-square-linear" width={16} aria-hidden="true" /></a>}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>

        <section className="cta" aria-labelledby="cta-title">
          <h2 id="cta-title">Open any NSE 500 stock and switch models on the same chart.</h2>
          <a className="btn btn-primary" {...linkProps(terminalHref(board?.rows[0]?.ticker ?? "RELIANCE"))}>Open the terminal</a>
        </section>
      </main>

      <footer className="footer">
        <p>PatternEdge is a research tool. Nothing here is investment advice, and we are not a SEBI-registered investment adviser.</p>
        <p>Daily data from NSE; live 5-minute bars built from Angel One SmartStream ticks.</p>
      </footer>
    </div>
  );
}

/** Last + day-change cells; only these re-render per live tick. */
function QuoteCells(p: { ticker: string; close: number; chg: number; asof: string }) {
  const q = useLiveQuote(p.ticker, p.close, p.chg, p.asof);
  return (
    <>
      <td className="num last" data-live={q.live || undefined}>{price(q.last)}</td>
      <td className={`num chg ${dirClass(q.chg)}`}>{pct(q.chg)}</td>
    </>
  );
}
