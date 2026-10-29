# PatternEdge platform — interface contract

Every component codes against this file. Change it only by agreement (tell the lead).

## Processes

```
Angel One SmartStream ──ticks──> [py] scripts/live_scan.py ──┐  bars, signals (pub/sub + keys)
NSE daily store / intraday store ─> [py] cpd/serve.py ───────┤  jobs: BLPOP cpd:jobs -> cpd:result:<id>
                                                             ▼
                                                  Redis 127.0.0.1:6380 (docker cpd-redis, no persistence, 128 MB)
                                                             ▼
                                   [bun] platform/server  (REST /api/*, WebSocket /ws, serves platform/web/dist)
                                                             ▼
                                   [react] platform/web   (Vite, lightweight-charts v5, no SSR)
```

- Models stay in Python (YOLO / sklearn / torch). Bun never runs models; it relays.
- Machine is shared with other workloads: Python jobs run `nice -n 19`, 2 threads (`CPD_THREADS=2`, `OMP_NUM_THREADS=2`).
- Redis key prefix: `cpd:`. Never touch the other Redis on 6379.
- Ports: Bun server **8787** (all interfaces). Vite dev **5174** proxies `/api` and `/ws` to 8787.
- Python env: `source ~/.cache/chart-pattern-detector/env.sh`, interpreter `~/.cache/chart-pattern-detector/scratch-venv/bin/python`
  (has pandas, torch, ultralytics, redis, websocket-client). Run from the repo root (`/home/server/cpd-platform`).

## Time and intervals

- `interval`: `"1D"` | `"FIVE_MINUTE"` | `"FIFTEEN_MINUTE"`.
- `time` in every candle/marker/point:
  - daily: `"YYYY-MM-DD"` string
  - intraday: integer epoch seconds of the **IST wall clock treated as UTC** (so the chart axis shows IST). Python: `int(pd.Timestamp(naive_ist).timestamp())`.

## Shared shapes (JSON)

```ts
type Candle = { time: Time; open: number; high: number; low: number; close: number; volume: number };
type Kind = "rule" | "lines" | "vision" | "forecast" | "edge" | "plan";
type Model = { id: string; name: string; kind: Kind; url: string; how: string; intraday: boolean };

type Marker = { time: Time; dir: 1 | -1 | 0; text: string };           // arrow below (1) / above (-1) / dot (0)
type Box    = { start: Time; end: Time; top: number; bottom: number; label: string; dir: 1|-1|0; conf: number; live: boolean };
type Line   = { points: { time: Time; value: number }[]; dir: 1|-1|0; label?: string };
type Level  = { title: "entry" | "stop" | "target" | string; price: number };
type Forecast = { median: Pt[]; lo: Pt[]; hi: Pt[] };                    // Pt = { time, value }; starts at last candle
type Score  = Pt[];                                                     // edge: score per bar (pane below price)
type PaperTrade = { signal: Time; why: string; entry: number; stop: number; target: number;
                    status: string; detail: string; R: number | null; entryTime: Time | null };

type RunResult = {
  id: string; model: string; kind: Kind; ticker: string; interval: string; asof: Time;
  candles: Candle[];               // last `bars` bars (default 400)
  markers: Marker[]; boxes: Box[]; lines: Line[]; levels: Level[];
  forecast: Forecast | null; score: Score | null;
  paperTrade: PaperTrade | null;   // latest bullish signal in the last `days` bars (default 10), TradePlan rules
  note: string;                    // one sentence: how to read this model's marks
  stats?: Record<string, number | string>;   // e.g. TradePlan p_target_first, expected_R, rank_bucket
  drivers?: { name: string; share: number; value: number }[];   // TradePlan "why"
  error?: string;
};

type Signal = { id: string; time: Time; ticker: string; interval: string; model: string; signal: string;
                dir: 1|-1; close: number; stop: number; target: number };

type BoardRow = { ticker: string; rank: number; group: string; p_win: number; eR: number; action: "BUY" | "NO TRADE";
                  close: number; chg: number; entry: number; stop: number; target: number; spark: number[] };  // spark = last 30 closes
type Board = { asof: string; rows: BoardRow[]; universe: number };      // top 15 TradePlan setups
```

Model style per kind (frontend renders by `kind`, not by model id):
- `rule`   → small arrows (markers) + for stock-pattern pivot polylines (`lines`)
- `lines`  → support (dir 1) / resistance (dir -1) trendlines, no markers
- `vision` → YOLO-style labelled rectangles (`boxes`); `live: true` boxes (right edge, causal) are bright, others muted
- `forecast` → median line + shaded lo–hi cone (`forecast`)
- `edge`   → `score` histogram pane under price + markers on top/bottom 5% days
- `plan`   → entry/stop/target zones (`levels`) + drivers list

## Redis

| key / channel | type | writer | content |
|---|---|---|---|
| `cpd:models` | string JSON `Model[]` | serve.py at start | |
| `cpd:universe` | string JSON `{[basket]: string[]}` | serve.py at start | |
| `cpd:board` | string JSON `Board` | serve.py at start + `refresh_board` job | |
| `cpd:jobs` | list (RPUSH by bun, BLPOP by py) | bun | `{id, type: "run"|"refresh_board"|"candles"|"yd_screen", model, ticker, interval, bars?, days?, profile?}` |
| `cpd:yd:screen` | string JSON `YdScreen` | serve.py `yd_screen` job | last Yd screen (the Yd chart reads its rank from it) |
| `cpd:result:<id>` | string JSON `RunResult`, TTL 600 s | serve.py | |
| `cpd:done` | channel | serve.py | job id, after the result key is set |
| `cpd:bar:<interval>` | channel | live_scan.py | `{ticker, interval, final: bool, ...Candle}`; forming bar ≤ 1 msg/s/ticker, final once at bar close |
| `cpd:last:<interval>` | hash ticker → JSON Candle | live_scan.py | latest (forming or final) bar per ticker |
| `cpd:signal` | channel | live_scan.py | `Signal` |
| `cpd:signals` | list JSON `Signal`, LPUSH + LTRIM 300 | live_scan.py | newest first |
| `cpd:status` | hash | live_scan.py / serve.py | `scanner_heartbeat` (epoch s), `scanner_symbols`, `serve_heartbeat` |

serve.py caches results per (model, ticker, interval, last bar time) so repeated views are instant.
Intraday candles for a ticker outside the live watchlist are fetched once from Angel history (60 days, one request) and stored; the evening `--once` backfill then keeps them growing.
Intraday candles for jobs = stored official candles (`~/.cache/chart-pattern-detector/intraday/<INTERVAL>/<T>.parquet`)
+ today's live bars (`intraday/live_<INTERVAL>.parquet` from the main scanner, `live_yd_<INTERVAL>.parquet` from the Yd worker); serve.py calls the throttled Angel history API only for a ticker with no stored candles.

## HTTP (Bun, port 8787)

| method path | response |
|---|---|
| GET `/api/health` | `{ok, redis, scanner_age_s, serve_age_s}` |
| GET `/api/models` | `Model[]` |
| GET `/api/universe` | `{[basket]: string[]}` |
| GET `/api/board` | `Board` |
| GET `/api/signals?limit=50` | `Signal[]` newest first |
| GET `/api/run?model=&ticker=&interval=&bars=&days=` | `RunResult` (enqueue job, wait on `cpd:done`, timeout 120 s → 504 `{error}`); identical in-flight requests share one job |
| GET `/*` | static files from `platform/web/dist` (SPA fallback to index.html) |

## WebSocket `/ws`

- client → server: `{"op":"sub","topics":[...]}` / `{"op":"unsub","topics":[...]}`
- topics: `bar:<interval>:<TICKER>`, `bars:<interval>` (all tickers, used by the board), `signals`, `board`
- server → client: `{"topic": string, "data": any}`; on `sub` to `bar:<interval>:<T>` the server first sends the cached `cpd:last` bar.
- Server pings every 30 s; client reconnects with backoff and re-subscribes.

## PatternEdge analogs (kind `edge`, 1D and intraday)

"This stock drew this shape before: what happened next, and is today's context the same?" `cpd/analog.py`.

Intraday (5m live, 15m, ...): same search over the stock's own intraday history (W=60, H=20 **bars**); analog
`start`/`end` are epoch contract times; NIFTY context is the previous session's; no volume-spike flag (time-of-day
volume profile). No score histogram: the GBM score is daily-trained, so `stats.score_bps`/`pct_rank` are the last daily ones.

- Query = the stock's last `W` daily bars (z-normalised log close; default W=60). Matches = the most similar past
  windows of the **same stock** (plus, if it helps out of sample, NSE500 peers), non-overlapping, each ending at least
  `H` bars before today so its outcome is known (H=20). No lookahead anywhere: context and results are as-of the window end.
- Context compared between today and each match: 50-DMA slope (the "MA slightly up vs flat" check), 200-DMA slope,
  close vs 50/200-DMA, NIFTY trend, volume trend, named patterns in the window (stock-pattern / HarmonicPattern /
  ta_patterns), latest quarterly results as known on that date (sales / profit YoY, days since announcement).
- Prediction = similarity × context-weighted outcomes of the matches.

RunResult additions for `PatternEdge (ours)` on 1D (existing `score` histogram stays):

```ts
type Analog = {
  start: Time; end: Time;          // the past window (end = its "today")
  sim: number;                     // shape similarity 0..1
  ctx: number;                     // context agreement with today 0..1
  fwd: number;                     // return over the next H bars after `end` (fraction, e.g. 0.062)
  mfe: number; mae: number;        // best / worst move within H bars (fractions)
  outcome: "target" | "stop" | "time";   // TradePlan rules from `end`: +2.5 ATR / -1.5 ATR / H bars
  ma50Slope: number;               // 50-DMA slope at `end` (% per 10 bars)
  results?: { q: string; salesYoY: number | null; profitYoY: number | null } | null;
  patterns: string[];              // named patterns inside the window
};
// RunResult (edge): candles = full daily history (so 2019 analogs can be scrolled to), plus
//   analogs: Analog[]  (best first, <= 10)
//   boxes: today's query window (live: true, label e.g. "7/10 analogs up · +4.1% / 20d")
//          + every analog window (live: false, label "2019-03 · +6.2%", dir = sign(fwd))
//   forecast: median / 10% / 90% of the analogs' paths, rescaled to today's close, H bars ahead
//   stats: { analog_n, analog_p_up, analog_median_fwd, analog_ctx_p_up, ma50_slope_now, ma50_slope_analogs, ... }
//   flags: string[]   // plain-English context mismatches, e.g. "50-DMA flat now (+0.0%); in the 6 analogs that
//                     //  went up it was rising (median +0.8%/10 bars)"
```

## Yd page (F&O, all timeframes)

- `/yd/:ticker?tf=1m|3m|5m|15m|1h|1D` (default 15m): the F&O stock list on the left (`universe.fo`: NSE stocks with stock
  futures, from the Angel scrip master) or the screen, a candle chart, and the Yd strategy panel.
- `GET /api/candles?ticker=&interval=` -> RunResult with candles (last 1000 bars) + the Yd strategy, cached 30 s. Job
  `{type: "candles", ticker, interval, profile: "yd"}`: intraday candles are fetched with the **Yd Angel One account**
  (the `#Yd` block in `.env`, its own login and rate limits) and topped up when the stored last bar is 2+ bars old.
  Intervals: `1D` (NSE daily store; Angel ONE_DAY for F&O stocks outside NSE500), `ONE_MINUTE`, `THREE_MINUTE`,
  `FIVE_MINUTE`, `FIFTEEN_MINUTE`, `ONE_HOUR`.

### Yd strategy (cpd/yd.py; the user's rules, not backtested)

Volume Shock + Opening-Range Breakout (A) / Pullback (B), long + short, 5-60 min holds. `VS_time(t) = CV(d,t) / mean
CV(d-k,t)` over the 5 previous regular sessions (today excluded, all 5 required, missing slot -> that day's last earlier
slot, partial/special sessions are never base days; n=10 and median supported in code) is the trigger; `VS_raw =
CV(d,t) / mean V(d-k)` is logged; `VS_EOD = V(d) / mean V(d-k)` is StockEdge's "Volume Shockers". OR = bars inside
09:15-09:30. Exits 1.5R / stop / 60 min / 15:15, stop first when both hit in one bar. `Q = min(⌊0.5% E / R⌋, ⌊80% E / O⌋)`.

RunResult additions (candles job, `profile: "yd"`):
- `markers`: entries (arrow, `"A 2.3x"` = setup + VS_time at the signal) and exits (dot, `"+1.5R target"`); skipped setups
  are only in `trades`. `lines`: OR high/low (dir 1/-1) and session VWAP (dir 0) per visible session.
- `paperTrade`: the newest non-skipped setup; `levels`: its entry/stop/target while pending or open.
- `trades: YdTrade[]` (oldest first): `{date, setup, side, signal, vs_time, vs_raw, entry, stop, target, status:
  pending|open|closed|skip, reason, entryTime, exitTime, exit?, R?, net_R?, qty?, binding?, costs? (₹), cost_R?, cost_R_slip?}`;
  qty and costs at ₹50,000 equity.
- `stats`: `VS_time, VS_raw, VS_EOD, VS_EOD_day, turnover20_cr, universe ("ok" or failed filters), results_day, rank,
  screen_slot, cost_R, breakeven, breakeven_slip, cost_basis`.

Timeframes: bar counts are minutes / bar length (time exit 60 min, retest window 40, chop window 30, stop order 10), so
**1m / 3m / 5m / 15m** run both setups (5m is the spec's native frame; OR = 15 / 5 / 3 / 1 bars). **1h**: VS_time per
hourly slot only; setups are not defined (the 09:15 bar spans past the opening range). **1D**: VS_EOD per day (dots at
>= 2x) for next-day planning.

`GET /api/yd/screen` -> `YdScreen` (job `yd_screen`, cached 60 s; the page refetches 20 s after each 5-minute close while
the market is open): `{asof, session, slot, universe, covered, top: string[], lists: {asm, gsm, band}, rows: [{ticker,
bar, vs_time, vs_raw, vs_eod, eod_day, close, turnover, ok, why, results, rank}]}`, rows by VS_time, `rank` 1.. among
rows passing the filters (top 5 = tradeable). Data: stored 5m candles + `live_yd_FIVE_MINUTE.parquet`, no Angel calls.
Filters: price ₹100-5,000; 20-day average turnover >= ₹50 cr (NSE500 daily store, 5m bars for the 5 names outside it);
not in ASM (NSE long + short term lists) or GSM (`sec_list.csv` remarks); not within 2% of the price band (`sec_list.csv`
band; F&O stocks have "No Band" and get NSE's 10% dynamic operating range, which NSE flexes in 5% steps intraday, so
that check is approximate). An unreachable NSE list shows "ASM/GSM unknown" and fails the row. Results day = an NSE
results announcement since the previous session's close (`cpd/results.py`), tagged, not excluded.

Live bars: systemd user unit `cpd-yd-live` runs `~/.cache/chart-pattern-detector/yd_scan.sh` (weekdays 09:12-15:35:
`scripts/live_scan.py --profile yd --watchlist fo --bars-only`, SmartStream on the Yd account for all F&O tokens).

Sources (checked 29-Oct-2026):
- StockEdge Volume Shockers: "today's traded volume is at least 1.5 times above their last 5 days' average traded
  volume" — https://web.stockedge.com/trending-stocks?section=volume-shockers (threshold here: 2.0).
- Zarattini, Barbon & Aziz, "A Profitable Day Trading Strategy for the U.S. Equity Market" (SFI 24-98,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284): 5-min OR; Relative Volume = first-5-min volume / 14-day
  average of first-5-min volume; trade only RV >= 100% and the top 20 by RV; stop 10% of 14-day ATR; no target, exit at
  the close; 1% risk per trade, 4x leverage cap.
- Costs: STT 0.025% intraday sell, NSE transaction 0.00297%, SEBI ₹10/crore, stamp 0.003% buy, GST 18% on brokerage +
  transaction + SEBI — https://support.rupeezy.in/support/solutions/articles/2100044912 (and
  https://hdfcsky.com/pricing/other-trading-charges, https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies
  for SEBI). Brokerage: Angel One intraday lower of ₹20 or 0.1% per order — https://www.angelone.in/exchange-transaction-charges
  (Zerodha-style 0.03% is cheaper; the code uses Angel's). IPFT ₹10/crore is not modelled.
- Tick size: https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf (< ₹250: 0.01, ≤ 1,000: 0.05, ≤ 5,000: 0.10 ...).
- ASM: https://archives.nseindia.com/invest/content/Long_Term_ASM.csv, .../Short_Term_ASM.csv; GSM + price bands:
  https://nsearchives.nseindia.com/content/equities/sec_list.csv (pages: https://www.nseindia.com/reports/asm, /reports/gsm);
  F&O dynamic range: https://www.nseindia.com/static/trade/price-band-operating-range-flex-faqs
