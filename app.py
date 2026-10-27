"""Chart Pattern Detector: pick a repo, pick an Indian stock/index (or upload), see what it detects/predicts.

    streamlit run app.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from cpd import data
from cpd.angel import INTERVALS
from cpd.bench import OUT
from cpd.vision import draw_boxes

REPOS = {
    # name: (kind, url, one-line how-it-works)
    "ta_patterns": ("rule", "https://github.com/AdventuresInDataScience/ta_patterns",
                    "300 numpy detectors (106 candlestick + 194 chart). Pivot-confirmed, no lookahead."),
    "stock-pattern": ("rule", "https://github.com/BennyThadikaran/stock-pattern",
                      "Pivot geometry: double tops/bottoms, H&S, triangles, flags, VCP, harmonics. We walk it forward bar by bar."),
    "chart_patterns": ("rule", "https://github.com/zeta-zetra/chart_patterns",
                       "scipy pivots + linear regression of highs/lows. Repo peeks 3-10 bars ahead; we delay its signals."),
    "TradingPatternScanner": ("rule", "https://github.com/white07S/TradingPatternScanner",
                              "Rolling-window masks over 3 bars. Repo peeks 1 bar ahead; we delay its signals."),
    "HarmonicPattern": ("rule", "https://github.com/xxwikkixx/HarmonicPattern",
                        "argrelextrema XABCD + Fibonacci ratio checks (Gartley, Bat, Butterfly, Crab). Close only."),
    "pytrendline": ("lines", "https://github.com/ednunezg/pytrendline",
                    "Brute-force support/resistance trendline fitting with pivot scoring. Visual only."),
    "foduucom-yolov8": ("vision", "https://github.com/foduucom/Stockmarket-pattern-detection",
                        "YOLOv8s on chart screenshots: H&S top/bottom, M_Head, W_Bottom, Triangle, StockLine."),
    "ChartScanAI": ("vision", "https://github.com/Omar-Karimov/ChartScanAI",
                    "YOLOv8 on mplfinance charts: Buy / Sell boxes."),
    "Kronos (mini)": ("forecast", "https://github.com/shiyu-coder/Kronos",
                      "Decoder-only transformer over tokenized OHLCV candles (K-line foundation model). Samples future bars."),
    "Chronos-2 (small)": ("forecast", "https://github.com/amazon-science/chronos-forecasting",
                          "Amazon's pretrained time-series transformer; quantile forecasts of the close."),
    "TimesFM 2.5": ("forecast", "https://github.com/google-research/timesfm",
                    "Google's 200M-param patch decoder foundation model. Heavy on CPU (~2 GB RAM)."),
    "VAE analog (nsarang)": ("forecast", "https://github.com/nsarang/big-data-stock-price-forecast",
                             "1D-conv VAE embeds 256-bar windows; forecast = futures of the nearest past windows."),
    "PatternEdge (ours)": ("edge", "", "All causal patterns + market context -> walk-forward gradient boosting. "
                                      "Ranks which pattern firings are worth trading today."),
}
DIR_COLOR = {1: "#16a34a", -1: "#dc2626", 0: "#64748b"}

st.set_page_config(page_title="Chart Pattern Detector", layout="wide")
st.title("Chart Pattern Detector")

@st.cache_data(ttl=60, show_spinner="Loading candles...")
def intraday(ticker, interval, minute):
    """Stored official candles + today's bars from the live scanner's tick stream. Only asks the (throttled)
    history API when the symbol has never been fetched."""
    from cpd import angel

    p = angel.STORE / interval / f"{ticker}.parquet"
    d = pd.read_parquet(p) if p.exists() else angel.history(ticker, interval, days=30)
    lp = angel.STORE / f"live_{interval}.parquet"
    if lp.exists():
        lv = pd.read_parquet(lp).query("ticker == @ticker").drop(columns="ticker")
        d = pd.concat([d, lv[lv.index > d.index[-1]]])
    return d


@st.fragment(run_every="60s")
def _auto_refresh():
    if st.session_state.get("_live") and src.startswith("Live intraday"):
        st.rerun(scope="app")


with st.sidebar:
    repo = st.selectbox("Repo / model", list(REPOS))
    kind, url, how = REPOS[repo]
    if url:
        st.markdown(f"[{url.split('github.com/')[1]}]({url})")
    st.caption(how)
    src = st.radio("Input", ["Indian market data", "Live intraday (Angel One)", "Upload CSV"] +
                   (["Upload chart image"] if kind == "vision" else []))
    df = None
    if src == "Indian market data":
        u = data.universe()
        basket = st.selectbox("Basket", list(u), index=list(u).index("nifty50"))
        ticker = st.selectbox("Symbol", u[basket])
        df = data.load(ticker, basket)
    elif src.startswith("Live intraday"):
        ticker = st.selectbox("Symbol", data.universe()["nse500"], index=data.universe()["nse500"].index("RELIANCE"))
        interval = st.selectbox("Bar", list(INTERVALS), index=2, format_func=lambda k: INTERVALS[k][0])
        live = st.toggle("Auto-refresh every minute", value=True)
        try:
            df = intraday(ticker, interval, pd.Timestamp.now().floor("1min"))
        except Exception as e:  # login / network: say so, don't crash the page
            st.error(f"Angel One: {e}")
    elif src == "Upload CSV":
        f = st.file_uploader("CSV with date,open,high,low,close[,volume]", type="csv")
        if f:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            df = df.set_index(pd.to_datetime(df.pop("date"))).sort_index()
            if "volume" not in df:
                df["volume"] = 0.0
            df = df[data.COLS].astype(float)
            ticker = f.name
    bars = st.slider("Bars to show", 60, 1000, 250, 10)
    key = f"{src}|{ticker}|{df.index[-1]}" if df is not None else ""  # cache key: new bar -> recompute
    days = st.slider("Buy signal window (last N bars)", 3, 30, 10)
st.session_state["_live"] = src.startswith("Live intraday") and live
_auto_refresh()


NO_EVENTS = pd.DataFrame(columns=["date", "pattern", "dir"])


@st.cache_data(show_spinner="Running detector...")
def run_rule(repo, ticker, n, _df):
    from cpd import rules

    return rules.DETECTORS[repo](_df, start=_df.index[-n])


@st.cache_data(show_spinner="Running YOLO on the chart as it looked each evening...")
def run_vision_live(repo, ticker, last, n, conf, days, _df):
    from cpd import vision

    return vision.live(repo, _df, os.path.join(tempfile.gettempdir(), f"cpd_live_{repo}.png"), days, n, conf)


@st.cache_data
def edge_preds(mtime):
    pr = pd.read_pickle(os.path.join(OUT, "edge_preds_PatternEdge.pkl"))
    pr["pct"] = pr.groupby("date").pred.rank(pct=True)
    return pr


def signal_panel(pt):
    if pt is None:
        st.info(f"No bullish signal in the last {days} bars.")
        return []
    st.markdown(f"**Buy signal {pt['signal'].date()}**: {pt['why'][:300]}  \n"
                f"Paper trade with the TradePlan rules: long at the next open, stop {lwchart.SL_ATR} ATR, "
                f"target {lwchart.TP_ATR} ATR, exit after {lwchart.H} bars.")
    c = st.columns(5)
    c[0].metric("Entry", f"{pt['entry']:,.2f}", str(pt["entry_date"].date()) if pt["entry_date"] is not None else "next open",
                delta_color="off")
    c[1].metric("Stop", f"{pt['stop']:,.2f}")
    c[2].metric("Target", f"{pt['target']:,.2f}")
    c[3].metric("Status", pt["status"][0], pt["status"][1], delta_color="off")
    c[4].metric("Result so far", "-" if pt["R"] is None else f"{pt['R']:+.2f} R")
    return [("entry", pt["entry"], "#64748b"), ("stop", pt["stop"], "#dc2626"), ("target", pt["target"], "#2563eb")]


@st.cache_data(show_spinner="Running model...")
def run_forecast(repo, ticker, _df, h):
    from cpd import forecast

    return forecast.forecast(repo, [_df], h)[0]


tab_plan, tab_run, tab_bench = st.tabs(["Trade plan", "Run a repo", "Benchmark (Indian data)"])

with tab_plan:
    from cpd import plan_ui

    plan_ui.render(ticker if df is not None and src != "Upload CSV" else "RELIANCE")


with tab_run:
    from cpd import lwchart

    if kind == "vision" and src == "Upload chart image":
        from cpd import vision

        img = st.file_uploader("Chart screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
        conf = st.slider("Confidence", 0.05, 0.9, vision.MODELS[repo]["conf"], 0.05)
        if img:
            p = os.path.join(tempfile.gettempdir(), "cpd_upload_" + img.name)
            open(p, "wb").write(img.getbuffer())
            dets = vision.detect_image(repo, p, conf)
            st.image(draw_boxes(p, dets), width="stretch")
            st.dataframe(pd.DataFrame(dets), width="stretch")
    elif df is None:
        st.info("Choose a symbol or upload data in the sidebar.")
    else:
        d = df.iloc[-bars:]
        intra = src.startswith("Live intraday")
        if intra:
            sig = os.path.join(os.path.dirname(OUT), "intraday", f"signals_{interval}.csv")
            if os.path.exists(sig):
                with st.expander("Live scanner: latest ChartScanAI signals across the watchlist", expanded=False):
                    st.dataframe(pd.read_csv(sig).tail(50)[::-1], hide_index=True, width="stretch")
            st.caption(f"Last finished {INTERVALS[interval][0]} bar: {df.index[-1]:%d-%b %H:%M} IST")
        ev, lines, fc, note = NO_EVENTS, [], None, ""
        if kind == "rule":
            ev = run_rule(repo, key, bars, df)
            if repo == "stock-pattern":  # the pattern's pivot points
                lines = [dict(points=sorted(r["info"]["points"].values()), color=DIR_COLOR[r.dir]) for _, r in ev.iterrows()]
                lines = [ln for ln in lines if ln["points"][0][0] >= d.index[0]]
            note = f"{len(ev)} events. Arrow = the bar the pattern became known (tradable from the next open)."
        elif kind == "lines":
            from cpd import rules

            n = min(bars, 250)
            with st.spinner("Fitting trendlines (O(n^3), ~10 s)..."):
                tl = rules.trendlines(df, n)
            lines = [dict(points=[(ln["dates"][ln["i0"]], ln["m"] * ln["i0"] + ln["b"]),
                                  (ln["dates"][ln["i1"]], ln["m"] * ln["i1"] + ln["b"])],
                          color=DIR_COLOR[1] if ln["kind"] == "support" else DIR_COLOR[-1]) for ln in tl]
            note = "Support (green) / resistance (red) trendlines. Visual only: no signals."
        elif kind == "vision":
            from cpd import vision

            n = st.slider("Bars rendered into the image", 40, 250, vision.MODELS[repo]["bars"], 10)
            conf = st.slider("Confidence", 0.05, 0.9, vision.MODELS[repo]["conf"], 0.05)
            ev = run_vision_live(repo, key, "", n, conf, days, df)
            img_p = os.path.join(tempfile.gettempdir(), f"cpd_app_{repo}.png")
            dets = vision.detect(repo, df, img_p, n, conf)
            for b in dets:  # today's boxes as brackets under (bullish) / over (bearish) the bars they cover
                seg = df.loc[b["start"]:b["end"]]
                y = seg.low.min() * 0.99 if b["dir"] > 0 else seg.high.max() * 1.01
                lines.append(dict(points=[(b["start"], y), (b["end"], y)], color=DIR_COLOR[b["dir"]]))
            note = (f"Brackets = every box on today's chart (incl. finished patterns, seen with hindsight). "
                    f"Arrows = boxes touching the right edge of the chart as it looked that evening, re-run for each of "
                    f"the last {days} bars: what you could have traded.")
        elif kind == "forecast":
            h = st.slider("Horizon (bars)", 5, 30, 10)
            if repo == "TimesFM 2.5":
                st.warning("TimesFM loads ~2 GB into RAM and takes ~15 s on CPU.")
            if st.button("Forecast") or repo != "TimesFM 2.5":
                q = run_forecast(repo, key, df, h)
                if intra:
                    stp = d.index.to_series().diff().mode()[0]
                    fx = pd.date_range(d.index[-1] + stp, periods=h, freq=stp)
                else:
                    fx = pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), periods=h)
                fc = (fx, q)
                chg = q[-1, 1] / d.close.iloc[-1] - 1
                ev = pd.DataFrame(dict(date=[d.index[-1]], dir=[int(np.sign(chg))], pattern=[f"{chg:+.1%} in {h} bars"]))
                note = "Blue = median forecast, dashed = 10%-90% band. Benchmark tab shows how often its sign is right."
        elif kind == "edge" and intra:
            st.info("PatternEdge is trained on daily bars (its inputs include 6-month returns and NSE breadth). "
                    "An intraday version needs retraining on the intraday history the live scanner is collecting.")
            st.stop()
        elif kind == "edge":
            p = os.path.join(OUT, "edge_preds_PatternEdge.pkl")
            if not os.path.exists(p):
                st.info("Run `python scripts/get_data.py` (or `python -m cpd.edge build && python -m cpd.edge fit`) first.")
                st.stop()
            pr = edge_preds(os.path.getmtime(p))
            mine = pr[pr.ticker.astype(str) == ticker]
            ev = pd.concat([mine[mine.pct >= 0.95].assign(dir=1, pattern="top 5% of NSE500"),
                            mine[mine.pct <= 0.05].assign(dir=-1, pattern="bottom 5%")])[["date", "dir", "pattern"]]
            note = (f"Arrows = days this stock ranked in the top / bottom 5% of NSE500 by expected 10-day excess return. "
                    f"Scores up to {pr.date.max().date()}." if len(mine) else f"{ticker} is not in the NSE500 universe.")
        pt = lwchart.paper_trade(d, ev, days) if kind != "lines" else None
        levels = signal_panel(pt) if kind != "lines" else []
        lwchart.show(d, ev, lines, levels, fc)
        st.caption(note)
        if kind == "rule":
            st.dataframe(ev.assign(info=ev["info"].astype(str)).sort_values("date", ascending=False), width="stretch")
        elif kind == "vision":
            with st.expander("Today's chart as the model sees it"):
                st.image(draw_boxes(img_p, dets), width="stretch")
                st.dataframe(pd.DataFrame(dets).drop(columns="box", errors="ignore"), width="stretch")
        elif kind == "edge":
            last = pr.date.max()
            today = pr[pr.date == last].sort_values("pred", ascending=False)
            c1, c2 = st.columns(2)
            fmt = lambda x: x[["ticker", "pred"]].assign(pred=(x.pred * 1e4).round(1)).rename(columns={"pred": "exp. 10d excess (bps)"})
            c1.subheader(f"Top long candidates ({last.date()})")
            c1.dataframe(fmt(today.head(15)), hide_index=True)
            c2.subheader("Top short candidates")
            c2.dataframe(fmt(today.tail(15)[::-1]), hide_index=True)
            imp = os.path.join(OUT, "edge_importance_PatternEdge.csv")
            if os.path.exists(imp):
                st.subheader("What the model looks at (permutation importance)")
                st.bar_chart(pd.read_csv(imp, index_col=0).iloc[:15, 0])

with tab_bench:
    s = os.path.join(OUT, "summary.csv")
    if not os.path.exists(s):
        st.info("Run `python -m cpd.bench run <method>` for each method, then `python -m cpd.bench report`.")
    else:
        sm = pd.read_csv(s)
        st.markdown("Every method scored the same way on **Nifty 50 stocks + NSE indices, 2018 to now**: signal at close *t*, "
                    "enter next open, exit close *t+10*. **excess_bps** = signed return minus what holding the same side "
                    "unconditionally would earn. Costs: 20 bps round trip in net_bps.")
        st.dataframe(sm.round(3), width="stretch", hide_index=True)
        st.bar_chart(sm.set_index("method")["excess_bps"])
        pp = os.path.join(OUT, "per_pattern.csv")
        if os.path.exists(pp):
            st.subheader("Per pattern")
            st.dataframe(pd.read_csv(pp).round(3), width="stretch", hide_index=True)
