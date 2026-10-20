"""Chart Pattern Detector: pick a repo, pick an Indian stock/index (or upload), see what it detects/predicts.

    streamlit run app.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cpd import data
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

with st.sidebar:
    repo = st.selectbox("Repo / model", list(REPOS))
    kind, url, how = REPOS[repo]
    if url:
        st.markdown(f"[{url.split('github.com/')[1]}]({url})")
    st.caption(how)
    src = st.radio("Input", ["Indian market data", "Upload CSV"] + (["Upload chart image"] if kind == "vision" else []))
    df = None
    if src == "Indian market data":
        u = data.universe()
        basket = st.selectbox("Basket", list(u), index=list(u).index("nifty50"))
        ticker = st.selectbox("Symbol", u[basket])
        df = data.load(ticker, basket)
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


def candles(d):
    fig = go.Figure(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="price"))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def mark_events(fig, d, ev):
    for dr, g in ev.groupby("dir"):
        y = np.where(dr > 0, d.low.reindex(g.date) * 0.985, d.high.reindex(g.date) * 1.015)
        fig.add_scatter(x=g.date, y=y, mode="markers", name={1: "bullish", -1: "bearish", 0: "neutral"}[dr],
                        marker=dict(symbol="triangle-up" if dr > 0 else "triangle-down" if dr < 0 else "circle",
                                    size=10, color=DIR_COLOR[dr]), text=g.pattern, hovertemplate="%{x|%Y-%m-%d}<br>%{text}")


@st.cache_data(show_spinner="Running detector...")
def run_rule(repo, ticker, n, _df):
    from cpd import rules

    return rules.DETECTORS[repo](_df, start=_df.index[-n])


@st.cache_data(show_spinner="Running model...")
def run_forecast(repo, ticker, _df, h):
    from cpd import forecast

    return forecast.forecast(repo, [_df], h)[0]


tab_plan, tab_run, tab_bench = st.tabs(["Trade plan", "Run a repo", "Benchmark (Indian data)"])

with tab_plan:
    from cpd import plan_ui

    plan_ui.render(ticker if df is not None and src == "Indian market data" else "RELIANCE")


with tab_run:
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
    elif kind == "rule":
        ev = run_rule(repo, ticker, bars, df)
        d = df.iloc[-bars:]
        fig = candles(d)
        mark_events(fig, d, ev)
        if repo == "stock-pattern":  # draw the pattern's pivot points
            for _, r in ev.iterrows():
                pts = sorted(r["info"]["points"].values())
                fig.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                line=dict(color=DIR_COLOR[r.dir], width=1.5), name=r.pattern, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(ev)} events. Marker date = the bar the pattern became known (tradable from next open).")
        st.dataframe(ev.assign(info=ev["info"].astype(str)).sort_values("date", ascending=False), width="stretch")
    elif kind == "lines":
        from cpd import rules

        n = min(bars, 250)
        with st.spinner("Fitting trendlines (O(n^3), ~10 s)..."):
            lines = rules.trendlines(df, n)
        d = df.iloc[-n:]
        fig = candles(d)
        for ln in lines:
            x = [ln["dates"][ln["i0"]], ln["dates"][ln["i1"]]]
            fig.add_scatter(x=x, y=[ln["m"] * ln["i0"] + ln["b"], ln["m"] * ln["i1"] + ln["b"]], mode="lines",
                            line=dict(color="#16a34a" if ln["kind"] == "support" else "#dc2626"), name=ln["kind"])
        st.plotly_chart(fig, width="stretch")
    elif kind == "vision":
        from cpd import vision

        n = st.slider("Bars rendered into the image", 40, 250, vision.MODELS[repo]["bars"], 10)
        conf = st.slider("Confidence", 0.05, 0.9, vision.MODELS[repo]["conf"], 0.05)
        p = os.path.join(tempfile.gettempdir(), f"cpd_app_{repo}.png")
        with st.spinner("Rendering chart and running YOLO..."):
            dets = vision.detect(repo, df, p, n, conf)
        st.image(draw_boxes(p, dets), width="stretch")
        st.caption("Rendered in the model's training style. Only boxes reaching the last bars (at_edge) are live signals; "
                   "boxes in the middle describe patterns that already played out.")
        st.dataframe(pd.DataFrame(dets).drop(columns="box", errors="ignore"), width="stretch")
    elif kind == "forecast":
        h = st.slider("Horizon (bars)", 5, 30, 10)
        if repo == "TimesFM 2.5":
            st.warning("TimesFM loads ~2 GB into RAM and takes ~15 s on CPU.")
        if st.button("Forecast") or repo != "TimesFM 2.5":
            q = run_forecast(repo, ticker, df, h)
            d = df.iloc[-bars:]
            fx = pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), periods=h)
            fig = candles(d)
            fig.add_scatter(x=list(fx) + list(fx[::-1]), y=list(q[:, 2]) + list(q[::-1, 0]), fill="toself",
                            fillcolor="rgba(37,99,235,0.15)", line=dict(width=0), name="q10-q90")
            fig.add_scatter(x=fx, y=q[:, 1], mode="lines", line=dict(color="#2563eb"), name="median")
            st.plotly_chart(fig, width="stretch")
            chg = q[-1, 1] / d.close.iloc[-1] - 1
            st.metric(f"Median {h}-bar change", f"{chg:+.2%}", help="Benchmark tab shows how often the sign of this is right.")
    elif kind == "edge":
        p = os.path.join(OUT, "edge_preds_PatternEdge.pkl")
        if not os.path.exists(p):
            st.info("Run `python -m cpd.edge build && python -m cpd.edge fit` first.")
        else:
            pr = pd.read_pickle(p)
            last = pr.date.max()
            today = pr[pr.date == last].sort_values("pred", ascending=False)
            c1, c2 = st.columns(2)
            c1.subheader(f"Top long candidates ({last.date()})")
            c1.dataframe(today.head(15)[["ticker", "pred"]].assign(pred=lambda x: (x.pred * 1e4).round(1)).rename(columns={"pred": "exp. 10d excess (bps)"}), hide_index=True)
            c2.subheader("Top short candidates")
            c2.dataframe(today.tail(15)[::-1][["ticker", "pred"]].assign(pred=lambda x: (x.pred * 1e4).round(1)).rename(columns={"pred": "exp. 10d excess (bps)"}), hide_index=True)
            mine = pr[pr.ticker.astype(str) == ticker].set_index("date")
            if len(mine):
                st.subheader(f"{ticker}: PatternEdge score history")
                st.caption("Expected 10-day return vs the NSE500 average, in bps. Above 0 = model prefers it to the average stock.")
                st.line_chart((mine.pred * 1e4).iloc[-bars:].rename("score (bps)"), height=260)
            else:
                st.caption(f"{ticker} is not in the NSE500 training universe.")
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
