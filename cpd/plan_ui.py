"""Streamlit 'Trade plan' tab: NSE stock (full model) or uploaded screenshot (image model)."""
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from cpd import chartcnn, data, lwchart, tradeplan as tp
from cpd.bench import OUT

NAMES = {
    "c_sma200": "distance from 200-day average", "c_sma50": "distance from 50-day average", "r60": "3-month return",
    "r120": "6-month return", "r20": "1-month return", "r5": "1-week return", "r1": "last day's return",
    "mkt_trend": "NIFTY trend (vs 200-day avg)", "mkt_r20": "NIFTY 1-month return", "mkt_vol20": "NIFTY volatility",
    "vol20": "stock volatility", "vol_ratio": "volatility now vs usual", "atr_pct": "daily range (ATR %)", "rsi14": "RSI(14)",
    "hi252": "distance below 52-week high", "lo252": "distance above 52-week low", "vol_z20": "volume vs usual",
    "rs20": "1-month strength vs NIFTY", "beta60": "beta to NIFTY", "adv1": "market breadth today", "adv10": "market breadth (10d)",
    "pos20": "position in 20-day range", "pos60": "position in 60-day range", "cnn_p": "chart-image CNN p(up)",
    "prior_sum3": "recent patterns' track record", "prior_max3": "best recent pattern's track record",
    "prior_min3": "worst recent pattern's track record", "d20": "20-bar move (in ATRs)", "d40": "40-bar move (in ATRs)",
    "d1": "last bar's move (in ATRs)", "d5": "5-bar move (in ATRs)", "d10": "10-bar move (in ATRs)", "ma20d": "distance from 20-bar avg (ATRs)",
    "ma50d": "distance from 50-bar avg (ATRs)", "atr_ch": "range expansion vs 20 bars ago", "hh20": "distance below 20-bar high (ATRs)",
    "ll20": "distance above 20-bar low (ATRs)", "gap": "today's gap", "body": "last candle body", "uwick": "last upper wick",
    "lwick": "last lower wick", "active_bull": "bullish patterns active", "active_bear": "bearish patterns active",
}


def _model(image=False):
    p = os.path.join(OUT, f"tradeplan_model{'_image' if image else ''}.pkl")
    return _load(p, os.path.getmtime(p)) if os.path.exists(p) else None


@st.cache_resource
def _load(p, mtime):  # mtime in the key: a daily refresh is picked up without restarting the app
    return pickle.load(open(p, "rb"))


def _cards(pl, fmt=lambda v: f"{v:,.2f}"):
    c = st.columns(5)
    c[0].metric("Action", pl["action"])
    c[1].metric("P(target before stop)", f"{pl['p_target_first']:.0%}")
    c[2].metric("Expected R after costs", f"{pl['expected_R']:+.2f}")
    c[3].metric("Stop", fmt(pl["stop"]), f"{-tp.SL_ATR:.1f} ATR", delta_color="off")
    c[4].metric("Target", fmt(pl["target"]), f"+{tp.TP_ATR:.1f} ATR", delta_color="off")
    st.caption(f"Entry: next open (reference {fmt(pl['entry'])}). Exit at stop, target, or after {pl['time_exit_bars']} bars. "
               f"Break-even P for this 1:{tp.TP_ATR / tp.SL_ATR:.2f} setup is ~{1 / (1 + tp.TP_ATR / tp.SL_ATR):.0%}; "
               f"time exits make the true break-even lower, which is why Expected R is modelled directly.")


def _why(model, x):
    d = tp.drivers(model, x)
    st.markdown("**Why**: inputs that pushed this plan up (🟢) or down (🔴) vs a typical stock, as a share of the total push:")
    for k, v in d.items():
        st.markdown(f"- {'🟢' if v > 0 else '🔴'} **{NAMES.get(k, k)}** = `{x[k]:.3g}` ({abs(v):.0%})")


def _cnn_view(img):
    occ = chartcnn.occlusion(img)
    c1, c2 = st.columns([1, 2])
    c1.image(np.kron(img, np.ones((3, 3), np.uint8)) * 255, caption="What the CNN sees (last 20 bars, JKX style)", clamp=True)
    c2.bar_chart(pd.Series(occ, index=[f"-{19 - i}" if i < 19 else "today" for i in range(20)], name="push towards up"), height=200)
    c2.caption("Bars that pushed the chart model towards 'up' (positive) or 'down' (negative).")


def _track_record():
    bt = os.path.join(OUT, "tradeplan_backtest.csv")
    if os.path.exists(bt):
        with st.expander("Track record of this exact rule (walk-forward 2018 → now, never-seen years)"):
            st.dataframe(pd.read_csv(bt), hide_index=True)
            yr = os.path.join(OUT, "tradeplan_by_year.csv")
            if os.path.exists(yr):
                st.dataframe(pd.read_csv(yr), hide_index=True)


def nse_plan(ticker):
    model = _model()
    tdy = os.path.join(OUT, "tradeplan_today.pkl")
    if model is None or not os.path.exists(tdy):
        st.info("Run `python -m cpd.tradeplan final` first.")
        return
    today = pd.read_pickle(tdy)
    today["ticker"] = today.ticker.astype(str)
    st.subheader(f"Best setups as of {model['asof']}")
    best = today.sort_values("rank").head(15)
    st.caption("Stocks are ranked every day. Expected R / P(target) = what plans of the same rank actually earned in "
               "2018-2026 (years the model never saw). Only ranks whose history clears costs get BUY. A ~27% target "
               "hit-rate can still be profitable: trades that hit neither level exit after 10 bars, on average in profit.")
    st.dataframe(best[["rank", "ticker", "rank_bucket", "p_win", "eR"]].rename(columns={
        "rank_bucket": "rank group", "p_win": "P(target first)", "eR": "Expected R (history)"}).round(3), hide_index=True, height=250)
    with st.expander("How each rank group did, 2018-2026 out of sample"):
        st.dataframe(model["rank_table"].round(3))
    row = today[today.ticker == ticker]
    if row.empty:
        st.info(f"{ticker} is not in the NSE500 model universe (or has no bar on {model['asof']}).")
        return
    row = row.iloc[0]
    df = data.load(ticker, "nse500").loc[: model["asof"]]
    pl = tp.plan(df, row.p_win, row.eR)
    st.subheader(f"{ticker}: plan for the next session")
    _cards(pl)
    lwchart.show(df.iloc[-250:], levels=[("entry", pl["entry"], "#64748b"), ("stop", pl["stop"], "#dc2626"),
                                         ("target", pl["target"], "#2563eb")], height=460)
    _why(model, row)
    _, img = chartcnn.features_for(df)
    _cnn_view(img)
    _track_record()


def image_plan():
    st.markdown("Upload a candlestick screenshot (TradingView, broker app…) with **green/red candles** and at least "
                f"**{tp.MIN_BARS} candles** visible. Volume pane optional.")
    f = st.file_uploader("Chart screenshot", type=["png", "jpg", "jpeg"])
    c1, c2 = st.columns(2)
    last_px = c1.number_input("Optional: current price", min_value=0.0, value=0.0)
    high_px = c2.number_input("Optional: highest price visible on the chart", min_value=0.0, value=0.0)
    model = _model(image=True)
    if model is None:
        st.info("Run `python -m cpd.tradeplan final --image` first.")
        return
    if not f:
        return
    p = os.path.join(tempfile.gettempdir(), "cpd_shot_" + f.name)
    open(p, "wb").write(f.getbuffer())
    try:
        r = tp.plan_image(p, model)
    except ValueError as e:
        st.error(str(e))
        return
    pl, cd = r["plan"], r["candles"]
    priced = last_px > 0 and high_px > last_px
    if priced:  # pixel -> price through two known points
        k = (high_px - last_px) / (cd.high.max() - cd.close.iloc[-1])
        conv = lambda v: last_px + (v - cd.close.iloc[-1]) * k
    else:
        conv = None
    _cards({**pl, **({kk: conv(pl[kk]) for kk in ("entry", "stop", "target")} if conv else {})},
           fmt=(lambda v: f"{v:,.2f}") if conv else (lambda v: "see chart"))
    im = Image.open(p).convert("RGB")
    H = im.height
    dr = ImageDraw.Draw(im)
    for name, v, c in [("entry", pl["entry"], (82, 81, 78)), ("stop", pl["stop"], (227, 73, 72)), ("target", pl["target"], (42, 120, 214))]:
        y = H - v
        dr.line([(0, y), (im.width, y)], fill=c, width=2)
        dr.text((5, y - 12), f"{name}" + (f" {conv(v):,.2f}" if conv else ""), fill=c)
    st.image(im, caption="Plan drawn on your chart", width="stretch")
    with st.expander(f"Check the reading: {len(cd)} candles reconstructed from your image"):
        fig = go.Figure(go.Candlestick(x=list(range(len(cd))), open=cd.open, high=cd.high, low=cd.low, close=cd.close))
        fig.update_layout(height=300, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    _why(model, r["features"])
    _cnn_view(r["cnn_image"])
    st.caption("Image model = only what a picture shows (shapes in ATR units + chart CNN), trained on NSE500 daily bars. "
               "Other timeframes (e.g. 1-minute) are an extrapolation: the shapes transfer, the statistics may not.")
    bt = os.path.join(OUT, "tradeplan_backtest_image.csv")
    if os.path.exists(bt):
        with st.expander("Track record of the image model (walk-forward 2018 → now)"):
            st.dataframe(pd.read_csv(bt), hide_index=True)


def render(ticker):
    mode = st.radio("Plan from", ["NSE stock (full model)", "Chart screenshot (image model)"], horizontal=True)
    nse_plan(ticker) if mode.startswith("NSE") else image_plan()
