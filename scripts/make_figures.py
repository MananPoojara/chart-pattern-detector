"""Regenerate the README figures in docs/img from results/ and live model runs.

    python scripts/make_figures.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image

from cpd import data
from cpd.bench import OUT

IMG = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(IMG, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#a3a29c"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6})


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def benchmark():
    s = pd.read_csv(os.path.join(OUT, "summary.csv"))
    s = s[~s.method.str.contains("Nifty50 stocks")].sort_values("excess_bps")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(s) + 1.2))
    col = [BLUE if v > 0 else RED for v in s.excess_bps]
    ax.barh(s.method, s.excess_bps, color=col, height=0.62)
    for i, (v, n, t) in enumerate(zip(s.excess_bps, s.n, s.t_excess)):
        ax.text(v + (4 if v >= 0 else -4), i, f"{v:+.0f}  (n={n:,}, t={t:.1f})", va="center",
                ha="left" if v >= 0 else "right", color=INK2, fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Excess 10-day return per signal (bps), vs holding the same stock unconditionally")
    ax.set_title("Out-of-sample 2018 → 2026, Indian market. Above 0 = the signal adds information.", loc="left", fontsize=11)
    lo, hi = s.excess_bps.min(), s.excess_bps.max()
    ax.set_xlim(lo - 80, hi + 110)
    ax.grid(axis="y", visible=False)
    save(fig, "benchmark.png")


def deciles():
    p = pd.read_pickle(os.path.join(OUT, "edge_preds_PatternEdge.pkl")).dropna(subset=["y"])
    p["dec"] = p.groupby("date").pred.transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)) + 1
    m = p.groupby(["date", "dec"]).y.mean().unstack().mean() * 1e4
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(m.index, m.values, color=[BLUE if v > 0 else RED for v in m.values], width=0.7)
    for x, v in m.items():
        ax.text(x, v + (3 if v > 0 else -3), f"{v:+.0f}", ha="center", va="bottom" if v > 0 else "top", color=INK2, fontsize=8.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("PatternEdge score decile each day (1 = most bearish, 10 = most bullish)")
    ax.set_ylabel("Next-10-day excess return (bps)")
    ax.set_title("NSE500, walk-forward 2018 → 2026: the ranking is monotonic", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    save(fig, "edge_deciles.png")


def ablation():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for tag, c, lab in [("PatternEdge", BLUE, "Patterns + context"), ("ContextOnly", ORANGE, "Context only"),
                        ("PatternsOnly", AQUA, "Patterns only")]:
        p = pd.read_pickle(os.path.join(OUT, f"edge_preds_{tag}.pkl")).dropna(subset=["y"])
        ic = p.groupby("date").apply(lambda g: g.pred.corr(g.y, method="spearman"), include_groups=False)
        yr = ic.groupby(ic.index.year).mean()
        ax.plot(yr.index, yr.values, color=c, lw=2, marker="o", ms=5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Mean daily rank IC")
    ax.set_title("What carries the signal? (rank correlation of score vs realised 10-day return)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    save(fig, "edge_ablation.png")


def detections(ticker="RELIANCE", basket="nifty50"):
    from cpd import forecast, rules, vision

    df = data.load(ticker, basket)
    d = df.iloc[-250:]
    # stock-pattern: pivot geometry drawn as lines
    ev = rules.stock_pattern(df, start=d.index[0])
    lines, colors = [], []
    for _, r in ev.iterrows():
        pts = sorted(r["info"]["points"].values())
        if pd.Timestamp(pts[0][0]) < d.index[0]:
            continue
        lines.append([(pd.Timestamp(a), b) for a, b in pts])
        colors.append(BLUE if r.dir > 0 else RED if r.dir < 0 else GRAY)
    ta = rules.ta_patterns(df, start=d.index[0], which="charts")
    ta = ta[ta.pattern.isin(ta.pattern.value_counts().index[:200])]
    up = pd.Series(np.nan, index=d.index)
    dn = pd.Series(np.nan, index=d.index)
    for _, r in ta.iterrows():
        (up if r.dir > 0 else dn)[r.date] = d.low[r.date] * 0.97 if r.dir > 0 else d.high[r.date] * 1.03
    ap = [mpf.make_addplot(up, type="scatter", marker="^", color=BLUE, markersize=18),
          mpf.make_addplot(dn, type="scatter", marker="v", color=RED, markersize=18)]
    st = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor=SURF, figcolor=SURF, gridcolor=GRID)
    kw = dict(alines=dict(alines=lines, colors=colors, linewidths=1.6)) if lines else {}
    fig, _ = mpf.plot(d, type="candle", style=st, addplot=ap, figsize=(11, 5), returnfig=True, volume=False,
                      title=f"\n{ticker}: stock-pattern geometry (lines) + ta_patterns chart events (▲ bull / ▼ bear)", **kw)
    save(fig, "rules_example.png")

    # YOLO models side by side
    for name in vision.MODELS:
        p = os.path.join(tempfile.gettempdir(), f"fig_{name}.png")
        dets = vision.detect(name, df, p)
        im = vision.draw_boxes(p, dets)
        im.thumbnail((1100, 1100))
        im.save(os.path.join(IMG, f"yolo_{name}.png"))
        print("wrote", f"yolo_{name}.png", [(x["label"], round(x["conf"], 2)) for x in dets])

    # forecast fan: Chronos-2 and Kronos from 30 bars ago, vs what happened
    h, back = 20, 20
    ctx = df.iloc[: len(df) - back]
    fig, ax = plt.subplots(figsize=(10, 4))
    hist = df.iloc[-120:]
    ax.plot(hist.index, hist.close, color=INK, lw=1.4, label="actual close")
    fx = df.index[len(df) - back :][:h]
    for name, c in [("Chronos-2 (small)", BLUE), ("Kronos (mini)", ORANGE)]:
        q = forecast.forecast(name, [ctx], back)[0][: len(fx)]
        ax.fill_between(fx, q[:, 0], q[:, 2], color=c, alpha=0.15, lw=0)
        ax.plot(fx, q[:, 1], color=c, lw=2, label=f"{name}: median, q10–q90 band")
    ax.axvline(ctx.index[-1], color=INK2, lw=0.8, ls="--")
    ax.text(ctx.index[-1], ax.get_ylim()[1], " forecast made here", color=INK2, fontsize=8.5, va="top")
    ax.set_title(f"{ticker}: foundation-model forecasts vs what actually happened", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "forecast_example.png")


def tradeplan():
    """Average net R of the top-5 plans per day, by year, for the three TradePlan variants."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    variants = [("_nocnn", "Context + patterns", ORANGE), ("", "+ chart CNN (full model)", BLUE), ("_image", "Screenshot-only model", AQUA)]
    w = 0.27
    for i, (tag, lab, c) in enumerate(variants):
        y = pd.read_csv(os.path.join(OUT, f"tradeplan_by_year{tag}.csv"), index_col=0).avg_R_net
        tot = pd.read_csv(os.path.join(OUT, f"tradeplan_backtest{tag}.csv")).set_index("rule").loc["TradePlan: top 5 per day", "avg_R_net"]
        ax.bar(y.index + (i - 1) * w, y.values, width=w - 0.03, color=c, label=f"{lab}  (all years: {tot:+.3f} R)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Avg R per trade, after costs")
    ax.set_xticks(y.index)
    ax.set_title("TradePlan: 5 best plans each day, out-of-sample 2018 → 2026 (1 R = the stop distance)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    save(fig, "tradeplan_by_year.png")


if __name__ == "__main__":
    benchmark()
    deciles()
    ablation()
    detections()
    tradeplan()
