"""Benchmark every repo on the same Indian data and the same yardstick.

  python -m cpd.bench run <method> [--universe nifty50,indices] [--start 2018-01-01]
  python -m cpd.bench report

`run` writes one CSV of signals per method (resumable, one ticker at a time).
`report` scores all of them with cpd.evaluate: 10-bar forward return, entry next open, 20 bps cost.
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from cpd import data, evaluate, rules

OUT = os.path.expanduser(os.environ.get("CPD_RESULTS", "~/.cache/chart-pattern-detector/results"))
H = 10
STEP = {"vision": 20, "Kronos (mini)": 120, "forecast": 20}  # bars between scans for the slow models


def universe(spec):
    u = data.universe()
    return [(b, t) for b in spec.split(",") for t in u[b]]


def signals(method, df, start):
    """-> DataFrame[date, pattern, dir, score] with every row known at close of `date`."""
    if method in rules.DETECTORS:
        kw = {"step": 3} if method == "stock-pattern" else {}  # ponytail: 3-bar scan step, events dated <=2 bars late
        ev = rules.DETECTORS[method](df, start=start, **kw)
        return ev.assign(score=ev["dir"].astype(float))[["date", "pattern", "dir", "score"]]
    i0 = int(df.index.searchsorted(pd.Timestamp(start)))
    from cpd import forecast, vision

    if method in vision.MODELS:
        rows, png = [], os.path.join(tempfile.gettempdir(), f"cpd_bench_{os.getpid()}.png")
        for t in range(i0, len(df) - H, STEP["vision"]):
            for d in vision.detect(method, df.iloc[: t + 1], png):
                if d["at_edge"] and d["dir"]:
                    rows.append((df.index[t], d["label"], d["dir"], d["conf"] * d["dir"]))
        return pd.DataFrame(rows, columns=["date", "pattern", "dir", "score"])
    if method in forecast.FORECASTERS:
        ts = list(range(max(i0, 300), len(df) - H, STEP.get(method, STEP["forecast"])))
        rows = []
        for k in range(0, len(ts), 8):
            b = ts[k : k + 8]
            kw = {"samples": 4} if method.startswith("Kronos") else {}  # ponytail: 4 paths/window keeps CPU time ~1.5 h
            q = forecast.forecast(method, [df.iloc[: t + 1] for t in b], H, **kw)
            for t, qq in zip(b, q):
                s = np.log(qq[-1, 1] / df.close.iloc[t])
                rows.append((df.index[t], "forecast", int(np.sign(s)), s))
        return pd.DataFrame(rows, columns=["date", "pattern", "dir", "score"])
    raise KeyError(method)


def run(method, spec, start):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"signals__{method.replace(' ', '_').replace('/', '_')}.csv")
    done = set(pd.read_csv(path, usecols=["ticker"]).ticker) if os.path.exists(path) else set()
    for basket, t in universe(spec):
        if t in done:
            continue
        df = data.load(t, basket)
        try:
            s = signals(method, df, start)
        except Exception as e:  # one bad series must not kill a multi-hour run
            print("FAIL", t, repr(e)[:200], flush=True)
            continue
        s.insert(0, "ticker", t)
        s.insert(0, "basket", basket)
        s.insert(0, "method", method)
        s.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        done.add(t)
        print(method, t, len(s), flush=True)


def attach_returns(sig: pd.DataFrame) -> pd.DataFrame:
    """Add fwd (10-bar fwd return), base (that ticker's mean fwd over the same period) and drop
    signals near suspected unadjusted corporate actions."""
    out = []
    for (b, t), g in sig.groupby(["basket", "ticker"]):
        df = data.load(t, b)
        fwd = evaluate.forward_return(df, H)
        bad = data.corporate_action_mask(df).astype(int).rolling(140).max().shift(-H).fillna(1).astype(bool)
        g = g.assign(date=pd.to_datetime(g.date))
        g = g[g.date.isin(df.index)]
        g = g[~bad.reindex(g.date).values]
        period = fwd[(fwd.index >= g.date.min()) & (fwd.index <= g.date.max())]
        out.append(g.assign(fwd=fwd.reindex(g.date).values, base=period.mean()))
    return pd.concat(out, ignore_index=True)


def report():
    rows, per_pattern = [], []
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("signals__"):
            continue
        sig = pd.read_csv(os.path.join(OUT, f))
        m = sig.method.iloc[0]
        sig = attach_returns(sig[sig.dir != 0])
        r = evaluate.score_signals(sig)
        if "forecast" in set(sig.pattern):  # rank IC: does the size of the forecast mean anything?
            r["ic"] = float(sig[["score", "fwd"]].dropna().corr("spearman").iloc[0, 1])
        rows.append(dict(method=m, **r))
        for p, g in sig.groupby("pattern"):
            if len(g) >= 30:
                per_pattern.append(dict(method=m, pattern=p, **evaluate.score_signals(g)))
    s = pd.DataFrame(rows).sort_values("excess_bps", ascending=False)
    pp = pd.DataFrame(per_pattern).sort_values("t_excess", ascending=False)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pp.to_csv(os.path.join(OUT, "per_pattern.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(pp.head(15).round(3).to_string(index=False))
    print(pp.tail(10).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("method", nargs="?")
    ap.add_argument("--universe", default="nifty50,indices")
    ap.add_argument("--start", default="2018-01-01")
    a = ap.parse_args()
    run(a.method, a.universe, a.start) if a.cmd == "run" else report()
