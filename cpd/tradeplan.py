"""TradePlan: turn PatternEdge + ChartCNN into a full, testable trade plan (entry, stop, target, probability).

What BullGPT-style tools output, but every number is learned and back-tested on years the model never saw.

Labels (triple barrier, Lopez de Prado): enter at next open; stop = entry - SL_ATR*ATR, target = entry + TP_ATR*ATR,
time exit after H bars. Label = 1 if the target is hit first. If both are touched in one bar we assume the stop.
Model: gradient boosting on context + pattern + chart-CNN features -> P(target before stop) (isotonic-calibrated)
       and E[R] (time exits are not full losses, so E[R] is modelled directly, not derived from P).
Expected move range: quantile models of the 10-bar return (q10/q50/q90).
Rule: trade only if expected R after costs > MIN_EDGE ("no setup" is a valid answer).

  python -m cpd.tradeplan fit        # walk-forward 2018..now -> results/tradeplan_preds.pkl + backtest
  python -m cpd.tradeplan final      # model on all data -> results/tradeplan_model.pkl (used by the app)
"""
import argparse
import os

import numpy as np
import pandas as pd

from cpd import data, evaluate
from cpd.bench import H, OUT
from cpd.edge import CTX_COLS, EMBARGO, pattern_features

SL_ATR, TP_ATR = 1.5, 2.5  # reward:risk 1.67
COST = 0.003  # round trip incl. STT, charges, slippage (fraction of price); converted to R per stock
MIN_EDGE = 0.05  # minimum CALIBRATED expected R per trade after costs (live plans)
TRAIN_START = "2013-01-01"  # ChartCNN was fitted on <= 2012, so its features are out-of-sample from here
LABELS = os.path.join(OUT, "tradeplan_labels.pkl")
RANK_BINS = [0, 5, 20, 50, 100, 250, 10_000]
RANK_LABELS = ["top 5", "6-20", "21-50", "51-100", "101-250", "rest"]


def triple_barrier(df, sl=SL_ATR, tp=TP_ATR, h=H):
    """Per bar t: outcome of a long entered at open[t+1]. Returns (hit_target, R multiple realised)."""
    e = df.open.shift(-1).values
    a = evaluate.atr(df).values
    pad = lambda x: np.r_[x[1:], [np.nan] * (h + 1)]
    hi = np.lib.stride_tricks.sliding_window_view(pad(df.high.values), h)[: len(df)]
    lo = np.lib.stride_tricks.sliding_window_view(pad(df.low.values), h)[: len(df)]
    cl = np.lib.stride_tricks.sliding_window_view(pad(df.close.values), h)[: len(df)]
    stop, targ = (e - sl * a)[:, None], (e + tp * a)[:, None]
    hs, ht = lo <= stop, hi >= targ
    first = lambda m: np.where(m.any(1), m.argmax(1), h + 1)
    fs, ft = first(hs), first(ht)
    win = ft < fs  # same bar -> stop (pessimistic)
    r_time = (cl[:, -1] - e) / (sl * a)
    r = np.where(win, tp / sl, np.where(fs <= h, -1.0, r_time))
    unknown = np.isnan(cl[:, -1])
    return np.where(unknown, np.nan, win.astype(float)), np.where(unknown, np.nan, r)


def labels():
    if os.path.exists(LABELS):
        return pd.read_pickle(LABELS)
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    out = np.full((len(panel), 3), np.nan, np.float32)
    for t, g in panel.groupby(panel.ticker.astype(str), sort=False, observed=True):
        df = data.load(t, "nse500")
        pos = df.index.get_indexer(g.date)
        w, r = triple_barrier(df)
        out[g.index, 0], out[g.index, 1] = w[pos], r[pos]
        out[g.index, 2] = evaluate.atr(df).values[pos] / df.close.values[pos]
    lab = pd.DataFrame(out, columns=["win", "R", "atr_frac"])
    lab.to_pickle(LABELS)
    return lab


SHAPE_COLS = ["d1", "d5", "d10", "d20", "d40", "pos20", "pos60", "ma20d", "ma50d", "atr_ch", "body", "uwick", "lwick",
              "gap", "vol_z20", "hh20", "ll20", "rsi14"]
SHAPE = os.path.join(OUT, "tradeplan_shape.pkl")
MIN_BARS = 80  # a screenshot needs at least this many candles


def shape_features(df):
    """Features that survive a screenshot: no price level, no history beyond ~60 bars.
    Every price quantity is a difference divided by ATR, so any linear pixel<->price mapping gives the same values."""
    from cpd.edge import _rsi

    c, o, h, l, v = df.close, df.open, df.high, df.low, df.volume
    a = evaluate.atr(df)
    f = pd.DataFrame(index=df.index)
    for k in (1, 5, 10, 20, 40):
        f[f"d{k}"] = (c - c.shift(k)) / a
    for n in (20, 60):
        lo, hi = l.rolling(n).min(), h.rolling(n).max()
        f[f"pos{n}"] = (c - lo) / (hi - lo)
    f["ma20d"] = (c - c.rolling(20).mean()) / a
    f["ma50d"] = (c - c.rolling(50).mean()) / a
    f["atr_ch"] = a / a.shift(20)
    f["body"], f["uwick"], f["lwick"] = (c - o) / a, (h - np.maximum(o, c)) / a, (np.minimum(o, c) - l) / a
    f["gap"] = (o - c.shift()) / a
    lv = np.log1p(v)
    f["vol_z20"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std()).fillna(0)
    f["hh20"] = (h.rolling(20).max() - c) / a
    f["ll20"] = (c - l.rolling(20).min()) / a
    f["rsi14"] = _rsi(c)
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


def shape_panel():
    if os.path.exists(SHAPE):
        return pd.read_pickle(SHAPE)
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    out = np.full((len(panel), len(SHAPE_COLS)), np.nan, np.float32)
    for t, g in panel.groupby(panel.ticker.astype(str), sort=False, observed=True):
        df = data.load(t, "nse500")
        out[g.index] = shape_features(df).values[df.index.get_indexer(g.date)]
    sp = pd.DataFrame(out, columns=SHAPE_COLS)
    sp.to_pickle(SHAPE)
    return sp


def features(panel, ev, y, train_rows, use_cnn=True, image_only=False):
    if image_only:
        return pd.concat([shape_panel(), pd.read_pickle(os.path.join(OUT, "cnn_features.pkl"))], axis=1)
    pf, _ = pattern_features(panel, ev, y, train_rows)
    X = pd.concat([panel[CTX_COLS + ["active_bull", "active_bear"]], pf], axis=1)
    cf = os.path.join(OUT, "cnn_features.pkl")
    if use_cnn and os.path.exists(cf):
        X = pd.concat([X, pd.read_pickle(cf)], axis=1)
    return X


def _models(X, win, ret, w_rows, cal_rows):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression

    kw = dict(max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=500, l2_regularization=1.0,
              max_bins=128, early_stopping=False, random_state=0)
    clf = HistGradientBoostingClassifier(**kw).fit(X[w_rows], win[w_rows])
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(X[cal_rows])[:, 1], win[cal_rows])
    qs = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw).fit(X[w_rows], ret[w_rows]) for q in (0.1, 0.5, 0.9)}
    return clf, iso, qs


def _r_model(X, R, w_rows):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=500, l2_regularization=1.0,
                                         max_bins=128, early_stopping=False, random_state=0).fit(X[w_rows], R[w_rows])


def fit(first_year=2018, use_cnn=True, image_only=False):
    tag = "_image" if image_only else "" if use_cnn else "_nocnn"
    panel, ev = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    lab = labels()
    d = panel.date.values
    y = (panel.fwd - panel.groupby("date").fwd.transform("mean")).values.astype(np.float32)  # for pattern priors
    ud = np.sort(panel.date.unique())
    win, ret = lab.win.values, panel.fwd.values
    preds = []
    for Y in range(first_year, pd.Timestamp(ud[-1]).year + 1):
        t0 = np.datetime64(f"{Y}-01-01")
        cut = ud[max(np.searchsorted(ud, t0) - EMBARGO, 0)]
        cal_start = ud[max(np.searchsorted(ud, np.datetime64(f"{Y - 1}-01-01")) - EMBARGO, 0)]
        base = (d >= np.datetime64(TRAIN_START)) & ~np.isnan(win) & ~np.isnan(ret)
        w_rows = base & (d < cal_start)  # fit
        cal_rows = base & (d >= np.datetime64(f"{Y - 1}-01-01")) & (d < cut)  # calibrate on the last year before Y
        test = (d >= t0) & (d < np.datetime64(f"{Y + 1}-01-01"))
        X = features(panel, ev, y, d < cut, use_cnn, image_only)
        clf, iso, qs = _models(X, win, ret, w_rows, cal_rows)
        rm = _r_model(X, lab.R.values, w_rows)
        p = iso.predict(clf.predict_proba(X[test])[:, 1])
        q = {f"q{int(k*100)}": m.predict(X[test]) for k, m in qs.items()}
        preds.append(panel.loc[test, ["ticker", "date"]].assign(p_win=p, pred_R=rm.predict(X[test]), win=win[test], R=lab.R.values[test],
                                                               cost_R=cost_r(lab.atr_frac.values[test]), **q))
        print(Y, "fit", int(w_rows.sum()), "test", int(test.sum()), "base win-rate", round(float(np.nanmean(win[test])), 3), flush=True)
    pr = pd.concat(preds, ignore_index=True)
    pr.to_pickle(os.path.join(OUT, f"tradeplan_preds{tag}.pkl"))
    backtest(pr, tag)


def cost_r(atr_frac):
    return COST / (SL_ATR * atr_frac)  # costs in units of the risk taken (stop distance)


def backtest(pr=None, tag=""):
    """Every test day: take the best-ranked plans (ranking is what transfers out of sample). Report what happened."""
    if pr is None:
        pr = pd.read_pickle(os.path.join(OUT, f"tradeplan_preds{tag}.pkl"))
    pr = pr.dropna(subset=["R"])
    pr = pr.assign(eR=pr.pred_R - pr.cost_R, net=pr.R - pr.cost_R)
    rows = []
    for name, m in [("every long (no model)", np.ones(len(pr), bool)),
                    ("TradePlan: top 20 per day", pr.groupby("date").eR.rank(ascending=False) <= 20),
                    ("TradePlan: top 5 per day", pr.groupby("date").eR.rank(ascending=False) <= 5)]:
        s = pr[m]
        rows.append(dict(rule=name, trades=len(s), per_day=round(len(s) / pr.date.nunique(), 1),
                         win_rate=round(s.win.mean(), 3), avg_R_net=round(s.net.mean(), 3),
                         pred_win=round(s.p_win.mean(), 3), years_positive=f"{(s.groupby(s.date.dt.year).net.mean() > 0).sum()}/{s.date.dt.year.nunique()}"))
    res = pd.DataFrame(rows)
    top5 = pr[pr.groupby("date").eR.rank(ascending=False) <= 5]
    yr = top5.groupby(top5.date.dt.year).agg(trades=("net", "size"), win_rate=("win", "mean"), avg_R_net=("net", "mean")).round(3)
    cal = pr.groupby(pd.cut(pr.p_win, [0, .3, .35, .4, .45, .5, .6, 1])).agg(n=("win", "size"), predicted=("p_win", "mean"), actual=("win", "mean")).round(3)
    res.to_csv(os.path.join(OUT, f"tradeplan_backtest{tag}.csv"), index=False)
    yr.to_csv(os.path.join(OUT, f"tradeplan_by_year{tag}.csv"))
    cal.to_csv(os.path.join(OUT, f"tradeplan_calibration{tag}.csv"))
    pd.set_option("display.width", 200)
    print(res.to_string(index=False), "\n", yr.to_string(), "\n", cal.to_string())
    return res


def final(image_only=False):
    """Fit on everything up to the latest labelled bar. Full model -> today's plans for every NSE500 stock;
    image model -> used for uploaded screenshots."""
    import pickle

    panel, ev = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    lab = labels()
    d = panel.date.values
    y = (panel.fwd - panel.groupby("date").fwd.transform("mean")).values.astype(np.float32)
    ud = np.sort(panel.date.unique())
    last = ud[-1]
    cal_start = ud[np.searchsorted(ud, last - np.timedelta64(365, "D"))]
    base = (d >= np.datetime64(TRAIN_START)) & ~np.isnan(lab.win.values) & ~np.isnan(panel.fwd.values)
    X = features(panel, ev, y, np.ones(len(panel), bool), image_only=image_only)
    clf, iso, qs = _models(X, lab.win.values, panel.fwd.values, base & (d < cal_start), base & (d >= cal_start))
    rm = _r_model(X, lab.R.values, base)
    model = dict(clf=clf, iso=iso, qs=qs, rm=rm, cols=list(X.columns), asof=str(pd.Timestamp(last).date()),
                 ref=X[base].median())  # typical values, used to explain a single plan
    # Displayed numbers are re-calibrated on the walk-forward (never-seen-year) history: raw model R/p were
    # ~0.1R too optimistic out of sample, so "27%" must mean 27%.
    from sklearn.isotonic import IsotonicRegression

    oos = pd.read_pickle(os.path.join(OUT, f"tradeplan_preds{'_image' if image_only else ''}.pkl")).dropna(subset=["R"])
    model["cal_R"] = IsotonicRegression(out_of_bounds="clip").fit(oos.pred_R - oos.cost_R, oos.R - oos.cost_R)
    model["cal_p"] = IsotonicRegression(out_of_bounds="clip").fit(oos.p_win, oos.win)
    # Cross-sectional plans: what did plans of the same DAILY RANK earn out of sample? (ranking is what transfers)
    oos = oos.assign(rank=(oos.pred_R - oos.cost_R).groupby(oos.date).rank(ascending=False), net=oos.R - oos.cost_R)
    model["rank_table"] = oos.groupby(pd.cut(oos["rank"], RANK_BINS, labels=RANK_LABELS), observed=True).agg(
        avg_R=("net", "mean"), win=("win", "mean"), trades=("net", "size"))
    if not image_only:
        _, model["priors"] = pattern_features(panel, ev, y, np.ones(len(panel), bool))
        latest = (panel.date == last).values
        plan = panel.loc[latest, ["ticker", "date"]].assign(
            p_win=iso.predict(clf.predict_proba(X[latest])[:, 1]), pred_R=rm.predict(X[latest]),
            cost_R=cost_r(lab.atr_frac.values[latest]), **{f"q{int(k*100)}": m.predict(X[latest]) for k, m in qs.items()})
        plan["eR_raw"] = plan.pred_R - plan.cost_R
        plan["rank"] = plan.eR_raw.rank(ascending=False)
        bucket = pd.cut(plan["rank"], RANK_BINS, labels=RANK_LABELS)
        rt = model["rank_table"]
        plan["rank_bucket"] = bucket.astype(str)
        plan["eR"] = bucket.map(rt.avg_R).astype(float)
        plan["p_win"] = bucket.map(rt.win).astype(float)
        pd.concat([plan, X[latest]], axis=1).to_pickle(os.path.join(OUT, "tradeplan_today.pkl"))
        print(plan.sort_values("eR", ascending=False).head(10))
    with open(os.path.join(OUT, f"tradeplan_model{'_image' if image_only else ''}.pkl"), "wb") as f:
        pickle.dump(model, f)


def drivers(model, x, k=6):
    """Which inputs moved this plan: swap each feature to its typical value and see how E[R] changes.
    Cheap, model-agnostic local explanation (a one-feature-at-a-time cousin of SHAP)."""
    x = pd.DataFrame([x[model["cols"]].astype(float).values], columns=model["cols"])
    base = model["rm"].predict(x)[0]
    alt = pd.concat([x] * len(model["cols"]), ignore_index=True)
    for i, c in enumerate(model["cols"]):
        alt.loc[i, c] = model["ref"][c]
    delta = base - model["rm"].predict(alt)
    s = pd.Series(delta, index=model["cols"])
    s = s / s.abs().sum()  # share of the total push (raw model R is over-confident, shares are not)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(k)


def plan(df, p_win, eR, atr_frac=None, sl=SL_ATR, tp=TP_ATR):
    """Levels for the last bar of df (any units). Entry reference = last close; the real entry is next open."""
    a = float(evaluate.atr(df).iloc[-1])
    e = float(df.close.iloc[-1])
    return dict(action="BUY" if eR > MIN_EDGE else "NO TRADE", entry=e, stop=e - sl * a, target=e + tp * a,
                p_target_first=float(p_win), expected_R=float(eR), time_exit_bars=H, atr=a)


def plan_image(path, model=None):
    """Screenshot -> candles -> image model -> plan. Costs assume a typical NSE500 ATR of 4% of price."""
    import pickle

    from cpd import chartcnn, digitize

    model = model or pickle.load(open(os.path.join(OUT, "tradeplan_model_image.pkl"), "rb"))
    df = digitize.digitize(path)
    if len(df) < MIN_BARS:
        raise ValueError(f"found {len(df)} candles; the model needs at least {MIN_BARS} visible")
    f = shape_features(df).iloc[-1].to_dict()
    cnn, img = chartcnn.features_for(df)
    f.update(cnn)
    x = pd.Series(f)
    X = pd.DataFrame([x[model["cols"]].values], columns=model["cols"])
    p = float(model["cal_p"].predict(model["iso"].predict(model["clf"].predict_proba(X)[:, 1]))[0])
    eR = float(model["cal_R"].predict([float(model["rm"].predict(X)[0]) - cost_r(0.04)])[0])
    return dict(plan=plan(df, p, eR), candles=df, cnn_image=img, drivers=drivers(model, x), features=x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "fit", "backtest", "final"])
    ap.add_argument("--no-cnn", action="store_true", help="ablation: without ChartCNN features")
    ap.add_argument("--image", action="store_true", help="screenshot model: only features readable from a chart image")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(use_cnn=not a.no_cnn, image_only=a.image)
    elif a.cmd == "final":
        final(image_only=a.image)
    else:
        {"labels": labels, "backtest": backtest}[a.cmd]()
