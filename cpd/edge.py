"""PatternEdge: what an ML engineer builds after seeing every repo above fail the same way.

Diagnosis of the repos (details in README):
  * rule detectors fire constantly (ta_patterns: ~8 events/bar) and treat every firing as equal;
  * vision models are trained on hindsight-labelled finished charts;
  * forecasters (Kronos/Chronos/TimesFM) predict ~a random walk on daily equities.
  None of them asks "WHEN does this pattern work?". So we:
  1. use every causal pattern as a candidate (ta_patterns, 300 detectors, verified no lookahead);
  2. learn a per-pattern prior from the TRAINING years only (shrunk mean signed excess return);
  3. add context the patterns ignore: trend, volatility regime, volume, relative strength vs NIFTY,
     market breadth (NSE advance/decline);
  4. gradient-boosted model -> expected 10-day excess return, cross-sectionally ranked each day;
  5. strict walk-forward: fit on years < Y (15-bar embargo), predict year Y, Y = 2018..now.
  Ablation `--no-patterns` answers whether patterns add anything over context alone.

  python -m cpd.edge build    # features for all NSE500 stocks (cached)
  python -m cpd.edge fit [--no-patterns | --no-context]   # walk-forward predictions -> results/signals__*.csv
  python -m cpd.edge fit --update   # nightly: refit the latest year only, keep earlier preds
"""
import argparse
import os

import numpy as np
import pandas as pd

from cpd import data, evaluate, rules
from cpd.bench import H, OUT

BREADTH = os.environ.get("CPD_BREADTH_CSV", "data/NSE_ADV_DEC.csv")
CACHE = os.path.join(OUT, "edge_panel.pkl")
EMBARGO = 15
CTX_COLS = ["r1", "r5", "r20", "r60", "r120", "vol20", "vol_ratio", "atr_pct", "rsi14", "c_sma50", "c_sma200",
            "hi252", "lo252", "vol_z20", "gap", "body", "uwick", "lwick", "pos20", "mkt_r20", "mkt_vol20",
            "mkt_trend", "rs20", "beta60", "adv1", "adv10", "dow"]


def _rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def market():
    m = data.load("NIFTY 50", "indices")
    lr = np.log(m.close).diff()
    b = pd.read_csv(BREADTH, parse_dates=["Date"]).set_index("Date")
    adv = (b.Advances / (b.Advances + b.Declines)).reindex(m.index).ffill()
    return pd.DataFrame({"mkt_lr": lr, "mkt_r20": np.log(m.close).diff(20), "mkt_vol20": lr.rolling(20).std(),
                         "mkt_trend": np.log(m.close / m.close.rolling(200).mean()), "adv1": adv, "adv10": adv.rolling(10).mean()})


def context(df, mk):
    c, o, h, l, v = df.close, df.open, df.high, df.low, df.volume
    lr = np.log(c).diff()
    a = evaluate.atr(df)
    m = mk.reindex(df.index)
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 20, 60, 120):
        f[f"r{n}"] = np.log(c).diff(n)
    f["vol20"] = lr.rolling(20).std()
    f["vol_ratio"] = f.vol20 / lr.rolling(120).std()
    f["atr_pct"] = a / c
    f["rsi14"] = _rsi(c)
    f["c_sma50"] = np.log(c / c.rolling(50).mean())
    f["c_sma200"] = np.log(c / c.rolling(200).mean())
    f["hi252"] = np.log(c / h.rolling(252, min_periods=60).max())
    f["lo252"] = np.log(c / l.rolling(252, min_periods=60).min())
    lv = np.log1p(v)
    f["vol_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["gap"] = np.log(o / c.shift())
    f["body"] = (c - o) / a
    f["uwick"] = (h - np.maximum(o, c)) / a
    f["lwick"] = (np.minimum(o, c) - l) / a
    f["pos20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min())
    for k in ("mkt_r20", "mkt_vol20", "mkt_trend", "adv1", "adv10"):
        f[k] = m[k]
    f["rs20"] = f.r20 - m.mkt_r20
    f["beta60"] = lr.rolling(60).cov(m.mkt_lr) / m.mkt_lr.rolling(60).var()
    f["dow"] = df.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


def build(basket="nse500", start="2003-01-01"):
    """Panel of (ticker, date) rows: context features + target. Pattern events (row, pattern, dir) are kept
    separately because their priors must be re-learned per walk-forward fold."""
    rules._path("ta_patterns/src")
    import ta_patterns as tap

    mk = market()
    frames, ev_row, ev_pat, ev_dir = [], [], [], []
    names, is_chart, row0 = None, None, 0
    for i, t in enumerate(data.universe()[basket]):
        df = data.load(t, basket)
        if len(df) < 400:
            continue
        a = (df.open.values, df.high.values, df.low.values, df.close.values)
        tc = tap.batch_all(*a, v=df.volume.values, chart_patterns=[])
        tg = tap.batch_all(*a, v=df.volume.values, candle_patterns=[])
        if names is None:
            names = list(tc.columns) + list(tg.columns)
            is_chart = np.r_[np.zeros(tc.shape[1], bool), np.ones(tg.shape[1], bool)]
        st = np.hstack([tc[names[: tc.shape[1]]].values, tg[names[tc.shape[1]:]].values]).astype(np.int8)
        f = context(df, mk)
        f["fwd"] = evaluate.forward_return(df, H).astype("float32")
        f["active_bull"] = (st > 0).sum(1).astype("float32")
        f["active_bear"] = (st < 0).sum(1).astype("float32")
        # latest H bars have no future yet: keep them (scored live, excluded from training via NaN target)
        bad = data.corporate_action_mask(df).astype(int).rolling(140).max().shift(-H).fillna(0).astype(bool).values
        keep = (df.index >= pd.Timestamp(start)) & ~bad & (np.arange(len(df)) >= 252) & f[CTX_COLS].notna().all(1).values
        pos = np.cumsum(keep) - 1 + row0  # global row id of kept bars
        prev = np.vstack([np.zeros((1, st.shape[1]), np.int8), st[:-1]])
        bi, pj = np.nonzero((st != 0) & (st != prev))  # event = state switches on
        ok = keep[bi]
        ev_row.append(pos[bi[ok]].astype(np.int32))
        ev_pat.append(pj[ok].astype(np.int16))
        ev_dir.append(st[bi[ok], pj[ok]])
        f = f[keep]
        f.insert(0, "ticker", t)
        frames.append(f.reset_index())
        row0 += int(keep.sum())
        print(i, t, int(keep.sum()), int(ok.sum()), flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel["ticker"] = panel.ticker.astype("category")
    ev = dict(row=np.concatenate(ev_row), pat=np.concatenate(ev_pat), dir=np.concatenate(ev_dir), names=names, is_chart=is_chart)
    pd.to_pickle((panel, ev), CACHE)
    return panel, ev


def pattern_features(panel, ev, y, train_rows, shrink=200):
    """Fold-specific pattern block over the last 3 bars (t-2..t).
    prior[p] = shrunk mean of dir*y over TRAINING events of pattern p (target encoding, no test leakage)."""
    n = len(panel)
    r, d = ev["row"], ev["dir"].astype(np.float32)
    tr = train_rows[r] & ~np.isnan(y[r])
    npat = len(ev["names"])
    s = np.bincount(ev["pat"][tr], weights=(d * y[r])[tr], minlength=npat)
    c = np.bincount(ev["pat"][tr], minlength=npat)
    prior = (s / (c + shrink)).astype(np.float32)
    contrib = d * prior[ev["pat"]]
    chart = ev["is_chart"][ev["pat"]]
    out = pd.DataFrame(index=panel.index)
    for k, m in [("n_bull3", d > 0), ("n_bear3", d < 0), ("chart_bull3", (d > 0) & chart), ("chart_bear3", (d < 0) & chart),
                 ("candle_bull3", (d > 0) & ~chart), ("candle_bear3", (d < 0) & ~chart)]:
        out[k] = np.bincount(r[m], minlength=n).astype(np.float32)
    out["prior_sum3"] = np.bincount(r, weights=contrib, minlength=n).astype(np.float32)
    mx, mn = np.zeros(n, np.float32), np.zeros(n, np.float32)
    np.maximum.at(mx, r, contrib)
    np.minimum.at(mn, r, contrib)
    out["prior_max3"], out["prior_min3"] = mx, mn
    g = out.groupby(panel.ticker.values, observed=True)  # rows are time-ordered within ticker
    agg = {k: "sum" for k in out.columns} | {"prior_max3": "max", "prior_min3": "min"}
    out = pd.concat([g[k].rolling(3, min_periods=1).agg(v).reset_index(level=0, drop=True).rename(k) for k, v in agg.items()], axis=1)
    return out.sort_index().astype(np.float32), pd.Series(prior, index=ev["names"])


ANALOG_COLS = ["analog_p_up", "analog_fwd", "analog_ctx_p_up", "analog_ctx_fwd", "analog_sim_mean", "analog_n",
               "ma50_slope", "ma200_slope"]


def fit(use_patterns=True, use_context=True, first_year=2018, analog=False, update=False):
    """analog=True: same model + cpd.analog features (results/analog_panel.pkl); predictions go to a *_candidate file,
    `python -m cpd.analog eval` decides whether they beat PatternEdge out of sample."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    panel, ev = pd.read_pickle(CACHE)
    extra = []
    if analog:
        from cpd.analog import PANEL
        a = pd.read_pickle(PANEL)
        extra = ANALOG_COLS + [c for c in ("salesYoY", "profitYoY", "days_since") if c in a and a[c].notna().mean() > 0.2]
        key = panel[["ticker", "date"]].astype({"ticker": str})
        a = key.merge(a.astype({"ticker": str})[["ticker", "date"] + extra], on=["ticker", "date"], how="left")
        for c in extra:
            panel[c] = a[c].values.astype("float32")
        del a, key
    dates = panel.date.values
    # target: 10-bar forward return minus that day's cross-sectional mean, winsorized
    y = (panel.fwd - panel.groupby("date").fwd.transform("mean")).values.astype(np.float32)
    lo, hi = np.nanpercentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    ud = np.sort(panel.date.unique())
    if update:  # nightly: refit only the latest year (earlier years' walk-forward preds don't change)
        first_year = pd.Timestamp(ud[-1]).year
    tag = "PatternEdge" if use_patterns and use_context else "ContextOnly" if use_context else "PatternsOnly"
    tag += "Analog" if analog else ""
    cols = (CTX_COLS if use_context else []) + (["active_bull", "active_bear"] if use_patterns else []) + extra
    preds, imps = [], []
    for Y in range(first_year, pd.Timestamp(ud[-1]).year + 1):
        t0 = np.datetime64(f"{Y}-01-01")
        cut = ud[max(np.searchsorted(ud, t0) - EMBARGO, 0)]  # purge: training targets end before the test year
        train = (dates < cut) & ~np.isnan(y)
        test = (dates >= t0) & (dates < np.datetime64(f"{Y + 1}-01-01"))
        X = panel[cols]
        if use_patterns:
            pf, priors = pattern_features(panel, ev, y, dates < cut)
            X = pd.concat([X, pf], axis=1)
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=500,
                                          l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=0)
        m.fit(X[train], y[train])
        p = m.predict(X[test])
        preds.append(panel.loc[test, ["ticker", "date", "fwd"]].assign(pred=p, y=y[test]))
        if Y == first_year:  # permutation importance on the first test year (rank-IC drop would be nicer; MSE is fine)
            sub = np.flatnonzero(test)[:: max(1, test.sum() // 20000)]
            pi = permutation_importance(m, X.iloc[sub], y[sub], n_repeats=3, random_state=0)
            imps.append(pd.Series(pi.importances_mean, index=X.columns))
        print(tag, Y, "train", int(train.sum()), "test", int(test.sum()),
              "rankIC", round(pd.Series(p).corr(pd.Series(y[test]), method="spearman"), 4), flush=True)
    pr = pd.concat(preds, ignore_index=True)
    old = os.path.join(OUT, f"edge_preds_{tag}.pkl")
    if update and not analog and os.path.exists(old):
        o = pd.read_pickle(old)
        pr = pd.concat([o[o.date < pd.Timestamp(f"{first_year}-01-01")], pr], ignore_index=True)
    if analog:
        pr.to_pickle(os.path.join(OUT, f"edge_preds_{tag}_candidate.pkl"))
        pd.concat(imps, axis=1).mean(1).sort_values(ascending=False).to_csv(os.path.join(OUT, f"edge_importance_{tag}.csv"))
        return pr
    pr.to_pickle(os.path.join(OUT, f"edge_preds_{tag}.pkl"))
    pd.concat(imps, axis=1).mean(1).sort_values(ascending=False).to_csv(os.path.join(OUT, f"edge_importance_{tag}.csv"))
    if use_patterns and use_context:
        priors.sort_values().to_csv(os.path.join(OUT, "edge_pattern_priors.csv"))
    signals_from_preds(pr, tag)


def signals_from_preds(pr, tag, q=0.1):
    """Daily cross-sectional ranking: long top decile (pred>0), short bottom decile (pred<0).
    Written in the bench signal format so `python -m cpd.bench report` scores it like every repo."""
    r = pr.groupby("date").pred.rank(pct=True)
    s = pr[((r >= 1 - q) & (pr.pred > 0)) | ((r <= q) & (pr.pred < 0))]
    nifty = set(data.universe()["nifty50"])
    for name, sub in [(f"{tag} (NSE500)", s), (f"{tag} (Nifty50 stocks)", s[s.ticker.astype(str).isin(nifty)])]:
        out = pd.DataFrame({"method": name, "basket": "nse500", "ticker": sub.ticker.astype(str), "date": sub.date,
                            "pattern": "edge", "dir": np.sign(sub.pred).astype(int), "score": sub.pred})
        out.to_csv(os.path.join(OUT, "signals__" + name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fit"])
    ap.add_argument("--no-patterns", action="store_true", help="ablation: context features only")
    ap.add_argument("--no-context", action="store_true", help="ablation: pattern features only")
    ap.add_argument("--analog", action="store_true", help="add cpd.analog features (candidate preds, see cpd.analog eval)")
    ap.add_argument("--update", action="store_true", help="refit the latest year only, keep earlier years' preds")
    a = ap.parse_args()
    build() if a.cmd == "build" else fit(use_patterns=not a.no_patterns, use_context=not a.no_context, analog=a.analog,
                                         update=a.update)
