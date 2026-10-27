"""PatternEdge analogs: "this stock drew this shape before - what happened next, and is today's context the same?"

Shape = last W daily closes, z-normalised log price; similarity = Pearson correlation (d^2 = 2W(1-rho)), one matrix
product over sliding windows. Matches are the stock's own past windows, greedily non-overlapping (>= max(W,H) apart so
their outcome periods don't overlap), each ending <= asof - max(H, 2W) so its outcome is known and it is not today's
trend. Context (50/200-DMA slope, close vs DMAs, NIFTY trend, volume trend, quarterly results, patterns) is compared
at each window end -> `ctx` 0..1. Volume never enters the shape distance.

  python -m cpd.analog build [--W 60]   # per-bar causal features for every NSE500 stock -> results/analog_panel*.pkl
  python -m cpd.analog peers            # own-history vs NSE500-peer pool (subsampled)
  python -m cpd.analog eval             # walk-forward comparison -> results/analog_eval.csv + log_analog.txt
  python -m cpd.analog                  # synthetic self-check
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

from cpd import data, evaluate
from cpd.bench import H as H_EVAL, OUT
from cpd.tradeplan import SL_ATR, TP_ATR

PANEL = os.path.join(OUT, "analog_panel.pkl")
GRID = (20, 40, 60, 90, 120)
# context components and the distance at which agreement falls to 1/e (~robust spread across NSE500 stocks)
SCALE = {"ma50_slope": 2.0, "ma200_slope": 1.0, "c_sma50": 0.07, "c_sma200": 0.15, "mkt_trend": 0.08,
         "mkt_r20": 0.05, "vol_trend": 0.4, "salesYoY": 0.25, "profitYoY": 0.5}
FLAT = {"ma50_slope": 0.3, "mkt_trend": 0.01, "profitYoY": 0.05, "salesYoY": 0.05}  # |x| below this = "flat"
SPIKE = 3.0  # volume / 50-bar median volume


# ------------------------------------------------------------------ inputs
def _market(mkt=None):
    if mkt is None:
        from cpd import edge
        mkt = edge.market()
    return mkt.assign(mkt_logc=mkt.mkt_lr.fillna(0).cumsum())


def _results(results, index):
    """`results`: DataFrame from cpd.results.asof, or a symbol to load it with. Optional: None when unavailable."""
    if results is None:
        return None
    try:
        if isinstance(results, str):
            from cpd import results as qr
            results = qr.asof(results, index)
        r = pd.DataFrame(results).reindex(index)
        return r if r[["salesYoY", "profitYoY"]].notna().any().any() else None
    except Exception:
        return None


def context(df, mkt=None, results=None):
    """Per-bar context, causal (each row uses bars <= that row only)."""
    c, v = df.close, df.volume
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    m = _market(mkt).reindex(df.index, method="ffill")
    f = pd.DataFrame(index=df.index)
    f["ma50_slope"] = (s50 / s50.shift(10) - 1) * 100
    f["ma200_slope"] = (s200 / s200.shift(10) - 1) * 100
    f["c_sma50"] = c / s50 - 1
    f["c_sma200"] = c / s200 - 1
    f["mkt_trend"], f["mkt_r20"] = m.mkt_trend, m.mkt_r20
    f["vol_trend"] = np.log(v.rolling(20).mean() / v.rolling(120).mean())
    f["vol_spike"] = v / v.rolling(50, min_periods=20).median().shift()
    f["turnover60"] = (c * v).rolling(60).median()
    f["nifty_logc"] = m.mkt_logc
    r = _results(results, df.index)
    for k in ("salesYoY", "profitYoY", "days_since"):
        f[k] = r[k].astype(float) if r is not None and k in r else np.nan
    f["q"] = r["q"].astype(str) if r is not None and "q" in r else None
    return f.replace([np.inf, -np.inf], np.nan)


def outcomes(df, ctx, H):
    """Per window end e: what happened over the next H bars (NaN when the future is unknown)."""
    c, hi, lo = df.close.values, df.high.values, df.low.values
    n = len(c)
    pad = lambda x: np.r_[x[1:], [np.nan] * H]
    C, HI, LO = (swv(pad(x), H)[:n] for x in (c, hi, lo))  # row e = bars e+1..e+H
    nl = ctx.nifty_logc.values
    nfut = np.r_[nl[H:], [np.nan] * H]
    a = evaluate.atr(df).values
    fwd = C[:, -1] / c - 1
    stop, targ = (c - SL_ATR * a)[:, None], (c + TP_ATR * a)[:, None]
    first = lambda m: np.where(m.any(1), m.argmax(1), H + 1)
    fs, ft = first(LO <= stop), first(HI >= targ)
    o = np.where(ft < fs, "target", np.where(fs <= H, "stop", "time"))  # same bar -> stop
    return pd.DataFrame({"fwd": fwd, "fwdEx": fwd - (np.exp(nfut - nl) - 1), "lx": np.log(C[:, -1] / c) - (nfut - nl),
                         "fwdAtr": (C[:, -1] - c) / a, "mfe": np.nanmax(HI, 1) / c - 1, "mae": np.nanmin(LO, 1) / c - 1,
                         "outcome": o}, index=df.index)


def _zwin(logc, W):
    """Row j = window ending at bar j+W-1, z-normalised. good[j] False for flat / NaN windows."""
    X = swv(logc, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = ((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)).astype(np.float32)
    good = np.isfinite(Z).all(1)
    Z[~good] = 0
    return Z, good


def _valid_ends(df, W, H, a_ok):
    """Bar e can be a match end if its window and outcome contain no corporate-action jump and outcome exists."""
    n = len(df)
    bad = data.corporate_action_mask(df).values.astype(int)
    cs = np.r_[0, np.cumsum(bad)]
    e = np.arange(n)
    lo, hi = np.clip(e - W + 1, 0, n), np.clip(e + H + 1, 0, n)
    return (cs[hi] - cs[lo] == 0) & (e >= W - 1) & (e + H < n) & a_ok


def _pick(c, k, gap):
    """Greedy top-k with suppression of anything within `gap` bars of a chosen end. c: -inf = not allowed."""
    c, out = c.copy(), []
    for _ in range(k):
        j = int(np.argmax(c))
        if c[j] == -np.inf:
            break
        out.append(j)
        c[max(0, j - gap + 1): j + gap] = -np.inf
    return out


def _rand_pick(cands, k, gap, rng):
    """k random allowed ends under the same spacing rule (cands sorted array of allowed ends)."""
    out = []
    for j in rng.choice(cands, min(len(cands), 6 * k), replace=False):
        if all(abs(j - o) >= gap for o in out):
            out.append(int(j))
            if len(out) == k:
                break
    return out


def _agree(F, f0, cols):
    """Context agreement 0..1 per row of F vs vector f0 (mean over available components)."""
    s = np.array([SCALE[c] for c in cols])
    with np.errstate(invalid="ignore"):
        d = np.exp(-np.abs(F - f0) / s)
    return np.where(np.isfinite(d).any(1), np.nanmean(np.where(np.isfinite(d), d, np.nan), 1), np.nan) \
        if d.size else np.full(len(F), np.nan)


def _state(x, key):
    if not np.isfinite(x):
        return "unknown"
    t = FLAT[key]
    return "flat" if abs(x) < t else ("rising" if x > 0 else "falling")


# ------------------------------------------------------------------ live: one query
def _patterns(df, s, e):
    try:
        from cpd import rules
        out = []
        # ponytail: stock-pattern scanned every 20 bars (~0.4 s/window); full step=3 scan is 4x slower
        for name, kw in (("stock-pattern", {"step": 20}), ("HarmonicPattern", {})):
            ev = rules.DETECTORS[name](df.iloc[max(0, s - 400): e + 1], start=df.index[s], **kw)  # detectors look back <= 400
            out += [str(p) for p in ev.pattern]
        return sorted(set(out))
    except Exception:
        return []


def find(df, asof=None, W=60, H=20, k=10, mkt=None, results=None, patterns=True, n_random=1000, seed=0):
    """-> (analogs: list[Analog] best first, summary: dict). See platform/CONTRACT.md "PatternEdge analogs".
    `results`: symbol (str) or a cpd.results.asof DataFrame; skipped gracefully when unavailable."""
    df = df[df.index <= pd.Timestamp(asof)] if asof is not None else df
    i = len(df) - 1
    ctx = context(df, mkt, results)
    out = outcomes(df, ctx, H)
    Z, good = _zwin(np.log(df.close.values), W)
    ends = np.arange(W - 1, len(df))
    ok = good & _valid_ends(df, W, H, np.isfinite(evaluate.atr(df).values))[ends] & (ends <= i - max(H, 2 * W))
    corr = Z @ Z[-1] / W
    gap = max(W, H)
    J = _pick(np.where(ok, corr, -np.inf), k, gap)
    cols = [c for c in SCALE if ctx[c].notna().iloc[-1] and ctx[c].iloc[ends[J]].notna().any()] if J else []
    F = ctx[cols].values
    agree = _agree(F[ends[J]], F[i], cols) if J else np.array([])
    pats_now = _patterns(df, i - W + 1, i) if patterns else []
    analogs = []
    for jj, j in enumerate(J):
        e = ends[j]
        pats = _patterns(df, e - W + 1, e) if patterns else []
        ag = agree[jj]
        if patterns and (pats or pats_now):  # pattern overlap as one more context component (Jaccard)
            jac = len(set(pats) & set(pats_now)) / len(set(pats) | set(pats_now))
            ag = (ag * len(cols) + jac) / (len(cols) + 1)
        r = ctx.iloc[e]
        res = None if not np.isfinite(r.salesYoY) and not np.isfinite(r.profitYoY) else {
            "q": r.q, "salesYoY": _num(r.salesYoY), "profitYoY": _num(r.profitYoY)}
        o = out.iloc[e]
        analogs.append({"start": _t(df.index[e - W + 1]), "end": _t(df.index[e]), "sim": float(max(corr[j], 0)),
                        "ctx": float(ag), "fwd": float(o.fwd), "fwdEx": float(o.fwdEx), "fwdAtr": float(o.fwdAtr),
                        "mfe": float(o.mfe), "mae": float(o.mae), "outcome": o.outcome,
                        "ma50Slope": _num(r.ma50_slope), "results": res, "patterns": pats})
    summary = _summary(df, ctx, out, analogs, [ends[j] for j in J], W, H, k, gap, ok, ends, n_random, seed, pats_now)
    return analogs, summary


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


def _t(ts):
    return ts.strftime("%Y-%m-%d")


def _summary(df, ctx, out, analogs, E, W, H, k, gap, ok, ends, n_random, seed, pats_now):
    i, c0 = len(df) - 1, float(df.close.iloc[-1])
    now = ctx.iloc[-1]
    s = {"analog_n": len(analogs), "W": W, "H": H, "ma50_slope_now": _num(now.ma50_slope), "patterns_now": pats_now}
    flags = []
    if not analogs:
        return {**s, "flags": ["no valid analogs (history too short)"], "forecast": None}
    fwd = np.array([a["fwd"] for a in analogs])
    fx = np.array([a["fwdEx"] for a in analogs])
    w = np.array([a["sim"] * a["ctx"] for a in analogs]) + 1e-9
    up = fwd > 0
    s.update(analog_p_up=float(up.mean()), analog_median_fwd=float(np.median(fwd)),
             analog_p_up_ex=float((fx > 0).mean()), analog_median_fwdEx=float(np.median(fx)),
             analog_ctx_p_up=float((w * up).sum() / w.sum()), analog_ctx_fwd=float((w * fwd).sum() / w.sum()),
             analog_ctx_fwdEx=float((w * fx).sum() / w.sum()),
             analog_sim_mean=float(np.mean([a["sim"] for a in analogs])),
             analog_ctx_mean=float(np.mean([a["ctx"] for a in analogs])),
             ma50_slope_analogs=_num(np.nanmedian([a["ma50Slope"] if a["ma50Slope"] is not None else np.nan for a in analogs])),
             outcomes={o: sum(a["outcome"] == o for a in analogs) for o in ("target", "stop", "time")})
    # forecast cone: analogs' normalised paths rescaled to today's close
    P = np.array([df.close.values[e: e + H + 1] / df.close.values[e] for e in E]) * c0
    q = np.nanpercentile(P, [10, 50, 90], axis=0)
    s["forecast"] = {"h": list(range(H + 1)), "lo": q[0].tolist(), "median": q[1].tolist(), "hi": q[2].tolist()}
    # significance vs random windows of the same stock under the same spacing + causal rules
    cands = ends[ok]
    if n_random and len(cands) >= k:
        rng = np.random.default_rng(seed)
        fxa = out.fwdEx.values
        R = [fxa[_rand_pick(cands, k, gap, rng)] for _ in range(n_random)]
        rp = np.array([(r > 0).mean() for r in R])
        rm = np.array([np.median(r) for r in R])
        pct = {"p_up_ex": float((rp < (fx > 0).mean()).mean() * 100 + (rp == (fx > 0).mean()).mean() * 50),
               "median_fwdEx": float((rm < np.median(fx)).mean() * 100)}
        s["analog_pctile"] = pct
        if max(pct["p_up_ex"], pct["median_fwdEx"]) < 90 and min(pct["p_up_ex"], pct["median_fwdEx"]) > 10:
            flags.append(f"no edge vs random windows (p_up {pct['p_up_ex']:.0f}th, median excess {pct['median_fwdEx']:.0f}th "
                         f"percentile of {n_random} random {k}-window sets)")
    # context mismatches vs the analogs that went up (or down, if most went down)
    grp = up if up.sum() >= (~up).sum() else ~up
    word = "went up" if grp is up else "went down"
    Er = np.array(E)[grp]
    for key, label, fmt in (("ma50_slope", "50-DMA", "{:+.1f}%/10 bars"), ("mkt_trend", "NIFTY vs its 200-DMA", "{:+.1%}"),
                            ("profitYoY", "Quarterly profit YoY", "{:+.0%}"), ("salesYoY", "Quarterly sales YoY", "{:+.0%}")):
        v, past = now[key], ctx[key].values[Er]
        past = past[np.isfinite(past)]
        if len(past) < 2 or not np.isfinite(v):
            continue
        m = float(np.median(past))
        st_now, st_past = _state(v, key), _state(m, key)
        if st_now != st_past:
            flags.append(f"{label} {st_now} now ({fmt.format(v)}); in the {grp.sum()} analogs that {word} it was "
                         f"{st_past} (median {fmt.format(m)})")
    # volume spike in today's window without results backing it
    vs = ctx.vol_spike.values[i - W + 1: i + 1]
    if np.nanmax(vs, initial=0) >= SPIKE:
        js = i - W + 1 + int(np.nanargmax(vs))
        ds, pg = ctx.days_since.values[js], ctx.profitYoY.values[js]
        backed = np.isfinite(ds) and ds <= 5 and np.isfinite(pg) and pg > 0
        if not backed:
            why = "no results data" if not np.isfinite(ds) else "no fresh positive results"
            flags.append(f"volume spike without results support ({np.nanmax(vs):.1f}x on {_t(df.index[js])}, {why}): "
                         "operator-style risk")
    s["flags"] = flags
    return s


# ------------------------------------------------------------------ panel: every bar, causal
FEAT_COLS = ["analog_n", "analog_sim_mean", "analog_p_up", "analog_fwd", "analog_ctx_p_up", "analog_ctx_fwd",
             "analog_rawfwd", "analog_rand_fwd", "analog_rand_p_up", "ma50_slope", "ma200_slope"]
TEST_COLS = ["ma_ag_n", "ma_ag_up", "ma_ag_fwd", "ma_dis_n", "ma_dis_up", "ma_dis_fwd",
             "nf_ag_n", "nf_ag_up", "nf_ag_fwd", "nf_dis_n", "nf_dis_up", "nf_dis_fwd",
             "pr_ag_n", "pr_ag_up", "pr_ag_fwd", "pr_dis_n", "pr_dis_up", "pr_dis_fwd"]


def features(df, W=60, H=H_EVAL, k=10, mkt=None, results=None, rows=None, block=512, seed=0):
    """Per-bar analog features (log excess-vs-NIFTY outcomes over H). Strictly causal: row i only sees windows ending
    <= i - max(H, 2W), whose outcomes end <= i. `rows`: bar positions to compute (default all)."""
    n = len(df)
    ctx = context(df, mkt, results)
    out = outcomes(df, ctx, H)
    lx, lraw = out.lx.values, np.log1p(out.fwd.values)
    Z, good = _zwin(np.log(df.close.values), W)  # Z row j <-> end bar j+W-1
    ok_end = np.zeros(n, bool)
    ok_end[W - 1:] = good
    ok_end &= _valid_ends(df, W, H, np.isfinite(evaluate.atr(df).values)) & np.isfinite(lx)
    cols = [c for c in SCALE if ctx[c].notna().any()]
    F = ctx[cols].values
    ma, nf = ctx.ma50_slope.values, ctx.mkt_trend.values
    pr = ctx.profitYoY.values
    lag, gap = max(H, 2 * W), max(W, H)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    rows = rows[rows >= W - 1 + lag]
    R = np.full((n, len(FEAT_COLS) + len(TEST_COLS)), np.nan, np.float32)
    rng = np.random.default_rng(seed)
    Zall = np.zeros((n, W), np.float32)
    Zall[W - 1:] = Z
    allowed = np.where(ok_end, 0.0, -np.inf).astype(np.float32)

    def split(sel, x):
        g = x[sel]
        return [len(g), (g > 0).mean() if len(g) else np.nan, g.mean() if len(g) else np.nan]

    for b0 in range(0, len(rows), block):
        rb = rows[b0: b0 + block]
        C = (Zall[rb] @ Zall[: rb[-1] + 1].T) / W
        for r, i in enumerate(rb):
            lim = i - lag + 1
            c = C[r, :lim] + allowed[:lim]
            J = np.array(_pick(c, k, gap), int)
            if len(J) < 3:
                continue
            sim = np.clip(C[r, J], 0, 1)
            ag = _agree(F[J], F[i], cols)
            w = sim * np.nan_to_num(ag, nan=0.5) + 1e-9
            x = lx[J]
            cands = np.flatnonzero(ok_end[:lim])
            Jr = _rand_pick(cands, k, gap, rng)
            xr = lx[Jr]
            m_ag = np.sign(ma[J]) == np.sign(ma[i])
            n_ag = (nf[J] > 0) == (nf[i] > 0)
            p_ok = np.isfinite(pr[J]) & np.isfinite(pr[i])
            p_ag = p_ok & ((pr[J] > 0) == (pr[i] > 0))
            R[i] = [len(J), sim.mean(), (x > 0).mean(), x.mean(), (w * (x > 0)).sum() / w.sum(), (w * x).sum() / w.sum(),
                    lraw[J].mean(), xr.mean(), (xr > 0).mean(), ma[i], ctx.ma200_slope.values[i],
                    *split(m_ag, x), *split(~m_ag, x), *split(n_ag, x), *split(~n_ag, x),
                    *split(p_ag, x), *split(p_ok & ~p_ag, x)]
    f = pd.DataFrame(R, index=df.index, columns=FEAT_COLS + TEST_COLS)
    for c in ("vol_trend", "turnover60", "salesYoY", "profitYoY", "days_since"):
        f[c] = ctx[c].astype(np.float32)
    return f.iloc[rows]


def _edge_rows():
    """(ticker -> dates) of the PatternEdge panel, so features line up with its rows."""
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    return {t: g.values for t, g in panel.groupby(panel.ticker.astype(str), observed=True).date}


def build(W=60, cols=None):
    from cpd import edge
    path = PANEL if W == 60 else os.path.join(OUT, f"analog_panel_W{W}.pkl")
    mk = edge.market()
    want = _edge_rows()
    frames = []
    for n, (t, dates) in enumerate(want.items()):
        df = data.load(t, "nse500")
        pos = df.index.get_indexer(pd.DatetimeIndex(dates))
        f = features(df, W=W, mkt=mk, results=t, rows=pos[pos >= 0])
        if cols:
            f = f[cols]
        frames.append(f.reset_index().assign(ticker=t))
        if n % 50 == 0:
            print("build W", W, n, t, len(f), flush=True)
    p = pd.concat(frames, ignore_index=True)
    p["ticker"] = p.ticker.astype("category")
    p.to_pickle(path)
    return p


# ------------------------------------------------------------------ peers: own history vs NSE500 pool (subsampled)
def peers(W=60, H=H_EVAL, k=10, qstep=20, lstep=5, start="2018-01-01"):
    """Queries every `qstep` bars from `start` for every stock; library = every `lstep`-th window of every stock.
    Peer pool excludes the query's own stock; a peer window is allowed if its outcome ended <= query date
    (calendar causal) and it ended >= 2W calendar-bars before the query. Spacing within a stock as in own-history.
    Scored on the same rows for both pools."""
    from cpd import edge
    mk = edge.market()
    want = _edge_rows()
    lib_Z, lib_x, lib_end, lib_known, lib_tk, qs = [], [], [], [], [], []
    for ti, (t, dates) in enumerate(want.items()):
        df = data.load(t, "nse500")
        ctx = context(df, mk)
        out = outcomes(df, ctx, H)
        Z, good = _zwin(np.log(df.close.values), W)
        e = np.arange(W - 1, len(df))
        ok = good & _valid_ends(df, W, H, np.isfinite(evaluate.atr(df).values))[e] & np.isfinite(out.lx.values[e])
        sel = np.flatnonzero(ok)[::lstep]
        lib_Z.append(Z[sel]), lib_x.append(out.lx.values[e[sel]].astype(np.float32))
        lib_end.append(df.index.values[e[sel]]), lib_tk.append(np.full(len(sel), ti, np.int16))
        lib_known.append(df.index.values[np.minimum(e[sel] + H, len(df) - 1)])
        dpos = df.index.get_indexer(pd.DatetimeIndex(dates))
        dpos = dpos[(dpos >= 0) & (df.index.values[np.maximum(dpos, 0)] >= np.datetime64(start))][::qstep]
        dpos = dpos[dpos >= W - 1 + 2 * W]
        for i in dpos:
            qs.append((t, ti, df.index[i], Z[i - W + 1]))
    Zl, xl = np.vstack(lib_Z), np.concatenate(lib_x)
    known, endd, tk = np.concatenate(lib_known), np.concatenate(lib_end), np.concatenate(lib_tk)
    print("peers: library", len(Zl), "queries", len(qs), flush=True)
    order = np.argsort(endd, kind="stable")  # time-sorted by window end
    Zl, xl, known, endd, tk = Zl[order], xl[order], known[order], endd[order], tk[order]
    rows = []
    gapd = np.timedelta64(int(max(W, H) * 1.45), "D")  # ~bars -> calendar days
    lagd = np.timedelta64(int(max(H, 2 * W) * 1.45), "D")
    for b0 in range(0, len(qs), 64):  # 64 x 550k float32 ~ 140 MB
        qb = qs[b0: b0 + 64]
        C = np.vstack([q[3] for q in qb]) @ Zl.T / W
        for r, (t, ti, d, _) in enumerate(qb):
            d64 = np.datetime64(d)
            lim = np.searchsorted(endd, d64 - lagd, side="right")  # library sorted by window end
            c = np.where(known[:lim] <= d64, C[r, :lim], -np.inf)
            own = tk[:lim] == ti
            res = {}
            for pool, pm in (("own", own), ("peer", ~own), ("all", None)):
                cc = c if pm is None else np.where(pm, c, -np.inf)
                m = min(k * 40, lim - 1)
                if m < k:
                    res[pool] = (np.nan,) * 3
                    continue
                top = np.argpartition(-cc, m)[:m]
                top = top[np.argsort(-cc[top])]
                pick = []
                for j in top:
                    if cc[j] == -np.inf:
                        break
                    if all(tk[j] != tk[p] or abs(endd[j] - endd[p]) >= gapd for p in pick):
                        pick.append(j)
                        if len(pick) == k:
                            break
                res[pool] = (xl[pick].mean(), (xl[pick] > 0).mean(), np.clip(C[r, pick], 0, 1).mean()) if len(pick) >= 3 else (np.nan,) * 3
            rows.append((t, d, *res["own"], *res["peer"], *res["all"]))
        if b0 % 6400 == 0:
            print("peers", b0, len(qs), flush=True)
    p = pd.DataFrame(rows, columns=["ticker", "date", "own_fwd", "own_p_up", "own_sim", "peer_fwd", "peer_p_up", "peer_sim",
                                    "all_fwd", "all_p_up", "all_sim"])
    p.to_pickle(os.path.join(OUT, f"analog_peers_W{W}.pkl"))
    return p


# ------------------------------------------------------------------ evaluation
def _daily_ic(df, col):
    d = df[["date", col, "y"]].dropna()
    d = d[d.groupby("date").y.transform("size") >= 30]
    rk = d.groupby("date")[[col, "y"]].rank(pct=True)
    rk["date"] = d.date.values
    return rk.groupby("date").apply(lambda g: g[col].corr(g.y)), d, rk


def _ic(df, col, step=H_EVAL):
    """Daily cross-sectional Spearman IC of `col` vs y. t-stat on every `step`-th date (non-overlapping labels)."""
    ic, d, rk = _daily_ic(df, col)
    ics = ic.iloc[::step]
    t = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 2 else np.nan
    q = rk[col]
    spread = (d.y[q >= 0.9].groupby(d.date[q >= 0.9]).mean() - d.y[q <= 0.1].groupby(d.date[q <= 0.1]).mean()).mean()
    yr = ic.groupby(ic.index.year).mean()
    return {"ic": ic.mean(), "t_nonoverlap": t, "spread_bps": spread * 1e4, "n_rows": len(d), "n_days": len(ic),
            **{f"ic_{y}": v for y, v in yr.items()}}


def _brier(df):
    """Brier of analog p_up (excess>0) for y>0 vs baseline = stock's own causal unconditional rate."""
    d = df.dropna(subset=["analog_p_up", "y"]).sort_values(["ticker", "date"])
    hit = (d.y > 0).astype(float)
    g = hit.groupby(d.ticker, observed=True)
    base = (g.cumsum() - hit).groupby(d.ticker, observed=True).shift(H_EVAL - 1)  # labels known by t: those ending <= t
    cnt = g.cumcount().groupby(d.ticker, observed=True).shift(H_EVAL - 1)
    base = (base / cnt).where(cnt > 100)
    m = base.notna() & (d.date >= "2018-01-01")
    bs = ((d.analog_p_up[m] - hit[m]) ** 2).mean()
    bc = ((d.analog_ctx_p_up[m] - hit[m]) ** 2).mean()
    b0 = ((base[m] - hit[m]) ** 2).mean()
    return {"brier_p_up": bs, "brier_ctx_p_up": bc, "brier_base": b0, "bss_p_up": 1 - bs / b0, "bss_ctx_p_up": 1 - bc / b0}


def _pairs(d, pfx):
    """Pair-level hit rate: sign(analog excess) == sign(query label), analogs split by context agreement."""
    out = {}
    up = (d.y > 0).values
    for g in ("ag", "dis"):
        n, u = d[f"{pfx}_{g}_n"].fillna(0).values, d[f"{pfx}_{g}_up"].fillna(0).values
        hits = np.where(up, u * n, (1 - u) * n).sum()
        out[f"{pfx}_{g}_pairs"] = n.sum()
        out[f"{pfx}_{g}_hit"] = hits / max(n.sum(), 1)
    return out


def eval_all(log):
    from cpd import edge
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    base = panel[["ticker", "date", "fwd"]].copy()
    del panel
    y = (base.fwd - base.groupby("date").fwd.transform("mean")).values
    lo, hi = np.nanpercentile(y, [1, 99])
    base["y"] = np.clip(y, lo, hi)
    base["ticker"] = base.ticker.astype(str)
    a = pd.read_pickle(PANEL)
    a["ticker"] = a.ticker.astype(str)
    df = base.merge(a, on=["ticker", "date"], how="left")
    pe = pd.read_pickle(os.path.join(OUT, "edge_preds_PatternEdge.pkl"))[["ticker", "date", "pred"]]
    pe["ticker"] = pe.ticker.astype(str)
    df = df.merge(pe.rename(columns={"pred": "patternedge"}), on=["ticker", "date"], how="left")
    candidate = os.path.join(OUT, "edge_preds_PatternEdgeAnalog_candidate.pkl")
    if os.path.exists(candidate):
        pa = pd.read_pickle(candidate)[["ticker", "date", "pred"]]
        pa["ticker"] = pa.ticker.astype(str)
        df = df.merge(pa.rename(columns={"pred": "patternedge_analog"}), on=["ticker", "date"], how="left")
    test = df[df.date >= "2018-01-01"]
    test = test[test.patternedge.notna()]  # identical row set for every signal
    liquid = test[test.turnover60 >= 5e7]
    rows = []
    sigs = [("a_analog_fwd", "analog_fwd"), ("a_analog_p_up", "analog_p_up"), ("a_analog_rawfwd", "analog_rawfwd"),
            ("b_analog_ctx_fwd", "analog_ctx_fwd"), ("b_analog_ctx_p_up", "analog_ctx_p_up"),
            ("c_random_fwd", "analog_rand_fwd"), ("c_random_p_up", "analog_rand_p_up"),
            ("ma50_slope", "ma50_slope"), ("ma_agree_fwd", "ma_ag_fwd"), ("ma_disagree_fwd", "ma_dis_fwd"),
            ("d_PatternEdge", "patternedge"), ("e_PatternEdge+analog", "patternedge_analog")]
    for univ, d in (("all", test), ("turnover>=5cr", liquid)):
        for name, col in sigs:
            if col in d:
                rows.append({"signal": name, "W": 60, "universe": univ, **_ic(d, col)})
    for W in GRID:
        p = os.path.join(OUT, f"analog_panel_W{W}.pkl")
        if W == 60 or not os.path.exists(p):
            continue
        g = pd.read_pickle(p)
        g["ticker"] = g.ticker.astype(str)
        dd = test[["ticker", "date", "y", "turnover60"]].merge(g, on=["ticker", "date"], how="left")
        for univ, d in (("all", dd), ("turnover>=5cr", dd[dd.turnover60 >= 5e7])):
            for name, col in (("a_analog_fwd", "analog_fwd"), ("b_analog_ctx_fwd", "analog_ctx_fwd"), ("c_random_fwd", "analog_rand_fwd")):
                rows.append({"signal": name, "W": W, "universe": univ, **_ic(d, col)})
    pp = os.path.join(OUT, "analog_peers_W60.pkl")
    if os.path.exists(pp):
        g = pd.read_pickle(pp)
        g["ticker"] = g.ticker.astype(str)
        dd = test[["ticker", "date", "y", "turnover60"]].merge(g, on=["ticker", "date"], how="inner")
        for univ, d in (("all", dd), ("turnover>=5cr", dd[dd.turnover60 >= 5e7])):
            for pool in ("own", "peer", "all"):
                rows.append({"signal": f"pool_{pool}_fwd (monthly-ish sample)", "W": 60, "universe": univ, **_ic(d, f"{pool}_fwd", step=1)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "analog_eval.csv"), index=False)
    pd.set_option("display.width", 250)
    log(res[["signal", "W", "universe", "ic", "t_nonoverlap", "spread_bps", "n_days"]].round(4).to_string())
    log("per-year IC (all universe, W=60):")
    yc = [c for c in res.columns if c.startswith("ic_")]
    log(res[(res.universe == "all") & (res.W == 60)].set_index("signal")[yc].round(4).to_string())
    for univ, d in (("all", test), ("turnover>=5cr", liquid)):
        log(f"[{univ}] Brier:", {k: round(v, 5) for k, v in _brier(d).items()})
        for pfx, what in (("ma", "50-DMA slope sign"), ("nf", "NIFTY above/below 200-DMA"), ("pr", "profit YoY sign")):
            pr = _pairs(d, pfx)
            log(f"[{univ}] analog->query hit rate by {what}: agree {pr[f'{pfx}_ag_hit']:.4f} (n={pr[f'{pfx}_ag_pairs']:.0f}) "
                f"disagree {pr[f'{pfx}_dis_hit']:.4f} (n={pr[f'{pfx}_dis_pairs']:.0f})")
    for univ, d in (("all", test), ("turnover>=5cr", liquid)):
        for x, b in (("analog_fwd", "analog_rand_fwd"), ("analog_ctx_fwd", "analog_fwd"), ("analog_ctx_p_up", "analog_p_up")):
            dd = d.dropna(subset=[x, b])
            diff = (_daily_ic(dd, x)[0] - _daily_ic(dd, b)[0]).dropna()
            ds = diff.iloc[::H_EVAL]
            log(f"[{univ}] IC({x}) - IC({b}) = {diff.mean():+.4f}, paired t (every {H_EVAL}th day) "
                f"{ds.mean() / ds.std() * np.sqrt(len(ds)):.2f}")
    _results_check(base, log)
    if "patternedge_analog" in test:
        dl = {}
        for univ, d in (("all", test), ("turnover>=5cr", liquid)):
            dd = d.dropna(subset=["patternedge_analog"])
            diff = (_daily_ic(dd, "patternedge_analog")[0] - _daily_ic(dd, "patternedge")[0]).dropna()
            ds = diff.iloc[::H_EVAL]
            log(f"[{univ}] (e)-(d) IC delta {diff.mean():+.4f}, paired t (every {H_EVAL}th day) "
                f"{ds.mean() / ds.std() * np.sqrt(len(ds)):.2f}, per-year:",
                diff.groupby(diff.index.year).mean().round(4).to_dict())
            dl[univ] = diff.mean()
        return res, dl["all"]
    return res, None


def _results_check(base, log):
    """Results context on the symbols cpd.results currently covers (recomputed here; the panel may predate the data)."""
    try:
        from cpd import edge, results as qr
        syms = sorted(set(qr.load().symbol) & set(base.ticker))
    except Exception as ex:
        log("results check skipped:", ex)
        return
    mk, fr = edge.market(), []
    for t in syms:
        df = data.load(t, "nse500")
        g = base[base.ticker == t]
        pos = df.index.get_indexer(pd.DatetimeIndex(g.date))
        f = features(df, mkt=mk, results=t, rows=pos[pos >= 0])
        fr.append(g.merge(f.reset_index(), on="date"))
    d = pd.concat(fr)
    d = d[d.date >= "2015-01-01"]
    pr = _pairs(d, "pr")
    log(f"results check ({len(syms)} symbols with filings, 2015+): analog->query hit rate by profit YoY sign: agree "
        f"{pr['pr_ag_hit']:.4f} (n={pr['pr_ag_pairs']:.0f}) disagree {pr['pr_dis_hit']:.4f} (n={pr['pr_dis_pairs']:.0f}); "
        f"pooled IC ctx_fwd {d.analog_ctx_fwd.corr(d.y, method='spearman'):.4f} vs fwd {d.analog_fwd.corr(d.y, method='spearman'):.4f}")


def main_eval():
    logf = open(os.path.join(OUT, "log_analog.txt"), "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log("=" * 30, pd.Timestamp.now())
    res, dlt = eval_all(log)
    if dlt is not None:
        log(f"(e) - (d) IC delta: {dlt:+.4f}")
    return res


# ------------------------------------------------------------------ self-check
def _selfcheck():
    rng = np.random.default_rng(1)
    n, W, H = 1500, 60, 20
    lr = rng.normal(0, 0.01, n)
    shape = np.sin(np.linspace(0, 3 * np.pi, W)) * 0.03 + np.linspace(0, 0.05, W)
    plant = 400
    c = 100 * np.exp(np.cumsum(lr))
    c[plant - W + 1: plant + 1] = c[plant - W] * np.exp(shape)
    c[plant + 1:] = c[plant] * np.exp(np.cumsum(lr[plant + 1:]))
    c[-W:] = c[-W - 1] * np.exp(shape)
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e6}, index=idx)
    mkt = pd.DataFrame({"mkt_lr": 0.0, "mkt_r20": 0.0, "mkt_trend": 0.0}, index=idx)
    an, s = find(df, W=W, H=H, k=5, mkt=mkt, patterns=False, n_random=200)
    assert an[0]["end"] == _t(idx[plant]), (an[0]["end"], _t(idx[plant]))
    assert an[0]["sim"] > 0.999
    cut = idx[n - 1 - max(H, 2 * W)]
    assert all(pd.Timestamp(a["end"]) <= cut for a in an)
    ends = sorted(idx.get_loc(pd.Timestamp(a["end"])) for a in an)
    assert all(b - a >= max(W, H) for a, b in zip(ends, ends[1:]))
    # panel features: row at the end must see the planted match; nothing may use bars after the row
    f = features(df, W=W, H=H, k=5, mkt=mkt)
    f2 = features(df.iloc[: n - 30], W=W, H=H, k=5, mkt=mkt)
    common = f2.index[f2.index.isin(f.index)]
    a1, a2 = f.loc[common, FEAT_COLS[:7]].values, f2.loc[common, FEAT_COLS[:7]].values
    assert np.allclose(np.nan_to_num(a1, nan=9), np.nan_to_num(a2, nan=9), atol=1e-5), "features depend on the future"
    print("self-check ok:", an[0]["start"], "->", an[0]["end"], "sim", round(an[0]["sim"], 4), "| flags:", s["flags"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="check", choices=["check", "build", "peers", "eval"])
    ap.add_argument("--W", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "check":
        _selfcheck()
    elif a.cmd == "build":
        build(a.W, cols=None if a.W == 60 else ["analog_fwd", "analog_ctx_fwd", "analog_rand_fwd", "analog_p_up", "analog_n"])
    elif a.cmd == "peers":
        peers(a.W)
    else:
        main_eval()
