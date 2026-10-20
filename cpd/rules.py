"""Adapters for the rule-based repos.

Every detector: fn(df, start=None) -> DataFrame[date, pattern, dir, info]
  df    daily OHLCV, DatetimeIndex, lowercase columns (cpd.data.load)
  date  the bar on which the pattern is KNOWN (no future bars used) -> safe to trade next open
  dir   +1 bullish, -1 bearish, 0 neutral/undirected
  start only report events on/after this date (walk-forward cost control)
Where a repo's own code peeks ahead we fix it here and say how in LAG notes.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

VENDOR = os.path.expanduser(os.environ.get("CPD_VENDOR", "~/.cache/chart-pattern-detector/vendor"))
warnings.filterwarnings("ignore")


def _path(p):
    p = os.path.join(VENDOR, p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _out(rows):
    return pd.DataFrame(rows, columns=["date", "pattern", "dir", "info"])


def _first_idx(df, start):
    return 0 if start is None else int(df.index.searchsorted(pd.Timestamp(start)))


def _edges(state: np.ndarray) -> np.ndarray:
    """Indices where a per-bar state switches on (or flips sign). Turns 'state' signals into events."""
    prev = np.r_[0, state[:-1]]
    return np.flatnonzero((state != 0) & (state != prev))


# ---------------------------------------------------------------- BennyThadikaran/stock-pattern
SP_FNS = dict(
    DBOT="find_double_bottom", DTOP="find_double_top", HNSD="find_hns", HNSU="find_reverse_hns",
    TRNG="find_triangles", VCPU="find_bullish_vcp", VCPD="find_bearish_vcp", FLAGU="find_bullish_flag",
    FLAGD="find_bearish_flag", UPTL="find_uptrend_line", DNTL="find_downtrend_line",
    ABCDU="find_bullish_abcd", ABCDD="find_bearish_abcd", BATU="find_bullish_bat", BATD="find_bearish_bat",
    GARTU="find_bullish_gartley", GARTD="find_bearish_gartley", CRABU="find_bullish_crab",
    CRABD="find_bearish_crab", BFLYU="find_bullish_butterfly", BFLYD="find_bearish_butterfly",
)
SP_DIR = {k: (0 if k == "TRNG" else 1 if k in ("DBOT", "HNSU", "UPTL") or k.endswith("U") else -1) for k in SP_FNS}
SP_PIVOT = dict(UPTL="low", FLAGU="low", DNTL="high", FLAGD="high")


def stock_pattern(df, start=None, step=1, lookback=160):
    """Repo scans 'as of the last bar'; we walk forward. get_max_min's centred window means a pivot
    is confirmed 5 bars after it happens, so pivots are masked until then (repo's backtest.py doesn't)."""
    _path("stock-pattern/src")
    import utils as sp

    d = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    pos = pd.Series(np.arange(len(d)), index=d.index)
    piv = {}
    for p in ("both", "high", "low"):
        a = sp.get_max_min(d, 6, 6, p)
        a["known"] = pos[a.index].values + 5
        piv[p] = a
    rows, seen = [], set()
    for t in range(max(_first_idx(df, start), 60), len(d), step):
        w = d.iloc[max(0, t - lookback) : t + 1]
        for code, fn in SP_FNS.items():
            a = piv[SP_PIVOT.get(code, "both")]
            p = a[(a.known <= t) & (a.index >= w.index[0])][["P", "V"]]
            if len(p) < 3:
                continue
            try:
                r = getattr(sp, fn)("SYM", w, p, {})
            except Exception:  # repo raises on some degenerate pivot sets
                r = None
            if r and (r["pattern"], r["start"]) not in seen:
                seen.add((r["pattern"], r["start"]))
                pts = {k: (str(v[0].date()), float(v[1])) for k, v in r["points"].items()}
                rows.append((d.index[t], r["pattern"], SP_DIR.get(r["pattern"], 0), {"start": str(r["start"].date()), "points": pts}))
    return _out(rows)


# ---------------------------------------------------------------- zeta-zetra/chart_patterns
CP_LAG = {"hs": 10, "ihs": 10}  # pivot half-window the repo looks ahead by (others: 3)


def chart_patterns(df, start=None):
    """Repo computes pivots with a centred window over the whole series -> a detection stamped at bar i
    needs bars up to i+3 (i+10 for H&S). We move every event forward by that lag."""
    _path("")
    from chart_patterns.chart_patterns.doubles import find_doubles_pattern
    from chart_patterns.chart_patterns.flag import find_flag_pattern
    from chart_patterns.chart_patterns.head_and_shoulders import find_head_and_shoulders
    from chart_patterns.chart_patterns.inverse_head_and_shoulders import find_inverse_head_and_shoulders
    from chart_patterns.chart_patterns.pennant import find_pennant
    from chart_patterns.chart_patterns.triangles import find_triangle_pattern

    i0 = max(0, _first_idx(df, start) - 120)  # warm-up for the repo's lookbacks
    raw = df.iloc[i0:].reset_index()
    runs = {
        "double_top": (find_doubles_pattern, dict(double="tops"), -1, "double"),
        "double_bottom": (find_doubles_pattern, dict(double="bottoms"), 1, "double"),
        "triangle_sym": (find_triangle_pattern, dict(triangle_type="symmetrical"), 0, "triangle"),
        "hs": (find_head_and_shoulders, {}, -1, "hs"),
        "ihs": (find_inverse_head_and_shoulders, {}, 1, "ihs"),
        "flag": (find_flag_pattern, {}, 0, "flag"),
        "pennant": (find_pennant, {}, 0, "pennant"),
    }
    rows = []
    for name, (fn, kw, dr, _) in runs.items():
        out = fn(raw.copy(), **kw)
        lag = CP_LAG.get(name, 3)
        for i in np.flatnonzero(out["chart_type"].values != ""):
            j = i + lag
            if j < len(raw):
                rows.append((raw["date"].iloc[j], name, dr, {"found_at": str(raw["date"].iloc[i].date())}))
    ev = _out(rows)
    return ev[ev.date >= pd.Timestamp(start)] if start is not None else ev


# ---------------------------------------------------------------- AdventuresInDataScience/ta_patterns
def ta_patterns(df, start=None, which="all"):
    """300 detectors (106 candlestick + 194 chart), causal by construction (verified on prefixes).
    Signals are per-bar states; we emit the bar where a state turns on."""
    _path("ta_patterns/src")
    import ta_patterns as tap

    kw = {"candle_patterns": "all", "chart_patterns": "all"}
    if which == "candles":
        kw["chart_patterns"] = []
    elif which == "charts":
        kw["candle_patterns"] = []
    tbl = tap.batch_all(df.open.values, df.high.values, df.low.values, df.close.values, v=df.volume.values, **kw)
    i0 = _first_idx(df, start)
    rows = []
    for col in tbl.columns:
        s = tbl[col].values.astype(int)
        for i in _edges(s):
            if i >= i0:
                rows.append((df.index[i], col, int(np.sign(s[i])), {}))
    return _out(rows).sort_values("date", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- white07S/TradingPatternScanner
def trading_pattern_scanner(df, start=None):
    """Repo masks use shift(-1) (1 bar ahead), so every label is delayed one bar here.
    Its 'patterns' are 3-bar wiggles; kept for completeness, not recommended."""
    _path("TradingPatternScanner")
    from tradingpatterns import tradingpatterns as tp

    d = df.rename(columns=str.title).copy()
    dirs = {"Head and Shoulder": -1, "Inverse Head and Shoulder": 1, "Double Top": -1, "Double Bottom": 1,
            "Ascending Triangle": 1, "Descending Triangle": -1, "Wedge Up": -1, "Wedge Down": 1,
            "Channel Up": 1, "Channel Down": -1, "Multiple Top": -1, "Multiple Bottom": 1}
    cols = {}
    for fn, col in [(tp.detect_head_shoulder, "head_shoulder_pattern"), (tp.detect_double_top_bottom, "double_pattern"),
                    (tp.detect_triangle_pattern, "triangle_pattern"), (tp.detect_wedge, "wedge_pattern"),
                    (tp.detect_channel, "channel_pattern"), (tp.detect_multiple_tops_bottoms, "multiple_top_bottom_pattern")]:
        cols[col] = fn(d.copy())[col].shift(1).fillna("").astype(str).values  # +1 bar: remove lookahead
    i0 = _first_idx(df, start)
    rows = []
    for col, lab in cols.items():
        for name, dr in dirs.items():
            for i in _edges((lab == name).astype(int)):
                if i >= i0:
                    rows.append((df.index[i], name, dr, {}))
    return _out(rows).sort_values("date", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- xxwikkixx/HarmonicPattern
def harmonic(df, start=None, err=0.10, window=400):
    """Close-only Gartley/Butterfly/Bat/Crab. Repo uses argrelextrema(order=10) on the whole series
    (10-bar lookahead); run on prefixes it is causal but pivots repaint. We run on prefixes."""
    np.NaN = np.nan  # numpy>=2 removed the alias the repo uses
    _path("HarmonicPattern/Harmonics")
    import harmonic_func as hf

    checks = {"Gartley": hf.is_Gartley, "Butterfly": hf.is_Butterfly, "Bat": hf.is_Bat, "Crab": hf.is_Crab}
    px = df["close"].values
    rows, last = [], {}
    for t in range(max(_first_idx(df, start), 60), len(px)):
        idx, pat, _, _ = hf.peak_detect(px[max(0, t - window) : t + 1])
        if len(pat) < 5:
            continue
        moves = np.diff(pat)
        for name, f in checks.items():
            s = f(moves, err)
            if s in (1, -1):
                key = (name, int(s), idx[0] + max(0, t - window))  # same X point = same setup
                if key not in last:
                    last[key] = t
                    rows.append((df.index[t], name, int(s), {}))
    return _out(rows)


# ---------------------------------------------------------------- ednunezg/pytrendline
def trendlines(df, bars=200):
    """Support/resistance trendlines for the LAST `bars` bars (O(n^3), keep small). Visual only:
    not a signal generator, so it is not part of the benchmark."""
    _path("pytrendline")
    import pytrendline

    w = df.iloc[-bars:].reset_index().rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
    cd = pytrendline.CandlestickData(df=w, time_interval="1d", datetime_col="Date")
    res = pytrendline.detect(candlestick_data=cd, trend_type=pytrendline.TrendlineTypes.BOTH,
                             first_pt_must_be_pivot=False, last_pt_must_be_pivot=False, all_pts_must_be_pivots=False,
                             trendline_must_include_global_maxmin_pt=False, min_points_required=3,
                             scan_from_date=None, ignore_breakouts=True, config={})
    lines = []
    for kind in ("support", "resistance"):
        t = res[f"{kind}_trendlines"]
        if len(t):
            for _, r in t.sort_values("score", ascending=False).head(4).iterrows():
                lines.append(dict(kind=kind, x0=r.starts_at_date, x1=w.Date.iloc[-1], m=r.m, b=r.b,
                                  i0=int(r.pointset_indeces[0]), i1=len(w) - 1, dates=list(w.Date)))
    return lines


DETECTORS = {
    "stock-pattern": stock_pattern,
    "chart_patterns": chart_patterns,
    "ta_patterns": ta_patterns,
    "TradingPatternScanner": trading_pattern_scanner,
    "HarmonicPattern": harmonic,
}

if __name__ == "__main__":
    from cpd import data

    df = data.load("RELIANCE", "nifty50")
    st = df.index[-250]
    for k, fn in DETECTORS.items():
        ev = fn(df, start=st)
        assert (ev.date >= st).all() and ev.dir.isin([-1, 0, 1]).all(), k
        print(k, len(ev), ev.pattern.value_counts().head(3).to_dict())
    # causality: detections up to bar T must not change when future bars are added
    cut = df.index[-60]
    for k in ("ta_patterns", "stock-pattern", "TradingPatternScanner"):
        a = DETECTORS[k](df.loc[:cut], start=st)
        b = DETECTORS[k](df, start=st)
        b = b[b.date <= cut]
        assert a[["date", "pattern", "dir"]].reset_index(drop=True).equals(b[["date", "pattern", "dir"]].reset_index(drop=True)), k
    print("lines", len(trendlines(df, 150)))
    print("ok")
