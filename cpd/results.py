"""Point-in-time quarterly results (NSE filings) as model features. Built by scripts/fetch_results.py.

quarterly.parquet: symbol, period_end, announced (first public broadcast, IST naive), consolidated,
sales / net_profit (INR crore), eps (basic, INR, as reported), source_url. First-announced values only.
The file may hold both bases for a quarter (standalone kept around the switch to consolidated reporting);
load() returns one row per symbol-quarter, consolidated preferred; load(all_bases=True) returns everything.
asof() computes each feature within one basis (consolidated first, standalone as per-field fallback).
q is the calendar quarter of period_end: '2019Q3' = quarter ended 30-Sep-2019 (Indian FY20 Q2).
actions.parquet: symbol, ex_date, factor (bonus/split share multipliers) -- used to put past EPS on today's share basis
using only actions already effective at each announcement, so eps_sue stays point-in-time.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

_REPO_COPY = os.path.join(os.path.dirname(__file__), "..", "data", "fundamentals", "quarterly.parquet")  # get_data.py
PATH = os.environ.get("CPD_FUNDAMENTALS") or (
    _REPO_COPY if os.path.exists(_REPO_COPY) and not os.path.exists(os.path.expanduser("~/.cache/chart-pattern-detector/fundamentals"))
    else os.path.expanduser("~/.cache/chart-pattern-detector/fundamentals/quarterly.parquet"))
COLS = ["q", "salesYoY", "profitYoY", "eps_sue", "days_since"]


@lru_cache(None)
def _read(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load(path: str = PATH, all_bases: bool = False) -> pd.DataFrame:
    df = _read(path)
    if all_bases:
        return df
    return (df.sort_values(["symbol", "period_end", "consolidated"])
              .drop_duplicates(["symbol", "period_end"], keep="last").reset_index(drop=True))


@lru_cache(None)
def load_actions(path: str = os.path.join(os.path.dirname(PATH), "actions.parquet")) -> pd.DataFrame:
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["symbol", "ex_date", "factor"])


def _yoy(cur, base):
    return (cur / base - 1.0) if np.isfinite(cur) and np.isfinite(base) and base > 0 else None


def _features(g: pd.DataFrame, acts=()) -> pd.DataFrame:
    """Per-quarter features for one symbol; uses only quarters announced up to that row's announcement.
    acts: [(ex_date, factor)]. EPS of quarter k is on the share basis at its announcement A_k; to compare at A_i
    divide by the product of factors with A_k < ex_date <= A_i."""
    g = g.sort_values("period_end").reset_index(drop=True)
    pe = g["period_end"]
    A = g["announced"].values
    raw_eps = g["eps"].to_numpy(float)
    key = {(p.year, p.month): i for i, p in enumerate(pe)}
    out = []
    for i, r in g.iterrows():
        adj = np.array([np.prod([f for t, f in acts if A[k] < t <= A[i]] or [1.0]) for k in range(len(g))])
        eps = raw_eps / adj
        j = key.get((pe[i].year - 1, pe[i].month))
        # the year-ago quarter must be the same basis and already public when this one is announced
        b = g.loc[j] if j is not None and g.at[j, "consolidated"] == r.consolidated and g.at[j, "announced"] < r.announced else None
        # seasonal random-walk surprise: d_t = EPS_t - EPS_{t-4}; scale = std of the 8 previous d's
        d = []
        for k in range(i, -1, -1):
            jj = key.get((pe[k].year - 1, pe[k].month))
            if jj is not None and g.at[k, "consolidated"] == g.at[jj, "consolidated"]:
                d.append(eps[k] - eps[jj])
            else:
                d.append(np.nan)
            if len(d) == 9:
                break
        prev = np.array(d[1:], float)
        sue = None
        if len(prev) == 8 and np.isfinite(prev).all() and np.isfinite(d[0]):
            sd = prev.std(ddof=1)
            sue = float(d[0] / sd) if sd > 0 else None
        out.append(dict(announced=r.announced, q=f"{pe[i].year}Q{(pe[i].month - 1) // 3 + 1}",
                        salesYoY=_yoy(r.sales, b.sales) if b is not None else None,
                        profitYoY=_yoy(r.net_profit, b.net_profit) if b is not None else None, eps_sue=sue))
    return pd.DataFrame(out, columns=["announced", "q", "salesYoY", "profitYoY", "eps_sue"])


def asof(symbol: str, dates, df: pd.DataFrame | None = None, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each date: features of the latest result announced strictly before that date (00:00 of the date).
    Calendar quarters: '2019Q3' = quarter ending Sep-2019. Rows with nothing announced yet are all None."""
    df = load(all_bases=True) if df is None else df
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    g = df[df["symbol"] == symbol]
    if g.empty:
        return pd.DataFrame({c: [None] * len(idx) for c in COLS}, index=idx)
    a = load_actions() if actions is None else actions
    a = a[a["symbol"] == symbol]
    acts = list(zip(pd.to_datetime(a["ex_date"]), a["factor"]))
    fc, fs = (_features(g[g["consolidated"] == b], acts).set_index("q") for b in (True, False))
    rows = []
    for q in sorted(set(fc.index) | set(fs.index)):
        c = fc.loc[q] if q in fc.index else None
        s_ = fs.loc[q] if q in fs.index else None
        r, used = {"q": q}, set()
        for k in ("salesYoY", "profitYoY", "eps_sue"):
            for name, src in (("c", c), ("s", s_)):
                if src is not None and pd.notna(src[k]):
                    r[k] = src[k]
                    used.add(name)
                    break
            else:
                r[k] = None
        # visible only once every basis it draws on is public
        r["announced"] = max([src.announced for name, src in (("c", c), ("s", s_)) if name in used and src is not None]
                             or [(c if c is not None else s_).announced])
        rows.append(r)
    f = pd.DataFrame(rows).sort_values("announced", kind="stable")
    # results announced on the same instant collapse to the later quarter
    f = f.drop_duplicates("announced", keep="last")
    pos = np.searchsorted(f["announced"].values, idx.normalize().values, side="left") - 1  # strictly before
    res = f.iloc[np.clip(pos, 0, None)].reset_index(drop=True)
    res["days_since"] = (idx.normalize() - pd.DatetimeIndex(res["announced"]).normalize()).days
    res = res[COLS].astype(object)
    res.loc[pos < 0, :] = None
    res = res.where(res.notna(), None)
    res.index = idx
    return res


if __name__ == "__main__":
    # synthetic: EPS = 1 + 0.1*n with a jump at the last quarter; sales +10%/yr; profit base <= 0 once
    q = pd.period_range("2015Q1", "2018Q4", freq="Q")
    n = np.arange(len(q))
    noise = np.array([0, .02, -.01, .03, -.02, .01, 0, -.03, .02, -.01, .01, 0, -.02, .03, .01, 0])
    syn = pd.DataFrame(dict(symbol="X", period_end=q.end_time.normalize(), consolidated=True,
                            announced=q.end_time.normalize() + pd.Timedelta(days=30, hours=16),
                            sales=100 * 1.1 ** (n / 4), net_profit=np.where(n == 3, -5.0, 10.0 + n),
                            eps=1 + 0.1 * n + noise))
    syn.loc[15, "eps"] += 1.0  # surprise in 2018Q4
    ann = syn.announced[15]  # 2019-01-30 16:00
    noacts = pd.DataFrame(columns=["symbol", "ex_date", "factor"])
    r = asof("X", [ann.normalize(), ann.normalize() + pd.Timedelta(days=1), "2015-01-01", "2019-03-01"], syn, noacts)
    assert r.iloc[0].q == "2018Q3", "same-day announcement must not be visible that day (strictly before)"
    assert r.iloc[1].q == "2018Q4" and r.iloc[1].days_since == 1
    assert r.iloc[2].isna().all(), "nothing announced yet"
    assert abs(r.iloc[1].salesYoY - 0.1) < 1e-9
    assert r.iloc[3].q == "2018Q4" and r.iloc[3].days_since == 30
    assert r.iloc[1].eps_sue > 5, r.iloc[1].eps_sue  # big jump vs ~0.02-sd history
    f = _features(syn)
    assert pd.isna(f.profitYoY[7]), "base <= 0 -> None"            # 2016Q4 vs -5
    assert f.salesYoY[:4].isna().all() and f.eps_sue[:12].isna().all()  # no year-ago / <8 diffs
    assert pd.notna(f.eps_sue[12])
    syn2 = syn.drop(index=4)  # missing 2016Q1 -> 2017Q1 YoY None
    assert pd.isna(_features(syn2).set_index("q").salesYoY["2017Q1"])
    syn3 = syn.copy(); syn3.loc[8, "consolidated"] = False  # basis switch -> no YoY across it
    assert pd.isna(_features(syn3).set_index("q").salesYoY["2018Q1"])
    # 1:1 bonus effective mid-2017: reported EPS halves from 2017Q3 on; adjusted SUE must match the no-bonus case
    syn4 = syn.copy(); syn4.loc[syn4.period_end >= "2017-09-30", "eps"] /= 2
    bonus = [(pd.Timestamp("2017-08-15"), 2.0)]
    s0, s4 = _features(syn).eps_sue.iloc[-1], _features(syn4, bonus).eps_sue.iloc[-1]
    assert abs(s0 - s4) < 1e-9, (s0, s4)  # z-score is scale-free once the basis is consistent
    assert abs(_features(syn4).eps_sue.iloc[-1] - s0) > 1  # unadjusted it would be garbage
    # switch to consolidated in 2017: consolidated has no YoY base yet -> standalone fills it, and becomes visible
    # only when the (later) standalone filing is public
    std = syn.copy(); std["consolidated"] = False; std["announced"] += pd.Timedelta(hours=2)
    con = syn[syn.period_end >= "2017-01-01"].copy(); con["sales"] *= 1.5
    mix = pd.concat([std, con])
    r5 = asof("X", ["2017-05-02"], mix, noacts).iloc[0]  # 2017Q1 announced 2017-04-30 16:00 / 18:00
    assert r5.q == "2017Q1" and abs(r5.salesYoY - 0.1) < 1e-9, r5
    r6 = asof("X", ["2018-05-02"], mix, noacts).iloc[0]   # 2018Q1: consolidated has its own base now
    assert abs(r6.salesYoY - 0.1) < 1e-9 and r6.days_since == 2, r6
    print("results.py self-check OK")
