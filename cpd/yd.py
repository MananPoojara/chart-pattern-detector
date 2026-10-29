"""Yd strategy: Volume Shock + Opening-Range Breakout / Pullback, NSE F&O stocks, intraday, long + short.

Rules (the user's design, NOT backtested; platform/CONTRACT.md "Yd"):
  VS_time(t) = CV(d,t) / agg CV(d-k,t)   the trigger: today's cumulative volume vs the same time on the n previous sessions
  VS_raw(t)  = CV(d,t) / agg V(d-k)      what a live StockEdge list shows (time-of-day biased, log only)
  VS_EOD     = V(d) / agg V(d-k)         StockEdge "Volume Shockers" (next-day planning)
  agg = mean | median over the n (5 | 10) previous regular sessions; today never in its own base; NaN until all n exist;
  a slot missing on a base day uses that day's last earlier slot; special sessions (Muhurat, partial days) are not base days.
  OR = high/low of the bars inside 09:15-09:30. A = first close beyond OR + VWAP with VS_time >= 2 (<= 11:00),
  B = breakout, retest, confirmation, stop order. Exit 1.5R / stop / 60 min / 15:15; stop and target in one bar -> stop.

Input: one ticker's intraday OHLCV, index = bar START (IST naive, Angel One convention). A bar is known at start + step,
and every value at bar i uses bars <= i only. Bar counts are minutes / step, so 1m/3m/5m/15m all run (5m is native).

Costs (verified Oct 2026; sources in platform/CONTRACT.md): Angel One intraday brokerage lower of ₹20 or 0.1% per order,
STT 0.025% sell, NSE transaction 0.00297%, SEBI ₹10/crore, stamp 0.003% buy, GST 18% on brokerage + transaction + SEBI.
"""
import numpy as np
import pandas as pd

H = pd.Timedelta
OPEN, OR_END, A_CUT, B_CUT, HARD = H("9h15min"), H("9h30min"), H("11h"), H("13h30min"), H("15h15min")
VS_MIN, TARGET_R, D_MIN, D_MAX, GAP = 2.0, 1.5, 0.006, 0.015, 0.5
MINUTES = dict(time_exit=60, retest=40, chop=30, valid=10)  # spec's 12 / 8 / 6 / 2 five-minute bars
BROKER, STT_SELL, TXN, SEBI, STAMP_BUY, GST, SLIP = (20.0, 0.001), 0.00025, 0.0000297, 0.000001, 0.00003, 0.18, 0.0003
RISK, CAP = 0.005, 0.8  # Q = min(floor(0.5% E / R), floor(80% E / O))


def tick(p):
    """NSE CM tick size by price (circular NSE/CMTR/67133, from 15-Apr-2025)."""
    p = abs(p)
    return 0.01 if p < 250 else 0.05 if p <= 1000 else 0.10 if p <= 5000 else 0.50 if p <= 10000 else 1.0 if p <= 20000 else 5.0


def step_of(df):
    return df.index.to_series().diff().dropna().mode()[0]


def volume_shock(df, n=5, how="mean"):
    """-> (bars: DataFrame vs_time, vs_raw like df.index; eod: Series vs_eod per regular session)."""
    day = df.index.normalize()
    tod = df.index - day
    cv = df.volume.groupby(day).cumsum()
    piv = cv.groupby([day, tod]).last().unstack().ffill(axis=1)  # missing slot -> that day's last earlier slot
    full = piv.iloc[:, -1]  # V(d) (so far, for an unfinished day)
    g = pd.Series(tod, index=day).groupby(level=0)
    reg = piv.index[((g.min() == OPEN) & (g.max() >= H("15h"))).reindex(piv.index).to_numpy()]
    agg = lambda x: getattr(x.rolling(n, min_periods=n), how)().to_numpy()
    pos = reg.searchsorted(piv.index) - 1  # last regular session strictly before each day
    pad = lambda a: np.concatenate([np.full((1,) + a.shape[1:], np.nan), a])[pos + 1]  # pos -1 -> no base yet
    bcv, bv = pad(agg(piv.loc[reg])), pad(agg(full.loc[reg]))
    r, c = piv.index.get_indexer(day), piv.columns.get_indexer(tod)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame(dict(vs_time=cv.to_numpy() / bcv[r, c], vs_raw=cv.to_numpy() / bv[r]), index=df.index)
        eod = pd.Series(full.to_numpy() / bv, index=piv.index)
    return out.replace([np.inf, -np.inf], np.nan), eod.loc[reg].replace([np.inf, -np.inf], np.nan)


def vs_eod_daily(d, n=5, how="mean"):
    """Daily bars: V(d) / agg of the n previous days. ponytail: Muhurat days aren't flagged in the daily store."""
    return d.volume / getattr(d.volume.shift(1).rolling(n, min_periods=n), how)()


def opening_range(df, step):
    """Per day orh / orl from the bars inside 09:15-09:30 (all of them present); empty when a bar spans past 09:30."""
    if step > OR_END - OPEN:
        return pd.DataFrame(columns=["orh", "orl"], index=pd.DatetimeIndex([]))
    day = df.index.normalize()
    m = (df.index - day >= OPEN) & (df.index - day < OR_END)
    o = df[m].groupby(day[m]).agg(orh=("high", "max"), orl=("low", "min"), k=("close", "size"))
    return o[o.k == (OR_END - OPEN) // step][["orh", "orl"]]


def vwap(df):
    day = df.index.normalize()
    pv = ((df.high + df.low + df.close) / 3 * df.volume).groupby(day).cumsum()
    return pv / df.volume.groupby(day).cumsum().replace(0, np.nan)


def costs(q, buy, sell):
    """₹ fees for one intraday round trip of q shares (buy value at `buy`, sell value at `sell`; shorts too)."""
    bv, sv = q * buy, q * sell
    brk = min(BROKER[0], BROKER[1] * bv) + min(BROKER[0], BROKER[1] * sv)
    reg = (TXN + SEBI) * (bv + sv)
    return brk + STT_SELL * sv + reg + STAMP_BUY * bv + GST * (brk + reg)


def size(equity, entry, stop, risk=RISK, cap=CAP):
    """-> (Q, binding limit)."""
    R = abs(entry - stop)
    qr, qc = (int(risk * equity // R) if R > 0 else 0), int(cap * equity // entry)
    return (qr, "risk") if qr <= qc else (qc, "80% cap")


def breakeven(cost_r, target=TARGET_R):
    """Win rate where p·target − (1−p)·1 − cost = 0."""
    return (1 + cost_r) / (target + 1)


def _day(b, orh, orw, vw, vs, step, nb, vs_min, a_cut, target):
    """One session, long logic. Shorts call this on the mirrored series (prices negated). -> list of setups."""
    t, o, h, l, c = b.index, b.open.to_numpy(), b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
    tod, n, out = t - t.normalize(), len(b), []
    live = lambda i: i >= n - 1 and b.attrs.get("last_day")  # newest bar of the data: the next bar isn't known yet

    def exits(rec, e, fill, S):
        R = fill - S
        T = fill + target * R
        rec.update(entry=fill, stop=S, target=T, entryTime=t[e], R_rs=R, status="open")
        for k in range(e, n):
            px = (min(S, o[k]) if k > e else S, "stop") if l[k] <= S else (T, "target") if h[k] >= T else \
                 (c[k], "time") if k - e + 1 >= nb["time_exit"] else (c[k], "15:15") if tod[k] + step >= HARD else None
            if px:
                rec.update(exit=px[0], exitTime=t[k], reason=px[1], status="closed", R=(px[0] - fill) / R)
                return rec
        if not b.attrs.get("last_day"):  # data ends mid-trade on an old day (gap in the store)
            rec.update(status="skip", reason="no bars to exit")
        return rec

    ok_d = lambda d, p: D_MIN * abs(p) <= d <= D_MAX * abs(p)
    start = np.searchsorted(tod, OR_END)  # no entries before the OR is known
    # Setup A: first close beyond OR + VWAP with VS_time >= vs_min (chop filter: <= 1/3 of the last 30 min below VWAP)
    for i in range(start, n):
        if tod[i] + step > a_cut:
            break
        if c[i] > orh and c[i] > vw[i] and vs[i] >= vs_min and (c[max(0, i - nb["chop"]):i] < vw[max(0, i - nb["chop"]):i]).sum() <= nb["chop"] // 3:
            P, S = c[i], min(l[i], orh - 0.1 * orw)
            rec = dict(setup="A", signal=t[i], vs_time=vs[i], P=P, stop=S, entry=P, target=P + target * (P - S))
            if not ok_d(P - S, P):
                rec.update(status="skip", reason=f"stop {abs((P - S) / P):.2%} outside 0.6-1.5%")
            elif live(i):
                rec.update(status="pending", reason="enter at the next bar's open")
            elif i + 1 >= n:
                rec.update(status="skip", reason="no next bar")
            elif o[i + 1] > P + GAP * (P - S) or o[i + 1] <= S:
                rec.update(status="skip", reason="next open gapped past P + 0.5d" if o[i + 1] > S else "opened through the stop")
            else:
                exits(rec, i + 1, o[i + 1], S)
            out.append(rec)
            break
    # Setup B: first breakout (close beyond OR, VS_time >= vs_min), retest + confirmation within 40 min, stop order 10 min
    br = next((i for i in range(start, n) if tod[i] + step <= B_CUT and c[i] > orh and vs[i] >= vs_min), None)
    if br is not None:
        rt = cf = None
        for j in range(br + 1, min(n, br + 1 + nb["retest"])):
            if c[j] < orh - 0.25 * orw:
                break
            rt = j if rt is None and l[j] <= orh + 0.25 * orw else rt
            if rt is not None and c[j] > o[j] and c[j] > vw[j] and c[j] > orh and tod[j] + step <= B_CUT:
                cf = j
                break
        if cf is not None:
            tk = tick(c[cf])
            trig, S = h[cf] + tk, l[rt:cf + 1].min() - tk
            rec = dict(setup="B", signal=t[cf], vs_time=vs[cf], P=trig, stop=S, entry=trig, target=trig + target * (trig - S))
            fill = next((k for k in range(cf + 1, min(n, cf + 1 + nb["valid"])) if h[k] >= trig), None)
            if not ok_d(trig - S, trig):
                rec.update(status="skip", reason=f"stop {abs((trig - S) / trig):.2%} outside 0.6-1.5%")
            elif fill is not None:
                exits(rec, fill, max(o[fill], trig), S)
            elif b.attrs.get("last_day") and n - 1 - cf < nb["valid"]:
                rec.update(status="pending", reason=f"stop order at {abs(trig):.2f}, valid {nb['valid'] - (n - 1 - cf)} more bar(s)")
            else:
                rec.update(status="skip", reason="stop order not filled in time")
            out.append(rec)
    return out


def analyze(df, n=5, how="mean", vs_min=VS_MIN, a_cut=A_CUT, target=TARGET_R, equity=50_000):
    """-> dict(vs, eod, orng, vwap, trades). trades: newest last, prices in rupees, R per trade, ₹ costs at size(equity)."""
    step = step_of(df)
    vs, eod = volume_shock(df, n, how)
    orng, vw = opening_range(df, step), vwap(df)
    nb = {k: max(1, round(H(minutes=m) / step)) for k, m in MINUTES.items()}
    day, trades = df.index.normalize(), []
    for d, (orh, orl) in orng.iterrows():
        m = day == d
        b = df[m].copy()
        b.attrs["last_day"] = d == day[-1]
        for side in (1, -1):
            mb = b if side == 1 else pd.DataFrame(dict(open=-b.open, high=-b.low, low=-b.high, close=-b.close), index=b.index)
            mb.attrs = b.attrs
            for r in _day(mb, side * (orh if side == 1 else orl), orh - orl, side * vw[m].to_numpy(),
                          vs.vs_time[m].to_numpy(), step, nb, vs_min, a_cut, target):
                for k in ("P", "stop", "entry", "target", "exit"):
                    if k in r:
                        r[k] = side * r[k]
                r.update(side=side, date=str(d.date()), vs_raw=vs.vs_raw.get(r["signal"]))
                if r["status"] != "skip":
                    q, bind = size(equity, abs(r["entry"]), abs(r["stop"]))
                    r.update(qty=q, binding=bind)
                    if q:
                        x = r.get("exit", r["target"])
                        buy, sell = (r["entry"], x) if side == 1 else (x, r["entry"])
                        risk = q * abs(r["entry"] - r["stop"])
                        r.update(costs=costs(q, buy, sell), slippage=SLIP * q * (buy + sell))
                        r.update(cost_R=r["costs"] / risk, cost_R_slip=(r["costs"] + r["slippage"]) / risk)
                        if "R" in r:
                            r["net_R"] = r["R"] - r["cost_R"]
                trades.append(r)
    trades.sort(key=lambda r: r["signal"])
    return dict(vs=vs, eod=eod, orng=orng, vwap=vw, trades=trades, step=step)


if __name__ == "__main__":
    # synthetic: 6 flat sessions of 5m bars (75 bars, 1000 shares each), then a day that breaks out on 3x volume
    idx = [pd.Timestamp(f"2026-10-{d:02d}") + OPEN + H(minutes=5 * i) for d in (19, 20, 21, 22, 23, 26, 27) for i in range(75)]
    df = pd.DataFrame(dict(open=100.0, high=100.5, low=99.5, close=100.0, volume=1000.0), index=pd.DatetimeIndex(idx))
    t = df.index.normalize() == pd.Timestamp("2026-10-27")
    df.loc[t, "volume"] = 3000.0
    T = lambda hm: pd.Timestamp("2026-10-27 " + hm)
    df.loc[T("09:30"), ["open", "high", "low", "close"]] = [100.0, 101.2, 100.3, 101.0]  # breakout bar
    df.loc[T("09:35"):T("09:50"), ["open", "high", "low", "close"]] = [101.0, 101.8, 100.9, 101.6]
    df.loc[T("09:55"), ["open", "high", "low", "close"]] = [101.6, 102.6, 101.5, 102.5]  # 1.5R = 101 + 1.5 * 0.7
    a = analyze(df)
    vs = a["vs"].vs_time
    assert np.isnan(vs[df.index.normalize() < pd.Timestamp("2026-10-26")]).all()  # need all 5 prior sessions
    assert abs(vs[T("09:15")] - 3.0) < 1e-9 and abs(vs[pd.Timestamp("2026-10-26 15:25")] - 1.0) < 1e-9
    assert abs(a["vs"].vs_raw[T("09:15")] - 3000 / 75000) < 1e-9 and abs(a["eod"].iloc[-1] - 3.0) < 1e-9
    assert a["orng"].loc[pd.Timestamp("2026-10-27")].tolist() == [100.5, 99.5]
    A = [r for r in a["trades"] if r["setup"] == "A" and r["side"] == 1]
    assert len(A) == 1 and A[0]["status"] == "closed" and A[0]["reason"] == "target", A
    assert A[0]["stop"] == 100.3 and A[0]["entry"] == 101.0 and abs(A[0]["R"] - 1.5) < 1e-9  # S = min(bar low, ORH - 0.1 ORW)
    assert not [r for r in a["trades"] if r["side"] == -1 and r["status"] != "skip"]
    assert size(50_000, 101.0, 100.3) == (357, "risk") and size(50_000, 4000, 3999) == (10, "80% cap")
    c = costs(100, 400, 400)  # ₹40,000 round trip: 20 + 20 + 10 STT + 2.376 + 0.08 + 1.2 stamp + 18% GST
    assert abs(c - (40 + 10 + 2.456 + 1.2 + 0.18 * 42.456)) < 1e-6, c
    assert abs(breakeven(0) - 0.4) < 1e-12 and tick(249) == 0.01 and tick(1200) == 0.1
    # same-bar stop + target -> stop; a live signal on the newest bar is pending
    df2 = df.copy()
    df2.loc[T("09:55"), ["high", "low"]] = [102.6, 100.0]
    assert [r["reason"] for r in analyze(df2)["trades"] if r["setup"] == "A" and r["side"] == 1] == ["stop"]
    p = [r for r in analyze(df.loc[:T("09:30")])["trades"] if r["setup"] == "A"]
    assert p and p[0]["status"] == "pending", p
    print("yd self-check ok", {k: A[0][k] for k in ("entry", "stop", "target", "exit", "R", "qty", "costs", "cost_R")})
