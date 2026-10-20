"""Read candles back out of a chart screenshot (TradingView, broker apps, mplfinance...).

Colour-based: bullish candles are green/teal, bearish are red. Each candle = one group of adjacent coloured
columns; wick extent = high/low, the thick part = body. A pane of bars sharing one baseline at the bottom is
the volume pane. Output prices are in PIXEL units (up = higher); everything downstream is scale-free, and
real prices are only needed to print levels (map via two known prices).
Limits: needs default-ish green/red candles; green/red indicator lines or overlapping drawings confuse it.
The app always shows the reconstruction so the user can check it.
"""
import numpy as np
import pandas as pd
from PIL import Image


def _masks(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    bull = (g - r > 40) & (g > 90)
    bear = (r - g > 40) & (r > 110) & (r - b > 30)
    return bull, bear


def _runs(col):
    """(start, end) of True runs in a 1-D bool array."""
    d = np.diff(np.r_[0, col.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _candles(bull, bear, r0, r1, max_width):
    """Candles inside rows r0..r1 (one pane). Returns per-candle dict with column span and the run's rows."""
    any_ = (bull | bear)[r0 : r1 + 1]
    out = []
    for a, b in _runs(any_.any(0)):
        if b - a + 1 > max_width:
            continue
        m = any_[:, a : b + 1]
        rr = np.flatnonzero(m.any(1))
        top, bot = rr.min(), rr.max()
        isbull = bull[r0 + top : r0 + bot + 1, a : b + 1].sum() >= bear[r0 + top : r0 + bot + 1, a : b + 1].sum()
        out.append(dict(x=(a + b) / 2, top=r0 + top, bot=r0 + bot, width=m[top : bot + 1].sum(1), bull=isbull))
    return out


def digitize(path, max_width=40, gap=4):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    bull, bear = _masks(rgb)
    rows = (bull | bear).any(1)
    # panes = bands of coloured rows separated by >= `gap` empty rows (price pane, volume pane, ...)
    bands = []
    for a, b in _runs(rows):
        if bands and a - bands[-1][1] <= gap:
            bands[-1] = (bands[-1][0], b)
        else:
            bands.append((a, b))
    panes = [(a, b, _candles(bull, bear, a, b, max_width)) for a, b in bands]
    panes = [p for p in panes if len(p[2]) >= 5]
    if not panes:
        raise ValueError("no green/red candles found")

    def is_volume(c):  # bars that share one baseline
        bots = np.array([x["bot"] for x in c])
        return np.mean(np.abs(bots - np.median(bots)) <= 1) > 0.8

    price = max((p for p in panes if not is_volume(p[2])), key=lambda p: p[1] - p[0], default=None)
    if price is None:
        raise ValueError("found bars but no candle pane")
    # volume pane: a band below the candles where coloured columns all end on one baseline (bars may touch)
    vol_col = None
    for a, b in bands:
        if a <= price[1]:
            continue
        m = (bull | bear)[a : b + 1]
        has = m.any(0)
        if has.sum() < 10:
            continue
        bottom = np.where(has, m.shape[0] - 1 - np.argmax(m[::-1], axis=0), -1)[has]
        if np.mean(np.abs(bottom - np.median(bottom)) <= 1) > 0.8:
            vol_col = m.sum(0)
            break
    H = rgb.shape[0]
    out = []
    for r in price[2]:
        wd = r["width"]
        thick = np.flatnonzero(wd >= max(2, 0.6 * wd.max())) if wd.max() >= 3 else np.arange(len(wd))
        btop, bbot = r["top"] + thick.min(), r["top"] + thick.max()
        o, cl = (H - bbot, H - btop) if r["bull"] else (H - btop, H - bbot)
        v = float(vol_col[int(round(r["x"]))]) if vol_col is not None else 0.0
        out.append(dict(x=r["x"], open=float(o), high=float(H - r["top"]), low=float(H - r["bot"]), close=float(cl), volume=float(v)))
    df = pd.DataFrame(out).sort_values("x").reset_index(drop=True)
    df.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))  # placeholder dates
    return df


def to_prices(df, px_ref, price_ref):
    """Map pixel units to prices given two (pixel, price) reference points, e.g. last close and visible high."""
    (p1, p2), (v1, v2) = px_ref, price_ref
    k = (v2 - v1) / (p2 - p1)
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = v1 + (df[c] - p1) * k
    return out


if __name__ == "__main__":
    # self-check: render real NSE bars TradingView-style, digitize, compare with the truth
    import os
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import mplfinance as mpf

    from cpd import data

    df = data.load("RELIANCE", "nifty50").iloc[-80:]
    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", wick="inherit", edge="inherit", volume="inherit")
    st = mpf.make_mpf_style(marketcolors=mc, facecolor="white", gridstyle="")
    p = os.path.join(tempfile.gettempdir(), "cpd_tv.png")
    mpf.plot(df.rename(columns=str.title), type="candle", style=st, volume=True, figsize=(12, 6), savefig=dict(fname=p, dpi=100))
    got = digitize(p)
    print("candles: true", len(df), "found", len(got))
    n = min(len(df), len(got))
    t, g = df.iloc[-n:], got.iloc[-n:]
    corr = np.corrcoef(np.diff(t.close.values), np.diff(g.close.values))[0, 1]
    rng = lambda d: (d.high - d.low).values
    print("close-change correlation", round(corr, 4), "| range correlation", round(np.corrcoef(rng(t), rng(g))[0, 1], 4),
          "| volume corr", round(np.corrcoef(t.volume.values, g.volume.values)[0, 1], 4))
    assert len(got) == len(df) and corr > 0.97, "digitizer self-check failed"
    print("ok")
