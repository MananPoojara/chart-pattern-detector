"""TradingView Lightweight Charts (v4.2) inside Streamlit, plus 'buy signal in the last N bars' paper-trade levels.

    show(df, events, lines=..., levels=..., forecast=...)
events: DataFrame with date, dir (+1/-1/0), pattern  -> arrows on the candles
"""
import json

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

from cpd import evaluate
from cpd.bench import H
from cpd.tradeplan import SL_ATR, TP_ATR

JS = "https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"
UP, DOWN, NEUTRAL = "#16a34a", "#dc2626", "#64748b"


def _t(x):
    """Daily bars -> 'YYYY-MM-DD'; intraday -> epoch seconds of the IST wall clock (the chart labels it as UTC, so
    the axis shows IST times)."""
    x = pd.Timestamp(x)
    return x.strftime("%Y-%m-%d") if x == x.normalize() else int(x.timestamp())


def _label(names):
    u = list(dict.fromkeys(map(str, names)))
    return u[0][:24] + (f" +{len(u) - 1}" if len(u) > 1 else "")


def show(df, events=None, lines=(), levels=(), forecast=None, height=520):
    """lines: [{"points": [(date, price), ...], "color": str}]; levels: [(name, price, color)];
    forecast: (dates, q) with q[:, 0..2] = q10, median, q90."""
    candles = [dict(time=_t(i), open=r.open, high=r.high, low=r.low, close=r.close) for i, r in df.iterrows()]
    vol = [dict(time=_t(i), value=float(v), color="rgba(22,163,74,.35)" if c >= o else "rgba(220,38,38,.35)")
           for i, v, o, c in zip(df.index, df.volume, df.open, df.close)]
    marks = []
    if events is not None and len(events):
        ev = events[events.date.between(df.index[0], df.index[-1])]
        for (d, dr), g in ev.groupby(["date", "dir"]):  # one arrow per bar and side; labels joined
            marks.append(dict(time=_t(d), position="belowBar" if dr > 0 else "aboveBar" if dr < 0 else "inBar",
                              color=UP if dr > 0 else DOWN if dr < 0 else NEUTRAL,
                              shape="arrowUp" if dr > 0 else "arrowDown" if dr < 0 else "circle",
                              text=_label(g.pattern)))
    marks.sort(key=lambda m: m["time"])
    for m in marks[:-12]:  # only the latest arrows carry text; busy detectors would bury the candles
        m["text"] = ""
    extra = [dict(data=[dict(time=_t(d), value=float(v)) for d, v in sorted(ln["points"])], color=ln["color"], dash=False)
             for ln in lines]
    if forecast is not None:
        fx, q = forecast
        base = dict(time=_t(df.index[-1]), value=float(df.close.iloc[-1]))
        for k, dash in [(1, False), (0, True), (2, True)]:
            extra.append(dict(data=[base] + [dict(time=_t(d), value=float(v)) for d, v in zip(fx, q[:, k])],
                              color="#2563eb", dash=dash))
    spec = dict(candles=candles, vol=vol, marks=marks, extra=extra,
                levels=[dict(title=n, price=float(p), color=c) for n, p, c in levels])
    components.html(f"""
<div id="c" style="height:{height}px"></div>
<script src="{JS}"></script>
<script>
const S = {json.dumps(spec)};
const dark = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
const el = document.getElementById('c');
const chart = LightweightCharts.createChart(el, {{
  height: {height}, autoSize: true,
  layout: {{background: {{color: dark ? '#0e1117' : '#ffffff'}}, textColor: dark ? '#cbd5e1' : '#334155'}},
  grid: {{vertLines: {{color: dark ? '#1e293b' : '#f1f5f9'}}, horzLines: {{color: dark ? '#1e293b' : '#f1f5f9'}}}},
  timeScale: {{rightOffset: 12, timeVisible: true, secondsVisible: false}}, crosshair: {{mode: 0}},
}});
const c = chart.addCandlestickSeries({{upColor: '{UP}', downColor: '{DOWN}', borderVisible: false,
                                      wickUpColor: '{UP}', wickDownColor: '{DOWN}'}});
c.setData(S.candles);
c.setMarkers(S.marks);
const v = chart.addHistogramSeries({{priceScaleId: '', priceFormat: {{type: 'volume'}}}});
v.priceScale().applyOptions({{scaleMargins: {{top: 0.82, bottom: 0}}}});
v.setData(S.vol);
for (const x of S.extra) {{
  const s = chart.addLineSeries({{color: x.color, lineWidth: 2, lineStyle: x.dash ? 2 : 0,
                                 priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false}});
  s.setData(x.data);
}}
for (const l of S.levels) c.createPriceLine({{price: l.price, color: l.color, lineWidth: 2, lineStyle: 2,
                                             axisLabelVisible: true, title: l.title}});
chart.timeScale().setVisibleLogicalRange({{from: S.candles.length - 120, to: S.candles.length + 8}});
</script>""", height=height + 10)


def paper_trade(df, events, days=10, sl=SL_ATR, tp=TP_ATR, h=H):
    """Latest bullish event in the last `days` bars -> long at the next open, stop/target in ATRs (TradePlan rules),
    and what has happened since. None if no bullish event."""
    ev = events[(events.dir > 0) & (events.date >= df.index[-days])] if len(events) else events
    if not len(ev):
        return None
    sig = ev.date.max()
    i = df.index.get_loc(sig)
    a = float(evaluate.atr(df).iloc[i])
    out = dict(signal=sig, why=", ".join(dict.fromkeys(map(str, ev[ev.date == sig].pattern))), atr=a)
    if i + 1 >= len(df):  # signal on the last bar: trade opens tomorrow
        e = float(df.close.iloc[-1])
        return dict(out, status=("pending", "enter at next open"), entry=e, stop=e - sl * a, target=e + tp * a, R=None, entry_date=None)
    e = float(df.open.iloc[i + 1])
    stop, target = e - sl * a, e + tp * a
    after = df.iloc[i + 1: i + 1 + h]
    hit_s, hit_t = np.flatnonzero(after.low.values <= stop), np.flatnonzero(after.high.values >= target)
    s0 = hit_s[0] if len(hit_s) else 99
    t0 = hit_t[0] if len(hit_t) else 99
    if t0 < s0:
        status, R = ("target hit", str(after.index[t0].date())), tp / sl
    elif s0 < 99:  # same bar counts as stop, like the backtest
        status, R = ("stopped", str(after.index[s0].date())), -1.0
    elif len(after) >= h:
        status, R = ("time exit", f"after {h} bars"), (after.close.iloc[-1] - e) / (sl * a)
    else:
        status, R = ("open", f"day {len(after)} of {h}"), (after.close.iloc[-1] - e) / (sl * a)
    return dict(out, status=status, entry=e, stop=stop, target=target, R=float(R), entry_date=df.index[i + 1])


if __name__ == "__main__":  # paper_trade self-check on a synthetic series
    idx = pd.bdate_range("2024-01-01", periods=60)
    px = pd.Series(100.0, index=idx)
    df = pd.DataFrame(dict(open=px, high=px + 1, low=px - 1, close=px, volume=1.0))
    df.iloc[55:, df.columns.get_loc("high")] = 200  # rally after the signal -> target
    ev = pd.DataFrame(dict(date=[idx[52]], dir=[1], pattern=["test"]))
    r = paper_trade(df, ev)
    assert r["entry"] == 100 and r["status"][0] == "target hit" and r["R"] == TP_ATR / SL_ATR, r
    df.iloc[55:, df.columns.get_loc("low")] = 0  # same bar hits both -> stop (pessimistic)
    assert paper_trade(df, ev)["R"] == -1.0
    assert paper_trade(df, ev.assign(dir=-1)) is None
    print("ok")
