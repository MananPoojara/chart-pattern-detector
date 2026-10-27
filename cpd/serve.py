"""Model service: BLPOP cpd:jobs -> RunResult JSON in cpd:result:<id> + PUBLISH cpd:done <id> (platform/CONTRACT.md).

    python -m cpd.serve             # serve forever (run with nice -n 19, CPD_THREADS=2)
    python -m cpd.serve --selftest  # one run per model kind on RELIANCE + one intraday run, asserts the shape
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CPD_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("CPD_THREADS", "2"))

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cpd import bus, data  # noqa: E402
from cpd.bench import OUT  # noqa: E402

REPOS = {  # id: (kind, url, how) -- same as app.py
    "ta_patterns": ("rule", "https://github.com/AdventuresInDataScience/ta_patterns",
                    "300 numpy detectors (106 candlestick + 194 chart). Pivot-confirmed, no lookahead."),
    "stock-pattern": ("rule", "https://github.com/BennyThadikaran/stock-pattern",
                      "Pivot geometry: double tops/bottoms, H&S, triangles, flags, VCP, harmonics. We walk it forward bar by bar."),
    "chart_patterns": ("rule", "https://github.com/zeta-zetra/chart_patterns",
                       "scipy pivots + linear regression of highs/lows. Repo peeks 3-10 bars ahead; we delay its signals."),
    "TradingPatternScanner": ("rule", "https://github.com/white07S/TradingPatternScanner",
                              "Rolling-window masks over 3 bars. Repo peeks 1 bar ahead; we delay its signals."),
    "HarmonicPattern": ("rule", "https://github.com/xxwikkixx/HarmonicPattern",
                        "argrelextrema XABCD + Fibonacci ratio checks (Gartley, Bat, Butterfly, Crab). Close only."),
    "pytrendline": ("lines", "https://github.com/ednunezg/pytrendline",
                    "Brute-force support/resistance trendline fitting with pivot scoring. Visual only."),
    "foduucom-yolov8": ("vision", "https://github.com/foduucom/Stockmarket-pattern-detection",
                        "YOLOv8s on chart screenshots: H&S top/bottom, M_Head, W_Bottom, Triangle, StockLine."),
    "ChartScanAI": ("vision", "https://github.com/Omar-Karimov/ChartScanAI", "YOLOv8 on mplfinance charts: Buy / Sell boxes."),
    "Kronos (mini)": ("forecast", "https://github.com/shiyu-coder/Kronos",
                      "Decoder-only transformer over tokenized OHLCV candles (K-line foundation model). Samples future bars."),
    "Chronos-2 (small)": ("forecast", "https://github.com/amazon-science/chronos-forecasting",
                          "Amazon's pretrained time-series transformer; quantile forecasts of the close."),
    "TimesFM 2.5": ("forecast", "https://github.com/google-research/timesfm",
                    "Google's 200M-param patch decoder foundation model. Heavy on CPU (~2 GB RAM)."),
    "VAE analog (nsarang)": ("forecast", "https://github.com/nsarang/big-data-stock-price-forecast",
                             "1D-conv VAE embeds 256-bar windows; forecast = futures of the nearest past windows."),
    "PatternEdge (ours)": ("edge", "", "All causal patterns + market context -> walk-forward gradient boosting. "
                                      "Ranks which pattern firings are worth trading today."),
    "TradePlan (ours)": ("plan", "", "Chart-image CNN + pattern track records + market context -> ranked next-session "
                                    "long plans with entry / stop / target, calibrated on never-seen years."),
}
MODELS = [dict(id=k, name=k, kind=v[0], url=v[1], how=v[2], intraday=v[0] not in ("edge", "plan") and not k.startswith("VAE")) for k, v in REPOS.items()]
KEYS = {"id", "model", "kind", "ticker", "interval", "asof", "candles", "markers", "boxes", "lines", "levels", "forecast",
        "score", "paperTrade", "note"}
IMG = os.path.join(tempfile.gettempdir(), f"cpd_serve_{os.getpid()}.png")
log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)


# ---------------------------------------------------------------- inputs
def _fresh():
    """Nightly data refresh rewrites the daily store: drop data.py's caches when its files change."""
    m = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in
              (os.path.join(data.STOCK_ROOT, "index.csv"), os.path.join(data.STOCK_ROOT, "bars.parquet"), data.INDEX_CSV))
    if getattr(_fresh, "m", m) != m:
        for f in (data.manifest, data._bars, data._indices, data.load):
            f.cache_clear()
    _fresh.m = m


def candles(ticker, interval):
    if interval == "1D":
        _fresh()
        try:
            return data.load(ticker)
        except IndexError:
            raise ValueError(f"unknown ticker {ticker}") from None
    from cpd import angel

    p = angel.STORE / interval / f"{ticker}.parquet"
    if not p.exists():  # never call the throttled history API from here
        raise ValueError(f"no stored {interval} candles for {ticker} (intraday covers the live watchlist only)")
    d = pd.read_parquet(p)
    lp = angel.STORE / f"live_{interval}.parquet"
    if lp.exists():
        lv = pd.read_parquet(lp)
        lv = lv[lv.ticker.astype(str) == ticker].drop(columns="ticker")
        d = pd.concat([d, lv[lv.index > d.index[-1]]])
    d = d[data.COLS].astype(float).sort_index()
    return d[~d.index.duplicated(keep="last")]


@lru_cache(4)
def _pickle(p, mtime):  # mtime in the key: nightly refreshes are picked up without a restart
    return pd.read_pickle(p) if p.endswith("today.pkl") or "preds" in p else pickle.load(open(p, "rb"))


def _res(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        raise ValueError(f"{name} missing: run the nightly model build first")
    return _pickle(p, os.path.getmtime(p))


@lru_cache(2)
def _edge(mtime):
    pr = _res("edge_preds_PatternEdge.pkl").copy()
    pr["ticker"] = pr.ticker.astype(str)
    pr["pct"] = pr.groupby("date").pred.rank(pct=True)
    return pr


def _today():
    t = _res("tradeplan_today.pkl").copy()
    t["ticker"] = t.ticker.astype(str)
    return t


# ---------------------------------------------------------------- run
def _markers(ev, d, tf):
    from cpd.lwchart import _label

    if not len(ev):
        return []
    ev = ev[ev.date.between(d.index[0], d.index[-1])]
    return [dict(time=tf(k[0]), dir=int(k[1]), text=_label(g.pattern)) for k, g in ev.groupby(["date", "dir"])]


def _paper(pt, tf):
    if pt is None:
        return None
    return dict(signal=tf(pt["signal"]), why=pt["why"][:300], entry=pt["entry"], stop=pt["stop"], target=pt["target"],
                status=pt["status"][0], detail=pt["status"][1], R=pt["R"],
                entryTime=None if pt["entry_date"] is None else tf(pt["entry_date"]))


def build(model, ticker, interval, bars, days, df):
    from cpd import lwchart

    kind = REPOS[model][0]
    daily = interval == "1D"
    tf = lambda x: bus.t(x, daily)
    d = df.iloc[-bars:]
    r = dict(model=model, kind=kind, ticker=ticker, interval=interval, asof=tf(d.index[-1]),
             candles=[dict(time=tf(i), open=o, high=h, low=lo, close=c, volume=v)
                      for i, o, h, lo, c, v in zip(d.index, d.open, d.high, d.low, d.close, d.volume)],
             markers=[], boxes=[], lines=[], levels=[], forecast=None, score=None, paperTrade=None, note="")
    ev = pd.DataFrame(columns=["date", "pattern", "dir"])
    if (kind in ("edge", "plan") or model.startswith("VAE")) and not daily:  # VAE needs 256-bar windows + their futures
        r["error"] = (f"{model} is trained on daily bars (6-month returns, NSE breadth, daily ranks); "
                      "pick 1D. An intraday version needs retraining on the intraday history being collected.")
        return r
    if kind == "rule":
        from cpd import rules

        ev = rules.DETECTORS[model](df, start=d.index[0])
        if model == "stock-pattern" and daily:  # pivot polylines; their points carry dates only, so daily only
            r["lines"] = [dict(points=[dict(time=p[0], value=p[1]) for p in sorted(x["info"]["points"].values())],
                               dir=int(x["dir"]), label=x["pattern"]) for _, x in ev.iterrows()]
            r["lines"] = [ln for ln in r["lines"] if ln["points"] and ln["points"][0]["time"] >= tf(d.index[0])]
        r["note"] = (f"{len(ev)} events. Arrow = the bar the pattern became known (tradable from the next open)."
                     + ("" if model != "stock-pattern" else " Lines join the pattern's pivots."))
    elif kind == "lines":
        from cpd import rules

        for ln in rules.trendlines(df, min(bars, 250)):
            r["lines"].append(dict(points=[dict(time=tf(ln["dates"][i]), value=ln["m"] * i + ln["b"]) for i in (ln["i0"], ln["i1"])],
                                   dir=1 if ln["kind"] == "support" else -1, label=ln["kind"]))
        r["note"] = "Support (green) / resistance (red) trendlines fitted on the visible bars. Visual only: no signals."
    elif kind == "vision":
        from cpd import vision

        ev = vision.live(model, df, IMG, days)
        for b in vision.detect(model, df, IMG):
            if b["end"] < d.index[0]:
                continue
            seg = df.loc[b["start"]:b["end"]]
            r["boxes"].append(dict(start=tf(max(b["start"], d.index[0])), end=tf(b["end"]), top=seg.high.max(),
                                   bottom=seg.low.min(), label=b["label"], dir=b["dir"], conf=b["conf"], live=bool(b["at_edge"])))
        r["note"] = (f"Boxes = what the model sees on today's chart (muted ones were drawn with hindsight; bright ones touch "
                     f"the right edge). Arrows = right-edge boxes on the chart as it looked at each of the last {days} bars.")
    elif kind == "forecast":
        from cpd import forecast

        h = 10
        q = forecast.forecast(model, [df], h)[0]
        if daily:
            fx = pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), periods=h)
        else:
            stp = d.index.to_series().diff().mode()[0]
            fx = pd.date_range(d.index[-1] + stp, periods=h, freq=stp)
        base = dict(time=tf(d.index[-1]), value=d.close.iloc[-1])
        r["forecast"] = {k: [base] + [dict(time=tf(x), value=v) for x, v in zip(fx, q[:, j])]
                         for k, j in (("lo", 0), ("median", 1), ("hi", 2))}
        chg = q[-1, 1] / d.close.iloc[-1] - 1
        ev = pd.DataFrame(dict(date=[d.index[-1]], dir=[int(np.sign(chg))], pattern=[f"{chg:+.1%} in {h} bars"]))
        r["stats"] = dict(horizon=h, median_change=chg)
        r["note"] = f"Line = median forecast of the close for the next {h} bars; shaded band = 10%-90% range."
    elif kind == "edge":
        pr = _edge(os.path.getmtime(os.path.join(OUT, "edge_preds_PatternEdge.pkl")))
        mine = pr[pr.ticker == ticker]
        if not len(mine):
            r["error"] = f"{ticker} is not in the NSE500 PatternEdge universe."
            return r
        vis = mine[mine.date.between(d.index[0], d.index[-1])]
        r["score"] = [dict(time=tf(x), value=v * 1e4) for x, v in zip(vis.date, vis.pred)]
        ev = pd.concat([mine[mine.pct >= 0.95].assign(dir=1, pattern="top 5% of NSE500"),
                        mine[mine.pct <= 0.05].assign(dir=-1, pattern="bottom 5%")])[["date", "dir", "pattern"]]
        last = mine.iloc[-1]
        r["stats"] = dict(scores_asof=str(pr.date.max().date()), score_bps=last.pred * 1e4, pct_rank=last.pct)
        r["note"] = ("Histogram = expected 10-day excess return vs NSE500 (bps). Arrows = days this stock ranked in the "
                     "top / bottom 5%.")
    elif kind == "plan":
        from cpd import tradeplan as tp
        from cpd.plan_ui import NAMES

        mdl, today = _res("tradeplan_model.pkl"), _today()
        row = today[today.ticker == ticker]
        if row.empty:
            r["error"] = f"{ticker} is not in the NSE500 TradePlan universe (or has no bar on {mdl['asof']})."
            return r
        row = row.iloc[0]
        pl = tp.plan(df.loc[:mdl["asof"]], row.p_win, row.eR)
        r["levels"] = [dict(title=k, price=pl[k]) for k in ("entry", "stop", "target")]
        r["stats"] = dict(action=pl["action"], p_target_first=pl["p_target_first"], expected_R=pl["expected_R"],
                          rank=int(row["rank"]), rank_bucket=row.rank_bucket, time_exit_bars=pl["time_exit_bars"], plan_asof=mdl["asof"])
        r["drivers"] = [dict(name=NAMES.get(k, k), share=v, value=float(row[k])) for k, v in tp.drivers(mdl, row).items()]
        r["note"] = (f"Plan for the session after {mdl['asof']}: enter at the next open, exit at stop, target or after "
                     f"{pl['time_exit_bars']} bars. {pl['action']} = expected R after costs vs the {tp.MIN_EDGE} R bar.")
        return r
    r["markers"] = _markers(ev, d, tf)
    if kind != "lines":
        r["paperTrade"] = _paper(lwchart.paper_trade(d, ev, min(days, len(d))), tf)
    return r


_cache = OrderedDict()


def run(job):
    model, ticker, interval = job.get("model", ""), str(job.get("ticker", "")).upper(), job.get("interval") or "1D"
    bars = min(max(int(job.get("bars") or 400), 30), 5000)
    days = min(max(int(job.get("days") or 10), 1), 60)
    r = dict(model=model, kind=REPOS.get(model, ("",))[0], ticker=ticker, interval=interval, asof=None, candles=[], markers=[],
             boxes=[], lines=[], levels=[], forecast=None, score=None, paperTrade=None, note="")
    try:
        if model not in REPOS:
            raise ValueError(f"unknown model {model!r}")
        df = candles(ticker, interval)
        key = (model, ticker, interval, bars, days, df.index[-1])
        if key not in _cache:
            _cache[key] = json.loads(bus.dumps(build(model, ticker, interval, bars, days, df)))
            while len(_cache) > 200:
                _cache.popitem(last=False)
        r = _cache[key]
    except Exception as e:
        log("run failed", model, ticker, interval, repr(e))
        r = dict(r, error=f"{type(e).__name__}: {e}")
    return dict(r, id=job.get("id", ""))


# ---------------------------------------------------------------- board + loop
def board():
    from cpd import tradeplan as tp

    mdl, today = _res("tradeplan_model.pkl"), _today()
    rows = []
    for _, x in today.sort_values("rank").head(15).iterrows():
        df = data.load(x.ticker, "nse500")
        pl = tp.plan(df.loc[:mdl["asof"]], x.p_win, x.eR)
        rows.append(dict(ticker=x.ticker, rank=int(x["rank"]), group=x.rank_bucket, p_win=x.p_win, eR=x.eR, action=pl["action"],
                         close=df.close.iloc[-1], chg=(df.close.iloc[-1] / df.close.iloc[-2] - 1) * 100,
                         entry=pl["entry"], stop=pl["stop"], target=pl["target"], spark=df.close.iloc[-30:].round(2).tolist()))
    return dict(asof=mdl["asof"], rows=rows, universe=len(today))


def handle(rc, job):
    t0 = time.time()
    if job.get("type") == "refresh_board":
        _fresh()
        try:
            out = board()
            rc.set("cpd:board", bus.dumps(out))
        except Exception as e:
            out = dict(error=f"{type(e).__name__}: {e}")
        out = dict(out, id=job.get("id", ""))
    else:
        out = run(job)
    rc.set(f"cpd:result:{out['id']}", bus.dumps(out), ex=600)
    rc.publish("cpd:done", out["id"])
    log(job.get("type", "run"), job.get("model", ""), job.get("ticker", ""), job.get("interval", ""),
        f"{time.time() - t0:.1f}s", out.get("error", "ok"))
    return out


def startup(rc):
    rc.set("cpd:models", bus.dumps(MODELS))
    rc.set("cpd:universe", bus.dumps(data.universe()))
    try:
        rc.set("cpd:board", bus.dumps(board()))
    except Exception as e:
        log("board failed:", repr(e))


def heartbeat():
    rc = bus.client(timeout=5)
    while True:
        try:
            rc.hset("cpd:status", "serve_heartbeat", int(time.time()))
        except Exception as e:
            log("heartbeat:", repr(e))
        time.sleep(10)


def main():
    rc = bus.client()
    startup(rc)
    threading.Thread(target=heartbeat, daemon=True).start()  # keeps beating during long jobs too
    log("serving cpd:jobs")
    while True:
        try:
            item = rc.blpop("cpd:jobs", timeout=30)
        except Exception as e:  # redis down: wait, redis-py reconnects on the next call
            log("redis:", repr(e))
            time.sleep(5)
            continue
        if not item:
            continue
        try:
            handle(rc, json.loads(item[1]))
        except Exception as e:  # bad job JSON or redis write failure: never crash the loop
            log("job failed:", repr(e), item[1][:200])


def check(r, daily):
    tt = str if daily else int
    assert KEYS <= set(r), KEYS - set(r)
    assert "error" not in r, r.get("error")
    assert r["candles"] and all(type(c["time"]) is tt for c in r["candles"]), "candle time"
    assert set(r["candles"][0]) == {"time", "open", "high", "low", "close", "volume"}
    times = [m["time"] for m in r["markers"]] + [b[k] for b in r["boxes"] for k in ("start", "end")] + \
            [p["time"] for ln in r["lines"] for p in ln["points"]] + [p["time"] for p in r["score"] or []] + \
            [p["time"] for v in (r["forecast"] or {}).values() for p in v]
    assert all(type(x) is tt for x in times), "mark time"
    assert all(m["dir"] in (1, -1, 0) for m in r["markers"])
    pt = r["paperTrade"]
    if pt:
        assert type(pt["signal"]) is tt and set(pt) >= {"entry", "stop", "target", "status", "detail", "R", "entryTime", "why"}
    json.dumps(r, allow_nan=False)


def selftest():
    rc = bus.client()
    startup(rc)
    b = json.loads(rc.get("cpd:board"))
    assert len(b["rows"]) == 15 and len(b["rows"][0]["spark"]) == 30 and set(b) == {"asof", "rows", "universe"}, b
    assert len(json.loads(rc.get("cpd:models"))) == len(REPOS)
    jobs = [("stock-pattern", "1D"), ("ta_patterns", "1D"), ("pytrendline", "1D"), ("ChartScanAI", "1D"),
            ("Chronos-2 (small)", "1D"), ("PatternEdge (ours)", "1D"), ("TradePlan (ours)", "1D"),
            ("ChartScanAI", "FIVE_MINUTE"), ("ta_patterns", "FIVE_MINUTE")]
    for i, (m, iv) in enumerate(jobs):
        t0 = time.time()
        handle(rc, dict(id=f"selftest-{i}", type="run", model=m, ticker="RELIANCE", interval=iv, bars=300, days=10))
        r = json.loads(rc.get(f"cpd:result:selftest-{i}"))
        check(r, iv == "1D")
        kind = REPOS[m][0]
        assert {"lines": r["lines"], "vision": r["boxes"], "forecast": r["forecast"], "edge": r["score"],
                "plan": r["levels"]}.get(kind, True), (m, "no marks")
        print(f"ok {m:20s} {iv:12s} {time.time() - t0:5.1f}s candles={len(r['candles'])} markers={len(r['markers'])} "
              f"boxes={len(r['boxes'])} lines={len(r['lines'])} paper={(r['paperTrade'] or {}).get('status')}", flush=True)
    r = handle(rc, dict(id="selftest-err", type="run", model="TradePlan (ours)", ticker="RELIANCE", interval="FIVE_MINUTE"))
    assert r["error"] and r["candles"] and type(r["candles"][0]["time"]) is int
    assert handle(rc, dict(id="selftest-bad", type="run", model="nope", ticker="RELIANCE"))["error"]
    t0 = time.time()
    handle(rc, dict(id="selftest-cached", type="run", model="ChartScanAI", ticker="RELIANCE", interval="1D", bars=300, days=10))
    assert time.time() - t0 < 1, "cache miss"
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
