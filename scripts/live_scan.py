"""Live intraday scanner (Angel One): ticks from the SmartStream WebSocket -> 5-min bars -> ChartScanAI on the chart
ending at each finished bar. Boxes touching the right edge are live signals -> intraday/signals_<INTERVAL>.csv, with
paper-trade levels (stop 1.5 ATR, target 2.5 ATR from the signal bar's close).

Context (earlier bars) is the local candle store; the candle-history API (which throttles hard) is only asked for
symbols with no stored candles, never per bar. Finished tick-built bars are appended to the store (official candles
win on overlap) and today's bars also go to intraday/live_<INTERVAL>.parquet for the app.
Bars and signals are also published to Redis (platform/CONTRACT.md); Redis being down never stops the scan.

  python scripts/live_scan.py [--interval FIVE_MINUTE] [--watchlist nifty50 | SBIN,TCS,...]
  python scripts/live_scan.py --once --backfill 730     # history only (after the close): grow the store
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import angel, bus, data, evaluate, vision  # noqa: E402
from cpd.tradeplan import SL_ATR, TP_ATR  # noqa: E402

MODEL = "ChartScanAI"
now = lambda: pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
OPEN, CLOSE = pd.Timedelta("9h15min"), pd.Timedelta("15h30min")


class Bars:
    """Tick -> OHLCV bars per symbol. Volume = change in the day's cumulative volume."""

    def __init__(self, step):
        self.step, self.lock = step, threading.Lock()
        self.cur, self.cum, self.done = {}, {}, {}
        self.t0 = now()  # bars that began before we started listening are partial: dropped

    def tick(self, sym, ts, px, cumvol):
        day0 = ts.normalize()
        if not OPEN <= ts - day0 < CLOSE:
            return
        start = day0 + OPEN + (ts - day0 - OPEN) // self.step * self.step
        with self.lock:
            b = self.cur.get(sym)
            if b is not None and start > b["t"]:
                self._finish(sym)
                b = None
            if b is None:
                if self.done.get(sym) and start <= self.done[sym][-1]["t"]:
                    return  # late tick for a bar already closed
                b = self.cur[sym] = dict(t=start, open=px, high=px, low=px, close=px, v0=self.cum.get(sym, cumvol))
            b["high"], b["low"], b["close"] = max(b["high"], px), min(b["low"], px), px
            self.cum[sym] = cumvol

    def _finish(self, sym):
        b = self.cur.pop(sym)
        b["volume"] = float(self.cum[sym] - b.pop("v0"))
        if b["t"] >= self.t0:
            self.done.setdefault(sym, []).append(b)

    def close_due(self, t):
        """Close every bar whose period has ended (illiquid symbols may get no tick in the next bar)."""
        with self.lock:
            for sym in [s for s, b in self.cur.items() if b["t"] + self.step <= t]:
                self._finish(sym)

    def frame(self, sym):
        with self.lock:
            rows = list(self.done.get(sym, []))
        return pd.DataFrame(rows).set_index("t").rename_axis("date")[data.COLS] if rows else None


class Pub:
    """Redis side: forming bars <= 1 msg/s per ticker, final bars once, cpd:last, signals, heartbeat. Never raises."""

    def __init__(self, interval, nsym, rc=None):
        self.iv, self.nsym, self.rc = interval, nsym, rc or bus.client(timeout=2)
        self.sent, self.nfinal, self.err, self.lock = {}, {}, 0.0, threading.Lock()

    def _run(self, f, *a):
        try:
            f(*a)
        except Exception as e:  # log at most once a minute; redis-py reconnects on the next call
            if time.time() - self.err > 60:
                self.err = time.time()
                print(f"{now():%H:%M:%S} redis: {e!r}", flush=True)

    def _candle(self, b):
        return dict(time=bus.t(b["t"], False), open=b["open"], high=b["high"], low=b["low"], close=b["close"], volume=b["volume"])

    def bars(self, bars):
        self._run(self._bars, bars)

    def _bars(self, bars):
        with self.lock:
            with bars.lock:
                cur = {s: dict(b, volume=float(bars.cum[s] - b["v0"])) for s, b in bars.cur.items()}
                fin = {s: d[self.nfinal.get(s, 0):] for s, d in bars.done.items()}
            msgs = []
            for s, d in fin.items():
                msgs += [(s, b, True) for b in d]
                self.nfinal[s] = self.nfinal.get(s, 0) + len(d)
            for s, b in cur.items():
                k = (b["t"], b["high"], b["low"], b["close"], b["volume"])
                if self.sent.get(s) != k:
                    self.sent[s] = k
                    msgs.append((s, b, False))
            p = self.rc.pipeline(transaction=False)
            for s, b, final in msgs:
                c = self._candle(b)
                p.publish(f"cpd:bar:{self.iv}", bus.dumps(dict(ticker=s, interval=self.iv, final=final, **c)))
                p.hset(f"cpd:last:{self.iv}", s, bus.dumps(c))
            p.hset("cpd:status", mapping=dict(scanner_heartbeat=int(time.time()), scanner_symbols=self.nsym))
            p.execute()

    def signals(self, rows):
        p = self.rc.pipeline(transaction=False)
        for r in rows:
            tm = bus.t(r["bar"], False)
            js = bus.dumps(dict(id=f"{r['ticker']}-{self.iv}-{tm}-{r['signal']}", time=tm, ticker=r["ticker"], interval=self.iv,
                                model=r["model"], signal=r["signal"], dir=r["dir"], close=r["close"], stop=r["stop"], target=r["target"]))
            p.lpush("cpd:signals", js)
            p.publish("cpd:signal", js)
        p.ltrim("cpd:signals", 0, 299)
        self._run(p.execute)


def detect(d):
    d = d.iloc[-vision.MODELS[MODEL]["bars"]:]
    return {(b["label"], b["dir"]) for b in vision.detect(MODEL, d, f"/tmp/cpd_scan_{os.getpid()}.png") if b["at_edge"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="FIVE_MINUTE", choices=list(angel.INTERVALS))
    ap.add_argument("--watchlist", default="nifty50", help="basket name or comma-separated NSE symbols")
    ap.add_argument("--backfill", type=int, default=60, help="days of history to fetch per symbol")
    ap.add_argument("--once", action="store_true", help="fetch history and exit (no live scan)")
    a = ap.parse_args()
    u = data.universe()
    syms = list(u[a.watchlist] if a.watchlist in u else a.watchlist.split(","))
    try:  # plus today's TradePlan board, so the home page's top setups get live prices
        import json
        syms += [r["ticker"] for r in json.loads(bus.client(timeout=2).get("cpd:board") or "{}").get("rows", [])]
    except Exception as e:
        print("board watchlist skipped:", e, flush=True)
    syms = [s for s in dict.fromkeys(syms) if s in angel.tokens()]
    step = pd.Timedelta(angel.INTERVALS[a.interval][0])
    log = lambda *m: print(f"{now():%H:%M:%S}", *m, flush=True)

    if a.once:  # plus every symbol someone opened on 5m (serve.py stores them), so their history keeps growing
        syms += [p.stem for p in (angel.STORE / a.interval).glob("*.parquet") if p.stem not in syms]
        for s in syms:
            try:
                angel.history(s, a.interval, days=a.backfill)
            except Exception as e:
                log(s, e)
        log("history done")
        sys.exit()

    bars, stop = Bars(step), threading.Event()
    by_token = {angel.tokens()[s]: s for s in syms}
    threading.Thread(target=angel.stream, args=(list(by_token), lambda t, ts, px, v: bars.tick(by_token[t], ts, px, v), stop),
                     daemon=True).start()
    log(f"streaming {len(syms)} symbols, {a.interval}")
    pub = Pub(a.interval, len(syms))

    def publish():  # 1 s loop: the tick callback stays cheap
        while not stop.wait(1):
            pub.bars(bars)
    threading.Thread(target=publish, daemon=True).start()
    ctx, prev, out = {}, {}, angel.STORE / f"signals_{a.interval}.csv"
    store = lambda s: angel.STORE / a.interval / f"{s}.parquet"
    for s in syms:  # stored candles are context right away; the throttled history API only fills symbols without any
        if store(s).exists():
            ctx[s] = pd.read_parquet(store(s))

    def load_context():  # history once per symbol, in the background: throttling must never delay a scan
        while len(ctx) < len(syms) and not stop.is_set():
            for s in [s for s in syms if s not in ctx]:
                try:
                    ctx[s] = angel.history(s, a.interval, days=a.backfill)
                except Exception as e:
                    log(s, "history:", e)
                    stop.wait(60)
    threading.Thread(target=load_context, daemon=True).start()
    while now() - now().normalize() < CLOSE + step and now().weekday() < 5:
        t0 = time.time()
        bars.close_due(now() - pd.Timedelta(seconds=2))
        live, rows, todo = [], [], []
        for s in syms:  # 1) store first, so the app/model service see the closed bar within seconds
            lb = bars.frame(s)
            if lb is None:
                continue
            live.append(lb.assign(ticker=s))
            if s not in ctx:
                continue
            # keep finished tick-built bars in the store (official candles win on overlap): history grows even
            # while the history API is throttling us
            d = pd.concat([ctx[s], lb])
            d = d[~d.index.duplicated(keep="first")].sort_index()
            if len(d) > len(ctx[s]):
                store(s).parent.mkdir(parents=True, exist_ok=True)
                d.to_parquet(store(s))
                ctx[s] = d
            if prev.get(s, (None,))[0] != d.index[-1] and len(d) >= 60:
                todo.append((s, d))
        if live:
            pd.concat(live).to_parquet(angel.STORE / f"live_{a.interval}.parquet")
        for s, d in todo:  # 2) then the (slower) detection
            labs = detect(d)
            new = labs - prev.get(s, (None, set()))[1]  # a pattern is reported once, on its first bar
            prev[s] = (d.index[-1], labs)
            atr, c0 = float(evaluate.atr(d).iloc[-1]), float(d.close.iloc[-1])
            rows += [dict(bar=d.index[-1], ticker=s, model=MODEL, signal=lab, dir=dr, close=c0,
                          stop=round(c0 - dr * SL_ATR * atr, 2), target=round(c0 + dr * TP_ATR * atr, 2))
                     for lab, dr in new if dr]
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            log("signals:", [(r["ticker"], r["signal"]) for r in rows])
            pub.signals(rows)
        log(f"scanned {len(prev)} symbols with live bars ({len(ctx)}/{len(syms)} have context) in {time.time() - t0:.0f}s")
        t = now()
        nxt = t.normalize() + OPEN + ((t - t.normalize() - OPEN) // step + 1) * step + pd.Timedelta(seconds=3)
        time.sleep(max(5.0, (nxt - now()).total_seconds()))
    pub.bars(bars)
    stop.set()
    log("market closed")
