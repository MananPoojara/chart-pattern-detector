"""Angel One SmartAPI: login (TOTP), NSE symbol tokens, intraday candles kept in a local Parquet store.

Credentials come from .env in the repo root (never committed):
  ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_CODE, ANGEL_ONE_MPIN, ANGEL_ONE_TOTP_SECRET
History lands in ~/.cache/chart-pattern-detector/intraday/<INTERVAL>/<SYMBOL>.parquet and grows on every call,
which is also the training set for intraday models later.
"""
import base64
import hmac
import json
import os
import struct
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from cpd.bench import OUT

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(OUT).parent / "intraday"
# interval -> (bar length, max days per request, per SmartAPI docs)
INTERVALS = {"ONE_MINUTE": ("1min", 30), "THREE_MINUTE": ("3min", 60), "FIVE_MINUTE": ("5min", 100),
             "FIFTEEN_MINUTE": ("15min", 200), "ONE_HOUR": ("60min", 400)}


def _env():
    e = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return {**e, **{k: v for k, v in os.environ.items() if k.startswith(("ANGEL_ONE_", "INSTRUMENT_"))}}


def totp(secret, t=None):
    """RFC 6238, 6 digits, 30 s."""
    key = base64.b32decode(secret.replace(" ", "").upper() + "=" * (-len(secret.replace(" ", "")) % 8))
    h = hmac.new(key, struct.pack(">Q", int(time.time() if t is None else t) // 30), "sha1").digest()
    o = h[-1] & 15
    return f"{(struct.unpack('>I', h[o:o + 4])[0] & 0x7FFFFFFF) % 1_000_000:06d}"


def _headers(e, jwt=None):
    h = {"Content-Type": "application/json", "Accept": "application/json", "X-UserType": "USER", "X-SourceID": "WEB",
         "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1", "X-MACAddress": "00:00:00:00:00:00",
         "X-PrivateKey": e["ANGEL_ONE_API_KEY"]}
    return {**h, "Authorization": f"Bearer {jwt}"} if jwt else h


_session = {}


def login():
    """JWT for today; logs in again after 6 h. Kept in memory only."""
    if _session.get("t", 0) > time.time() - 6 * 3600:
        return _session["jwt"]
    e = _env()
    for attempt in range(4):
        resp = requests.post(e.get("ANGEL_ONE_BASE_URL", "https://apiconnect.angelone.in") +
                             "/rest/auth/angelbroking/user/v1/loginByPassword", headers=_headers(e), timeout=20,
                             json=dict(clientcode=e["ANGEL_ONE_CLIENT_CODE"], password=e["ANGEL_ONE_MPIN"],
                                       totp=totp(e["ANGEL_ONE_TOTP_SECRET"])))
        try:
            r = resp.json()
            break
        except ValueError:  # throttled: HTML/empty body
            r = {"message": f"HTTP {resp.status_code}"}
            time.sleep(15 * (attempt + 1))
    if not r.get("status"):
        raise RuntimeError(f"Angel One login failed: {r.get('message')} ({r.get('errorcode')})")
    _session.update(jwt=r["data"]["jwtToken"], feed=r["data"]["feedToken"], t=time.time())
    return _session["jwt"]


def stream(token_list, on_quote, stop):
    """SmartStream v2 (WebSocket), Quote mode, NSE cash. Calls on_quote(token, ts, ltp, day_volume) per tick until
    stop (threading.Event) is set; reconnects on drops. ts = exchange time, IST, tz-naive."""
    import threading

    import websocket

    def on_open(ws):
        ws.send(json.dumps(dict(correlationID="cpd", action=1,
                                params=dict(mode=2, tokenList=[dict(exchangeType=1, tokens=list(token_list))]))))

    def on_message(ws, msg):
        if isinstance(msg, bytes) and len(msg) >= 75:  # Quote packet, little-endian (SmartStream v2 docs)
            tok = msg[2:27].split(b"\0")[0].decode()
            ts_ms, ltp = struct.unpack_from("<qq", msg, 35)
            vol = struct.unpack_from("<q", msg, 67)[0]
            on_quote(tok, pd.Timestamp(ts_ms, unit="ms") + pd.Timedelta(hours=5, minutes=30), ltp / 100, vol)

    def heartbeat(ws):
        while not stop.wait(25):
            try:
                ws.send("ping")
            except Exception:
                return

    while not stop.is_set():
        e = _env()
        try:
            login()
        except Exception as err:
            print("stream login:", err, flush=True)
            stop.wait(30)
            continue
        ws = websocket.WebSocketApp(e.get("ANGEL_ONE_WS_URL", "wss://smartapisocket.angelone.in/smart-stream"),
                                    header={"Authorization": _session["jwt"], "x-api-key": e["ANGEL_ONE_API_KEY"],
                                            "x-client-code": e["ANGEL_ONE_CLIENT_CODE"], "x-feed-token": _session["feed"]},
                                    on_open=on_open, on_message=on_message,
                                    on_error=lambda ws, err: print("stream error:", err, flush=True))
        threading.Thread(target=heartbeat, args=(ws,), daemon=True).start()
        ws.run_forever()
        stop.wait(5)  # dropped: reconnect


@lru_cache(None)
def tokens():
    """NSE cash symbol -> SmartAPI token, from the scrip master (35 MB, slow server; equity tokens are stable,
    so a copy is reused for a week)."""
    p = STORE / "scrip_master.json"
    if not p.exists() or time.time() - p.stat().st_mtime > 7 * 86400:
        STORE.mkdir(parents=True, exist_ok=True)
        e = _env()
        url = e.get("INSTRUMENT_MASTER_URL") or e.get("ANGEL_ONE_SCRIP_MASTER_URL")
        tmp = p.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(15, 120)) as r, open(tmp, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(p)
    m = pd.DataFrame(json.loads(p.read_text()))
    m = m[(m.exch_seg == "NSE") & m.symbol.str.endswith("-EQ")]
    return dict(zip(m.name, m.token))


def _fetch(symbol, interval, start, end):
    e = _env()
    body = dict(exchange="NSE", symboltoken=tokens()[symbol], interval=interval,
                fromdate=f"{start:%Y-%m-%d %H:%M}", todate=f"{end:%Y-%m-%d %H:%M}")
    for attempt in range(4):
        resp = requests.post(e.get("ANGEL_ONE_BASE_URL", "https://apiconnect.angelone.in") +
                             "/rest/secure/angelbroking/historical/v1/getCandleData",
                             headers=_headers(e, login()), json=body, timeout=30)
        time.sleep(1.0)  # documented 3/s, but bursts get 403 "exceeding access rate"; 1/s holds
        try:
            r = resp.json()
        except ValueError:  # empty/HTML body when throttled
            r = {"message": f"rate: HTTP {resp.status_code} non-JSON"}
        if r.get("status"):
            break
        if attempt == 3 or "rate" not in str(r.get("message", "")).lower():
            raise RuntimeError(f"{symbol}: {r.get('message')} ({r.get('errorcode')})")
        time.sleep(20 * (attempt + 1))
    d = pd.DataFrame(r.get("data") or [], columns=["date", "open", "high", "low", "close", "volume"])
    d["date"] = pd.to_datetime(d.date.str[:19])  # exchange time (IST), tz dropped
    return d.set_index("date").astype(float)


def history(symbol, interval="FIVE_MINUTE", days=30):
    """Stored + newly fetched candles; only finished bars (the forming bar is dropped)."""
    step, max_days = INTERVALS[interval]
    p = STORE / interval / f"{symbol}.parquet"
    old = pd.read_parquet(p) if p.exists() else None
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    first = now.normalize() - pd.Timedelta(days=days)
    if old is not None and len(old):  # extend backwards if asked for more history, and forwards to now
        spans = ([(first, old.index[0])] if old.index[0] - first > pd.Timedelta(days=3) else []) + [(old.index[-1], now)]
    else:
        spans = [(first, now)]
    parts = [old] if old is not None else []
    for start, stop in spans:
        while start < stop:
            end = min(start + pd.Timedelta(days=max_days), stop)
            parts.append(_fetch(symbol, interval, start, end))
            start = end
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[d.index + pd.Timedelta(step) <= now]
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p)
    return d


if __name__ == "__main__":
    assert totp("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", 59) == "287082"  # RFC 6238 test vector (6-digit)
    d = history("SBIN", days=5)
    print(d.tail(3), len(d))
