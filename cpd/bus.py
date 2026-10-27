"""Redis (docker cpd-redis, 127.0.0.1:6380) + JSON helpers shared by serve.py and live_scan.py. See platform/CONTRACT.md."""
import json
import math
import os

import numpy as np
import pandas as pd


def client(timeout=None):
    import redis

    return redis.Redis(host=os.environ.get("CPD_REDIS_HOST", "127.0.0.1"), port=int(os.environ.get("CPD_REDIS_PORT", "6380")),
                       decode_responses=True, socket_connect_timeout=3, socket_timeout=timeout, health_check_interval=30)


def t(x, daily):
    """Contract time: daily -> 'YYYY-MM-DD'; intraday -> epoch seconds of the IST wall clock treated as UTC."""
    x = pd.Timestamp(x)
    return x.strftime("%Y-%m-%d") if daily else int(x.timestamp())


def clean(o, nd=4):
    """JSON-safe: numpy -> python, NaN/inf -> None, floats rounded."""
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), nd) if math.isfinite(o) else None
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def dumps(o):
    return json.dumps(clean(o), allow_nan=False, separators=(",", ":"))
