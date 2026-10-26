"""Indian market data loader (NSE stocks + NSE indices).

Data lives outside git. Point env vars at your private copies:
  CPD_STOCK_ROOT  dir containing index.csv + daily/ CSVs, or index.csv + bars.parquet (scripts/get_data.py)
  CPD_INDEX_CSV   Ticker,Date,Open,High,Low,Close,Volume  (dd-mm-yyyy)
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

STOCK_ROOT = os.environ.get("CPD_STOCK_ROOT", "data/stocks")
INDEX_CSV = os.environ.get("CPD_INDEX_CSV", "data/idx_data.csv")
COLS = ["open", "high", "low", "close", "volume"]


@lru_cache(None)
def manifest() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(STOCK_ROOT, "index.csv"), usecols=["basket", "ticker", "path", "first", "last", "bars"])
    return m


@lru_cache(None)
def _bars():
    p = os.path.join(STOCK_ROOT, "bars.parquet")
    return pd.read_parquet(p).groupby("path") if os.path.exists(p) else None


@lru_cache(None)
def _indices() -> pd.DataFrame:
    d = pd.read_csv(INDEX_CSV, thousands=",")
    d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
    return d


def universe() -> dict[str, list[str]]:
    """basket -> tickers. Indices are their own basket."""
    out = {"indices": sorted(_indices()["Ticker"].unique())}
    for b, g in manifest().groupby("basket"):
        out[b] = sorted(g["ticker"].unique())
    return out


@lru_cache(256)
def load(ticker: str, basket: str | None = None) -> pd.DataFrame:
    """Daily OHLCV, DatetimeIndex 'date', lowercase columns."""
    if basket == "indices" or (basket is None and ticker in universe()["indices"]):
        d = _indices().query("Ticker == @ticker").rename(columns=str.lower).set_index("date")
    else:
        m = manifest()
        row = m[(m.ticker == ticker) & ((m.basket == basket) if basket else True)].iloc[0]
        b = _bars()
        d = (b.get_group(row.path) if b is not None else
             pd.read_csv(os.path.join(STOCK_ROOT, row.path), parse_dates=["date"])).set_index("date")
    d = d[COLS].astype(float).sort_index()
    d["volume"] = d["volume"].fillna(0)
    return d[~d.index.duplicated()]


def corporate_action_mask(df: pd.DataFrame, thresh: float = 0.35) -> pd.Series:
    """True on bars whose close-to-close log move exceeds `thresh`.
    The stock cache is NOT verified split/bonus adjusted; such jumps are fake patterns."""
    return np.log(df["close"]).diff().abs() > thresh
