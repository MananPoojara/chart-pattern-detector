"""One yardstick for every repo: a signal is (date, direction ±1, score).
We measure what happens AFTER the date, using only bars after it."""
import numpy as np
import pandas as pd
from scipy import stats


def forward_return(df: pd.DataFrame, h: int = 10) -> pd.Series:
    """Signal known at close[t] -> enter next open[t+1] -> exit close[t+h]. Log return; NaN if future unknown."""
    return np.log(df["close"].shift(-h)) - np.log(df["open"].shift(-1))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def score_signals(sig: pd.DataFrame, h: int = 10, cost_bps: float = 20) -> dict:
    """sig needs columns: dir (±1), fwd (log fwd return over h), base (unconditional mean fwd of that ticker).
    Edge = signed return minus what you'd get holding the same side unconditionally."""
    sig = sig.dropna(subset=["fwd"])
    if len(sig) == 0:
        return {"n": 0}
    signed = sig["dir"] * sig["fwd"]
    excess = sig["dir"] * (sig["fwd"] - sig["base"])
    # signals on the same day are correlated; t-stat over daily means, not over raw signals
    daily = excess.groupby(sig["date"]).mean() if "date" in sig else excess
    t = stats.ttest_1samp(daily, 0).statistic if len(daily) > 2 else np.nan
    return {
        "n": len(sig),
        "hit_rate": float((signed > 0).mean()),
        "mean_signed_bps": float(signed.mean() * 1e4),
        "net_bps": float(signed.mean() * 1e4 - cost_bps),
        "excess_bps": float(excess.mean() * 1e4),
        "t_excess": float(t),
        # India: shorting cash equities overnight is restricted, so the long side alone matters
        "long_n": int((sig["dir"] > 0).sum()),
        "long_excess_bps": float(excess[sig["dir"] > 0].mean() * 1e4) if (sig["dir"] > 0).any() else np.nan,
    }
