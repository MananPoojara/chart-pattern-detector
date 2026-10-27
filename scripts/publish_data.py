"""Pack the data + trained models so people who clone the repo can run everything without training,
and upload them to the `data` GitHub release (overwrites the previous upload).

  python scripts/publish_data.py [--no-upload]

data.tar   : stocks/index.csv + stocks/bars.parquet (NSE500, daily OHLCV), idx_data.csv, NSE_ADV_DEC.csv,
             fundamentals/quarterly.parquet (NSE quarterly results, point-in-time)
models.tar : trained TradePlan / ChartCNN / PatternEdge outputs the app reads, plus the benchmark tables
"""
import argparse
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import data  # noqa: E402
from cpd.bench import OUT  # noqa: E402

REPO = "MananPoojara/chart-pattern-detector"
TAG = "data"
MODEL_FILES = ["tradeplan_model.pkl", "tradeplan_model_image.pkl", "tradeplan_today.pkl", "cnn_i20.pt", "cnn_pca.pkl",
               "edge_preds_PatternEdge.pkl", "edge_importance_PatternEdge.csv", "summary.csv", "per_pattern.csv",
               "tradeplan_backtest.csv", "tradeplan_backtest_image.csv", "tradeplan_by_year.csv", "analog_eval.csv"]


def _add_bytes(tar, name, b):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime = len(b), time.time()
    tar.addfile(ti, io.BytesIO(b))


def pack(stage):
    root = Path(data.STOCK_ROOT)
    man = pd.read_csv(root / "index.csv")[["basket", "ticker", "path", "first", "last", "bars"]]  # no private source paths
    bars = pd.concat([pd.read_csv(root / p, usecols=["date"] + data.COLS).assign(path=p) for p in man.path.unique()])
    bars["date"] = pd.to_datetime(bars.date)
    pq = io.BytesIO()
    bars.to_parquet(pq, index=False, compression="zstd")
    with tarfile.open(stage / "data.tar", "w") as t:
        _add_bytes(t, "stocks/index.csv", man.to_csv(index=False).encode())
        _add_bytes(t, "stocks/bars.parquet", pq.getvalue())
        t.add(data.INDEX_CSV, "idx_data.csv")
        t.add(os.environ.get("CPD_BREADTH_CSV", "data/NSE_ADV_DEC.csv"), "NSE_ADV_DEC.csv")
        from cpd import results  # NSE quarterly results with announcement dates (PatternEdge analog context)
        if os.path.exists(results.PATH):
            t.add(results.PATH, "fundamentals/quarterly.parquet")
    with tarfile.open(stage / "models.tar", "w") as t:
        for f in MODEL_FILES:
            t.add(os.path.join(OUT, f), f)
    asof = pd.read_pickle(os.path.join(OUT, "tradeplan_model.pkl"))["asof"]
    return pd.Timestamp(asof).date(), man.ticker.nunique(), bars.date.max().date()


def upload(stage, notes):
    gh = lambda *a: subprocess.run(["gh", *a, "-R", REPO], capture_output=True, text=True)
    if gh("release", "view", TAG).returncode:
        r = gh("release", "create", TAG, "--title", "Data + trained models", "--notes", notes)
        assert r.returncode == 0, r.stderr
    else:
        gh("release", "edit", TAG, "--notes", notes)
    r = gh("release", "upload", TAG, str(stage / "data.tar"), str(stage / "models.tar"), "--clobber")
    assert r.returncode == 0, r.stderr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    stage = Path(OUT).parent / "release"
    stage.mkdir(exist_ok=True)
    asof, n, last = pack(stage)
    notes = (f"Daily NSE data up to **{last}** ({n} NSE500 stocks + 5 NSE indices + breadth) and models with plans "
             f"as of {asof}. Refreshed every trading night.\n\n`python scripts/get_data.py` downloads and unpacks both.")
    print(notes, {f.name: f"{f.stat().st_size / 1e6:.1f} MB" for f in stage.glob("*.tar")})
    if not a.no_upload:
        upload(stage, notes)
        print("uploaded")
