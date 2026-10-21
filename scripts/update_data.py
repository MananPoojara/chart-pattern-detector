"""Bring the private data copy up to date. Sources are only READ; output goes to --out.

  python scripts/update_data.py --native-root "<AmiBroker 'Updated Till DD-MM-YYYY' folder>" \
      --decoder-dir "<Book-to-rust>/library/market-data" --out ~/.cache/chart-pattern-detector/data

Stocks : existing series (CPD_STOCK_ROOT) + newer bars decoded from the AmiBroker database (audited decoder),
         then any later trading days from NSE's official bhavcopy. If the overlap with the old series disagrees
         (> 0.5%: split/bonus re-adjusted by the vendor), the whole series is replaced by the new decode.
Indices: CPD_INDEX_CSV (+ --index-csv newer copy) + NSE ind_close_all files for missing days.
Breadth: CPD_BREADTH_CSV + advances/declines counted from NSE bhavcopy for missing days.
"""
import argparse
import io
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import data  # noqa: E402

NSE = "https://nsearchives.nseindia.com/content"
HDR = {"User-Agent": "Mozilla/5.0"}
INDEX_NAMES = {"Nifty 50": "NIFTY 50", "Nifty Bank": "NIFTYBANK", "Nifty IT": "NIFTYIT",
               "Nifty Midcap 100": "NIFTYMIDCAP100", "Nifty Midcap Select": "NIFTYMIDCAPSELECT"}


def _get(url):
    r = requests.get(url, headers=HDR, timeout=30)
    return r.content if r.status_code == 200 else None


def bhavcopy(day):
    """NSE cash-market bhavcopy (UDiFF format) for one date -> DataFrame of EQ rows, or None if not published."""
    raw = _get(f"{NSE}/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
    if raw is None:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        b = pd.read_csv(z.open(z.namelist()[0]))
    b = b[b.SctySrs.isin(["EQ", "BE", "BZ"])]
    return pd.DataFrame({"ticker": b.TckrSymb, "open": b.OpnPric, "high": b.HghPric, "low": b.LwPric,
                         "close": b.ClsPric, "prev": b.PrvsClsgPric, "volume": b.TtlTradgVol.astype(float)})


def index_close(day):
    raw = _get(f"{NSE}/indices/ind_close_all_{day:%d%m%Y}.csv")
    if raw is None:
        return None
    t = pd.read_csv(io.BytesIO(raw))
    names = {k.lower(): v for k, v in INDEX_NAMES.items()}  # NSE's capitalisation varies ("NIFTY Midcap 100")
    t = t[t["Index Name"].str.lower().isin(names)]
    num = lambda c: pd.to_numeric(t[c], errors="coerce")
    return pd.DataFrame({"Ticker": t["Index Name"].str.lower().map(names), "Date": day.strftime("%d-%m-%Y"),
                         "Open": num("Open Index Value"), "High": num("High Index Value"), "Low": num("Low Index Value"),
                         "Close": num("Closing Index Value"), "Volume": num("Volume")})


def trading_days(start, end):
    return [d for d in pd.bdate_range(start, end)]


def update_stocks(native_root, decoder_dir, out, until):
    sys.path.insert(0, str(decoder_dir))
    from native_import import decode_valid_tail

    src_root = Path(data.STOCK_ROOT)
    man = pd.read_csv(src_root / "index.csv")
    dst = out / "stocks"
    log, done = [], {}
    for _, r in man.iterrows():
        if r.path in done:  # same CSV shared by several baskets
            continue
        old = pd.read_csv(src_root / r.path)
        new_rows = None
        if native_root and isinstance(r.base_source, str):
            # same symbol file in the newer database: swap the dated folder names
            rel = Path(r.base_source)
            db_folder, sym_rel = rel.parts[-3], Path(*rel.parts[-2:])
            new_db = next((p for p in Path(native_root).iterdir()
                           if re.sub(r"\d\d-\d\d-\d{4}", "", p.name).strip().lower() ==
                           re.sub(r"\d\d-\d\d-\d{4}", "", db_folder).strip().lower()), None)
            f = new_db / sym_rel if new_db else None
            if f is not None and f.exists():
                try:  # strict decode, or the audited clean-suffix rule for files with bad early bars
                    new_rows = pd.DataFrame(decode_valid_tail(f.read_bytes(), sym_rel.name)[0])
                except Exception as e:  # keep the old series rather than guess
                    log.append((r.ticker, "decode_failed", str(e)[:80]))
        status = "unchanged"
        if new_rows is not None and len(new_rows):
            ov = old.merge(new_rows[["date", "close"]], on="date", suffixes=("", "_new")).tail(20)
            if len(ov) and (np.abs(ov.close_new / ov.close - 1) > 0.005).any():
                new_rows = new_rows[new_rows.date >= old.date.iloc[0]]  # vendor re-adjusted history
                old, status = new_rows[old.columns], "replaced_readjusted"
            else:
                add = new_rows[new_rows.date > old.date.iloc[-1]]
                if len(add):
                    old, status = pd.concat([old, add[old.columns]], ignore_index=True), f"+{len(add)} native"
        done[r.path] = old
        log.append((r.ticker, status, old.date.iloc[-1]))
    # later days from NSE bhavcopy
    last = max(d.date.iloc[-1] for d in done.values())
    for day in trading_days(pd.Timestamp(last) + pd.Timedelta(days=1), until):
        b = bhavcopy(day)
        if b is None:
            continue
        b = b.set_index("ticker")
        n = 0
        for _, r in man.drop_duplicates("path").iterrows():
            d = done[r.path]
            if r.ticker in b.index and d.date.iloc[-1] < day.strftime("%Y-%m-%d"):
                q = b.loc[r.ticker]
                if isinstance(q, pd.DataFrame):
                    q = q.iloc[0]
                if abs(np.log(q.prev / d.close.iloc[-1])) > 0.02:  # our last close must equal NSE's previous close
                    log.append((r.ticker, f"bhav_skip_{day.date()}", "prev-close mismatch (corporate action?)"))
                    continue
                row = {c: 0.0 for c in d.columns}
                row.update(date=day.strftime("%Y-%m-%d"), open=q.open, high=q.high, low=q.low, close=q.close, volume=q.volume)
                done[r.path] = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
                n += 1
        print("bhavcopy", day.date(), "rows added", n, flush=True)
    for path, d in done.items():
        p = dst / path
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p, index=False)
    m2 = man.copy()
    m2["last"] = m2.path.map(lambda p: done[p].date.iloc[-1])
    m2["bars"] = m2.path.map(lambda p: len(done[p]))
    m2.to_csv(dst / "index.csv", index=False)
    pd.DataFrame(log, columns=["ticker", "status", "detail"]).to_csv(out / "update_log.csv", index=False)
    return pd.DataFrame(log, columns=["ticker", "status", "detail"])


def update_indices(extra_csv, out, until):
    a = pd.read_csv(data.INDEX_CSV, thousands=",")
    if extra_csv:
        a = pd.concat([a, pd.read_csv(extra_csv, thousands=",")])
    a["_d"] = pd.to_datetime(a.Date, format="%d-%m-%Y")
    a = a.drop_duplicates(["Ticker", "_d"], keep="last")
    for day in trading_days(a._d.max() + pd.Timedelta(days=1), until):
        t = index_close(day)
        if t is not None and len(t):
            t["_d"] = day
            a = pd.concat([a, t])
            print("index close", day.date(), len(t), flush=True)
    a = a.sort_values(["Ticker", "_d"]).drop(columns="_d")
    a.to_csv(out / "idx_data.csv", index=False)
    return a.groupby("Ticker").Date.last()


def update_breadth(breadth_csv, out, until):
    b = pd.read_csv(breadth_csv, parse_dates=["Date"])
    rows = []
    for day in trading_days(b.Date.max() + pd.Timedelta(days=1), until):
        c = bhavcopy(day)
        if c is None:
            continue
        up, dn = (c.close > c.prev), (c.close < c.prev)
        rows.append(dict(Date=day, Advances=int(up.sum()), Declines=int(dn.sum()), Unchanged=int((~up & ~dn).sum()),
                         UpVolume=float(c.volume[up].sum()), DownVolume=float(c.volume[dn].sum()),
                         AdvVolume=float(c.volume[up].sum()), DecVolume=float(c.volume[dn].sum())))
    b = pd.concat([b, pd.DataFrame(rows)], ignore_index=True)
    b.to_csv(out / "NSE_ADV_DEC.csv", index=False, date_format="%Y-%m-%d")
    print("breadth days added", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-root", type=Path, help="newer AmiBroker 'Updated Till ...' folder (read only)")
    ap.add_argument("--decoder-dir", type=Path, help="folder with native_import.py")
    ap.add_argument("--index-csv", type=Path, help="newer index CSV to merge before NSE top-up")
    ap.add_argument("--out", type=Path, default=Path.home() / ".cache/chart-pattern-detector/data")
    ap.add_argument("--until", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    a = ap.parse_args()
    if a.out.resolve() in {Path(data.STOCK_ROOT).resolve(), Path(data.STOCK_ROOT).resolve().parent}:
        sys.exit("refusing to write into the source data folder")
    a.out.mkdir(parents=True, exist_ok=True)
    until = pd.Timestamp(a.until)
    print(update_indices(a.index_csv, a.out, until))
    update_breadth(os.environ.get("CPD_BREADTH_CSV"), a.out, until)
    log = update_stocks(a.native_root, a.decoder_dir, a.out, until)
    print(log.status.str.replace(r"\+\d+ native", "+N native", regex=True).value_counts())
    print("latest dates:", log.detail.value_counts().head(5).to_dict())
