"""Point-in-time quarterly results for NSE500 from NSE filings -> fundamentals/quarterly.parquet.

Source (all NSE, no yfinance):
  filings list   /api/corporates-financial-results  (legacy, quarters to Dec-2024)
                 /api/integrated-filing-results     (SEBI integrated filing, Jan-2025 on)
  figures        the filing's XBRL (mid-2018 on) or NSE's archived results page (HTML, before that)
  announced      first public broadcast: the earliest results announcement on /api/corporate-announcements
                 (the PDF outcome, often hours or days before the XBRL) that names this period end, or sits in
                 the 3 days before the filing, or is the board-meeting-day outcome; capped by the filing broadcast.
  splits/bonus   /api/corporates-corporateActions -> actions.parquet (cpd.results puts EPS on one share basis)
Units: sales and net_profit in INR crore; eps in INR per share (basic, quarterly, not annualised).
Banks: sales = Interest Earned. Insurers: sales = net premium (life) / premium earned (general); NSE has structured
insurer results only from mid-2024. net_profit = attributable to owners when reported, else profit for the period.
Rows per (symbol, quarter, basis): consolidated whenever filed; standalone where no consolidated exists and for the
first 3 years of consolidated reporting (bridges YoY/SUE over the switch). cpd.results.load() returns one row per
quarter, consolidated preferred. Revisions dropped: first broadcast wins.
Known gap: ~2018Q1-Q3 legacy filings carry neither XBRL nor an HTML page on NSE, so those quarters are missing.

Raw downloads are cached under fundamentals/raw/<SYM>/ so the build step never re-hits NSE.
  python scripts/fetch_results.py              # incremental: symbols with new filings since last run
  python scripts/fetch_results.py --backfill   # full history for all NSE500, resumable
  python scripts/fetch_results.py --build      # rebuild the parquet from the cache only
"""
import argparse
import fcntl
import gzip
import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(os.path.expanduser("~/.cache/chart-pattern-detector/fundamentals"))
RAW, OUT, STATE = ROOT / "raw", ROOT / "quarterly.parquet", ROOT / "state.json"
API = "https://www.nseindia.com/api/"
FIRST_QUARTER = pd.Timestamp("2014-01-01")  # 2 years of warm-up so YoY / SUE exist from 2016
INTEG = "type=Integrated%20Filing-%20Financials"


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


class NSE:
    """One session, >= 1.1 s between requests, backoff on 401/403/429/5xx."""
    H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
         "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
         "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"}

    def __init__(self):
        self.last, self.s = 0.0, None
        self.n = 0

    def _prime(self):
        self.s = requests.Session()
        self.s.headers.update(self.H)
        for u in ("https://www.nseindia.com/", self.H["Referer"]):
            self._wait()
            try:
                self.s.get(u, timeout=20)
            except requests.RequestException:
                pass

    def _wait(self):
        time.sleep(max(0.0, self.last + 1.1 - time.time()))
        self.last = time.time()

    def get(self, url):
        if self.s is None:
            self._prime()
        for attempt in range(5):
            self._wait()
            self.n += 1
            try:
                r = self.s.get(url, timeout=40)
            except requests.RequestException as e:
                code, r = str(e)[:60], None
            else:
                code = r.status_code
                if code == 200:
                    return r.content
                if code == 404:
                    return None
            pause = 30 * 2 ** attempt
            log(f"  {code} on {url[:90]} -> sleep {pause}s")
            time.sleep(pause)
            if code in (401, 403):
                self._prime()
        return None


# ---------- fetch (fills the cache) ----------

def _save_json(p, b):
    try:
        json.loads(b)
    except (ValueError, TypeError):
        return False
    p.write_bytes(b)
    return True


def fetch_symbol(nse, sym, refresh=True):
    d = RAW / sym
    (d / "docs").mkdir(parents=True, exist_ok=True)
    q = requests.utils.quote(sym)
    today = datetime.now().strftime("%d-%m-%Y")
    lists = [("legacy.json", f"corporates-financial-results?index=equities&symbol={q}&period=Quarterly", False),
             ("integrated.json", f"integrated-filing-results?index=equities&symbol={q}&{INTEG}&size=1000", True),
             ("ann.json", f"corporate-announcements?index=equities&symbol={q}&from_date=01-10-2013&to_date={today}", True),
             ("ca.json", f"corporates-corporateActions?index=equities&symbol={q}&from_date=01-01-2013&to_date={today}", True)]
    for name, path, live in lists:
        if (d / name).exists() and not (refresh and live):
            continue
        b = nse.get(API + path)
        if name == "legacy.json" and b is not None and b.strip() == b"[]":  # insurers live under their own index
            b = nse.get(API + path.replace("index=equities", "index=insurance"))
        if b is None or not _save_json(d / name, b):
            log(f"  {sym}: list {name} failed")
    for f in choose(filings(sym)):
        for c in f["cands"]:  # first broadcast; fall back to a later one only if its doc is gone
            p = d / "docs" / (Path(c["url"]).name + ".gz")
            if p.exists():
                break
            b = nse.get(c["url"])
            if b and len(b) > 500:
                p.write_bytes(gzip.compress(b))
                break


# ---------- filings list -> one candidate set per quarter ----------

def _ts(s):
    return pd.to_datetime(s, format="mixed", dayfirst=True) if s else pd.NaT


def filings(sym):
    d, out = RAW / sym, []
    if (d / "legacy.json").exists():
        for x in json.loads((d / "legacy.json").read_text()) or []:
            url = x.get("xbrl") or ""
            kind = "xbrl" if url.endswith(".xml") else "html"
            if kind == "html":
                url = x.get("resultDetailedDataLink") or ""
            end = _ts(x.get("toDate") or x.get("periodEnd"))
            start = _ts(x.get("fromDate")) if x.get("fromDate") else end - pd.Timedelta(days=90)  # insurers: end only
            if not url or pd.isna(end) or not 80 <= (end - start).days <= 100:
                continue
            out.append(dict(end=end, cons=x.get("consolidated") == "Consolidated", url=url, kind=kind,
                            bcast=_ts(x.get("broadCastDate"))))
    if (d / "integrated.json").exists():
        for x in json.loads((d / "integrated.json").read_text()).get("data") or []:
            url = x.get("xbrl") or ""
            if not url.endswith(".xml"):
                continue
            out.append(dict(end=_ts(x.get("qe_Date")), cons=x.get("consolidated") == "Consolidated", url=url,
                            kind="xbrl", bcast=_ts(x.get("broadcast_Date"))))
    return [f for f in out if not pd.isna(f["bcast"]) and f["end"] >= FIRST_QUARTER]


def choose(fs):
    """[(end, consolidated, candidates by broadcast)] to download/parse. Consolidated whenever filed; standalone where
    there is no consolidated, and for the first 3 years of the consolidated series, so that YoY / SUE survive the
    switch from standalone-only to consolidated reporting (SEBI made quarterly consolidated mandatory in FY20)."""
    by = {}
    for f in fs:
        by.setdefault((f["end"], f["cons"]), []).append(f)
    first_cons = min((e for e, c in by if c), default=None)
    res = []
    for (end, cons), g in sorted(by.items()):
        if cons or (end, True) not in by or end <= first_cons + pd.Timedelta(days=1100):
            res.append(dict(end=end, cons=cons, cands=sorted(g, key=lambda f: f["bcast"])))
    return res


# ---------- parsers ----------

# first non-zero tag wins (filers sometimes put 0 in the attributable line); insurers: net premium, PAT, EPS
SALES = ["RevenueFromOperations", "InterestEarned", "TotalRevenueFromOperations", "NetPremiumIncome",
         "PremiumEarned", "Income"]
PROFIT = ["ProfitOrLossAttributableToOwnersOfParent", "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
          "ProfitLossForPeriod", "ProfitLossForThePeriod", "ProfitLossAfterTaxAndExtraordinaryItems",
          "ProfitLossAfterTaxBeforeExtraordinaryItems", "ProfitLossAfterTax"]
EPS = ["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations", "BasicEarningsPerShareAfterExtraordinaryItems",
       "BasicEarningsLossPerShareFromContinuingOperations", "BasicEarningsPerShareBeforeExtraordinaryItems",
       "BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
       "BasicAndDilutedEPSBeforeExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized"]


def parse_xbrl(b, end):
    root = ET.fromstring(b)
    ctx = {}
    for c in root.iter():
        if not c.tag.endswith("}context"):
            continue
        dims = any(e.tag.endswith("explicitMember") or e.tag.endswith("typedMember") for e in c.iter())
        s = [e.text for e in c.iter() if e.tag.endswith("}startDate")]
        e_ = [e.text for e in c.iter() if e.tag.endswith("}endDate")]
        if s and e_ and not dims:
            ctx[c.get("id")] = (pd.Timestamp(s[0]), pd.Timestamp(e_[0]))
    # 2018-22 NSE files often omit the plain contexts (OneD/FourD); their period is still stated as facts
    rp = {}
    for el in root:
        n = el.tag.split("}")[-1]
        if n in ("DateOfStartOfReportingPeriod", "DateOfEndOfReportingPeriod") and el.text:
            rp.setdefault(el.get("contextRef"), {})[n[6]] = pd.Timestamp(el.text.strip())
    for k, v in rp.items():
        if k not in ctx and len(v) == 2:
            ctx[k] = (v["S"], v["E"])
    if "OneD" not in ctx and rp.get("OneD", {}).get("E", end) == end:  # bank files: end date only; OneD = quarter
        ctx["OneD"] = (end - pd.Timedelta(days=90), end)
    good = {k for k, (s, e) in ctx.items() if e == end and 80 <= (e - s).days <= 100}
    facts, meta = {}, {}
    for el in root:
        name = el.tag.split("}")[-1]
        if name in ("DateOfStartOfBoardMeeting", "DateOfBoardMeetingWhenFinancialResultsWereApproved") and el.text:
            meta.setdefault("board", el.text.strip())
        elif el.get("contextRef") in good and el.text and el.text.strip():
            facts.setdefault(name, el.text.strip())

    def pick(names, scale):
        vs = []
        for n in names:
            try:
                vs.append(float(facts[n]) / scale)
            except (KeyError, ValueError):
                pass
        return next((v for v in vs if v), vs[0] if vs else None)
    return dict(sales=pick(SALES, 1e7), net_profit=pick(PROFIT, 1e7), eps=pick(EPS, 1),
                board=_ts(meta.get("board")))


H_SALES = [r"^net sales\s*/\s*income from operations", r"^revenue from operations", r"^interest earned",
           r"^income from operations", r"^total income from operations", r"^net sales"]
H_PROFIT = [r"after (taxes,? )?minority interest", r"attributable to (the )?(owners|equity holders|shareholders)",
            r"^consolidated net profit", r"^net profit\s*\(?\+?\)?\s*/\s*\(?loss\)?.{0,6} for the period",
            r"^net profit.*for the period", r"^profit\s*/?\s*\(?loss\)? for the period"]
H_EPS = [r"^basic eps for continu\w* and discontinued", r"^basic eps after extraordinary", r"^basic eps for continuing",
         r"^basic eps before extraordinary", r"^basic eps", r"after extraordinary.*\) basic", r"before extraordinary.*\) basic",
         r"^earnings per share.*\) basic"]
NUM = re.compile(r"^\(?-?[\d,]+(\.\d+)?\)?$")


def parse_html(b, end=None):
    t = re.sub(r"(?is)<(script|style).*?</\1>|<!--.*?-->", " ", b.decode("latin-1"))
    lines = [html.unescape(x).replace("\xa0", " ").strip() for x in re.split(r"<[^>]+>", t)]
    lines = [re.sub(r"\s+", " ", x) for x in lines if x.strip()]
    unit = 1.0
    for x in lines:
        m = re.search(r"Rs\.?\s*in\s*(lakhs|lacs|crores?|thousands|millions)", x, re.I)
        if m:
            unit = {"l": 100.0, "c": 1.0, "t": 1e4, "m": 10.0}[m.group(1)[0].lower()]
            break
    vals, label, head = {}, None, ""
    for x in lines:
        if x.lower().startswith("segment reporting"):
            break
        if NUM.match(x) and label:
            v = x.replace(",", "")
            v = -float(v.strip("()")) if v.startswith("(") else float(v)
            key = f"{head} {label}" if re.match(r"\([a-z]+\)\s*(basic|diluted)", label, re.I) else label
            vals.setdefault(key.lower(), v)
            label = None
        elif not NUM.match(x) and x != "-":
            if label:
                head = label  # two labels in a row: the first is a section header, e.g. "Earnings per share (...)"
            label = x
        else:
            label = None

    def pick(pats, scale):
        vs = [v / scale for p in pats for k, v in vals.items() if re.search(p, k)]
        return next((v for v in vs if v), vs[0] if vs else None)
    np_ = pick(H_PROFIT, unit)
    # NSE's typed 2016-18 pages sometimes carry a wrong tax line (finance cost copied in), corrupting the PAT rows;
    # the filer's own comprehensive-income lines stay right, so PAT = total comprehensive income - OCI wins on conflict
    tci, oci = pick([r"^total comprehensive income"], unit), pick([r"^other comprehensive income"], unit)
    if tci is not None and oci is not None and tci - oci and (np_ is None or abs(tci - oci - np_) > 0.01 * abs(np_)):
        np_ = tci - oci
    return dict(sales=pick(H_SALES, unit), net_profit=np_, eps=pick(H_EPS, 1), board=pd.NaT)


# ---------- announcement time ----------

RES_TXT = re.compile(r"financial result|results? for the (quarter|period)", re.I)
NOT_RES = re.compile(r"intimation|to be held|newspaper|will be held|is scheduled|trading window|e-voting", re.I)
ENDED = re.compile(r"(?:period|quarter|year)\s+ended\s+(?:on\s+)?([0-9]{1,2}(?:st|nd|rd|th)?[\s-]*[A-Za-z]{3,9},?[\s-]*[0-9]{4}"
                   r"|[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})", re.I)


def announcements(sym):
    """[(time, kind, period_end)]: kind 'res' = mentions results, 'bm' = bare 'Outcome of Board Meeting'."""
    p = RAW / sym / "ann.json"
    try:
        a = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        a = []
    out = []
    for x in a if isinstance(a, list) else []:
        txt, desc = x.get("attchmntText") or "", x.get("desc") or ""
        if NOT_RES.search(desc) or NOT_RES.search(txt):
            continue
        if desc == "Financial Result Updates" or RES_TXT.search(txt):
            m = ENDED.search(txt)
            pe = pd.to_datetime(re.sub(r"(st|nd|rd|th)\b", "", m.group(1)), errors="coerce") if m else pd.NaT
            out.append((_ts(x.get("an_dt")), "res", pe))
        elif desc == "Outcome of Board Meeting":
            out.append((_ts(x.get("an_dt")), "bm", pd.NaT))
    return out


def first_broadcast(end, bcast, board, anns):
    """Earliest public results announcement, never later than the filing broadcast:
    - a results announcement naming this period end, any time in (period end, broadcast]
    - an undated results announcement within 3 days before the broadcast
    - a bare 'Outcome of Board Meeting' on/after the board-meeting day (when the filing states that day)."""
    hits = [bcast]
    for t, kind, pe in anns:
        if pd.isna(t) or not end < t <= bcast:
            continue
        if kind == "res" and (pe == end or (pd.isna(pe) and t >= bcast - timedelta(days=3))):
            hits.append(t)
        elif kind == "bm" and not pd.isna(board) and t >= board.normalize():
            hits.append(t)
    return min(hits)


# ---------- splits / bonuses (EPS basis changes) ----------

def actions(sym):
    """[(ex_date, factor)]: shares multiply by factor on ex_date (bonus a:b -> (a+b)/b, split FV x->y -> x/y)."""
    p = RAW / sym / "ca.json"
    try:
        a = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        a = []
    out = []
    for x in a if isinstance(a, list) else []:
        s, f = (x.get("subject") or "").lower(), 1.0
        m = re.search(r"bonus\s*(\d+)\s*:\s*(\d+)", s)
        if m and int(m.group(2)):
            f *= (int(m.group(1)) + int(m.group(2))) / int(m.group(2))
        m = re.search(r"(split|sub-?division|consolidation)\D*?(?:rs|re)\.?\s*([\d.]+)\D*?to\s*(?:rs|re)\.?\s*([\d.]+)", s)
        if m and float(m.group(3)):
            f *= float(m.group(2)) / float(m.group(3))
        if f != 1.0:
            out.append((_ts(x.get("exDate")), f))
    return out


# ---------- build ----------

def build():
    rows, acts = [], []
    for d in sorted(RAW.iterdir()) if RAW.exists() else []:
        sym, anns = d.name, announcements(d.name)
        acts += [dict(symbol=sym, ex_date=t, factor=f) for t, f in actions(sym)]
        for q in choose(filings(sym)):
            for c in q["cands"]:
                p = d / "docs" / (Path(c["url"]).name + ".gz")
                if not p.exists():
                    continue
                try:
                    b = gzip.decompress(p.read_bytes())
                    v = parse_xbrl(b, q["end"]) if c["kind"] == "xbrl" else parse_html(b)
                except Exception as e:  # noqa: BLE001 - a broken file must not stop the build
                    log(f"  parse fail {sym} {p.name}: {e!s:.60}")
                    continue
                if v["sales"] is None and v["net_profit"] is None:
                    continue
                rows.append(dict(symbol=sym, period_end=q["end"], announced=first_broadcast(q["end"], c["bcast"], v["board"], anns),
                                 consolidated=q["cons"], sales=v["sales"], net_profit=v["net_profit"], eps=v["eps"],
                                 source_url=c["url"]))
                break
    df = pd.DataFrame(rows, columns=["symbol", "period_end", "announced", "consolidated", "sales", "net_profit", "eps",
                                     "source_url"])
    df = df.sort_values(["symbol", "period_end"]).reset_index(drop=True)
    pd.DataFrame(acts, columns=["symbol", "ex_date", "factor"]).drop_duplicates().to_parquet(ROOT / "actions.parquet", index=False)
    tmp = OUT.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)
    log(f"built {OUT}: {len(df)} rows, {df.symbol.nunique()} symbols, "
        f"{df.period_end.min() if len(df) else '-'} .. {df.period_end.max() if len(df) else '-'}")
    return df


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "last_run": None}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=0))


def universe():
    from cpd import data
    return data.universe()["nse500"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="full history for every NSE500 symbol (resumable)")
    ap.add_argument("--build", action="store_true", help="only rebuild the parquet from the cache")
    ap.add_argument("--symbols", nargs="*", help="limit to these symbols (forces re-fetch)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.build:
        return build()
    lock = open(ROOT / ".lock", "w")
    try:  # one downloader at a time (nightly cron vs a running backfill)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another fetch_results run holds the lock; exiting")
    nse, st, uni = NSE(), state(), universe()
    started = datetime.now()
    if a.symbols:
        todo = a.symbols
    elif a.backfill:
        todo = [s for s in uni if s not in st["done"]]
    else:  # incremental: who filed since the last run (3-day overlap) + any symbol never fetched
        since = pd.Timestamp(st["last_run"] or started - timedelta(days=7)) - timedelta(days=3)
        b = nse.get(API + f"integrated-filing-results?index=equities&{INTEG}&from_date={since:%d-%m-%Y}"
                    f"&to_date={started:%d-%m-%Y}&size=5000")
        if b is None:
            sys.exit("filings feed unavailable; try again later")
        new = {x["symbol"] for x in json.loads(b).get("data") or []}
        todo = sorted((new & set(uni)) | (set(uni) - set(st["done"])))
    log(f"{len(todo)} symbols to fetch")
    for i, sym in enumerate(todo, 1):
        try:
            fetch_symbol(nse, sym)
        except Exception as e:  # noqa: BLE001 - one bad symbol must not kill a multi-hour run
            log(f"  {sym}: {e!s:.100}")
            continue
        st["done"][sym] = datetime.now().isoformat(timespec="seconds")
        save_state(st)
        log(f"[{i}/{len(todo)}] {sym}  requests={nse.n}")
        if i % 50 == 0:
            build()
    if not a.symbols:
        st["last_run"] = started.isoformat(timespec="seconds")
        save_state(st)
    build()


if __name__ == "__main__":
    main()
