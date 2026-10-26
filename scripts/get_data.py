"""Download the published NSE daily data + trained models (refreshed every trading night). No training needed.

  python scripts/get_data.py            # data -> ./data, models -> ~/.cache/chart-pattern-detector/results
"""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd.bench import OUT  # noqa: E402

URL = "https://github.com/MananPoojara/chart-pattern-detector/releases/download/data/"
DATA = Path(__file__).resolve().parents[1] / "data"

for name, dest in [("data.tar", DATA), ("models.tar", Path(OUT))]:
    print("downloading", name, "...", flush=True)
    raw = urllib.request.urlopen(URL + name, timeout=120).read()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    print(f"  {len(raw) / 1e6:.0f} MB -> {dest}")
print("done: streamlit run app.py")
