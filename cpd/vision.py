"""YOLOv8 chart-image detectors: foduucom/stockmarket-pattern-detection-yolov8 and Omar-Karimov/ChartScanAI.

Two entry points per model:
  detect_image(path)            -> boxes on any screenshot the user uploads
  detect(df, bars)              -> we render OHLC the way the model was trained, then map boxes back to dates
Both models were trained on hand-labelled *finished* charts, so a box can describe bars the labeller saw
after the pattern. For trading we only accept boxes that end at the right edge of a chart that stops at t.
"""
import os
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from PIL import Image, ImageDraw

from cpd.rules import VENDOR

MODELS = {
    "foduucom-yolov8": dict(
        weights=("foduucom/stockmarket-pattern-detection-yolov8", "model.pt"),
        style=dict(style="charles", ylabel="Price", figsize=(6.4, 6.4)), bars=90, conf=0.25,
        dirs={"Head and shoulders bottom": 1, "Head and shoulders top": -1, "M_Head": -1, "W_Bottom": 1, "Triangle": 0, "StockLine": 0},
    ),
    "ChartScanAI": dict(
        weights=os.path.join(VENDOR, "ChartScanAI/weights/custom_yolov8.pt"),
        style=dict(style="yahoo", axisoff=True, figsize=(18, 6.5)), bars=180, conf=0.30,
        dirs={"Buy": 1, "Sell": -1},
    ),
}


@lru_cache(None)
def _model(name):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("CPD_THREADS", "2")))  # ultralytics forces up to 8 threads on import
    w = MODELS[name]["weights"]
    if isinstance(w, tuple):
        from huggingface_hub import hf_hub_download

        w = hf_hub_download(*w)
    return YOLO(w)


def detect_image(name, path, conf=None):
    m = MODELS[name]
    r = _model(name).predict(path, conf=conf or m["conf"], device="cpu", verbose=False)[0]
    return [dict(label=r.names[int(c)], conf=float(p), box=[float(v) for v in b.tolist()], dir=m["dirs"].get(r.names[int(c)], 0))
            for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


def render(name, df, path):
    """Render df in the model's training style. Returns px->bar-index mapper (no bbox_inches, so pixels are exact)."""
    d = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
    fig, axes = mpf.plot(d, type="candle", volume=False, returnfig=True, **MODELS[name]["style"])
    fig.savefig(path, dpi=fig.dpi)
    fig.canvas.draw()
    inv = axes[0].transData.inverted()
    plt.close(fig)
    return lambda x: inv.transform((x, 0))[0]


def detect(name, df, path, bars=None, conf=None):
    """Run on the last `bars` bars of df. Each detection gets the dates it spans and `at_edge`:
    True if the box reaches the last 5 bars (the only boxes usable as a live signal)."""
    w = df.iloc[-(bars or MODELS[name]["bars"]):]
    px2bar = render(name, w, path)
    out = []
    for d in detect_image(name, path, conf):
        b0, b1 = (min(max(int(round(px2bar(x))), 0), len(w) - 1) for x in (d["box"][0], d["box"][2]))
        out.append(dict(d, start=w.index[b0], end=w.index[b1], at_edge=b1 >= len(w) - 5))
    return out


def draw_boxes(path, dets):
    im = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for d in dets:
        c = {1: "#16a34a", -1: "#dc2626", 0: "#64748b"}[d["dir"]]
        dr.rectangle(d["box"], outline=c, width=3)
        dr.text((d["box"][0] + 4, d["box"][1] + 4), f"{d['label']} {d['conf']:.2f}", fill=c)
    return im


if __name__ == "__main__":
    import tempfile

    from cpd import data

    df = data.load("RELIANCE", "nifty50")
    for n in MODELS:
        dets = detect(n, df, os.path.join(tempfile.gettempdir(), f"cpd_{n}.png"))
        assert all(d["start"] <= d["end"] <= df.index[-1] for d in dets), n
        print(n, [(d["label"], round(d["conf"], 2), str(d["start"].date()), str(d["end"].date()), d["at_edge"]) for d in dets])
