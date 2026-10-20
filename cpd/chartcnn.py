"""ChartCNN: a chart-image model that PREDICTS (Jiang, Kelly & Xiu 2023, "(Re-)Imag(in)ing Price Trends").

Difference from ChartScanAI/foduucom: no hand-drawn boxes. The image is the last N bars ending at t and
the label is what happened AFTER t, so nothing in the training signal comes from hindsight.

Image (JKX "I20"): 20 days x 3 px wide, 64 px tall, black background, white pixels.
  per day: high-low vertical bar in the middle column, open tick left, close tick right;
  20-day moving average as a line; volume bars in the bottom 1/5. Each image is scaled to its own
  min/max, so the CNN sees shape only (price level is irrelevant -> works on any stock / timeframe).

  python -m cpd.chartcnn build      # images + labels for NSE500 -> cache
  python -m cpd.chartcnn train      # train once on <= TRAIN_END, validate, save weights
  python -m cpd.chartcnn predict    # p(up) + embedding for every edge-panel row
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cpd import data, evaluate
from cpd.bench import H, OUT

torch.set_num_threads(int(os.environ.get("CPD_THREADS", "2")))
D, IMG_H, IMG_W = 20, 64, 60
VOL_H = IMG_H // 5  # bottom rows for volume
PX_H = IMG_H - VOL_H - 1  # price rows (one blank separator row)
CACHE = os.path.join(OUT, "cnn_images.npz")
WEIGHTS = os.path.join(OUT, "cnn_i20.pt")
TRAIN_END, VAL_END = "2010-12-31", "2012-12-31"  # CNN never sees 2013+ (the stacker trains on 2013+)


def render(o, h, l, c, v, ma):
    """Vectorised JKX renderer. Inputs: arrays (N, D). Returns uint8 (N, IMG_H, IMG_W) in {0,1}."""
    n = len(c)
    lo = np.nanmin(np.minimum(l, np.where(np.isnan(ma), np.inf, ma)), axis=1, keepdims=True)
    hi = np.nanmax(np.maximum(h, np.where(np.isnan(ma), -np.inf, ma)), axis=1, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    def row(p):  # price -> row index from the top
        r = np.round((p - lo) / span * (PX_H - 1))
        return (PX_H - 1 - np.clip(r, 0, PX_H - 1)).astype(np.int64)

    img = np.zeros((n, IMG_H, IMG_W), np.int16)
    ii = np.arange(n)[:, None]
    x = np.arange(D)[None, :] * 3
    rh, rl = row(h), row(l)  # rh <= rl (top is 0)
    # high-low bars via +1/-1 then cumsum down the rows
    np.add.at(img, (ii.repeat(D, 1), rh, x + 1), 1)
    ok = rl + 1 < IMG_H
    np.add.at(img, (ii.repeat(D, 1)[ok], (rl + 1)[ok], (x + 1).repeat(n, 0)[ok]), -1)
    img = np.cumsum(img, axis=1)
    img[:, PX_H:, :] = 0
    img[ii, row(o), x] = 1
    img[ii, row(c), x + 2] = 1
    m = ~np.isnan(ma)
    for dx in range(3):  # MA drawn across the whole 3-px day so it reads as a line
        img[ii.repeat(D, 1)[m], row(np.where(m, ma, lo))[m], (x + dx).repeat(n, 0)[m]] = 1
    vmax = np.nanmax(v, axis=1, keepdims=True)
    vh = np.round(np.where(vmax > 0, v / np.where(vmax > 0, vmax, 1), 0) * VOL_H).astype(np.int64)
    for k in range(VOL_H):  # bar of height vh at the bottom
        img[:, IMG_H - 1 - k, :][:, x[0] + 1] = (vh > k).astype(np.int16)
    return (img > 0).astype(np.uint8)


def windows(df, idx):
    """Arrays (len(idx), D) of the D bars ending at each index in idx (inclusive)."""
    ma = df.close.rolling(D).mean().values
    k = np.asarray(idx)[:, None] + np.arange(-D + 1, 1)[None, :]
    a = lambda s: s[k]
    return a(df.open.values), a(df.high.values), a(df.low.values), a(df.close.values), a(df.volume.values), a(ma)


def excursions(df, h=H):
    """Max favourable / adverse excursion over the next h bars from entry at next open, in ATR units.
    These set data-driven targets and stops (instead of a fixed 2:1)."""
    e = df.open.shift(-1).values
    hi = np.lib.stride_tricks.sliding_window_view(np.r_[df.high.values, [np.nan] * (h + 1)][1:], h)[: len(df)]
    lo = np.lib.stride_tricks.sliding_window_view(np.r_[df.low.values, [np.nan] * (h + 1)][1:], h)[: len(df)]
    a = evaluate.atr(df).values
    with np.errstate(invalid="ignore"):
        up = (np.max(hi, 1) - e) / a  # NaN near the end (future unknown)
        dn = (e - np.min(lo, 1)) / a
    return up.astype(np.float32), dn.astype(np.float32)


def build(basket="nse500", start="2003-01-01"):
    """One row per (ticker, day): packed image bits + labels. Same row order and filter as the edge panel."""
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    keys = panel[["ticker", "date"]].astype({"ticker": str})
    imgs, ups, dns = [], [], []
    for i, (t, g) in enumerate(keys.groupby("ticker", sort=False)):
        df = data.load(t, basket)
        pos = df.index.get_indexer(g.date)
        im = render(*windows(df, pos))
        imgs.append(np.packbits(im.reshape(len(pos), -1), axis=1))
        up, dn = excursions(df)
        ups.append(up[pos])
        dns.append(dn[pos])
        if i % 50 == 0:
            print(i, t, len(pos), flush=True)
    order = np.concatenate([g.index.values for _, g in keys.groupby("ticker", sort=False)])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    bits = np.concatenate(imgs)[inv]
    np.savez(CACHE, bits=bits, mfe=np.concatenate(ups)[inv], mae=np.concatenate(dns)[inv])
    print("images", bits.shape)


def unpack(bits):
    return np.unpackbits(bits, axis=1)[:, : IMG_H * IMG_W].reshape(-1, 1, IMG_H, IMG_W).astype(np.float32)


class Net(nn.Module):
    """JKX I20 CNN: 3 blocks of [5x3 conv, BN, LeakyReLU, 2x1 max-pool], then a linear head.
    Channels 64/128/256 in the paper; 16/32/64 here for CPU (the paper reports little sensitivity)."""

    def __init__(self, ch=(16, 32, 64)):
        super().__init__()
        layers, cin = [], 1
        for k, cout in enumerate(ch):
            layers += [nn.Conv2d(cin, cout, (5, 3), stride=(3, 1) if k == 0 else 1, dilation=(2, 1) if k == 0 else 1,
                                 padding=(12, 1) if k == 0 else (2, 1)),
                       nn.BatchNorm2d(cout), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
            cin = cout
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.features(torch.zeros(1, 1, IMG_H, IMG_W)).numel()
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(n, 2)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def embed(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def _labels(panel):
    """Beat the day's average NSE500 stock over the next 10 bars? JKX used the raw sign, but on NSE the raw sign
    is mostly market direction (val loss stayed above log 2), so we remove the market each day."""
    ex = panel.fwd - panel.groupby("date").fwd.transform("mean")
    return (ex.values > 0).astype(np.int64)


def train(epochs=8, batch=256, max_train=300_000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    z = np.load(CACHE)
    bits = z["bits"]
    y = _labels(panel)
    ok = panel.fwd.notna().values
    d = panel.date.values
    tr = np.flatnonzero(ok & (d <= np.datetime64(TRAIN_END)))
    va = np.flatnonzero(ok & (d > np.datetime64(TRAIN_END)) & (d <= np.datetime64(VAL_END)))
    tr = rng.choice(tr, min(max_train, len(tr)), replace=False)
    va = rng.choice(va, min(60_000, len(va)), replace=False)
    print("train", len(tr), "val", len(va), "up-rate", y[tr].mean().round(3), flush=True)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    best, bad = 1e9, 0
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        rng.shuffle(tr)
        for k in range(0, len(tr), batch):
            b = tr[k : k + batch]
            opt.zero_grad()
            loss = lossf(net(torch.from_numpy(unpack(bits[b]))), torch.from_numpy(y[b]))
            loss.backward()
            opt.step()
        p = predict_proba(net, bits[va])
        vl = float(lossf(torch.log(torch.from_numpy(np.c_[1 - p, p]).clamp_min(1e-7)), torch.from_numpy(y[va])))
        acc = float(((p > 0.5) == y[va]).mean())
        print(f"epoch {ep} {time.time()-t0:.0f}s val_loss {vl:.4f} val_acc {acc:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(net.state_dict(), WEIGHTS)
        else:
            bad += 1
            if bad >= 2:  # early stopping, as in the paper
                break


def load_net():
    net = Net()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return net.eval()


@torch.no_grad()
def predict_proba(net, bits, batch=2048, emb=False):
    net.eval()
    ps, es = [], []
    for k in range(0, len(bits), batch):
        x = torch.from_numpy(unpack(bits[k : k + batch]))
        e = net.embed(x)
        ps.append(torch.softmax(net.head(e), 1)[:, 1].numpy())
        if emb:
            es.append(e.numpy().astype(np.float16))
    net.train()
    return (np.concatenate(ps), np.concatenate(es)) if emb else np.concatenate(ps)


def predict_all(n_pca=8, chunk=10_000):
    """CNN p(up) + a small PCA of its embedding for every panel row -> features for the trade-plan model.
    Streams in 10k chunks: the embedding is 11,520 floats per chart (~0.5 GB per chunk)."""
    from sklearn.decomposition import PCA

    panel, _ = pd.read_pickle(os.path.join(OUT, "edge_panel.pkl"))
    bits = np.load(CACHE)["bits"]
    net = load_net()
    d = panel.date.values
    fit_rows = np.sort(np.random.default_rng(0).choice(np.flatnonzero(d <= np.datetime64(VAL_END)), 10_000, replace=False))
    _, e = predict_proba(net, bits[fit_rows], emb=True)
    pca = PCA(n_pca, random_state=0).fit(e.astype(np.float32))  # fitted on CNN-era rows only
    pd.to_pickle(pca, os.path.join(OUT, "cnn_pca.pkl"))
    p = np.empty(len(bits), np.float32)
    comps = np.empty((len(bits), n_pca), np.float32)
    for k in range(0, len(bits), chunk):
        pk, ek = predict_proba(net, bits[k : k + chunk], emb=True)
        p[k : k + chunk], comps[k : k + chunk] = pk, pca.transform(ek.astype(np.float32))
        print("scored", k + len(pk), flush=True)
    out = pd.DataFrame(comps, columns=[f"cnn_pc{i}" for i in range(n_pca)])
    out.insert(0, "cnn_p", p)
    out.to_pickle(os.path.join(OUT, "cnn_features.pkl"))
    ex = (panel.fwd - panel.groupby("date").fwd.transform("mean")).values
    for lab, m in [("val", (d > np.datetime64(TRAIN_END)) & (d <= np.datetime64(VAL_END))), ("test 2013+", d > np.datetime64(VAL_END))]:
        m &= ~np.isnan(ex)
        print(lab, "acc", round(float(((p[m] > 0.5) == (ex[m] > 0)).mean()), 4),
              "rankIC vs excess fwd", round(pd.Series(p[m]).corr(pd.Series(ex[m]), method="spearman"), 4), flush=True)


def features_for(df, n_pca=8):
    """CNN p(up) + PCA embedding for the LAST bar of any OHLCV frame (e.g. a digitized screenshot)."""
    net = load_net()
    pca = pd.read_pickle(os.path.join(OUT, "cnn_pca.pkl"))
    im = render(*windows(df.reset_index(drop=True), [len(df) - 1]))
    p, e = predict_proba(net, np.packbits(im.reshape(1, -1), axis=1), emb=True)
    out = dict(cnn_p=float(p[0]))
    out.update({f"cnn_pc{i}": float(v) for i, v in enumerate(pca.transform(e.astype(np.float32))[0][:n_pca])})
    return out, im[0]


def occlusion(img):
    """Which days drove the CNN: blank each day's 3 pixel columns, measure the drop in p(up)."""
    net = load_net()
    imgs = np.repeat(img[None], D + 1, axis=0).copy()
    for k in range(D):
        imgs[k + 1, :, 3 * k : 3 * k + 3] = 0
    p = predict_proba(net, np.packbits(imgs.reshape(D + 1, -1), axis=1))
    return p[0] - p[1:]  # >0: this day pushed the model towards "up"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "predict"])
    a = ap.parse_args()
    {"build": build, "train": train, "predict": predict_all}[a.cmd]()
