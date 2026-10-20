"""Forecasting repos: Kronos, Chronos, TimesFM, nsarang VAE-analog.

forecast(name, ctxs, h) -> np.ndarray (B, h, 3): q10, q50, q90 of the CLOSE for the next h bars,
one row per context DataFrame (daily OHLCV, history up to and including bar t).
A forecaster becomes a directional signal via sign(q50[-1] / close[t] - 1).
"""
import os
import pickle
import sys
import types
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from cpd.rules import VENDOR, _path

torch.set_num_threads(int(os.environ.get("CPD_THREADS", "2")))
QS = [0.1, 0.5, 0.9]
CTX = 512


@lru_cache(None)
def _load(name):
    if name.startswith("Kronos"):
        _path("Kronos")
        from model import Kronos, KronosPredictor, KronosTokenizer

        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        return KronosPredictor(Kronos.from_pretrained("NeoQuasar/Kronos-mini"), tok, device="cpu", max_context=2048)
    if name.startswith("Chronos"):
        from chronos import BaseChronosPipeline

        return BaseChronosPipeline.from_pretrained("autogluon/chronos-2-small", device_map="cpu", torch_dtype=torch.float32)
    if name.startswith("TimesFM"):
        import timesfm

        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=128, normalize_inputs=True, per_core_batch_size=16,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=True, fix_quantile_crossing=True))
        return m
    if name.startswith("VAE"):
        class _Stub(dict):
            def __setstate__(self, s):
                pass

        class _U(pickle.Unpickler):  # checkpoint pickles lightning objects; we only need the weights
            def find_class(self, mod, n):
                return _Stub if mod.startswith("pytorch_lightning") else super().find_class(mod, n)

        _path("big-data-stock-price-forecast")
        from core.nn.vae import VAE

        ck = torch.load(os.path.join(VENDOR, "big-data-stock-price-forecast/checkpoints/masknet-epoch=23-loss_val=0.14.ckpt"),
                        map_location="cpu", weights_only=False, pickle_module=types.SimpleNamespace(Unpickler=_U, load=pickle.load, __name__="pickle"))
        vae = VAE(ck["hyper_parameters"]["latent_dim"])
        vae.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("vae.")})
        return vae.eval()
    raise KeyError(name)


def _kronos(ctxs, h, samples=8):
    p = _load("Kronos")
    cols = ["open", "high", "low", "close", "volume"]
    xs = [c.iloc[-CTX:] for c in ctxs]
    n = min(len(x) for x in xs)  # predict_batch needs equal lengths
    xs = [x.iloc[-n:] for x in xs]
    xt = [pd.Series(x.index) for x in xs]
    yt = [pd.Series(pd.bdate_range(x.index[-1] + pd.Timedelta(days=1), periods=h)) for x in xs]  # NSE holidays ignored
    # predict() averages its samples; to get a distribution, batch `samples` copies with sample_count=1
    outs = p.predict_batch([x[cols].reset_index(drop=True) for x in xs for _ in range(samples)],
                           [t for t in xt for _ in range(samples)], [t for t in yt for _ in range(samples)],
                           pred_len=h, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    paths = np.stack([o["close"].values for o in outs]).reshape(len(xs), samples, h)
    return np.moveaxis(np.quantile(paths, QS, axis=1), 0, -1)


def _chronos(ctxs, h):
    p = _load("Chronos")
    q, _ = p.predict_quantiles([torch.tensor(c.close.values[-CTX:], dtype=torch.float32)[None] for c in ctxs],
                               prediction_length=h, quantile_levels=QS)
    return np.stack([x[0].numpy() for x in q])


def _timesfm(ctxs, h):
    _, q = _load("TimesFM").forecast(horizon=h, inputs=[c.close.values[-CTX:] for c in ctxs])
    return np.asarray(q)[:, :, [1, 5, 9]]  # cols: mean, q0.1..q0.9


def _vae(ctxs, h, L=256, k=10):
    """nsarang's method: VAE-embed every 256-bar z-scored window of the SAME series, find nearest past
    windows (L1 in [mu, logvar]), average their futures. Repo averages top-2; we take q10/q50/q90 over top-k.
    Library only holds windows whose future ended before t (repo notebook does not enforce this)."""
    vae = _load("VAE")
    out = []
    for c in ctxs:
        px = c.close.values[-2500:]
        W = np.lib.stride_tricks.sliding_window_view(px, L)
        mu_, sd_ = W.mean(1, keepdims=True), W.std(1, keepdims=True) + 1e-8
        with torch.no_grad():
            m, lv = vae.encode(torch.tensor((W - mu_) / sd_, dtype=torch.float32)[:, None])
        e = torch.cat([m, lv], 1)
        q = len(W) - 1
        lib = e[: max(q - h + 1, 1)]
        idx = torch.topk(torch.cdist(lib, e[q : q + 1], p=1)[:, 0], min(k, len(lib)), largest=False).indices.tolist()
        fut = np.stack([(px[j + L : j + L + h] - mu_[j]) / sd_[j] for j in idx if j + L + h <= len(px)])
        out.append(np.quantile(fut * sd_[q] + mu_[q], QS, axis=0).T)
    return np.stack(out)


FORECASTERS = {
    "Kronos (mini)": _kronos,
    "Chronos-2 (small)": _chronos,
    "TimesFM 2.5": _timesfm,
    "VAE analog (nsarang)": _vae,
}


def forecast(name, ctxs, h=10, **kw):
    if isinstance(ctxs, pd.DataFrame):
        ctxs = [ctxs]
    return FORECASTERS[name](ctxs, h, **kw)


if __name__ == "__main__":
    import time

    from cpd import data

    df = data.load("RELIANCE", "nifty50")
    for n in ["Chronos-2 (small)", "VAE analog (nsarang)", "Kronos (mini)"]:
        t = time.time()
        f = forecast(n, [df.iloc[:-10], df.iloc[:-30]], 10)
        assert f.shape == (2, 10, 3) and np.all(f[..., 0] <= f[..., 2] + 1e-6), n
        print(n, f"{time.time()-t:.1f}s", "last close", df.close.iloc[-11], "q50 d10", f[0, -1, 1].round(1), "actual", df.close.iloc[-1])
