# -*- coding: utf-8 -*-
"""Fault-progression target on FD002: does the rollout carry the approach to failure?

Protocol. Windows are the last 161 cycles of each published record (public_domain late=True), on
which the drift inside a window is the approach to failure rather than the first hundred cycles of
a healthy engine; channels are normalised per operating point on the training records only. The
model sees six channels and four samples and rolls out with the true operating settings.

Target. The health index is the mean of five normalised degradation channels (T24, T30, T50, Nf,
Ps30), read as a ten-cycle block centred on the horizon and on the anchor. The reported quantity is
the drift the rollout misses, divided by the drift that actually occurs over the same block, so a
predictor that holds the anchor scores exactly 1.00 and one that knows nothing about the plant can
do no better. A second reference is the mean drift measured on the training records, applied to the
test records: it represents "learn the average ramp and nothing else".

usage: python run_public_drift.py FD002 0 1
"""
import json
import os
import sys
import time

import numpy as np
import torch

import public_domain
import train

HORIZONS = (25, 50, 100)
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public4")
os.makedirs(OUT, exist_ok=True)

CHS = [0, 1, 2, 3, 4]
T0, L, BLK = 60, 4, 10


def health(Z):
    """Health index per record and cycle: mean of five channels, ten-cycle trailing block."""
    s = Z[:, :, CHS].mean(-1)
    c = np.cumsum(s, axis=1)
    out = c.copy()
    out[:, BLK:] = (c[:, BLK:] - c[:, :-BLK]) / BLK
    out[:, :BLK] = c[:, :BLK] / np.arange(1, BLK + 1)
    return out


def drift_error(dom, dd, model):
    """Drift error at each horizon, plus the two references."""
    Z = dd["yte"].cpu().numpy()
    H = health(Z)
    anchor = H[:, T0]
    out, ref = {}, {}
    for h in HORIZONS:
        true_d = H[:, T0 + h] - anchor
        rms = float(np.sqrt((true_d ** 2).mean())) + 1e-12
        ref[h] = rms
    preds, true_d = {}, {}
    with torch.enable_grad():
        for e in range(Z.shape[0]):
            w = dd["yte"][e, T0 - L + 1:T0 + 1].unsqueeze(0)
            prev = dd["yte"][e, T0]
            z = model.encode(w)
            seq = []
            for k in range(max(HORIZONS)):
                z = model.step(z, dd["ute"][e, T0 + k].unsqueeze(0))
                z = z.detach()
                if model.structure().get("latent", 0) > 0 and train.RESIDUAL_DECODE:
                    prev = prev + model.decode(z)[0].detach()
                else:
                    prev = model.decode(z)[0].detach()
                seq.append(prev.clone())
            P = torch.stack(seq).cpu().numpy()                     # (h, dy)
            hp = health(np.concatenate([Z[e:e + 1, :T0 + 1], P[None]], axis=1))
            for h in HORIZONS:
                preds.setdefault(h, []).append(hp[0, T0 + h] - anchor[e])
                true_d.setdefault(h, []).append(H[e, T0 + h] - anchor[e])
    for h in HORIZONS:
        pr = np.array(preds[h])
        out["drift_%d" % h] = round(float(np.sqrt(((pr - np.array(true_d[h])) ** 2).mean())
                                       / ref[h]), 3)
    hs = np.array(HORIZONS, dtype=float)
    ns = np.array([out["drift_%d" % h] + 1e-12 for h in HORIZONS])
    out["drift_beta"] = round(float(np.polyfit(np.log(hs), np.log(ns), 1)[0]), 3)
    return out


def reference(dom):
    """Mean-ramp baseline: learn the average drift on the training records, apply it to the test
    records, score it against the drift that actually occurs there. Persistence is 1.00."""
    Xtr, _ = dom.prebuilt()[0]
    Xte, _ = dom.prebuilt()[1]
    Htr, Hte = health(Xtr), health(Xte)
    ref = {}
    for h in HORIZONS:
        dm = float((Htr[:, T0 + h] - Htr[:, T0]).mean())
        d = Hte[:, T0 + h] - Hte[:, T0]
        rms = float(np.sqrt((d ** 2).mean())) + 1e-12
        ref["ramp_%d" % h] = round(float(np.sqrt(((dm - d) ** 2).mean()) / rms), 3)
    return ref


subset = sys.argv[1]
seeds = [int(x) for x in sys.argv[2:] if not x.startswith("--")] or [0]
only = os.environ.get("ONLY_MODELS")
KMAX = int(os.environ.get("KMAX", "30"))

dom = public_domain.make(subset=subset, condnorm=True, late=True)
ref = reference(dom)
print("references", json.dumps(ref), flush=True)

for model in MODELS:
    if only and model not in only.split(","):
        continue
    for seed in seeds:
        path = os.path.join(OUT, "D5%s_late_K%d_%s_s%d.json" % (subset, KMAX, model, seed))
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        n_tr = int(dom.prebuilt()[0][0].shape[0])
        m, mdl = train.train("D5", model, epochs=60, seed=seed, n_train=n_tr, steps=dom.steps,
                             L=4, Kmax=KMAX, curriculum=True, horizons=HORIZONS, dom=dom,
                             return_model=True)
        dd = train.build_data(dom, 0, 0, dom.steps, seed)
        m.update(drift_error(dom, dd, mdl))
        m["n_train"] = n_tr
        m["n_test"] = int(dom.prebuilt()[1][0].shape[0])
        m["dataset"] = "NASA C-MAPSS %s, late windows, per-condition normalised (public)" % subset
        m["subset"] = subset
        m["target"] = "ten-cycle health block, mean of five channels"
        m["references"] = ref
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print("%s late %s s%d drift @25=%s @50=%s @100=%s beta=%s (%.0fs)"
              % (subset, model, seed, m["drift_25"], m["drift_50"], m["drift_100"],
                 m["drift_beta"], time.time() - t0), flush=True)
print(subset, "drift grid done", flush=True)
