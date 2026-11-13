# -*- coding: utf-8 -*-
"""Standard RUL scoring on FD002, from a rollout rather than a regression head.

Training uses the last 161 cycles of every training record, normalised per operating point
(public_domain late=True, condnorm=True), exactly as the degradation grid. The failure threshold is
the health index reached at the end of those records, averaged over the training pool; a predicted
remaining life is the number of cycles the rollout needs to cross it.

Test data are the published test records (`test_FD002.txt`) truncated where the benchmark hands out
its RUL (`RUL_FD002.txt`). For each engine the window is its last 161 cycles, the anchor is the last
observed cycle, and the rollout uses the measured operating settings for at most CAP cycles.

Model-free references, scored on the same engines:
  local slope   linear trend of the health index over the last sixteen observed cycles
  ramp slope    the average drift rate measured on the training pool
  fleet median  the median published RUL (an oracle floor, not a predictor)
"""
import json
import os
import sys
import time

import numpy as np
import torch

import public_domain
import train

CAP = 160                # the rollout cannot be steered past the end of the record
CHS = [0, 1, 2, 3, 4]
BLK = 10
T0 = 160                 # anchor = last observed cycle of the 161-cycle window
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rul")
os.makedirs(OUT, exist_ok=True)
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]


def health1(s):
    """Ten-cycle trailing block mean of a one-dimensional health sequence."""
    c = np.cumsum(s)
    out = c.copy()
    out[BLK:] = (c[BLK:] - c[:-BLK]) / BLK
    out[:BLK] = c[:BLK] / np.arange(1, BLK + 1)
    return out


def health(Z):
    """Same filter along the cycle axis of a (records, cycles, channels) array."""
    return np.stack([health1(Z[i][:, CHS].mean(-1)) for i in range(Z.shape[0])])


def crossing(H, thr):
    """Cycles until the health index reaches the threshold, capped at CAP."""
    hit = np.argmax(H >= thr) if (H >= thr).any() else None
    return int(hit) if hit is not None else CAP


subset = sys.argv[1] if len(sys.argv) > 1 else "FD002"
seeds = [int(x) for x in sys.argv[2:] if not x.startswith("--")] or [0]
only = os.environ.get("ONLY_MODELS")

dom = public_domain.make(subset=subset, condnorm=True, late=True)
Xtr, Utr = dom.prebuilt()[0]
dd = train.build_data(dom, 0, 0, dom.steps, 0)
mean, std = dd["mean"], dd["std"]

rul = np.loadtxt(os.path.join(public_domain.DATA, "RUL_FD002.txt"), dtype=float).ravel()
recs = public_domain._records(os.path.join(public_domain.DATA, "test_FD002.txt"))
keep = [(r[0][-(dom.steps + 1):], r[1][-(dom.steps + 1):]) for r in recs
        if r[0].shape[0] >= dom.steps + 1]
Xn = np.stack([(k[0] - dom.cmu[public_domain.condition_labels(k[1])])
               / dom.csd[public_domain.condition_labels(k[1])] for k in keep])
Un = np.stack([k[1][:-1] for k in keep])
rul = rul[:Xn.shape[0]]

Htr = health(Xtr)
thr = float(Htr[:, -1].mean())
rate = float((Htr[:, -1] - Htr[:, 0]).mean() / (Htr.shape[1] - 1))
print("engines %d  threshold %.3f  train ramp %.4f per cycle" % (len(rul), thr, rate), flush=True)

Hte = health(Xn)
h_last = Hte[:, -1]
slope = np.array([np.polyfit(np.arange(16), Hte[i, -16:], 1)[0] for i in range(Xn.shape[0])])
ref, preds_ref = {}, {}
preds_ref["local_slope"] = [crossing(np.maximum(h_last[i] + max(slope[i], 1e-4) * np.arange(CAP), 0),
                                     thr) for i in range(Xn.shape[0])]
preds_ref["ramp_slope"] = [crossing(np.maximum(h_last[i] + rate * np.arange(CAP), 0), thr)
                           for i in range(Xn.shape[0])]
preds_ref["fleet_median"] = [int(np.median(rul))] * len(rul)
for name, p in preds_ref.items():
    e = np.array(p, float) - rul
    ref[name] = {"median_abs": round(float(np.median(np.abs(e))), 1),
                 "rmse": round(float(np.sqrt((e ** 2).mean())), 1),
                 "within20": round(float((np.abs(e) <= 20).mean()), 3)}
print("references", json.dumps(ref), flush=True)

for model in MODELS:
    if only and model not in only.split(","):
        continue
    for seed in seeds:
        path = os.path.join(OUT, "D5%s_rul_%s_s%d.json" % (subset, model, seed))
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        m, mdl = train.train("D5", model, epochs=60, seed=seed, n_train=int(Xtr.shape[0]),
                             steps=dom.steps, L=4, Kmax=30, curriculum=True,
                             horizons=(25, 50, 100), dom=dom, return_model=True)
        preds = []
        mt = torch.tensor(mean, dtype=torch.float32, device=train.DEV)
        st = torch.tensor(std, dtype=torch.float32, device=train.DEV)
        with torch.enable_grad():
            for e in range(Xn.shape[0]):
                w0 = Xn[e, T0 - 3:T0 + 1][:, dom.obs_idx]
                w = ((torch.tensor(w0, dtype=torch.float32, device=train.DEV) - mt) / st).unsqueeze(0)
                prev = w[0, -1]
                z = mdl.encode(w)
                seq = [prev.detach().cpu().numpy()]
                uu = torch.tensor(Un[e], dtype=torch.float32, device=train.DEV)
                for k in range(CAP):
                    z = mdl.step(z, uu[k].unsqueeze(0))
                    z = z.detach()
                    if mdl.structure().get("latent", 0) > 0 and train.RESIDUAL_DECODE:
                        prev = prev + mdl.decode(z)[0].detach()
                    else:
                        prev = mdl.decode(z)[0].detach()
                    seq.append(prev.cpu().numpy())
                P = np.stack(seq)
                # the rollout is in the z-score space of train.py; the threshold lives in the
                # per-operating-point space of public_domain, so map the sequence back first
                Pc = P * np.asarray(std)[None, :] + np.asarray(mean)[None, :]
                preds.append(crossing(health1(Pc[:, CHS].mean(-1)), thr))
        preds = np.array(preds, float)
        err = preds - rul
        out = {k: m[k] for k in ("params", "structure", "train_time_s", "final_loss") if k in m}
        out.update({"rul_median_abs": round(float(np.median(np.abs(err))), 1),
                    "rul_rmse": round(float(np.sqrt((err ** 2).mean())), 1),
                    "rul_within20": round(float((np.abs(err) <= 20).mean()), 3),
                    "rul_capped": round(float((preds >= CAP).mean()), 3),
                    "rul_predictions": [int(x) for x in preds],
                    "references": ref, "threshold": round(thr, 3), "n_test": int(len(rul)),
                    "model": model, "seed": seed,
                    "dataset": "NASA C-MAPSS FD002 test records with published RUL"})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("%s rul %s s%d median|e|=%.1f rmse=%.1f within20=%.3f capped=%.2f (%.0fs)"
              % (subset, model, seed, out["rul_median_abs"], out["rul_rmse"],
                 out["rul_within20"], out["rul_capped"], time.time() - t0), flush=True)
print(subset, "rul grid done", flush=True)
