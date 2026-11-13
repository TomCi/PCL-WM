# -*- coding: utf-8 -*-
"""Closed-loop MPC benchmark with repeated controller runs.

One trained model per configuration (seed 0), then three controller runs over
disjoint blocks of eight evaluation episodes, each with its own sampling seed.
The reported number is the mean over the 24 episode-runs; the spread is the
range of the three run means.
"""
import json
import time

import numpy as np
import torch

import domains
import mpc
import train as trainer

DEV = trainer.DEV
MODELS = ["MLP-MD", "PI-Lat", "PCL-WM w/o symplectic", "NODE", "PCL-WM"]
RUNS = 3
EP_PER_RUN = 8


def control_run(net, d, dom, L, eps, steps=100, H=15, N=128, iters=3,
                n_elite=16, w_u=0.02, ctl_seed=0):
    torch.manual_seed(1000 + ctl_seed)
    yte = d["yte"]
    pos_idx = [0, 1, 2]
    ep_err = []
    for e in eps:
        t0 = 50
        with torch.no_grad():
            w = yte[e, t0 - L + 1:t0 + 1].unsqueeze(0)
            z = net.encode(w)
            prev = w[:, -1]
        mu = torch.zeros(1, H, dom.du, device=DEV).reshape(1, H * dom.du)
        sig = torch.ones_like(mu) * 0.5
        errs = []
        for k in range(steps):
            ref = yte[e, t0 + k:t0 + k + H]
            if ref.shape[0] < H:
                break
            for _ in range(iters):
                cand = mu + sig * torch.randn(N, H * dom.du, device=DEV)
                u_seq = cand.reshape(N, H, dom.du)
                with torch.no_grad():
                    cost = mpc.rollout_batch(net, z, prev, u_seq, pos_idx, ref)
                    cost = cost + w_u * (cand ** 2).mean(dim=1)
                elite = cand[cost.topk(n_elite, largest=False).indices]
                mu = elite.mean(0, keepdim=True)
                sig = elite.std(0, keepdim=True) + 1e-3
            mu = torch.cat([mu[:, dom.du:], mu[:, -dom.du:]], dim=1)
            u0 = mu[:, :dom.du]
            with torch.enable_grad():
                z = net.step(z, u0)
                prev = prev + net.decode(z) if net.structure().get("latent", 0) > 0 else net.decode(z)
            z, prev = z.detach(), prev.detach()
            errs.append(float(((prev[0, pos_idx] - yte[e, t0 + k + 1, pos_idx]) ** 2).mean().sqrt()))
        ep_err.append(float(np.mean(errs)))
    return ep_err


if __name__ == "__main__":
    dom = domains.make("D1")
    d = trainer.build_data(dom, 240, RUNS * EP_PER_RUN, 400, 0)
    out = []
    for name in MODELS:
        t0 = time.time()
        met, net = trainer.train("D1", name, epochs=60, seed=0, Kmax=30, return_model=True)
        net.eval()
        runs, eps = [], []
        for r in range(RUNS):
            block = list(range(r * EP_PER_RUN, (r + 1) * EP_PER_RUN))
            e = control_run(net, d, dom, met["L"], block, ctl_seed=r)
            eps.extend(e)
            runs.append(float(np.mean(e)))
            print("  %s run %d rmse %.4f" % (name, r, runs[-1]), flush=True)
        rec = {"model": name, "domain": "D1", "runs": runs,
               "episodes": eps,
               "mean": float(np.mean(eps)), "min": float(min(eps)), "max": float(max(eps)),
               "wall_s": round(time.time() - t0, 1)}
        print(json.dumps({k: rec[k] for k in ("model", "mean", "min", "max", "wall_s")}), flush=True)
        out.append(rec)
    with open("mpc_rep_D1.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("mpc repeats done", flush=True)
