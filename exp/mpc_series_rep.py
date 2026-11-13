# -*- coding: utf-8 -*-
"""Closed-loop tracking curves under the repeated-run protocol, for Fig. 6.

Same protocol as the table: one trained model per configuration, three controller
runs over disjoint blocks of eight episodes, each with its own sampling seed.
Per-step error is stored for every episode so the figure can show both the
average trajectory and the episode-level mean with its run-to-run range.
"""
import json
import time

import numpy as np
import torch

import domains
import mpc
import train as trainer

DEV = trainer.DEV
MODELS = [("MLP-MD", "MLP-MD"), ("PI-Lat", "PI-Lat"), ("abl-no-symplectic", "PCL-WM w/o symplectic"),
          ("NODE", "NODE"), ("PCL-WM", "PCL-WM")]
RUNS = 3
EP_PER_RUN = 8


def control_curve(net, d, dom, L, eps, steps=100, H=15, N=128, iters=3,
                  n_elite=16, w_u=0.02, ctl_seed=0):
    """Return per-episode error curves and the episode RMSE."""
    torch.manual_seed(1000 + ctl_seed)
    yte = d["yte"]
    pos_idx = [0, 1, 2]
    curves, rmses = [], []
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
        curves.append(errs)
        rmses.append(float(np.mean(errs)))
    return curves, rmses


if __name__ == "__main__":
    dom = domains.make("D1")
    d = trainer.build_data(dom, 240, RUNS * EP_PER_RUN, 400, 0)
    out = []
    for name, label in MODELS:
        t0 = time.time()
        met, net = trainer.train("D1", name, epochs=60, seed=0, Kmax=30, return_model=True)
        net.eval()
        all_curves, all_rmse, run_means, run_curves = [], [], [], []
        for r in range(RUNS):
            block = list(range(r * EP_PER_RUN, (r + 1) * EP_PER_RUN))
            c, rm = control_curve(net, d, dom, met["L"], block, ctl_seed=r)
            all_curves.extend(c)
            all_rmse.extend(rm)
            run_means.append(float(np.mean(rm)))
            run_curves.append(np.mean(np.array(c), axis=0).tolist())
            print("  %s run %d rmse %.4f (%d steps)" % (label, r, run_means[-1], len(c[0])), flush=True)
        rec = {"model": label, "registry": name, "domain": "D1",
               "series": all_curves, "run_curves": run_curves, "run_means": run_means,
               "mpc_pos_rmse": float(np.mean(all_rmse)), "wall_s": round(time.time() - t0, 1)}
        out.append(rec)
        with open("mpc_series_rep_D1.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("%s mean %.4f wall %.0fs" % (label, rec["mpc_pos_rmse"], rec["wall_s"]), flush=True)
    print("mpc series repeats done", flush=True)
