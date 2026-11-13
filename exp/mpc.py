# -*- coding: utf-8 -*-
"""Closed-loop MPC benchmark on D1 / D2.

The learned model is used inside a CEM random-shooting MPC, the plant is the
exact simulator, so the reported error is a true closed-loop error that includes
model bias.  Metric: RMSE of measured position over the episode (normalised).
"""
import json
import sys

import numpy as np
import torch

import domains
import train as trainer

DEV = trainer.DEV


def rollout_batch(net, z0, prev0, u_seq, pos_idx, ref):
    """Batched H-step rollout; returns mean squared position tracking cost per sample."""
    N, H, _ = u_seq.shape
    z = z0.repeat((N,) + (1,) * (z0.dim() - 1))
    prev = prev0.repeat(N, 1)
    latent = net.structure().get("latent", 0) > 0
    cost = None
    for j in range(H):
        with torch.enable_grad():
            z = net.step(z, u_seq[:, j])
            prev = prev + net.decode(z) if latent else net.decode(z)
        z, prev = z.detach(), prev.detach()
        term = ((prev[:, pos_idx] - ref[j, pos_idx]) ** 2).mean(dim=1)
        cost = term if cost is None else cost + term
    return cost / H


def run(domain, model_name, steps=100, H=15, N=128, iters=3, n_elite=16,
        w_u=0.02, epochs=60, seed=0, episodes=4):
    dom = domains.make(domain)
    d = trainer.build_data(dom, 240, 8, 400, seed)
    metrics, net = trainer.train(domain, model_name, epochs=epochs, seed=seed,
                                 Kmax=30, return_model=True)
    net.eval()
    L = metrics["L"]
    yte = d["yte"]
    pos_idx = [0, 1, 2]
    ep_err = []
    for e in range(episodes):
        t0 = 50
        with torch.no_grad():
            w = yte[e, t0 - L + 1:t0 + 1].unsqueeze(0)
            z = net.encode(w)
            prev = w[:, -1]
        u_last = torch.zeros(1, dom.du, device=DEV)
        mu = u_last.repeat(1, H)
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
                    cost = rollout_batch(net, z, prev, u_seq, pos_idx, ref)
                    cost = cost + w_u * (cand ** 2).mean(dim=1)
                elite = cand[cost.topk(n_elite, largest=False).indices]
                mu = elite.mean(0, keepdim=True)
                sig = elite.std(0, keepdim=True) + 1e-3
            # warm start the next solve with the previous solution, shifted by one step
            mu = torch.cat([mu[:, dom.du:], mu[:, -dom.du:]], dim=1)
            u0 = mu[:, :dom.du]
            with torch.enable_grad():
                z = net.step(z, u0)
                prev = prev + net.decode(z) if net.structure().get("latent", 0) > 0 else net.decode(z)
            z, prev = z.detach(), prev.detach()
            errs.append(float(((prev[0, pos_idx] - yte[e, t0 + k + 1, pos_idx]) ** 2).mean().sqrt()))
        ep_err.append(float(np.mean(errs)))
    return {"domain": domain, "model": model_name,
            "mpc_pos_rmse": round(float(np.mean(ep_err)), 4),
            "train": {k: metrics[k] for k in ("nrmse_25", "nrmse_100", "nrmse_200", "beta", "params")}}


if __name__ == "__main__":
    dom = sys.argv[1] if len(sys.argv) > 1 else "D1"
    mods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["MLP-MD", "PI-Lat", "PCL-WM"]
    out = []
    for m in mods:
        r = run(dom, m)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        out.append(r)
    with open(f"mpc_{dom}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
