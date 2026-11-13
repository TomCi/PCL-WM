# -*- coding: utf-8 -*-
"""Measure the certified horizon of Eq. (16) against the observed crossing horizon.

L is the Lipschitz constant of the learned latent step map, estimated by central
finite differences taken on the latent states the open-loop rollout actually visits,
h is the step size the model learned, e0 is the first-step position error in
normalised units and the tolerance is eps = 0.1, the scale used throughout the tables.
"""
import json
import math
import time

import numpy as np
import torch

import domains
import train as trainer

DEV = trainer.DEV
EPS_TOL = 0.1
MODELS = [("PCL-WM", "PCL-WM"), ("abl-no-cert", "PCL-WM w/o certificate")]
SEEDS = [0, 1]


def open_loop(net, d, dom, L, t0=50, steps=200):
    """Roll every test episode forward with the recorded inputs; return error curves."""
    pos_idx = [0, 1, 2]
    yte = d["yte"]
    curves, latents = [], []
    for e in range(yte.shape[0]):
        with torch.no_grad():
            w = yte[e, t0 - L + 1:t0 + 1].unsqueeze(0)
            z = net.encode(w)
            prev = w[:, -1]
        errs, store = [], []
        for k in range(steps):
            u = d["ute"][e, t0 + k:t0 + k + 1]
            store.append((z.detach().clone(), u.clone()))
            with torch.enable_grad():
                z = net.step(z, u)
                prev = prev + net.decode(z)
            z, prev = z.detach(), prev.detach()
            errs.append(float(((prev[0, pos_idx] - yte[e, t0 + k + 1, pos_idx]) ** 2).mean().sqrt()))
        curves.append(errs)
        latents.append(store)
    return np.array(curves), latents


def lipschitz(net, latents, n_sample=400, eps=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    flat = [item for ep in latents for item in ep]
    idx = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
    vals = []
    for i in idx:
        z, u = flat[i]
        v = torch.randn_like(z)
        v = v / v.norm()
        with torch.enable_grad():
            zp = net.step(z + eps * v, u)
            zm = net.step(z - eps * v, u)
        zp, zm = zp.detach(), zm.detach()
        vals.append(float((zp - zm).norm()) / (2 * eps))
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 95))


if __name__ == "__main__":
    dom = domains.make("D1")
    out = []
    for name, label in MODELS:
        for seed in SEEDS:
            t0 = time.time()
            met, net = trainer.train("D1", name, epochs=60, seed=seed, Kmax=30, return_model=True)
            net.eval()
            d = trainer.build_data(dom, 240, 16, 400, seed)
            curves, latents = open_loop(net, d, dom, met["L"])
            mean_curve = curves.mean(axis=0)
            e0 = float(mean_curve[0])
            h = float(net.h.detach())
            Lm, Lp = lipschitz(net, latents, seed=seed)
            rho = 1.0 + Lm * h
            n_cert = math.log(1.0 - EPS_TOL * (1.0 - rho) / e0) / math.log(rho) if rho > 1 else float("nan")
            cross = np.where(mean_curve > EPS_TOL)[0]
            n_obs = int(cross[0] + 1) if len(cross) else None
            rec = {"model": label, "registry": name, "seed": seed, "h": round(h, 4),
                   "L_mean": round(Lm, 4), "L_p95": round(Lp, 4), "rho": round(rho, 4),
                   "e0": round(e0, 5), "n_cert": round(n_cert, 2),
                   "n_obs": n_obs, "ratio": round(n_cert / n_obs, 3) if n_obs else None,
                   "err_at_50": round(float(mean_curve[49]), 4),
                   "err_at_100": round(float(mean_curve[99]), 4),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec)
            print("%-24s seed %d  h %.3f  L %.2f  rho %.3f  e0 %.4f  N_cert %.1f  N_obs %s  ratio %s"
                  % (label, seed, h, Lm, rho, e0, n_cert, n_obs, rec["ratio"]), flush=True)
            with open("cert_probe_D1.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
    print("cert probe done", flush=True)
