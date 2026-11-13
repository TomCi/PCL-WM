# -*- coding: utf-8 -*-
"""Train and evaluate one world model on one simulated domain (v2, partial obs)."""
import argparse
import json
import time

import numpy as np
import torch

import domains
import models

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
RESIDUAL_DECODE = True


def build_data(dom, n_train, n_test, steps, seed, noise=0.01):
    if hasattr(dom, "prebuilt"):
        Xtr, Utr = dom.prebuilt()[0]
        Xte, Ute = dom.prebuilt()[1]
    else:
        Xtr, Utr = dom.sample(n_train, steps, seed=seed)
        Xte, Ute = dom.sample(n_test, steps, seed=seed + 991)
    rng = np.random.default_rng(seed + 7)
    Otr = dom.obs(Xtr, noise, rng)
    Ote = dom.obs(Xte, noise, rng)
    mean = Otr.reshape(-1, dom.dy).mean(0)
    std = Otr.reshape(-1, dom.dy).std(0) + 1e-8
    umean = Utr.reshape(-1, dom.du).mean(0) if dom.du else np.zeros(0)
    ustd = (Utr.reshape(-1, dom.du).std(0) + 1e-8) if dom.du else np.zeros(0)
    t = lambda a: torch.tensor(np.asarray(a), dtype=DTYPE, device=DEV)
    return {
        "ytr": t((Otr - mean) / std), "utr": t((Utr - umean) / ustd),
        "yte": t((Ote - mean) / std), "ute": t((Ute - umean) / ustd),
        "mean": mean, "std": std, "xte": Xte,
    }


def sample_batch(d, L, K, bs, rng, n_ep, T):
    e = rng.integers(0, n_ep, bs)
    t = rng.integers(L - 1, T - K, bs)
    w = torch.stack([d["ytr"][i, j - L + 1:j + 1] for i, j in zip(e, t)])
    y = torch.stack([d["ytr"][i, j + 1:j + K + 1] for i, j in zip(e, t)])
    u = torch.stack([d["utr"][i, j:j + K] for i, j in zip(e, t)])
    return w, y, u


def train(domain, model_name, epochs=120, seed=0, n_train=240, steps=400,
          L=4, batch=64, lr=1.5e-3, Kmax=10, curriculum=True, hidden=128, batches=25,
          return_model=False, horizons=(25, 50, 100, 200), dom=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dom = dom if dom is not None else domains.make(domain)
    d = build_data(dom, n_train, 24, steps, seed)
    model = models.build(model_name, dom.dy, dom.du, L=L, hidden=hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w_reg = {"gauge": 0.01, "cert": 0.005, "smooth": 0.01}
    latent = model.structure().get("latent", 0) > 0
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        K = Kmax if not curriculum else min(Kmax, 2 + ep * (Kmax - 2) // max(1, epochs // 3))
        model.train()
        losses = []
        for b in range(batches):
            w, y, u = sample_batch(d, L, K, batch, rng, n_train, steps)
            z = model.encode(w)
            z0n = z.norm(dim=1).detach()
            rmax = (z0n.mean() * 1.15).clamp(min=1e-3)
            loss = 0.0
            prev = w[:, -1]
            if latent and RESIDUAL_DECODE:
                loss = 0.5 * ((prev + model.decode(z) - prev) ** 2).mean()
            elif latent:
                loss = 0.5 * ((model.decode(z) - prev) ** 2).mean()
            for k in range(K):
                z = model.step(z, u[:, k])
                if latent:
                    growth = (z.norm(dim=1) - z0n).relu() ** 2
                    loss = loss + 0.02 * growth.mean()
                if latent and RESIDUAL_DECODE:
                    pred = prev + model.decode(z)
                else:
                    pred = model.decode(z)
                prev = pred
                loss = loss + (0.9 ** k) * ((pred - y[:, k]) ** 2).mean()
            if b % 3 == 0:
                reg = model.regularizers({"z0": model.encode(w), "u": u[:, 0],
                                          "z1": model.step(model.encode(w), u[:, 0])})
                for k2, v in reg.items():
                    loss = loss + w_reg.get(k2, 0.02) * v
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        hist.append(float(np.mean(losses)))
    metrics = evaluate(model, d, dom, L, horizons=horizons)
    metrics.update({"domain": domain, "model": model_name, "seed": seed,
                    "epochs": epochs, "n_train": n_train, "L": L,
                    "train_time_s": round(time.time() - t0, 1),
                    "final_loss": round(hist[-1], 5),
                    "dy": dom.dy, "dx": dom.dx,
                    "structure": model.structure(),
                    "params": int(sum(p.numel() for p in model.parameters()))})
    if return_model:
        return metrics, model
    return metrics


def evaluate(model, d, dom, L, horizons=(25, 50, 100, 200), t0=50, n_eval=16):
    model.eval()
    yte, ute = d["yte"], d["ute"]
    full = (dom.dy == dom.dx)
    errs = {h: [] for h in horizons}
    drift = {h: [] for h in horizons}
    one_step, lat = [], []
    for e in range(min(n_eval, yte.shape[0])):
        w = yte[e, t0 - L + 1:t0 + 1].unsqueeze(0)
        preds = []
        prev = yte[e, t0]
        with torch.enable_grad():
            z = model.encode(w)
            ts = time.time()
            for k in range(max(horizons)):
                z = model.step(z, ute[e, t0 + k].unsqueeze(0))
                z = z.detach()
                if model.structure().get("latent", 0) > 0 and RESIDUAL_DECODE:
                    prev = prev + model.decode(z)[0].detach()
                    preds.append(prev)
                else:
                    preds.append(model.decode(z)[0].detach())
        lat.append((time.time() - ts) / max(horizons) * 1000)
        P = torch.stack(preds)
        for h in horizons:
            errs[h].append(float(((P[h - 1] - yte[e, t0 + h]) ** 2).mean().sqrt()))
            if full:
                p_np = P[h - 1].cpu().numpy() * d["std"] + d["mean"]
                e_pred = float(np.mean(dom.energy(p_np)))
                e_true = float(np.mean(dom.energy(d["xte"][e, t0 + h])))
                drift[h].append(abs(e_pred - e_true) / (abs(e_true) + 1e-9) * 100.0)
        one_step.append(float(((P[0] - yte[e, t0 + 1]) ** 2).mean().sqrt()))
    out = {}
    for h in horizons:
        out[f"nrmse_{h}"] = round(float(np.mean(errs[h])), 4)
        out[f"drift_{h}"] = round(float(np.mean(drift[h])), 2) if full else None
    hs = np.array(horizons, dtype=float)
    ns = np.array([out[f"nrmse_{h}"] + 1e-12 for h in horizons])
    out["beta"] = round(float(np.polyfit(np.log(hs), np.log(ns), 1)[0]), 3)
    out["one_step"] = round(float(np.mean(one_step)), 4)
    out["latency_ms"] = round(float(np.mean(lat)), 3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="D1")
    ap.add_argument("--model", default="PCL-WM")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--kmax", type=int, default=10)
    a = ap.parse_args()
    m = train(a.domain, a.model, a.epochs, a.seed, a.n_train, a.steps,
              L=a.L, Kmax=a.kmax, curriculum=not a.no_curriculum)
    print(json.dumps(m, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
