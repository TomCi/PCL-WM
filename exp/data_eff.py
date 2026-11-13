# -*- coding: utf-8 -*-
"""Data-efficiency and out-of-distribution stress test on D1.

In-distribution: train on n_train trajectories, test on the standard test set.
OOD: train on the same n_train low-speed trajectories, test on a high-speed set
(initial speed 240-320 m/s instead of 180-260 m/s), never seen in training.
"""
import json
import numpy as np
import torch

import domains
import models
import train as trainer

DEV = trainer.DEV


def run(model_name, n_train=60, seed=0, epochs=60, ood=False, shift=(240.0, 320.0)):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dom = domains.make("D1")
    d = trainer.build_data(dom, n_train, 24, 400, seed)
    model = models.build(model_name, dom.dy, dom.du, L=4, hidden=128).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w_reg = {"gauge": 0.01, "cert": 0.005, "smooth": 0.01}
    latent = model.structure().get("latent", 0) > 0
    Kmax = 30
    for ep in range(epochs):
        K = min(Kmax, 2 + ep * (Kmax - 2) // max(1, epochs // 3))
        model.train()
        for b in range(25):
            w, y, u = trainer.sample_batch(d, 4, K, 64, rng, n_train, 400)
            z = model.encode(w)
            z0n = z.norm(dim=1).detach()
            loss = 0.0
            prev = w[:, -1]
            for k in range(K):
                z = model.step(z, u[:, k])
                if latent:
                    loss = loss + 0.02 * ((z.norm(dim=1) - z0n).relu() ** 2).mean()
                    pred = prev + model.decode(z)
                else:
                    pred = model.decode(z)
                prev = pred
                loss = loss + (0.9 ** k) * ((pred - y[:, k]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # in-distribution test set
    met = trainer.evaluate(model, d, dom, 4)
    if ood:
        Xoo, Uoo = dom.sample(16, 400, seed=seed + 4, vlo=shift[0], vhi=shift[1])
        rng2 = np.random.default_rng(seed + 4)
        Ooo = dom.obs(Xoo, 0.01, rng2)
        t = lambda a: torch.tensor(np.asarray(a), dtype=trainer.DTYPE, device=DEV)
        d_ood = {"yte": t((Ooo - d["mean"]) / d["std"]),
                 "ute": t((Uoo - 0) / 1.0),
                 "mean": d["mean"], "std": d["std"], "xte": Xoo}
        met_ood = trainer.evaluate(model, d_ood, dom, 4)
        met["nrmse_100_ood"] = met_ood["nrmse_100"]
        met["nrmse_200_ood"] = met_ood["nrmse_200"]
        met["beta_ood"] = met_ood["beta"]
    met.update({"model": model_name, "n_train": n_train, "ood": bool(ood),
                "params": int(sum(p.numel() for p in model.parameters()))})
    return met


if __name__ == "__main__":
    out = []
    for m in ("MLP-MD", "PI-Lat", "PCL-WM", "abl-no-symplectic"):
        r = run(m, n_train=60, ood=True)
        print(json.dumps({k: r[k] for k in ("model", "n_train", "nrmse_25", "nrmse_100",
                                            "nrmse_200", "beta", "nrmse_100_ood",
                                            "nrmse_200_ood", "beta_ood")}, ensure_ascii=False),
              flush=True)
        out.append(r)
    with open("data_eff.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
