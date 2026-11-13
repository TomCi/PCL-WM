# -*- coding: utf-8 -*-
"""Second schedule: the latent models get 120 epochs on the public fleet.

The latent family carries ten times the parameters of the direct predictors and on measured (noise-
dominated) records it needs more optimisation steps to settle; this run checks convergence rather than
re-tuning anything.
"""
import json
import os
import sys
import time

import public_domain
import train

HORIZONS = (25, 50, 100)
MODELS = ["GRU-LD", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public120")
os.makedirs(OUT, exist_ok=True)
seeds = [int(x) for x in sys.argv[1:]] or [0]

for model in MODELS:
    for seed in seeds:
        path = os.path.join(OUT, f"D5_{model}_s{seed}.json")
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        dom = public_domain.make()
        n_tr = int(dom.prebuilt()[0][0].shape[0])
        m = train.train("D5", model, epochs=120, seed=seed, n_train=n_tr, steps=dom.steps,
                        L=4, Kmax=30, curriculum=True, horizons=HORIZONS, dom=dom)
        m["n_train"] = n_tr
        m["n_test"] = int(dom.prebuilt()[1][0].shape[0])
        m["dataset"] = "NASA C-MAPSS FD001 (public)"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print(f"{model} s{seed} @25={m['nrmse_25']} @50={m['nrmse_50']} @100={m['nrmse_100']} "
              f"beta={m['beta']} one={m['one_step']} ({time.time() - t0:.0f}s)", flush=True)
print("public120 done", flush=True)
