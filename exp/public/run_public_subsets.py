# -*- coding: utf-8 -*-
"""Run the baseline grid on a C-MAPSS subset beyond FD001 (D5b, multi-condition).

Protocol is identical to run_public.py: 60 epochs, K up to 30, four-step history window,
horizons 25/50/100, two seeds per configuration. FD002 and FD004 add the three operating
settings as a three-channel exogenous input.

usage: python run_public_subsets.py FD002 0 1  [FD002 2 3 --models PCL-WM,abl-no-symplectic]
"""
import json
import os
import sys
import time

import public_domain
import train

HORIZONS = (25, 50, 100)
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public2")
os.makedirs(OUT, exist_ok=True)

subset = sys.argv[1]
seeds = [int(x) for x in sys.argv[2:] if not x.startswith("--")] or [0]
only = os.environ.get("ONLY_MODELS")

for model in MODELS:
    if only and model not in only.split(","):
        continue
    for seed in seeds:
        path = os.path.join(OUT, f"D5{subset}_{model}_s{seed}.json")
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        dom = public_domain.make(subset=subset)
        n_tr = int(dom.prebuilt()[0][0].shape[0])
        m = train.train("D5", model, epochs=60, seed=seed, n_train=n_tr, steps=dom.steps,
                        L=4, Kmax=30, curriculum=True, horizons=HORIZONS, dom=dom)
        m["n_train"] = n_tr
        m["n_test"] = int(dom.prebuilt()[1][0].shape[0])
        m["dataset"] = "NASA C-MAPSS %s (public)" % subset
        m["subset"] = subset
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print(f"{subset} {model} s{seed} @25={m['nrmse_25']} @50={m['nrmse_50']} "
              f"@100={m['nrmse_100']} beta={m['beta']} ({time.time() - t0:.0f}s)", flush=True)
print(subset, "grid done", flush=True)
