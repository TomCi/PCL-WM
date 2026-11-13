# -*- coding: utf-8 -*-
"""Run the full baseline grid on the public C-MAPSS fleet (domain D5).

Protocol is identical to the simulated domains: 60 epochs, K up to 30, four-step history window,
one-step-normalised NRMSE, horizons 25/50/100 (the record length limits the horizon to 160 cycles).
"""
import json
import os
import sys
import time

import public_domain
import train

HORIZONS = (25, 50, 100)
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public")
os.makedirs(OUT, exist_ok=True)

seeds = [int(x) for x in sys.argv[1:]] or [0]
only = os.environ.get("ONLY_MODELS")

for model in MODELS:
    if only and model not in only.split(","):
        continue
    for seed in seeds:
        path = os.path.join(OUT, f"D5_{model}_s{seed}.json")
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        dom = public_domain.make()
        n_tr = int(dom.prebuilt()[0][0].shape[0])
        m = train.train("D5", model, epochs=60, seed=seed, n_train=n_tr, steps=dom.steps,
                        L=4, Kmax=30, curriculum=True, horizons=HORIZONS, dom=dom)
        m["n_train"] = int(dom.prebuilt()[0][0].shape[0])
        m["n_test"] = int(dom.prebuilt()[1][0].shape[0])
        m["dataset"] = "NASA C-MAPSS FD001 (public)"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print(f"{model} s{seed} @25={m['nrmse_25']} @50={m['nrmse_50']} @100={m['nrmse_100']} "
              f"beta={m['beta']} one={m['one_step']} ({time.time() - t0:.0f}s)", flush=True)
print("public grid done", flush=True)
