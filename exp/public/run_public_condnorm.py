# -*- coding: utf-8 -*-
"""FD002 with per-operating-point normalisation (the obvious repair, run to check it).

Protocol is identical to run_public_subsets.py; the only change is that each sensor channel is
centred and scaled per operating point on the training records (public_domain.condnorm=True),
so the step function of the operating point is removed before the window is formed.

usage: python run_public_condnorm.py FD002 0 1
"""
import json
import os
import sys
import time

import public_domain
import train

HORIZONS = (25, 50, 100)
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public3")
os.makedirs(OUT, exist_ok=True)

subset = sys.argv[1]
seeds = [int(x) for x in sys.argv[2:] if not x.startswith("--")] or [0]
only = os.environ.get("ONLY_MODELS")

for model in MODELS:
    if only and model not in only.split(","):
        continue
    for seed in seeds:
        path = os.path.join(OUT, "D5%s_cn_%s_s%d.json" % (subset, model, seed))
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        dom = public_domain.make(subset=subset, condnorm=True)
        n_tr = int(dom.prebuilt()[0][0].shape[0])
        m = train.train("D5", model, epochs=60, seed=seed, n_train=n_tr, steps=dom.steps,
                        L=4, Kmax=30, curriculum=True, horizons=HORIZONS, dom=dom)
        m["n_train"] = n_tr
        m["n_test"] = int(dom.prebuilt()[1][0].shape[0])
        m["dataset"] = "NASA C-MAPSS %s, per-condition normalised (public)" % subset
        m["subset"] = subset
        m["condnorm"] = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print("%s cn %s s%d @25=%s @50=%s @100=%s beta=%s (%.0fs)"
              % (subset, model, seed, m["nrmse_25"], m["nrmse_50"], m["nrmse_100"], m["beta"],
                 time.time() - t0), flush=True)
print(subset, "condnorm grid done", flush=True)
