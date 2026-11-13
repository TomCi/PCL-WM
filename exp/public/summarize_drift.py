# -*- coding: utf-8 -*-
"""Tabulate the fault-progression grid: two seeds, mean drift error at each horizon."""
import glob
import json
import os

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]


def load(subset, prefix="late"):
    rows = {}
    for p in glob.glob(os.path.join(EXP, "results_public4", "D5%s_%s_*_s*.json" % (subset, prefix))):
        m = json.load(open(p, encoding="utf-8"))
        rows.setdefault(m["model"], []).append(m)
    return rows


for subset in ("FD002", "FD004"):
    rows = load(subset)
    if not rows:
        continue
    ref = list(rows.values())[0][0]["references"]
    print("\n=== %s  (references: persistence 1.00; ramp %s/%s/%s)"
          % (subset, ref["ramp_25"], ref["ramp_50"], ref["ramp_100"]))
    print("%-22s %8s %8s %8s %8s %8s %10s" % ("model", "@25", "@50", "@100", "beta",
                                              "nrmse100", "params"))
    for m in MODELS:
        if m not in rows:
            continue
        r = rows[m]
        d = [np.mean([x["drift_%d" % h] for x in r]) for h in (25, 50, 100)]
        b = np.mean([x["drift_beta"] for x in r])
        n100 = np.mean([x["nrmse_100"] for x in r])
        print("%-22s %8.3f %8.3f %8.3f %8.2f %8.3f %10d"
              % (m, d[0], d[1], d[2], b, n100, r[0]["params"]))
