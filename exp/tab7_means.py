# -*- coding: utf-8 -*-
"""Two-seed energy deviation on the thermal network (D3)."""
import json
import os

MODELS = ["NODE", "MLP-MD", "PI-Lat", "GRU-LD", "PCL-WM", "HNN", "abl-no-symplectic"]

out = []
for m in MODELS:
    vals = []
    for s in (0, 1):
        p = os.path.join("results_v2", "D3_%s_s%d.json" % (m, s))
        vals.append(json.load(open(p, encoding="utf-8"))["drift_200"])
    out.append({"model": m, "s0": vals[0], "s1": vals[1], "mean": (vals[0] + vals[1]) / 2})
    print("%-20s %8.2f  %8.2f  mean %8.2f" % (m, vals[0], vals[1], (vals[0] + vals[1]) / 2))
json.dump(out, open("tab7_means.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved tab7_means.json")
