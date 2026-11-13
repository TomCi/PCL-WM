# -*- coding: utf-8 -*-
"""Two-seed means for the RUL grid, plus the three references."""
import glob
import io
import json
import os

import numpy as np

D = r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\results_rul"
MODELS = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
out = io.open(os.path.join(os.path.dirname(D), "rul_summary.txt"), "w", encoding="utf-8")
rows = []
for m in MODELS:
    fs = sorted(glob.glob(os.path.join(D, "D5FD002_rul_%s_s*.json" % m)))
    if not fs:
        continue
    ds = [json.load(io.open(f, encoding="utf-8")) for f in fs]
    med = np.mean([d["rul_median_abs"] for d in ds])
    rm = np.mean([d["rul_rmse"] for d in ds])
    w20 = np.mean([d["rul_within20"] for d in ds])
    cap = np.mean([d["rul_capped"] for d in ds])
    rows.append((m, med, rm, w20, cap, len(ds)))
    out.write("%-20s seeds=%d  median|e|=%6.1f  rmse=%6.1f  within20=%.3f  capped=%.2f\n"
              % (m, len(ds), med, rm, w20, cap))
ref = json.load(io.open(sorted(glob.glob(os.path.join(D, "*.json")))[0], encoding="utf-8"))["references"]
out.write("\nreferences (87 engines, published RUL):\n")
for k, v in ref.items():
    out.write("  %-12s median|e|=%5.1f  rmse=%5.1f  within20=%.3f\n"
              % (k, v["median_abs"], v["rmse"], v["within20"]))
out.close()
print(io.open(os.path.join(os.path.dirname(D), "rul_summary.txt"), encoding="utf-8").read())
