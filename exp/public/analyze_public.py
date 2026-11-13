# -*- coding: utf-8 -*-
"""Collect the public-domain (C-MAPSS FD001) results and add the persistence baseline."""
import glob
import json
import os

import numpy as np
import torch

import public_domain
import train

HOR = (25, 50, 100)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_public")

# persistence baseline on the same held-out windows
dom = public_domain.make()
d = train.build_data(dom, 68, 16, dom.steps, 0)
yte = d["yte"].cpu().numpy()
base = {}
for h in HOR:
    base[h] = float(np.mean(np.sqrt(((yte[:, 50] - yte[:, 50 + h]) ** 2).mean(1))))
base["one"] = float(np.mean(np.sqrt(((yte[:, 49] - yte[:, 50]) ** 2).mean(1))))
hs = np.array(HOR, dtype=float)
ns = np.array([base[h] for h in HOR])
base["beta"] = float(np.polyfit(np.log(hs), np.log(ns), 1)[0])

rows = {}
for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
    r = json.load(open(p, encoding="utf-8"))
    if "model" not in r:
        continue
    rows.setdefault(r["model"], []).append(r)

order = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM", "abl-no-symplectic"]
print("persistence: one=%.4f @25=%.4f @50=%.4f @100=%.4f beta=%.3f" % (
    base["one"], base[25], base[50], base[100], base["beta"]))
print("%-20s %8s %8s %8s %8s %7s %7s %7s %7s" % ("model", "@25", "@50", "@100", "one", "beta", "params", "lat_ms", "t_s"))
for m in order:
    if m not in rows:
        continue
    rs = rows[m]
    def avg(k):
        return float(np.mean([x[k] for x in rs]))
    print("%-20s %8.3f %8.3f %8.3f %8.3f %7.3f %7d %7.2f %7.0f" % (
        m, avg("nrmse_25"), avg("nrmse_50"), avg("nrmse_100"), avg("one_step"),
        avg("beta"), rs[0]["params"], avg("latency_ms"), avg("train_time_s")))

with open(os.path.join(OUT, "persistence.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): round(v, 4) for k, v in base.items()}, f, indent=1)
