# -*- coding: utf-8 -*-
"""Two-seed means for Table 4 and the per-seed spread that the text now reports."""
import glob
import json
import os
import statistics

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
GROUPS = {
    "PCL-WM (full)": ["D1_PCL-WM_s%d.json" % s for s in (0, 1)],
    "w/o symplectic step": ["D1_abl-no-symplectic_s%d.json" % s for s in (0, 1)],
    "w/o gauge penalty": ["D1_abl-no-gauge_s%d.json" % s for s in (0, 1)],
    "free dissipation": ["D1_abl-free-dissipation_s%d.json" % s for s in (0, 1)],
    "w/o certificate": ["D1_abl-no-cert_s%d.json" % s for s in (0, 1)],
    "w/o curriculum": ["D1_abl-no-curriculum_s%d.json" % s for s in (0, 1)],
    "latent width 16": ["D1_abl-dz16_s%d.json" % s for s in (0, 1)],
    "latent width 64": ["D1_abl-dz64_s%d.json" % s for s in (0, 1)],
}

for name, files in GROUPS.items():
    rs = [json.load(open(os.path.join(RES, f), encoding="utf-8")) for f in files]
    m25 = statistics.mean(r["nrmse_25"] for r in rs)
    m50 = statistics.mean(r["nrmse_50"] for r in rs)
    m200 = statistics.mean(r["nrmse_200"] for r in rs)
    mb = statistics.mean(r["beta"] for r in rs)
    mo = statistics.mean(r["one_step"] for r in rs)
    p200 = [r["nrmse_200"] for r in rs]
    print('  ("%s", "%.3f", "%.3f", "%.3f", "%.2f", "%.4f", "%s"),  spread %.3f-%.3f'
          % (name, m25, m50, m200, mb, mo, "{:,}".format(rs[0]["params"]), min(p200), max(p200)))
