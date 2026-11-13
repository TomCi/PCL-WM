# -*- coding: utf-8 -*-
"""Redraw the ablation panel (Fig. 4) as two-seed means with the per-seed range as an error bar."""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.use()
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")
FI = os.path.join(HERE, "figs_v2")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.unicode_minus": False,
})

GROUPS = [
    ("PCL-WM", ["D1_PCL-WM_s0.json", "D1_PCL-WM_s1.json"]),
    ("abl-no-symplectic", ["D1_abl-no-symplectic_s0.json", "D1_abl-no-symplectic_s1.json"]),
    ("abl-no-cert", ["D1_abl-no-cert_s0.json", "D1_abl-no-cert_s1.json"]),
    ("abl-dz64", ["D1_abl-dz64_s0.json", "D1_abl-dz64_s1.json"]),
    ("abl-no-curriculum", ["D1_abl-no-curriculum_s0.json", "D1_abl-no-curriculum_s1.json"]),
    ("abl-free-dissipation", ["D1_abl-free-dissipation_s0.json", "D1_abl-free-dissipation_s1.json"]),
    ("abl-no-gauge", ["D1_abl-no-gauge_s0.json", "D1_abl-no-gauge_s1.json"]),
    ("abl-dz16", ["D1_abl-dz16_s0.json", "D1_abl-dz16_s1.json"]),
]

rows, vals, lo, hi = [], [], [], []
for name, files in GROUPS:
    v = [json.load(open(os.path.join(RES, f), encoding="utf-8"))["nrmse_200"] for f in files]
    rows.append(name)
    vals.append(statistics.mean(v))
    lo.append(statistics.mean(v) - min(v))
    hi.append(max(v) - statistics.mean(v))

vals = np.array(vals)
o = np.argsort(-vals)
rows = [rows[i] for i in o]
vals = vals[o]
lo = np.array(lo)[o]
hi = np.array(hi)[o]

fig, ax = plt.subplots(figsize=(3.9, 2.35))
ys = np.arange(len(rows))
cols = [figstyle.ACCENT if r == "PCL-WM" else figstyle.NEUTRAL for r in rows]
ax.barh(ys, vals, height=0.6, color=cols, lw=0, alpha=0.92)
ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor="0.2", elinewidth=0.7,
            capsize=2.2, capthick=0.7, zorder=3)
for y, v, h in zip(ys, vals, hi):
    ax.text(v + h + 0.03, y, f"{v:.2f}", va="center", ha="left", fontsize=6.4)
ax.set_yticks(ys)
ax.set_yticklabels([figstyle.DISPLAY.get(r, r) for r in rows])
ax.set_xlabel("NRMSE at 200 steps (D1)")
ax.set_xlim(0, max(np.array(vals) + np.array(hi)) * 1.14)
figstyle.clean(ax, grid="x")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FI, "v2_ablation.png"), bbox_inches="tight", dpi=400)
fig.savefig(os.path.join(FI, "v2_ablation.pdf"), bbox_inches="tight")
print("written", os.path.join(FI, "v2_ablation.png"))
