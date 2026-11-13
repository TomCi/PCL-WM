# -*- coding: utf-8 -*-
"""Figure 5: data efficiency and distribution shift, now with two seeds per configuration.

Bars are the mean of the two seeds; the error bars are the range between them.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.use()

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "savefig.dpi": 400, "mathtext.fontset": "dejavuserif",
})

with open(os.path.join(HERE, "data_eff_means.json"), encoding="utf-8") as f:
    rows = json.load(f)

order = ["MLP-MD", "PI-Lat", "PCL-WM", "PCL-WM w/o symplectic"]
SHORT = {"MLP-MD": "MLP-MD", "PI-Lat": "PI-Lat", "PCL-WM": "PCL-WM",
         "PCL-WM w/o symplectic": "PCL-WM\nw/o sympl."}
rows = sorted(rows, key=lambda r: order.index(r["model"]))
labels = [SHORT[r["model"]] for r in rows]
idv = np.array([r["nrmse_200"] for r in rows])
ood = np.array([r["nrmse_200_ood"] for r in rows])
id_err = np.array([[r["nrmse_200"] - min(r["nrmse_200_s0"], r["nrmse_200_s1"]) for r in rows],
                   [max(r["nrmse_200_s0"], r["nrmse_200_s1"]) - r["nrmse_200"] for r in rows]])
ood_err = np.array([[r["nrmse_200_ood"] - min(r["nrmse_200_ood_s0"], r["nrmse_200_ood_s1"]) for r in rows],
                    [max(r["nrmse_200_ood_s0"], r["nrmse_200_ood_s1"]) - r["nrmse_200_ood"] for r in rows]])

fig, ax = plt.subplots(figsize=(3.55, 2.2))
x = np.arange(len(rows))
w = 0.32
ekw = dict(elinewidth=0.7, capsize=2.0, capthick=0.7, ecolor="0.2")
ax.bar(x - w / 2, idv, w, yerr=id_err, color=figstyle.NEUTRAL, lw=0,
       error_kw=ekw, label="in distribution")
ax.bar(x + w / 2, ood, w, yerr=ood_err, color=figstyle.ACCENT, lw=0,
       error_kw=ekw, label="shifted initial states")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("NRMSE at 200 steps")
top = max((idv + id_err[1]).max(), (ood + ood_err[1]).max())
ax.set_ylim(0, top * 1.06)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, handlelength=1.6, columnspacing=1.6)
figstyle.clean(ax, grid="y")
fig.subplots_adjust(top=0.86)
fig.savefig(os.path.join(HERE, "figs_v2", "v2_data_eff.png"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figs_v2", "v2_data_eff.pdf"), bbox_inches="tight")
print("written", os.path.join(HERE, "figs_v2", "v2_data_eff.png"))
