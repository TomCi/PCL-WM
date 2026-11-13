# -*- coding: utf-8 -*-
"""Aggregate the v2 grid (partial observability) into tables and figures."""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.use()

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")
FI = os.path.join(HERE, "figs_v2")
os.makedirs(FI, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 400,
    "mathtext.fontset": "dejavuserif",
})

FAM = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM"]
ABL = ["abl-no-symplectic", "abl-no-gauge", "abl-free-dissipation", "abl-no-cert",
       "abl-no-curriculum", "abl-dz16", "abl-dz64"]
DOMS = ["D1", "D2", "D3", "D9"]
COL = {"MLP-MD": "#4a6fa5", "GRU-LD": "#6b8f71", "NODE": "#8c6d9c",
       "HNN": "#a8845c", "PI-Lat": "#7d8c99", "PCL-WM": "#b03a2e"}


def load():
    rows = []
    for f in sorted(os.listdir(RES)):
        if f.endswith(".json") and not f.startswith("aggregate"):
            with open(os.path.join(RES, f), encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return rows


KEYS = ("nrmse_25", "nrmse_50", "nrmse_100", "nrmse_200", "beta",
        "one_step", "latency_ms", "params", "train_time_s")


def agg(rows):
    g = defaultdict(list)
    for r in rows:
        if "wall_s" in r or "train_time_s" in r:
            g[(r["domain"], r["model"])].append(r)
    out = {}
    for k, v in g.items():
        d = {}
        for name in KEYS:
            vals = [float(x[name]) for x in v if x.get(name) is not None]
            d[name] = round(float(np.mean(vals)), 3) if vals else None
        d["n"] = len(v)
        out[k] = d
    return out


def table_main(A):
    L = ["| Model | D1@25 | D1@200 | D2@200 | D3@200 | D9@200 | $\\beta$(D1) | $\\beta$(D2) | latency (ms) | params |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in FAM:
        def g(d, k, f="{:.2f}"):
            return f.format(A[(d, m)][k]) if (d, m) in A and A[(d, m)][k] is not None else "-"
        L.append(f"| {m} | {g('D1','nrmse_25')} | {g('D1','nrmse_200')} | {g('D2','nrmse_200')} | "
                 f"{g('D3','nrmse_200')} | {g('D9','nrmse_200')} | {g('D1','beta')} | "
                 f"{g('D2','beta')} | {g('D1','latency_ms')} | {g('D1','params','{:.0f}')} |")
    return "\n".join(L)


def table_abl(A):
    L = ["| Configuration | NRMSE@25 | NRMSE@100 | NRMSE@200 | $\\beta$ | one-step | params |",
         "|---|---|---|---|---|---|---|"]
    for m in ["PCL-WM"] + ABL:
        r = A.get(("D1", m))
        if not r:
            L.append(f"| {m} | - | - | - | - | - | - |")
            continue
        L.append(f"| {m} | {r['nrmse_25']:.3f} | {r['nrmse_100']:.3f} | {r['nrmse_200']:.3f} | "
                 f"{r['beta']:.2f} | {r['one_step']:.4f} | {r['params']:.0f} |")
    return "\n".join(L)


def fig_err(A):
    hs = np.array([25, 50, 100, 200])
    style = {"MLP-MD": ("-", "o"), "GRU-LD": ("-", "s"), "NODE": ("--", "D"),
             "HNN": (":", "v"), "PI-Lat": ("-.", "P"), "PCL-WM": ("-", "o")}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    handles = []
    for ax, dom, tag in zip(axes, ["D1", "D2"], ["(a)", "(b)"]):
        for m in FAM:
            if (dom, m) not in A:
                continue
            ls, mk = style[m]
            lw = 1.5 if m == "PCL-WM" else 0.9
            ln, = ax.semilogy(hs, [A[(dom, m)][f"nrmse_{h}"] for h in hs], ls=ls, marker=mk,
                              ms=2.5, lw=lw, color=figstyle.PAL[m], label=m,
                              markerfacecolor=figstyle.PAL[m], markeredgewidth=0)
            if dom == "D1":
                handles.append(ln)
        ax.set_xlabel("Horizon (steps)")
        ax.set_xticks(hs)
        ax.set_xticklabels(["25", "50", "100", "200"])
        figstyle.clean(ax, grid="both")
        figstyle.panel(ax, tag)
    axes[0].set_ylabel("NRMSE (normalised)")
    fig.subplots_adjust(top=0.90, wspace=0.28)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.045), ncol=6, frameon=False,
               handlelength=1.9, columnspacing=1.4)
    fig.savefig(os.path.join(FI, "v2_err_growth.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FI, "v2_err_growth.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_beta(A):
    rows = [m for m in FAM if ("D1", m) in A]
    vals = [A[("D1", m)]["beta"] for m in rows]
    o = np.argsort(vals)
    rows = [rows[i] for i in o]
    vals = [vals[i] for i in o]
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ys = np.arange(len(rows))
    ax.barh(ys, vals, height=0.58, color=[figstyle.PAL[m] for m in rows],
            alpha=0.9, lw=0)
    ax.axvline(1.0, color="0.35", lw=0.7, ls="--", zorder=3)
    ax.annotate("", xy=(1.0, -0.95), xytext=(1.0, len(rows) - 0.42),
                arrowprops=dict(arrowstyle="-", lw=0.7, ls="--", color="0.35"),
                annotation_clip=False)
    ax.text(1.03, -0.72, r"$\beta=1$: error stays linear; $\beta>1$ accelerates",
            fontsize=6.2, color="0.33", ha="left", va="center", clip_on=False)
    for y, v in zip(ys, vals):
        ax.text(v - 0.03, y, f"{v:.2f}", va="center", ha="right", fontsize=6.4,
                color="white", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(rows)
    ax.set_xlabel(r"Error-growth exponent $\beta$ (D1)")
    ax.set_xlim(0, max(vals) * 1.06)
    ax.set_ylim(-1.15, len(rows) - 0.45)
    figstyle.clean(ax, grid="x")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FI, "v2_beta.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FI, "v2_beta.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_abl(A):
    rows = [m for m in ["PCL-WM"] + ABL if ("D1", m) in A and A[("D1", m)]["nrmse_200"]]
    vals = [A[("D1", m)]["nrmse_200"] for m in rows]
    o = np.argsort(vals)
    rows = [rows[i] for i in o]
    vals = [vals[i] for i in o]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ys = np.arange(len(rows))
    ax.barh(ys, vals, height=0.55,
            color=["#b03a2e" if r == "PCL-WM" else "#8b9aa8" for r in rows], lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.replace("abl-", "") for r in rows])
    ax.set_xlabel("NRMSE at 200 steps (D1)")
    ax.grid(True, axis="x", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FI, "v2_ablation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FI, "v2_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load()
    A = agg(rows)
    with open(os.path.join(RES, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in A.items()}, f, ensure_ascii=False, indent=1)
    mpc = {}
    for f in ("mpc_D1.json",):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for r in json.load(fh):
                    mpc[r["model"]] = r["mpc_pos_rmse"]
    md = ["# PCL-WM experiments, v2 (partial observability)", "",
          f"Runs: {len(rows)} configurations; 60 epochs; K curriculum up to 30; "
          f"window L=4; measured channels only.", "",
          "## Table A. Accuracy, error growth and cost", "", table_main(A), "",
          "## Table B. Ablations (D1)", "", table_abl(A), ""]
    if mpc:
        md += ["## Table C. Closed-loop MPC on D1 (position RMSE, normalised)", "",
               "| Model | MPC RMSE |", "|---|---|"]
        for k, v in mpc.items():
            md.append(f"| {k} | {v:.3f} |")
        md.append("")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    fig_err(A)
    fig_beta(A)
    fig_abl(A)
    print("\n".join(md))


if __name__ == "__main__":
    main()
