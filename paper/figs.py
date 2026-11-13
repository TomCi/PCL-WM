# -*- coding: utf-8 -*-
"""Build the manuscript figures from measured data (journal rcParams)."""
import json
import os
import shutil

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "exp")
FIG = os.path.join(HERE, "figs")
sys.path.insert(0, EXP)
import figstyle
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "savefig.dpi": 400,
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": "white",
})

ACC = "#b03a2e"
GREY = "#8b9aa8"


def save_fig(fig, name, pad=0.05):
    """Write the manuscript formats: PNG for the DOCX, vector PDF and LZW TIFF for submission."""
    base = os.path.join(FIG, name)
    fig.savefig(base + ".png", bbox_inches="tight", pad_inches=pad)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(base + ".tif", bbox_inches="tight", pad_inches=pad, dpi=400,
                pil_kwargs={"compression": "tiff_lzw"})


def fig1():
    """(a) architecture block diagram, (b) map of the evaluation."""
    figstyle.use()
    fig = plt.figure(figsize=(6.10, 4.55))
    ax = fig.add_axes([0.005, 0.500, 0.99, 0.495])
    ax.set_xlim(0, 9.7)
    ax.set_ylim(0, 3.42)
    ax.axis("off")

    SHARED = dict(fc="#f2f5f8", ec="#7c8b99")
    CANON = dict(fc="#e8eef7", ec="#3f6699")
    RESID = dict(fc="#f8eceb", ec=figstyle.ACCENT)

    def box(x, y, w, h, text, style):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    linewidth=0.7, edgecolor=style["ec"],
                                    facecolor=style["fc"]))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
                linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.7, color="0.3",
                                     shrinkA=0, shrinkB=0))

    box(0.05, 1.34, 1.48, 0.92, "observation\nwindow\n$o_{t-L+1:t}$", SHARED)
    box(1.78, 1.34, 1.22, 0.92, "encoder\n$E_\\phi$", SHARED)
    box(3.26, 2.26, 1.62, 0.84, "canonical pair\n$(q,p)$, $m$ dof", CANON)
    box(3.26, 0.50, 1.62, 0.84, "residual block\n$\\xi$, $r$ channels", RESID)
    box(5.18, 2.26, 1.98, 0.84, "Hamiltonian $H_\\theta$\n+ Verlet map", CANON)
    box(5.18, 0.50, 1.98, 0.84, "port-Hamiltonian\n$(J_r-R)\\nabla H_r+Gu$", RESID)
    box(7.60, 1.34, 1.28, 0.92, "residual\ndecoder\n$D_\\psi$", SHARED)

    arrow(1.55, 1.80, 1.76, 1.80)
    arrow(3.02, 1.80, 3.24, 2.66)
    arrow(3.02, 1.80, 3.24, 0.94)
    arrow(4.90, 2.68, 5.16, 2.68)
    arrow(4.90, 0.92, 5.16, 0.92)
    arrow(7.18, 2.68, 7.58, 2.16)
    arrow(7.18, 0.92, 7.58, 1.44)

    ax.text(4.07, 3.38, "gauge: $\\bar{q}=0$", fontsize=7.0, ha="center", color="0.25")
    ax.text(6.17, 3.38, "$R=LL^{\\top}\\succeq 0$", fontsize=7.0, ha="center", color="0.25")
    ax.text(8.24, 0.92, "$\\hat{o}_{k+1}=\\hat{o}_k+D_\\psi(z_{k+1})$", fontsize=7.0,
            ha="center")

    key = [(0.05, SHARED, "shared"), (1.55, CANON, "canonical branch"),
           (3.62, RESID, "residual branch")]
    for x, style, text in key:
        ax.add_patch(FancyBboxPatch((x, 0.04), 0.20, 0.12,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    linewidth=0.7, edgecolor=style["ec"],
                                    facecolor=style["fc"]))
        ax.text(x + 0.28, 0.10, text, fontsize=6.9, va="center", color="0.3")
    ax.text(-0.02, 3.50, "a", fontsize=8.5, fontweight="bold", ha="left", va="top")

    # ---- (b) the map of the evaluation: one row per question the paper asks -----------------
    axb = fig.add_axes([0.005, 0.005, 0.99, 0.465])
    axb.set_xlim(0, 9.7)
    axb.set_ylim(0, 6.15)
    axb.axis("off")
    axb.text(-0.02, 6.30, "b", fontsize=8.5, fontweight="bold", ha="left", va="top")

    ROWS = [
        ("short horizon,\n25-100 steps",
         "parity with the direct predictors: 0.17 against 0.17 at 100 steps",
         "Table 2", "plain"),
        ("long horizon,\n200 steps",
         "structure moves the error exponent: $\\beta$ 1.33 against 0.96 without the Verlet map",
         "Tables 2, 3", "accent"),
        ("partial\nobservability",
         "the prior is what survives six channels standing in for twenty-one: 3.27 against 0.22",
         "Table 2, D9", "accent"),
        ("closed-loop\ncontrol",
         "all five inside 0.065-0.074 under MPC; the prior is not what makes them stable",
         "Table 6", "plain"),
        ("invariants",
         "energy error 1.3-1.5 per cent from known equations, 40-3276 per cent learned",
         "Table 7", "plain"),
        ("public fleet,\nFD001",
         "boundedness rather than accuracy: 1.14 against 14.07 without the Verlet map",
         "Table 10", "accent"),
    ]
    TOP = 5.98
    H = 0.97
    for i, (lab, sent, anch, kind) in enumerate(ROWS):
        y = TOP - (i + 1) * H
        if i % 2 == 0:
            axb.add_patch(plt.Rectangle((0, y), 9.7, H, facecolor="#f4f6f8",
                                        edgecolor="none", zorder=0))
        col = figstyle.ACCENT if kind == "accent" else "0.60"
        axb.add_patch(plt.Rectangle((0.06, y + H / 2 - 0.05), 0.05, 0.10,
                                    facecolor=col, edgecolor="none", zorder=2))
        axb.text(0.24, y + H / 2, lab, fontsize=7.1, va="center", ha="left",
                 linespacing=1.3, color="0.16", zorder=2)
        axb.text(2.45, y + H / 2, sent, fontsize=6.9, va="center", ha="left",
                 color="0.22", zorder=2)
        axb.text(9.66, y + H / 2, anch, fontsize=6.6, va="center", ha="right",
                 color="0.42", zorder=2)
    axb.plot([0, 9.7], [TOP, TOP], color="0.30", linewidth=0.8, zorder=2)
    axb.plot([0, 9.7], [TOP - 6 * H, TOP - 6 * H], color="0.30", linewidth=0.8, zorder=2)
    axb.text(0.06, TOP - 6 * H - 0.22,
             "the accent marks the rows where the prior, and not the model capacity, is what is measured",
             fontsize=6.4, color="0.45", va="center")

    save_fig(fig, "fig1_architecture", pad=0.03)
    plt.close(fig)


def raster_fig(src_name, dst_name):
    """Copy a raster produced by the experiment scripts and add TIFF and PDF wrappers."""
    from PIL import Image
    src = os.path.join(EXP, "figs_v2", src_name)
    dst = os.path.join(FIG, dst_name)
    shutil.copyfile(src, dst + ".png")
    im = Image.open(src)
    dpi = im.info.get("dpi", (400, 400))
    im.convert("RGB").save(dst + ".tif", compression="tiff_lzw", dpi=dpi)
    pdf_src = src[:-4] + ".pdf"
    if os.path.exists(pdf_src):
        shutil.copyfile(pdf_src, dst + ".pdf")
    else:
        im.convert("RGB").save(dst + ".pdf", resolution=float(dpi[0]))


def fig2():
    raster_fig("v2_err_growth.png", "fig2_error_growth")


def fig3():
    raster_fig("v2_beta.png", "fig3_beta")


def fig4():
    raster_fig("v2_ablation.png", "fig4_ablation")


def fig5():
    raster_fig("v2_data_eff.png", "fig5_data_efficiency")


def fig6():
    """Closed loop: error against the step, and the per-run mean with its spread."""
    figstyle.use()
    with open(os.path.join(EXP, "mpc_series_rep_D1.json"), encoding="utf-8") as f:
        rows = json.load(f)
    rows.sort(key=lambda r: r["mpc_pos_rmse"])
    labels = [r["model"] for r in rows]
    mpc = [r["mpc_pos_rmse"] for r in rows]
    lo = [min(r["run_means"]) for r in rows]
    hi = [max(r["run_means"]) for r in rows]
    col = {m: figstyle.PAL.get(m, figstyle.NEUTRAL) for m in labels}

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
    ax = axes[0]
    handles = []
    for r in rows:
        s = np.mean(np.array(r["series"]), axis=0)
        rc = np.array(r["run_curves"])
        x = np.arange(1, len(s) + 1)
        c = col[r["model"]]
        ax.fill_between(x, rc.min(axis=0), rc.max(axis=0), color=c, alpha=0.12, lw=0)
        ln, = ax.semilogy(x, s, lw=1.6 if r["model"] == "PCL-WM" else 0.9, color=c,
                          label=figstyle.DISPLAY.get(r["model"], r["model"]))
        handles.append(ln)
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel("Position error (normalised)")
    figstyle.clean(ax, grid="both")
    figstyle.panel(ax, "(a)")

    ax = axes[1]
    y = np.arange(len(rows))
    ax.barh(y, mpc, height=0.62, color=[col[m] for m in labels], lw=0, alpha=0.92)
    ax.errorbar(mpc, y, xerr=[np.array(mpc) - np.array(lo), np.array(hi) - np.array(mpc)],
                fmt="none", ecolor="0.2", elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=3)
    for yy, v, h in zip(y, mpc, hi):
        ax.text(v + (h - v) + 0.0016, yy, f"{v:.3f}", va="center", ha="left", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels([figstyle.DISPLAY.get(m, m) for m in labels], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("Closed-loop position RMSE")
    ax.set_xlim(0, max(hi) * 1.20)
    figstyle.clean(ax, grid="x")
    figstyle.panel(ax, "(b)")

    fig.subplots_adjust(top=0.85, wspace=0.30)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
               handlelength=1.8, columnspacing=1.1)
    save_fig(fig, "fig6_closedloop")
    plt.close(fig)


def fig7():
    """Measured-fleet panel: one held-out engine trace plus the error-growth comparison."""
    figstyle.use()
    with open(os.path.join(EXP, "public_traces.json"), encoding="utf-8") as f:
        tr = json.load(f)
    with open(os.path.join(EXP, "results_public", "persistence.json"), encoding="utf-8") as f:
        base = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
    ax = axes[0]
    t = np.arange(1, tr["H"] + 1)
    truth = np.array(tr["truth"])[0][:, 4]
    smooth = np.convolve(truth, np.ones(11) / 11, mode="valid")
    ax.plot(t, truth, color="0.70", lw=0.6, label="measured, per cycle")
    ax.plot(t[5:len(t) - 5], smooth, color="0.12", lw=1.2, label="measured, 11-cycle mean")
    ax.plot(t, np.array(tr["PI-Lat"])[0][:, 4], color=figstyle.PAL["PI-Lat"], lw=1.0,
            ls="--", label="PI-Lat")
    ax.plot(t, np.array(tr["PCL-WM"])[0][:, 4], color=figstyle.ACCENT, lw=1.5,
            label="PCL-WM (full)")
    ax.set_xlabel("Cycle after the anchor point")
    ax.set_ylabel("$P_{s30}$ (normalised)")
    figstyle.clean(ax, grid="both")
    figstyle.panel(ax, "(a)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.02, 0.93),
              handlelength=2.0, labelspacing=0.25)

    ax = axes[1]
    models = [("MLP-MD", figstyle.PAL["MLP-MD"], "-", "o"),
              ("PI-Lat", figstyle.PAL["PI-Lat"], "--", "P"),
              ("PCL-WM", figstyle.ACCENT, "-", "o"),
              ("abl-no-symplectic", figstyle.PAL["PCL-WM w/o symplectic"], "--", "X")]
    hs = [25, 50, 100]
    ax.plot(hs, [base[str(h)] for h in hs], color="0.55", lw=1.1, ls=":",
            label="persistence", marker="s", ms=2.2)
    for name, c, ls, mk in models:
        vals = []
        for h in hs:
            fs = [os.path.join(EXP, "results_public", f"D5_{name}_s{s}.json") for s in (0, 1)]
            vals.append(float(np.mean([json.load(open(p, encoding="utf-8"))[f"nrmse_{h}"]
                                       for p in fs])))
        ax.plot(hs, vals, color=c, lw=1.6 if name == "PCL-WM" else 0.9, ls=ls,
                marker=mk, ms=2.4, label=figstyle.DISPLAY.get(name, name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Horizon (cycles)")
    ax.set_ylabel("Normalised RMSE")
    figstyle.clean(ax, grid="both")
    figstyle.panel(ax, "(b)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=2,
              handlelength=1.8, labelspacing=0.25, columnspacing=1.0)

    fig.subplots_adjust(top=0.86, wspace=0.30)
    save_fig(fig, "fig7_public")
    plt.close(fig)


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
