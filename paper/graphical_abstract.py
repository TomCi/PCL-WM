# -*- coding: utf-8 -*-
"""Graphical abstract for the manuscript: 13 x 5.2 cm, Elsevier 2.5:1."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "exp"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle

figstyle.use()
INK = figstyle.INK
RED = figstyle.ACCENT
MUTED = "#6d7a86"

W, H = 13.0, 5.2
fig = plt.figure(figsize=(W / 2.54, H / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, lines, fc="#f2f5f8", ec="#8d9aa6", fs=5.9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
                                linewidth=0.7, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=INK, linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.7, color="#5b6873", shrinkA=1, shrinkB=1))


def head(x, y, text, fs=6.4):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=fs, color="#5b6873",
            fontweight="bold")


# ---------------- plant ----------------
head(0.15, 4.72, "PLANT")
box(0.15, 2.35, 2.75, 1.45, "measured channels\n(window of $L$ steps)")
ax.text(0.15, 1.90, "the full state is never seen", fontsize=5.5, color=MUTED,
        style="italic")
arrow(2.92, 3.07, 3.35, 3.07)

# ---------------- model ----------------
ax.add_patch(FancyBboxPatch((3.35, 0.85), 4.10, 3.60,
                            boxstyle="round,pad=0.04,rounding_size=0.12",
                            linewidth=0.8, edgecolor="#c3ccd6", facecolor="white",
                            linestyle="--"))
ax.text(3.48, 4.50, "PCL-WM", fontsize=6.8, color=RED, fontweight="bold",
        va="bottom")

box(3.50, 2.30, 0.80, 0.95, "encoder\n$E_\\phi$", fs=5.7)
box(4.55, 3.05, 1.60, 1.20,
    "canonical pair\n$(q,p)$\nVerlet map", fc="#e8eef7", ec="#3f6699", fs=5.7)
box(4.55, 1.35, 1.60, 1.20,
    "residual $\\xi$\nport-Hamiltonian\n$R=LL^\\top$", fc="#f8eceb", ec=RED, fs=5.7)
box(6.70, 2.30, 0.78, 0.95, "decoder\n$D_\\psi$", fs=5.7)

arrow(4.32, 2.85, 4.52, 3.55)
arrow(4.32, 2.70, 4.52, 1.95)
arrow(6.18, 3.65, 6.67, 3.05)
arrow(6.18, 1.95, 6.67, 2.60)

ax.text(5.38, 1.15, "priors: canonical · symplectic · dissipative",
        ha="center", va="center", fontsize=5.1, color=MUTED)

# ---------------- verdict ----------------
head(7.95, 4.72, "WHAT EACH PRIOR BUYS")
ax.text(7.95, 4.20, "200-step error, projectile", fontsize=5.4, color=MUTED,
        va="bottom")

x0, sc = 9.35, 2.35
for i, (name, val, col) in enumerate([("PCL-WM (full)", 0.793, RED),
                                      ("w/o symplectic", 0.343, figstyle.NEUTRAL)]):
    y = 3.62 - i * 0.62
    ax.text(9.25, y + 0.15, name, ha="right", va="center", fontsize=5.7, color=INK)
    ax.add_patch(plt.Rectangle((x0, y), val * sc, 0.30, facecolor=col, edgecolor="none"))
    ax.text(x0 + val * sc + 0.08, y + 0.15, "%.2f" % val, ha="left", va="center",
            fontsize=5.7, color=INK)

ax.text(7.95, 2.55, "the component carries the ranking,", fontsize=5.5, color=MUTED,
        style="italic", va="center")
ax.text(7.95, 2.20, "not the architecture as a whole", fontsize=5.5, color=MUTED,
        style="italic", va="center")

ax.text(7.95, 1.62, "partly observed plant: 3.27 vs 0.22", fontsize=5.7, color=INK,
        va="center")
ax.text(7.95, 1.22, "closed loop: 0.065–0.074, all five", fontsize=5.7, color=INK,
        va="center")
ax.text(0.15, 0.34, "projectile · quadrotor · thermal network · reaction–diffusion · C-MAPSS fleet",
        fontsize=5.2, color="#95a1ac", va="center")

out_png = os.path.join(HERE, "figs", "fig_graphical_abstract.png")
out_pdf = os.path.join(HERE, "figs", "fig_graphical_abstract.pdf")
out_tif = os.path.join(HERE, "figs", "fig_graphical_abstract.tif")
fig.savefig(out_png, dpi=400, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
fig.savefig(out_tif, dpi=400, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
print("saved %.1f x %.1f cm" % (W, H))
