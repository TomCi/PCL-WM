# -*- coding: utf-8 -*-
"""Shared figure style for the manuscript: one palette, one typography, one axis look."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# semantic colours: the method carries the accent, baselines stay muted
PAL = {
    "MLP-MD": "#4a6fa5",
    "GRU-LD": "#5f8a72",
    "NODE": "#8c6d9c",
    "HNN": "#9d7a52",
    "PI-Lat": "#6b7a87",
    "PCL-WM": "#b03a2e",
    "PCL-WM w/o symplectic": "#a6605a",
}
NEUTRAL = "#93a2b0"
NEUTRAL_LIGHT = "#bcc6d0"
ACCENT = "#b03a2e"
INK = "#222222"

# the registry names never reach a journal figure
DISPLAY = {
    "PCL-WM": "PCL-WM (full)",
    "abl-no-symplectic": "w/o symplectic map",
    "abl-no-cert": "w/o certificate term",
    "abl-no-gauge": "w/o gauge condition",
    "abl-free-dissipation": "free dissipation",
    "abl-no-curriculum": "w/o curriculum",
    "abl-dz16": "latent width 16",
    "abl-dz64": "latent width 64",
}

BASE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.4,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "grid.linestyle": ":",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.55,
    "savefig.dpi": 400,
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "0.15",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "0.15",
    "ytick.color": "0.15",
}


def use():
    plt.rcParams.update(BASE)


def clean(ax, grid="y"):
    """Open frame: keep the axes that carry meaning, drop the box."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, which="major")
    ax.set_axisbelow(True)
    return ax


def panel(ax, tag, dx=0.0, dy=1.012):
    """Panel letter printed just above each axes, bold, left-aligned."""
    ax.text(dx, dy, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", clip_on=False)
