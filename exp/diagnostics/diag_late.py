# -*- coding: utf-8 -*-
"""Is the approach-to-failure learnable inside a hundred cycles?

Late-window protocol: the last 161 cycles of every FD002 record, per-operating-point normalised.
Reference predictors inside the window, no model trained:
  persistence      hold the anchor value
  linear trend      least-squares line through the last 16 observed samples of the channel
Error is the drift the predictor misses, divided by the drift that actually occurs, so persistence
scores exactly 1.00 by construction.
"""
import numpy as np

import public_domain
from train import build_data

CH = 0  # T24, the channel that carries the thermal degradation

for late in (False, True):
    d = public_domain.make(subset="FD002", condnorm=True, late=late)
    dd = build_data(d, 0, 0, d.steps, 0)
    Z = dd["yte"].cpu().numpy()                     # (n, steps+1, 6) normalised observations
    t0, L = 60, 4
    for h in (25, 50, 100):
        drift = Z[:, t0 + h, CH] - Z[:, t0, CH]
        rms = np.sqrt((drift ** 2).mean()) + 1e-12
        e_pers = np.sqrt((drift ** 2).mean()) / rms
        lin = []
        for i in range(Z.shape[0]):
            seg = Z[i, t0 - 16 + 1:t0 + 1, CH]
            k = np.arange(seg.size)
            slope = np.polyfit(k, seg, 1)[0]
            lin.append((slope * h - drift[i]) ** 2)
        e_lin = np.sqrt(np.mean(lin)) / rms
        print("late=%-5s @%3d  drift rms %.3f  persistence %.2f  linear %.2f"
              % (late, h, rms, e_pers, e_lin))
