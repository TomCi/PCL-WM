# -*- coding: utf-8 -*-
"""Does a smoothed, multi-channel health index have a learnable trend?

Same late-window protocol. The reference predictors are still persistence and a linear trend,
but the drift is read from a ten-cycle moving average of the normalised degradation channels, and
from the mean of five such channels, so sensor noise is averaged out before the trend is judged.
"""
import numpy as np

import public_domain
from train import build_data

CHS = [0, 1, 2, 3, 4]  # T24, T30, T50, Nf, Ps30


def smooth(Z, w=10):
    """Ten-cycle trailing mean, applied along the cycle axis."""
    c = np.cumsum(Z, axis=1)
    out = c.copy()
    out[:, w:] = (c[:, w:] - c[:, :-w]) / w
    out[:, :w] = c[:, :w] / np.arange(1, w + 1)
    return out


for late in (True,):
    d = public_domain.make(subset="FD002", condnorm=True, late=late)
    dd = build_data(d, 0, 0, d.steps, 0)
    Z = dd["yte"].cpu().numpy()
    t0 = 60
    for chan, name in ((CHS, "mean of five"), ([0], "T24 only")):
        S = smooth(Z[:, :, chan].mean(-1))
        # systematic part: the mean drift over records, in units of the record-to-record scatter
        for h in (25, 50, 100):
            drift = S[:, t0 + h] - S[:, t0]
            rms = np.sqrt((drift ** 2).mean()) + 1e-12
            lin = []
            for i in range(S.shape[0]):
                seg = S[i, t0 - 16 + 1:t0 + 1]
                k = np.arange(seg.size)
                lin.append((np.polyfit(k, seg, 1)[0] * h - drift[i]) ** 2)
            print("late=%s  %-12s @%3d  mean drift %+.3f  rms %.3f  linear %.2f"
                  % (late, name, h, drift.mean(), rms, np.sqrt(np.mean(lin)) / rms))
