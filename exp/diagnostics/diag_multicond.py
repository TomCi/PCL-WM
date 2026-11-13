# -*- coding: utf-8 -*-
"""Quantify why the multi-condition subsets do not support this protocol."""
import os

import numpy as np

import public_domain

CENT = np.array([[0.001, 0.0003, 100], [10, 0.25, 100], [20, 0.7, 100],
                 [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)


def labels(osu):
    sh = osu.shape[:-1]
    flat = osu.reshape(-1, 3)
    d2 = ((flat[:, None, :] - CENT[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).reshape(sh)


def persistence(subset):
    d = public_domain.make(subset=subset)
    (Xtr, Utr), (Xte, Ute) = d.prebuilt()
    Otr = d.obs(Xtr, 0.01, np.random.default_rng(7))
    Ote = d.obs(Xte, 0.01, np.random.default_rng(7))
    mean = Otr.reshape(-1, d.dy).mean(0)
    std = Otr.reshape(-1, d.dy).std(0) + 1e-8
    Zte = (Ote - mean) / std
    rms = np.sqrt((Zte ** 2).mean())
    one = np.sqrt((np.diff(Zte, axis=1) ** 2).mean()) / rms
    return d, Ute, float(one)


for subset in ("FD001", "FD002"):
    d, Ute, one = persistence(subset)
    print("%s: observed persistence one-step NRMSE %.3f" % (subset, one))
    if Ute.shape[-1]:
        L = labels(Ute)
        same = (L[:, 1:] == L[:, :-1]).mean()
        print("   consecutive cycles in the same operating point: %.3f" % same)
        print("   distinct operating points in a 160-cycle window: %.1f of 6"
              % np.mean([len(np.unique(l)) for l in L]))
        # how much of the sensor variance is explained by the operating point alone
        Xte = d.prebuilt()[1][0]
        across = np.mean([Xte[..., j].std() for j in d.obs_idx])
        within = np.mean([np.mean([Xte[L == c][:, j].std() for c in range(6) if (L == c).any()])
                          for j in d.obs_idx])
        print("   observed-channel std across conditions %.1f, within a condition %.1f"
              % (across, within))
