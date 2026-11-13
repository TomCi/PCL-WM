# -*- coding: utf-8 -*-
"""Floor comparison: predicting the window mean is the best any model can do on a
condition-switching record. Compare that floor with what the models actually reach."""
import numpy as np

import public_domain
from train import build_data

for subset in ("FD001", "FD002"):
    d = public_domain.make(subset=subset)
    (Xtr, Utr), (Xte, Ute) = d.prebuilt()
    d_ = build_data(d, 0, 0, d.steps, 0)
    Z = d_["yte"].cpu().numpy()                     # normalised test observations
    rms = np.sqrt((Z ** 2).mean())
    for h in (25, 50, 100):
        pred = Z[:, :1, :].repeat(h, axis=1)        # hold the last observed sample
        e = np.sqrt(((Z[:, 1:h + 1, :] - pred) ** 2).mean()) / rms
        m = np.zeros_like(Z[:, 1:h + 1, :])         # hold the mean
        e2 = np.sqrt(((Z[:, 1:h + 1, :] - m) ** 2).mean()) / rms
        print("%s @%d  persistence %.3f  window mean %.3f" % (subset, h, e, e2))
