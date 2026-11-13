# -*- coding: utf-8 -*-
"""Is the FD002 file the genuine published one? Check degradation trends per subset."""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cmapss")
for f in ("train_FD001.txt", "train_FD002.txt", "train_FD004.txt"):
    a = np.loadtxt(os.path.join(DATA, f))
    r = a[a[:, 0] == 1]
    u = a[:, 0]
    print(f, "cycles", r.shape[0], "cols", a.shape[1], "units", len(np.unique(u)))
    print("  T24 first/mid/last %.1f %.1f %.1f" % (r[0, 6], r[len(r) // 2, 6], r[-1, 6]))
    print("  T24 distinct values", len(np.unique(np.round(r[:, 6], 3)),
                                       ), "of", r.shape[0], "cycles")
    print("  Nf  first/mid/last %.1f %.1f %.1f" % (r[0, 11], r[len(r) // 2, 11], r[-1, 11]))
