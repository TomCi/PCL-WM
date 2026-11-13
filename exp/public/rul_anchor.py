# -*- coding: utf-8 -*-
"""Is the failure threshold even reachable from the test anchors?"""
import io
import os

import numpy as np

import public_domain as pd

dom = pd.make(subset="FD002", condnorm=True, late=True)
CHS = [0, 1, 2, 3, 4]
BLK = 10


def health1(s):
    c = np.cumsum(s)
    out = c.copy()
    out[BLK:] = (c[BLK:] - c[:-BLK]) / BLK
    out[:BLK] = c[:BLK] / np.arange(1, BLK + 1)
    return out


Xtr, Utr = dom.prebuilt()[0]
thr = float(np.stack([health1(Xtr[i][:, CHS].mean(-1))[-1] for i in range(Xtr.shape[0])]).mean())
print("threshold from train ends %.3f" % thr)

recs = pd._records(os.path.join(pd.DATA, "test_FD002.txt"))
keep = [r for r in recs if r[0].shape[0] >= dom.steps + 1]
Xn = np.stack([(k[0][-(dom.steps + 1):] - dom.cmu[pd.condition_labels(k[1][-(dom.steps + 1):])])
               / dom.csd[pd.condition_labels(k[1][-(dom.steps + 1):])] for k in keep])
H = np.stack([health1(Xn[i][:, CHS].mean(-1)) for i in range(Xn.shape[0])])
h0 = H[:, -1]
rul = np.loadtxt(os.path.join(pd.DATA, "RUL_FD002.txt"))[:Xn.shape[0]]
print("engines kept %d" % len(h0))
print("anchor health: min %.3f q25 %.3f med %.3f q75 %.3f max %.3f"
      % (h0.min(), np.percentile(h0, 25), np.median(h0), np.percentile(h0, 75), h0.max()))
print("anchor health >= threshold: %d of %d" % ((h0 >= thr).sum(), h0.size))
print("anchor health, engines with RUL<=20: med %.3f ; RUL>100: med %.3f"
      % (np.median(h0[rul <= 20]), np.median(h0[rul > 100])))
