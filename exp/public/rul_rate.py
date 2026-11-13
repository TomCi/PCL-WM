# -*- coding: utf-8 -*-
"""Implied drift rate of each rollout, read back from its predicted remaining life."""
import glob
import io
import json
import os

import numpy as np

import public_domain as pd

CHS = [0, 1, 2, 3, 4]
BLK = 10
D = r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\results_rul"


def health1(s):
    c = np.cumsum(s)
    out = c.copy()
    out[BLK:] = (c[BLK:] - c[:-BLK]) / BLK
    out[:BLK] = c[:BLK] / np.arange(1, BLK + 1)
    return out


dom = pd.make(subset="FD002", condnorm=True, late=True)
recs = pd._records(os.path.join(pd.DATA, "test_FD002.txt"))
keep = [r for r in recs if r[0].shape[0] >= dom.steps + 1]
Xn = np.stack([(k[0][-(dom.steps + 1):] - dom.cmu[pd.condition_labels(k[1][-(dom.steps + 1):])])
               / dom.csd[pd.condition_labels(k[1][-(dom.steps + 1):])] for k in keep])
h0 = np.stack([health1(Xn[i][:, CHS].mean(-1))[-1] for i in range(Xn.shape[0])])
rul = np.loadtxt(os.path.join(pd.DATA, "RUL_FD002.txt"))[:Xn.shape[0]]
thr = 1.072
gap = np.maximum(thr - h0, 1e-3)
obs_rate = gap / np.maximum(rul, 1)

print("observed rate to threshold: med %.4f per cycle" % np.median(obs_rate))
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    d = json.load(io.open(p, encoding="utf-8"))
    pr = np.array(d["rul_predictions"], float)
    pred_rate = gap / np.maximum(pr, 1)
    ratio = pred_rate / obs_rate
    print("%-32s implied/predic rate ratio med %.2f  q25 %.2f q75 %.2f  within 0.5-2x: %.2f"
          % (os.path.basename(p), np.median(ratio), np.percentile(ratio, 25),
             np.percentile(ratio, 75), float(((ratio >= 0.5) & (ratio <= 2)).mean())))
