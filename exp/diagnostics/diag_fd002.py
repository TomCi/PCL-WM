# -*- coding: utf-8 -*-
"""Diagnose the FD002 fit: is the 1.06 one-step error a protocol artefact or a real result?"""
import numpy as np
import public_domain

d = public_domain.make(subset="FD002")
(Xtr, Utr), (Xte, Ute) = d.prebuilt()
print("train", Xtr.shape, "test", Xte.shape, "du", Utr.shape[-1])

Otr = d.obs(Xtr, 0.01, np.random.default_rng(7))
Ote = d.obs(Xte, 0.01, np.random.default_rng(7))
mean = Otr.reshape(-1, d.dy).mean(0)
std = Otr.reshape(-1, d.dy).std(0) + 1e-8
print("obs mean", np.round(mean, 1))
print("obs std ", np.round(std, 3))

Ztr = (Otr - mean) / std
Zte = (Ote - mean) / std

# persistence one-step, normalised by signal rms
rms = np.sqrt((Zte ** 2).mean())
pers = np.sqrt((np.diff(Zte, axis=1) ** 2).mean()) / rms
print("persistence one-step normalised", round(float(pers), 4))

# how much does the operating condition move the sensors?
cond = np.rint(Utr[:, 0, 0]).astype(int)
for c in np.unique(cond):
    m = cond == c
    print("cond", c, "share %.3f" % m.mean(),
          "T24 mean %6.1f std %6.2f" % (Xtr[m, :, 1].mean(), Xtr[m, :, 1].std()))

# within-window condition switches
sw = (np.abs(np.diff(Utr[:, :, 0], axis=1)) > 0.5).sum(1)
print("windows with a condition switch:", int((sw > 0).sum()), "of", sw.shape[0])
