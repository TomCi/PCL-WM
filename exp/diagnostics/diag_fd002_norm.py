# -*- coding: utf-8 -*-
"""Check whether per-condition normalisation turns FD002 into a learnable signal."""
import numpy as np

import public_domain

CENT = np.array([[0.001, 0.0003, 100], [10, 0.25, 100], [20, 0.7, 100],
                 [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)


def labels(osu):
    sh = osu.shape[:-1]
    flat = osu.reshape(-1, 3)
    d2 = ((flat[:, None, :] - CENT[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).reshape(sh)


d = public_domain.make(subset="FD002")
recs = d.recs
steps = d.steps

ok = [r for r in recs if r[0].shape[0] >= steps + 1]
n_test = d.n_test
train, test = ok[:-n_test], ok[-n_test:]

S = np.stack([r[0][:steps + 1] for r in train])          # (n, T, 21)
U = np.stack([r[1][:steps + 1] for r in train])          # os for the same rows
L = labels(U)

mu = np.zeros((6, 21))
sd = np.zeros((6, 21))
for c in range(6):
    sel = S[L == c]
    mu[c], sd[c] = sel.mean(0), sel.std(0) + 1e-8

Z = (S - mu[L]) / sd[L]
Zo = Z[..., d.obs_idx]
rms = np.sqrt((Zo ** 2).mean())
pers = np.sqrt((np.diff(Zo, axis=1) ** 2).mean()) / rms
print("condition share", np.round(np.bincount(L.ravel(), minlength=6) / L.size, 3))
print("obs normalised mean %s" % np.round(Zo.reshape(-1, 6).mean(0), 3))
print("obs normalised std  %s" % np.round(Zo.reshape(-1, 6).std(0), 3))
print("persistence one-step, per-condition normalised %.4f" % pers)

Ste = np.stack([r[0][:steps + 1] for r in test])
Lte = labels(np.stack([r[1][:steps + 1] for r in test]))
Zte = ((Ste - mu[Lte]) / sd[Lte])[..., d.obs_idx]
rms_te = np.sqrt((Zte ** 2).mean())
pers_te = np.sqrt((np.diff(Zte, axis=1) ** 2).mean()) / rms_te
print("test persistence one-step %.4f" % pers_te)

# degradation signature: does the normalised observed signal drift within the window?
drift = (Zte[:, -1, :] - Zte[:, 0, :]).mean(0)
print("test normalised drift over 160 cycles %s" % np.round(drift, 3))
