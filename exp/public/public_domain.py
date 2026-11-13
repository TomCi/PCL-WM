# -*- coding: utf-8 -*-
"""Public-data domain: NASA C-MAPSS turbofan fleet, run-to-failure.

Nothing here is simulated. Each window is an excerpt of a published run-to-failure record; the model
sees six of the twenty-one sensor channels and has to roll the remaining hidden channels forward.

Subsets: FD001 (one operating point, no exogenous input), FD002 and FD004 (six operating
conditions, the three operating settings are supplied as a three-channel exogenous input).

Record layout: unit, cycle, os1, os2, os3, s1..s21.
"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cmapss")

FILES = {
    "FD001": "train_FD001.txt",
    "FD002": "train_FD002.txt",
    "FD004": "train_FD004.txt",
}

# measured channels as 0-based indices into the 21 sensor columns:
# s2 (T24), s3 (T30), s4 (T50), s8 (Nf), s11 (Ps30), s17 (htBleed). The other fifteen are hidden.
OBS_IDX = [1, 2, 3, 7, 10, 16]

# the six operating points of FD002/FD004 as (os1, os2, os3); every record in these subsets
# visits all six, and it switches between them from one cycle to the next
CENT = np.array([[0.001, 0.0003, 100.0], [10.0, 0.25, 100.0], [20.0, 0.7, 100.0],
                 [25.0, 0.62, 60.0], [35.0, 0.84, 100.0], [42.0, 0.84, 100.0]])


def condition_labels(U):
    """Nearest operating point for every row of a (..., 3) array of operating settings."""
    sh = U.shape[:-1]
    flat = np.asarray(U, dtype=float).reshape(-1, 3)
    d2 = ((flat[:, None, :] - CENT[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).reshape(sh)


def _records(path):
    a = np.loadtxt(path, dtype=float)
    out = []
    for u in np.unique(a[:, 0]):
        r = a[a[:, 0] == u]
        out.append((r[:, 5:], r[:, 2:5]))
    return out


class CMAPSS:
    """Windows of the published fleet, one window per engine."""

    name = "D5"
    obs_idx = OBS_IDX
    dx = 21
    steps = 160

    def __init__(self, path=None, steps=160, n_test=16, subset="FD001", condnorm=False,
                 late=False):
        self.subset = subset
        self.du = 0 if subset == "FD001" else 3
        self.recs = _records(path or os.path.join(DATA, FILES[subset]))
        self.steps = steps
        self.n_test = n_test
        self.late = late
        self._train, self._test = self._split()
        self.condnorm = False
        if condnorm and self.du:
            self._condition_normalise()

    def _condition_normalise(self):
        """Subtract, per operating point, the mean and scale of each sensor channel.

        On the multi-condition subsets the raw reading is a step function of the operating point,
        so a window is not a trajectory. Removing the per-point mean and scale leaves the drift
        that the record accumulates, which is the quantity the protocol is meant to score. Stats
        are fitted on the training records only and then applied to both pools.
        """
        S = np.concatenate([r[0] for r in self._train], 0)
        L = condition_labels(np.concatenate([r[1] for r in self._train], 0))
        mu = np.zeros((len(CENT), S.shape[-1]))
        sd = np.zeros_like(mu)
        for c in range(len(CENT)):
            sel = S[L == c]
            mu[c] = sel.mean(0)
            sd[c] = sel.std(0) + 1e-3
        self.cmu, self.csd = mu, sd
        self.recs = [((S_i - mu[condition_labels(U_i)]) / sd[condition_labels(U_i)], U_i)
                     for S_i, U_i in self.recs]
        self._train, self._test = self._split()
        self.condnorm = True

    @property
    def dy(self):
        return len(self.obs_idx)

    def _split(self):
        need = self.steps + 1
        ok = [r for r in self.recs if r[0].shape[0] >= need]
        if len(ok) <= self.n_test:
            raise RuntimeError("not enough long records: %d" % len(ok))
        return ok[:-self.n_test], ok[-self.n_test:]

    def _pack(self, pool):
        if self.late:
            # the last cycles of the record, so a window ends at failure and the drift inside it
            # is the approach to failure rather than the first hundred cycles of a healthy engine
            X = np.stack([r[0][-(self.steps + 1):] for r in pool])
            U = (np.stack([r[1][-(self.steps + 1):-1] for r in pool])
                 if self.du else np.zeros((X.shape[0], self.steps, 0)))
            return X, U
        X = np.stack([r[0][:self.steps + 1] for r in pool])
        if self.du:
            U = np.stack([r[1][:self.steps] for r in pool])
        else:
            U = np.zeros((X.shape[0], self.steps, 0))
        return X, U

    def prebuilt(self):
        return self._pack(self._train), self._pack(self._test)

    def sample(self, n_ep, steps, seed, **kw):
        rng = np.random.default_rng(seed)
        X, U = self._pack(self._train if n_ep >= 100 else self._test)
        if n_ep < X.shape[0]:
            idx = rng.choice(X.shape[0], n_ep, replace=False)
            X, U = X[idx], U[idx]
        return X, U

    def obs(self, s, noise=0.0, rng=None):
        o = s[..., self.obs_idx].copy()
        if noise <= 0.0:
            return o
        return o + rng.normal(0, noise, size=o.shape) * np.std(o, axis=(0, 1))

    def energy(self, s):
        return np.zeros(s.shape[:-1])


def make(**kw):
    return CMAPSS(**kw)


if __name__ == "__main__":
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    d = CMAPSS(subset=subset)
    (Xtr, Utr), (Xte, Ute) = d.prebuilt()
    print("subset", subset, "records", len(d.recs), "train", Xtr.shape, "test", Xte.shape,
          "du", Utr.shape[-1])
    print("sensor std", np.round(Xtr.reshape(-1, d.dx).std(0), 4))
