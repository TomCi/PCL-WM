# -*- coding: utf-8 -*-
"""EKF / UKF baselines with the *true* plant equations (oracle-physics filters).

The filter sees only the measured channels, is initialised from the measured
history with the unobserved components set to their training mean, and is then
rolled out with the true dynamics.  The score is the same normalised RMSE on the
same windows as every learned model, so the comparison is like for like.
"""
import json
import os

import numpy as np

import domains
from train import build_data

DT = domains.DT
OBS_N = 0.01          # observation noise level, 1% of the channel std (as in training)
L = 4                 # history window used for the filter
T0 = 50
HORIZONS = (25, 50, 100, 200)
N_EVAL = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_filters")


def rk4(dom, x, u):
    k1 = dom.f(x, u)
    k2 = dom.f(x + 0.5 * DT * k1, u)
    k3 = dom.f(x + 0.5 * DT * k2, u)
    k4 = dom.f(x + DT * k3, u)
    return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def jac(dom, x, u, h=1e-6):
    n = x.size
    F = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = h * max(1.0, abs(x[i]))
        F[:, i] = (rk4(dom, x + d, u) - rk4(dom, x - d, u)) / (2.0 * d[i])
    return F


def ekf_filter(dom, y_phys, u_seq, x_mean, x_scale, R):
    """Sequential EKF over the history window; returns posterior mean/cov at T0."""
    x = x_mean.copy()
    x[dom.obs_idx] = y_phys[0]
    P = np.diag(x_scale ** 2)
    for i, o in enumerate(dom.obs_idx):
        P[o, o] = R[i]
    H = np.zeros((dom.dy, dom.dx))
    for i, o in enumerate(dom.obs_idx):
        H[i, o] = 1.0
    for k in range(len(y_phys) - 1):
        n = len(y_phys) - 1
        u = u_seq[k] if len(u_seq) > k else np.zeros(dom.du)
        x = rk4(dom, x, u)
        F = jac(dom, x, u)
        P = F @ P @ F.T + np.diag((1e-3 * x_scale) ** 2)
        y = y_phys[k + 1]
        S = H @ P @ H.T + np.diag(R)
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y - x[dom.obs_idx])
        P = (np.eye(dom.dx) - K @ H) @ P
        P = 0.5 * (P + P.T)
    return x, P


def ukf_filter(dom, y_phys, u_seq, x_mean, x_scale, R):
    n = dom.dx
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    lam = alpha ** 2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1.0 - alpha ** 2 + beta)
    gam = np.sqrt(n + lam)

    x = x_mean.copy()
    x[dom.obs_idx] = y_phys[0]
    P = np.diag(x_scale ** 2)
    for i, o in enumerate(dom.obs_idx):
        P[o, o] = R[i]
    H = np.zeros((dom.dy, dom.dx))
    for i, o in enumerate(dom.obs_idx):
        H[i, o] = 1.0

    def sigmas(x, P):
        s = np.linalg.cholesky(P + 1e-12 * np.eye(n))
        S = np.zeros((2 * n + 1, n))
        S[0] = x
        for i in range(n):
            S[1 + i] = x + gam * s[:, i]
            S[1 + n + i] = x - gam * s[:, i]
        return S

    for k in range(len(y_phys) - 1):
        u = u_seq[k] if len(u_seq) > k else np.zeros(dom.du)
        S = sigmas(x, P)
        Sx = np.array([rk4(dom, si, u) for si in S])
        x = wm @ Sx
        P = np.diag((1e-3 * x_scale) ** 2).copy()
        for i in range(2 * n + 1):
            d = Sx[i] - x
            P = P + wc[i] * np.outer(d, d)
        Sy = Sx @ H.T
        ybar = wm @ Sy
        Syy = np.diag(R).copy()
        Sxy = np.zeros((n, dom.dy))
        for i in range(2 * n + 1):
            dy = Sy[i] - ybar
            Syy = Syy + wc[i] * np.outer(dy, dy)
            Sxy = Sxy + wc[i] * np.outer(Sx[i] - x, dy)
        K = Sxy @ np.linalg.inv(Syy)
        x = x + K @ (y_phys[k + 1] - ybar)
        P = P - K @ Syy @ K.T
        P = 0.5 * (P + P.T)
    return x, P


def run(domain, kind):
    dom = domains.make(domain)
    d = build_data(dom, 240, 24, 400, 0, noise=OBS_N)
    Xtr, Utr = dom.sample(240, 400, seed=0)
    umean = Utr.reshape(-1, dom.du).mean(0)
    ustd = Utr.reshape(-1, dom.du).std(0) + 1e-8
    x_mean = Xtr.reshape(-1, dom.dx).mean(0)
    x_scale = Xtr.reshape(-1, dom.dx).std(0) + 1e-8
    mean, std = d["mean"], d["std"]
    R = (OBS_N * np.std(Xtr[:, :, dom.obs_idx].reshape(-1, dom.dy), axis=0)) ** 2

    errs = {h: [] for h in HORIZONS}
    one = []
    for e in range(N_EVAL):
        yte = d["yte"][e].cpu().numpy() if hasattr(d["yte"], "cpu") else np.asarray(d["yte"][e])
        ute = d["ute"][e].cpu().numpy() if hasattr(d["ute"], "cpu") else np.asarray(d["ute"][e])
        y_phys = lambda arr: arr * std + mean
        u_seq = ute * ustd + umean          # back to the raw control the plant was driven with
        ywin = y_phys(yte[T0 - L + 1:T0 + 1])
        filt = ekf_filter if kind == "ekf" else ukf_filter
        x, _ = filt(dom, ywin, u_seq[T0 - L + 1:T0 + 1], x_mean, x_scale, R)
        preds = []
        for k in range(max(HORIZONS)):
            x = rk4(dom, x, u_seq[T0 + k])
            preds.append((x[dom.obs_idx] - mean) / std)
        for h in HORIZONS:
            errs[h].append(float(np.sqrt(np.mean((preds[h - 1] - yte[T0 + h]) ** 2))))
        one.append(float(np.sqrt(np.mean((preds[0] - yte[T0 + 1]) ** 2))))
    out = {f"nrmse_{h}": round(float(np.mean(errs[h])), 4) for h in HORIZONS}
    hs = np.array(HORIZONS, float)
    ns = np.array([out[f"nrmse_{h}"] + 1e-12 for h in HORIZONS])
    out["beta"] = round(float(np.polyfit(np.log(hs), np.log(ns), 1)[0]), 3)
    out["one_step"] = round(float(np.mean(one)), 4)
    out["domain"] = domain
    out["kind"] = kind
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for domain in ("D1", "D2", "D3", "D9"):
        for kind in ("ekf", "ukf"):
            r = run(domain, kind)
            rows.append(r)
            print(f"{domain} {kind.upper()} "
                  f"@25={r['nrmse_25']} @50={r['nrmse_50']} @100={r['nrmse_100']} @200={r['nrmse_200']} "
                  f"beta={r['beta']} one={r['one_step']}", flush=True)
            with open(os.path.join(OUT, f"{domain}_{kind}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
    print("filters done", flush=True)


if __name__ == "__main__":
    main()
