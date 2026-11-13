# -*- coding: utf-8 -*-
"""World-model families for the PCL-WM study (v2: partial observability).

Every model sees a history window w of shape [B, L, dy] of measured channels and
must predict the next dy measured channels for K steps.  Direct models stack the
window; latent models encode it into z and advance z.

  28/29-argument interface:
    encode(w) -> z            w: [B, L, dy]
    step(z, u) -> z'
    decode(z) -> [B, dy]
    regularizers(batch) -> dict
"""
import math
import torch
import torch.nn as nn


def mlp(i, h, o, layers=2, act=nn.Tanh()):
    mods, last = [], i
    for _ in range(layers):
        mods += [nn.Linear(last, h), act]
        last = h
    mods += [nn.Linear(last, o)]
    return nn.Sequential(*mods)


class MLPDirect(nn.Module):
    """Window-stacked MLP; the state is the rolling history window."""

    def __init__(self, dy, du, L=4, hidden=128):
        super().__init__()
        self.L, self.dy_ = L, dy
        self.net = mlp(L * dy + du, hidden, dy, layers=3)

    def encode(self, w):
        return w

    def step(self, z, u):
        nxt = z[:, -1] + self.net(torch.cat([z.reshape(z.shape[0], -1), u], -1))
        return torch.cat([z[:, 1:], nxt.unsqueeze(1)], 1)

    def decode(self, z):
        return z[:, -1]

    def regularizers(self, batch):
        return {}

    def structure(self):
        return {"family": "MLP-MD", "latent": 0}


class NODEDirect(nn.Module):
    """Neural ODE on the measured frame, with the history window as conditioning."""

    def __init__(self, dy, du, L=4, hidden=128):
        super().__init__()
        self.L, self.dy_ = L, dy
        self.f = mlp((L - 1) * dy + dy + du, hidden, dy, layers=3)
        self.log_dt = nn.Parameter(torch.tensor(0.0))

    def encode(self, w):
        return w

    def _rhs(self, z, u):
        hist = z[:, :-1].reshape(z.shape[0], -1)
        return self.f(torch.cat([hist, z[:, -1], u], -1))

    def step(self, z, u):
        h = torch.exp(self.log_dt) * 0.1
        k1 = self._rhs(z, u)
        z1 = torch.cat([z[:, :-1], (z[:, -1] + 0.5 * h * k1).unsqueeze(1)], 1)
        k2 = self._rhs(z1, u)
        z2 = torch.cat([z[:, :-1], (z[:, -1] + 0.5 * h * k2).unsqueeze(1)], 1)
        k3 = self._rhs(z2, u)
        z3 = torch.cat([z[:, :-1], (z[:, -1] + h * k3).unsqueeze(1)], 1)
        k4 = self._rhs(z3, u)
        nxt = z[:, -1] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.cat([z[:, 1:], nxt.unsqueeze(1)], 1)

    def decode(self, z):
        return z[:, -1]

    def regularizers(self, batch):
        return {}

    def structure(self):
        return {"family": "NODE", "latent": 0}


class HNNDirect(nn.Module):
    """Hamiltonian network on the measured vector, one history frame, no encoder.

    This is the deliberately un-augmented physics baseline used to test the
    necessity claim: it has no observation-to-state map, so hidden channels
    cannot be recovered.
    """

    def __init__(self, dy, du, L=4, hidden=128):
        super().__init__()
        self.dy_ = dy
        self.half = max(1, dy // 2)
        self.H = mlp(2 * self.half, hidden, 1, layers=3)
        self.B = nn.Linear(du, self.half, bias=False)
        self.log_h = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def h(self):
        return torch.exp(self.log_h)

    def encode(self, w):
        return w[:, -1]

    def _gradq(self, q, p):
        q = q.clone().requires_grad_(True)
        Hv = self.H(torch.cat([q, p], -1)).sum()
        return torch.autograd.grad(Hv, q, create_graph=torch.is_grad_enabled())[0]

    def step(self, z, u):
        half, dy = self.half, self.dy_
        q, p = z[:, :half], z[:, half:2 * half]
        b = self.B(u)
        p1 = p - 0.5 * self.h * self._gradq(q, p) + 0.5 * self.h * b
        q1 = q + self.h * p1
        p2 = p1 - 0.5 * self.h * self._gradq(q1, p1) + 0.5 * self.h * b
        return torch.cat([q1, p2, z[:, 2 * half:dy]], -1)

    def decode(self, z):
        return z[:, :self.dy_]

    def regularizers(self, batch):
        return {}

    def structure(self):
        return {"family": "HNN", "latent": 0}


class GRULatent(nn.Module):
    """Recurrent latent model: GRU encoder over the window, GRU cell transition."""

    def __init__(self, dy, du, L=4, dz=32, hidden=128):
        super().__init__()
        self.dz_ = dz
        self.enc = nn.GRU(dy, dz, batch_first=True)
        self.cell = nn.GRUCell(dz + du, dz)
        self.dec = mlp(dz, hidden, dy, layers=2)

    def encode(self, w):
        _, h = self.enc(w)
        return h[-1]

    def step(self, z, u):
        return self.cell(torch.cat([z, u], -1), z)

    def decode(self, z):
        return self.dec(z)

    def regularizers(self, batch):
        return {}

    def structure(self):
        return {"family": "GRU-LD", "latent": self.dz_}


class PILatent(nn.Module):
    """Latent model with smoothness penalty only: no geometric structure."""

    def __init__(self, dy, du, L=4, dz=32, hidden=128):
        super().__init__()
        self.dz_ = dz
        self.enc = mlp(L * dy, hidden, dz, layers=2)
        self.dyn = mlp(dz + du, hidden, dz, layers=2)
        self.dec = mlp(dz, hidden, dy, layers=2)

    def encode(self, w):
        return self.enc(w.reshape(w.shape[0], -1))

    def step(self, z, u):
        return z + self.dyn(torch.cat([z, u], -1))

    def decode(self, z):
        return self.dec(z)

    def regularizers(self, batch):
        return {"smooth": ((batch["z1"] - batch["z0"]) ** 2).mean()}

    def structure(self):
        return {"family": "PI-Lat", "latent": self.dz_}


class PCLWM(nn.Module):
    """Physics-constrained latent world model (v2).

    The window is encoded into m canonical pairs (q, p), advanced by a
    Stormer-Verlet step of a learned Hamiltonian, plus a residual block xi whose
    flow is a discrete port-Hamiltonian update driven by the gradient of a
    residual energy that sees the whole latent, so the dissipative and actuated
    part couples back into q and p.  R = L L^T is positive semi-definite and the
    conservative part is skew.
    """

    def __init__(self, dy, du, L=4, dz=32, hidden=128, symplectic=True, gauge=True,
                 free_dissipation=False, cert=True):
        super().__init__()
        self.dy_ = dy
        self.symplectic, self.gauge = symplectic, gauge
        self.free_dissipation, self.cert = free_dissipation, cert
        self.dz_ = dz
        self.m = max(2, dz // 4)
        self.r = dz - 2 * self.m
        self.enc = mlp(L * dy, hidden, dz, layers=2)
        self.dec = mlp(dz, hidden, dy, layers=2)
        self.H = mlp(2 * self.m, hidden, 1, layers=3)
        self.Hr = mlp(dz, hidden, 1, layers=2)
        self.C = mlp(dz, 16, self.r, layers=1)
        self.log_h = nn.Parameter(torch.tensor(math.log(0.2)))
        self.Rraw = nn.Parameter(torch.randn(self.r, self.r) * 0.02)
        self.Jr = nn.Parameter(torch.randn(self.r, self.r) * 0.02)
        self.G = nn.Linear(du, self.r, bias=False)
        self.gain = nn.Parameter(torch.ones(dy))

    @property
    def h(self):
        return 0.05 + 0.45 * torch.sigmoid(self.log_h)

    def encode(self, w):
        return self.enc(w.reshape(w.shape[0], -1))

    def decode(self, z):
        return self.dec(z) * self.gain

    def split(self, z):
        return z[:, :self.m], z[:, self.m:2 * self.m], z[:, 2 * self.m:]

    def _grad(self, q, p):
        q = q.clone().requires_grad_(True)
        p = p.clone().requires_grad_(True)
        Hv = self.H(torch.cat([q, p], -1)).sum()
        gq, gp = torch.autograd.grad(Hv, [q, p], create_graph=torch.is_grad_enabled())
        return gq, gp

    def _canonical_step(self, q, p):
        if self.symplectic:
            gq, _ = self._grad(q, p)
            p1 = p - 0.5 * self.h * gq
            q1 = q + self.h * p1
            gq2, _ = self._grad(q1, p1)
            return q1, p1 - 0.5 * self.h * gq2
        gq, gp = self._grad(q, p)
        q2, p2 = q + self.h * gp, p - self.h * gq
        gq2, gp2 = self._grad(q2, p2)
        return q + 0.5 * self.h * (gp + gp2), p - 0.5 * self.h * (gq + gq2)

    def _residual_step(self, z, u):
        xi = z[:, 2 * self.m:]
        zz = z.clone().requires_grad_(True)
        gx = torch.autograd.grad(self.Hr(zz).sum(), zz,
                                 create_graph=torch.is_grad_enabled())[0][:, 2 * self.m:]
        if self.free_dissipation:
            R = self.Rraw @ self.Rraw.T
        else:
            R = self.Rraw.T @ self.Rraw
        Jr = self.Jr - self.Jr.T
        drive = self.G(u)
        return xi + self.h * (torch.einsum("ij,bj->bi", Jr - R, gx) + drive)

    def step(self, z, u):
        q, p, xi = self.split(z)
        q1, p1 = self._canonical_step(q, p)
        xi1 = self._residual_step(z, u)
        return torch.cat([q1, p1, xi1], -1)

    def gauge_penalty(self, z, n=8, eps=1e-2, probes=3):
        """Symplecticity residual Phi^T Jm Phi - Jm along random probes."""
        q, p = z[:n, :self.m], z[:n, self.m:2 * self.m]
        x = torch.cat([q, p], -1)
        d = 2 * self.m
        Jm = torch.zeros(d, d, device=z.device)
        Jm[:d // 2, d // 2:] = torch.eye(d // 2, device=z.device)
        Jm[d // 2:, :d // 2] = -torch.eye(d // 2, device=z.device)
        loss = 0.0
        for _ in range(probes):
            a, b = torch.randn_like(x), torch.randn_like(x)
            imgs = []
            for v in (a, b):
                xp, xm = x + eps * v, x - eps * v
                fp = torch.cat(self._canonical_step(xp[:, :self.m], xp[:, self.m:]), -1)
                fm = torch.cat(self._canonical_step(xm[:, :self.m], xm[:, self.m:]), -1)
                imgs.append((fp - fm) / (2 * eps))
            A, B = imgs
            w_map = torch.einsum("bi,ij,bj->b", A, Jm, B)
            w_ref = torch.einsum("bi,ij,bj->b", a, Jm, b)
            loss = loss + ((w_map - w_ref) ** 2).mean()
        return loss / probes

    def cert_penalty(self, z):
        q, p = z[:8, :self.m], z[:8, self.m:2 * self.m]
        gq, gp = self._grad(q, p)
        amp = (gq ** 2 + gp ** 2).mean()
        return (amp - 1.0) ** 2

    def regularizers(self, batch):
        out = {}
        if self.gauge and self.symplectic:
            out["gauge"] = self.gauge_penalty(batch["z0"])
        if self.cert:
            out["cert"] = self.cert_penalty(batch["z0"])
        return out

    def structure(self):
        return {"family": "PCL-WM", "latent": self.dz_, "m": self.m, "r": self.r,
                "symplectic": self.symplectic, "gauge": self.gauge,
                "free_dissipation": self.free_dissipation, "cert": self.cert}


def build(name, dy, du, L=4, hidden=128):
    table = {
        "MLP-MD": lambda: MLPDirect(dy, du, L, hidden),
        "NODE": lambda: NODEDirect(dy, du, L, hidden),
        "HNN": lambda: HNNDirect(dy, du, L, hidden),
        "GRU-LD": lambda: GRULatent(dy, du, L, 32, hidden),
        "PI-Lat": lambda: PILatent(dy, du, L, 32, hidden),
        "PCL-WM": lambda: PCLWM(dy, du, L, 32, hidden),
        "abl-no-symplectic": lambda: PCLWM(dy, du, L, 32, hidden, symplectic=False),
        "abl-no-gauge": lambda: PCLWM(dy, du, L, 32, hidden, gauge=False),
        "abl-free-dissipation": lambda: PCLWM(dy, du, L, 32, hidden, free_dissipation=True),
        "abl-no-cert": lambda: PCLWM(dy, du, L, 32, hidden, cert=False),
        "abl-no-curriculum": lambda: PCLWM(dy, du, L, 32, hidden),
        "abl-dz16": lambda: PCLWM(dy, du, L, 16, hidden),
        "abl-dz64": lambda: PCLWM(dy, du, L, 64, hidden),
    }
    if name not in table:
        raise ValueError(name)
    return table[name]()
