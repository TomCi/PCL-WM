# -*- coding: utf-8 -*-
"""Simulated engineering domains for the PCL-WM experiments.

Each domain exposes dx, du, sample(n_ep, steps, seed), energy(states), obs(states).
Plants are integrated with classical RK4 at a fixed internal step.
"""
import numpy as np

DT = 0.02


class _Base:
    """Sensor model: only obs_idx channels are measured; velocities stay hidden."""

    def obs(self, s, noise=0.0, rng=None):
        o = s[..., self.obs_idx].copy()
        if noise <= 0.0:
            return o
        return o + rng.normal(0, noise, size=o.shape) * np.std(o, axis=(0, 1))

    @property
    def dy(self):
        return len(self.obs_idx)


class Projectile(_Base):
    """6-DoF guided projectile: position, velocity, attitude, body rates (12), 3 fins."""

    name = "D1"
    dx, du = 12, 3
    obs_idx = [0, 1, 2, 6, 7, 8]
    m, g = 8.0, 9.81

    def f(self, x, u):
        v, ang, om = x[3:6], x[6:9], x[9:12]
        speed = np.linalg.norm(v) + 1e-6
        rho, S, Cd = 1.2, 0.018, 0.22
        drag = -0.5 * rho * speed * S * Cd * v / self.m
        qd = 0.5 * rho * speed * speed * S
        lift = qd * 0.35 * u * 0.5 / self.m
        grav = np.array([0.0, 0.0, -self.g])
        dang = np.array([om[0] * np.cos(ang[2]) - om[1] * np.sin(ang[2]),
                         om[0] * np.sin(ang[2]) + om[1] * np.cos(ang[2]),
                         om[2]])
        dome = np.array([0.02 * u[0] - 0.35 * om[0],
                         0.02 * u[1] - 0.35 * om[1],
                         0.01 * u[2] - 0.30 * om[2]])
        return np.concatenate([v, drag + lift + grav, dang, dome])

    def sample(self, n_ep, steps, seed, vlo=180.0, vhi=260.0):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_ep, steps + 1, self.dx))
        U = rng.normal(0, 1, size=(n_ep, steps, self.du)) * 0.35
        for e in range(n_ep):
            v0, th = rng.uniform(vlo, vhi), rng.uniform(0.6, 0.9)
            x = np.array([0.0, 0.0, 0.0, v0 * np.cos(th), 0.0, v0 * np.sin(th),
                          0.05, 0.02, 0.03, 0.0, 0.0, 0.0])
            X[e, 0] = x
            for k in range(steps):
                u = U[e, k]
                k1 = self.f(x, u); k2 = self.f(x + 0.5 * DT * k1, u)
                k3 = self.f(x + 0.5 * DT * k2, u); k4 = self.f(x + DT * k3, u)
                x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                X[e, k + 1] = x
        return X, U

    def energy(self, s):
        v, z = s[..., 3:6], s[..., 2]
        return 0.5 * self.m * np.sum(v * v, axis=-1) + self.m * self.g * z


class Quadrotor(_Base):
    """Planar quadrotor: pos3, vel3, quat4, rates3, rotor4 = 17 states, 4 controls."""

    name = "D2"
    dx, du = 17, 4
    obs_idx = [0, 1, 2, 6, 7, 8, 9]
    m, g, L = 1.2, 9.81, 0.25
    J = np.array([0.02, 0.02, 0.04])

    def f(self, x, u):
        v, q, om, rot = x[3:6], x[6:10], x[10:13], x[13:17]
        rot_new = rot + DT * (6.0 * u - 6.0 * rot)
        thrust = 2.2 * rot_new ** 2
        F = np.zeros(3)
        F[2] = thrust.sum() / self.m - self.g
        F[0] = (thrust[1] + thrust[2] - thrust[0] - thrust[3]) * 0.06 / self.m
        F[1] = (thrust[2] + thrust[0] - thrust[1] - thrust[3]) * 0.06 / self.m
        tau = np.array([(thrust[2] + thrust[3] - thrust[0] - thrust[1]) * self.L * 0.5,
                        (thrust[0] + thrust[2] - thrust[1] - thrust[3]) * self.L * 0.5,
                        (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * 0.04])
        tauc = np.array([(thrust[1] + thrust[2] - thrust[0] - thrust[3]) * 0.02, 0.0, 0.0])
        dome = (tau - tauc) / self.J - 0.6 * om / self.J
        wx, wy, wz = om
        dq = 0.5 * np.array([-q[1] * wx - q[2] * wy - q[3] * wz,
                             q[0] * wx + q[2] * wz - q[3] * wy,
                             q[0] * wy - q[1] * wz + q[3] * wx,
                             q[0] * wz + q[1] * wy - q[2] * wx])
        qn = q + DT * dq
        qn = qn / (np.linalg.norm(qn) + 1e-9)
        return np.concatenate([v, F, qn, dome, (rot_new - rot) / DT])

    def sample(self, n_ep, steps, seed):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_ep, steps + 1, self.dx))
        U = rng.normal(1.15, 0.06, size=(n_ep, steps, self.du))
        for e in range(n_ep):
            x = np.array([0.0, 0.0, 1.0 + rng.normal(0, 0.08),
                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 1.15, 1.15, 1.15, 1.15])
            X[e, 0] = x
            for k in range(steps):
                u = U[e, k]
                k1 = self.f(x, u); k2 = self.f(x + 0.5 * DT * k1, u)
                k3 = self.f(x + 0.5 * DT * k2, u); k4 = self.f(x + DT * k3, u)
                x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                X[e, k + 1] = x
        return X, U

    def energy(self, s):
        v, z = s[..., 3:6], s[..., 2]
        return 0.5 * self.m * np.sum(v * v, axis=-1) + self.m * self.g * z


class Thermal(_Base):
    """Eight-node lumped RC network with three heat inputs."""

    name = "D3"
    dx, du = 8, 3
    obs_idx = list(range(8))
    C = np.array([12.0, 8.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0])
    G = 2.2

    def f(self, x, u):
        d = np.zeros(self.dx)
        for i in range(self.dx):
            nb = [(i - 1) % self.dx, (i + 1) % self.dx]
            d[i] = sum(self.G * (x[j] - x[i]) for j in nb) * 0.5 / self.C[i]
        d[0] += 1.5 * u[0] / self.C[0]
        d[3] += 1.5 * u[1] / self.C[3]
        d[6] += 1.5 * u[2] / self.C[6]
        d -= 0.02 * (x - 300.0) / self.C
        return d

    def sample(self, n_ep, steps, seed):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_ep, steps + 1, self.dx))
        U = rng.normal(0.0, 1.0, size=(n_ep, steps, self.du)) * 0.5
        for e in range(n_ep):
            x = 300.0 + rng.normal(0, 6.0, size=self.dx)
            X[e, 0] = x
            for k in range(steps):
                u = U[e, k]
                k1 = self.f(x, u); k2 = self.f(x + 0.5 * DT * k1, u)
                k3 = self.f(x + 0.5 * DT * k2, u); k4 = self.f(x + DT * k3, u)
                x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                X[e, k + 1] = x
        return X, U

    def energy(self, s):
        return 0.5 * np.sum(self.C * (s - 300.0) ** 2, axis=-1)


class ReactionDiffusion(_Base):
    """1-D reaction-diffusion field, 24 cells x 3 species, point actuation."""

    name = "D9"
    N, nsp = 24, 3
    dx, du = N * 3, 2
    obs_idx = [c * 3 + s for c in range(0, 24, 3) for s in range(3)]
    D = np.array([0.28, 0.16, 0.10])

    def f(self, x, u):
        fld = x.reshape(self.nsp, self.N)
        lap = np.roll(fld, 1, axis=1) - 2 * fld + np.roll(fld, -1, axis=1)
        react = np.stack([-fld[0] * fld[1] + 0.02,
                          fld[0] * fld[1] - fld[1] * fld[2] - 0.01,
                          fld[1] * fld[2] - 0.02 * fld[2]])
        d = self.D[:, None] * lap + react
        d[0, 3:7] += 4.0 * u[0]
        d[2, 17:21] += 4.0 * u[1]
        return d.reshape(-1)

    def sample(self, n_ep, steps, seed):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_ep, steps + 1, self.dx))
        U = rng.normal(0.0, 1.0, size=(n_ep, steps, self.du)) * 0.6
        for e in range(n_ep):
            base = np.stack([1.0 + 0.25 * rng.normal(size=self.N),
                             0.9 + 0.20 * rng.normal(size=self.N),
                             0.4 + 0.15 * rng.normal(size=self.N)]).reshape(-1)
            x = np.clip(base, 0.02, None)
            X[e, 0] = x
            for k in range(steps):
                u = U[e, k]
                k1 = self.f(x, u); k2 = self.f(x + 0.5 * DT * k1, u)
                k3 = self.f(x + 0.5 * DT * k2, u); k4 = self.f(x + DT * k3, u)
                x = np.clip(x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, None)
                X[e, k + 1] = x
        return X, U

    def energy(self, s):
        fld = s.reshape(*s.shape[:-1], self.nsp, self.N)
        return np.sum(fld ** 2, axis=(-1, -2))


DOMAINS = {"D1": Projectile, "D2": Quadrotor, "D3": Thermal, "D9": ReactionDiffusion}


def make(domain):
    return DOMAINS[domain]()
