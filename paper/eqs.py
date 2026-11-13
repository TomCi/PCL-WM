# -*- coding: utf-8 -*-
"""Render every displayed equation of the manuscript to a PNG (mathtext, no LaTeX)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EQD = os.path.join(HERE, "eqs")
os.makedirs(EQD, exist_ok=True)

plt.rcParams.update({"mathtext.fontset": "dejavuserif", "font.size": 10})

# Mathtext (matplotlib) supports a reduced command set; inequalities are written
# with the unicode glyphs that DejaVu Serif actually ships with.
EQS = {
    1: r"$z_k=(q_k,\ p_k,\ \xi_k)\in\mathbb{R}^{2m+r}$",
    2: r"$\dot{q}=\partial H/\partial p,\qquad \dot{p}=-\partial H/\partial q$",
    3: r"$p_{k+\frac{1}{2}}=p_k-\frac{h}{2}\nabla_q H(q_k,p_{k+\frac{1}{2}})$",
    4: r"$q_{k+1}=q_k+h\nabla_p H(q_k,p_{k+\frac{1}{2}}),\quad p_{k+1}=p_{k+\frac{1}{2}}-\frac{h}{2}\nabla_q H(q_{k+1},p_{k+\frac{1}{2}})$",
    5: r"$\dot{\xi}=(J_r-R)\nabla H_r(\xi)+G\,u$",
    6: r"$R=L\,L^{\top},\qquad R\succeq 0$",
    7: r"$J=\sum_{i=1}^{m}\left(e_i e_{m+i}^{\top}-e_{m+i} e_i^{\top}\right)$",
    8: r"$z_{k+1}=F_\theta(z_k,u_k)$",
    9: r"$\hat{o}_{k+1}=\hat{o}_k+D_\theta(z_{k+1})$",
    10: r"$\bar{q}=\frac{1}{B}\sum_{b=1}^{B}q^{(b)},\qquad \mathcal{L}_{\mathrm{gauge}}=\Vert\bar{q}\Vert_2^2$",
    11: r"$\mathcal{S}(\Phi)=\Phi^{\top}J\,\Phi-J,\qquad \mathcal{L}_{\mathrm{symp}}=\Vert\mathcal{S}(\Phi)\Vert_F^2$",
    12: r"$\mathcal{L}=\sum_{k=1}^{K}\gamma^{k}\,\Vert\hat{o}_{k}-o_{k}\Vert_2^{2}$",
    13: r"$e_{k+1}=(1+Lh)\,e_k+\delta_k$",
    14: r"$e_K=E_{\mathrm{enc}}+E_{\mathrm{dec}}+E_{H}+E_{\mathrm{int}}+E_{\mathrm{data}}$",
    15: r"$\frac{dH}{dt}=-(\nabla H)^{\top}R\,\nabla H+(\nabla H)^{\top}G\,u$",
    16: r"$N_{\mathrm{cert}}=\ln\left(1-\frac{\varepsilon(1-\rho)}{e_0}\right)/\ln\rho,\qquad \rho=1+Lh$",
    17: r"$\ln e(k)=\beta\ln k+c$",
    18: r"$\mathcal{J}=\sum_{i=0}^{H-1}\Vert y_i-y_i^{\mathrm{ref}}\Vert_Q^2+\lambda\Vert u_i\Vert_2^2$",
}


def render(num, tex, fs=11):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, tex, fontsize=fs)
    fig.savefig(os.path.join(EQD, "eq%02d.png" % num), dpi=400,
                bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return t


def main():
    ok, bad = [], []
    for n, tex in EQS.items():
        try:
            render(n, tex)
            ok.append(n)
        except Exception as exc:  # noqa: BLE001
            bad.append((n, str(exc)[:120]))
    print("rendered:", ok)
    if bad:
        print("FAILED:")
        for n, msg in bad:
            print(" ", n, msg)


if __name__ == "__main__":
    main()
