# -*- coding: utf-8 -*-
"""LaTeX -> MathML -> OMML, so the displayed equations are native Word objects.

MathML comes from latex2mathml; the MathML -> OMML step uses the stylesheet that
ships with Office (MML2OMML.XSL), which is the same conversion Word itself uses.
"""
import io
import os
import re

from lxml import etree
import latex2mathml.converter

XSL_CANDIDATES = [
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL",
]

EQS = {
    1: r"z_k=(q_k,\ p_k,\ \xi_k)\in\mathbb{R}^{2m+r}",
    2: r"\dot{q}=\partial H/\partial p,\qquad \dot{p}=-\partial H/\partial q",
    3: r"p_{k+\frac{1}{2}}=p_k-\frac{h}{2}\nabla_q H(q_k,p_{k+\frac{1}{2}})",
    4: r"q_{k+1}=q_k+h\nabla_p H(q_k,p_{k+\frac{1}{2}}),\quad p_{k+1}=p_{k+\frac{1}{2}}-\frac{h}{2}\nabla_q H(q_{k+1},p_{k+\frac{1}{2}})",
    5: r"\dot{\xi}=(J_r-R)\nabla H_r(\xi)+G\,u",
    6: r"R=L\,L^{\mathsf{T}},\qquad R\succeq 0",
    7: r"J=\sum_{i=1}^{m}\left(e_i e_{m+i}^{\mathsf{T}}-e_{m+i} e_i^{\mathsf{T}}\right)",
    8: r"z_{k+1}=F_\theta(z_k,u_k)",
    9: r"\hat{o}_{k+1}=\hat{o}_k+D_\theta(z_{k+1})",
    10: r"\bar{q}=\frac{1}{B}\sum_{b=1}^{B}q^{(b)},\qquad \mathcal{L}_{\mathrm{gauge}}=\|\bar{q}\|_2^2",
    11: r"\mathcal{S}(\Phi)=\Phi^{\mathsf{T}}J\,\Phi-J,\qquad \mathcal{L}_{\mathrm{symp}}=\|\mathcal{S}(\Phi)\|_F^2",
    12: r"\mathcal{L}=\sum_{k=1}^{K}\gamma^{k}\,\|\hat{o}_{k}-o_{k}\|_2^{2}",
    13: r"e_{k+1}=(1+Lh)\,e_k+\delta_k",
    14: r"e_K=E_{\mathrm{enc}}+E_{\mathrm{dec}}+E_{H}+E_{\mathrm{int}}+E_{\mathrm{data}}",
    15: r"\frac{dH}{dt}=-(\nabla H)^{\mathsf{T}}R\,\nabla H+(\nabla H)^{\mathsf{T}}G\,u",
    16: r"N_{\mathrm{cert}}=\ln\left(1-\frac{\varepsilon(1-\rho)}{e_0}\right)/\ln\rho,\qquad \rho=1+Lh",
    17: r"\ln e(k)=\beta\ln k+c",
    18: r"\mathcal{J}=\sum_{i=0}^{H-1}\|y_i-y_i^{\mathrm{ref}}\|_Q^2+\lambda\|u_i\|_2^2",
}

_cache = {}


def _xsl_path():
    for p in XSL_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError("MML2OMML.XSL not found")


M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def _convert_one(tex):
    mathml = latex2mathml.converter.convert(tex)
    tree = etree.fromstring(mathml.encode("utf-8"))
    xslt = etree.parse(_xsl_path())
    transform = etree.XSLT(xslt)
    omml = transform(tree).getroot()
    if not omml.tag.endswith("oMath"):
        raise ValueError("stylesheet returned %s" % omml.tag)
    return omml


def _spacer(n=4):
    r = etree.SubElement(etree.Element(M_NS + "tmp"), M_NS + "r")
    t = etree.SubElement(r, M_NS + "t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = " " * n
    return r


def to_omml(tex):
    """Return the OMML (m:oMath) element for one LaTeX string, spacing included."""
    if tex in _cache:
        return _cache[tex]
    tex = tex.replace("\,", " ")
    parts = [x.strip() for x in re.split(r"\\qquad|\\quad", tex) if x.strip()]
    pieces = [_convert_one(x) for x in parts]
    root = etree.Element(M_NS + "oMath", nsmap={"m": M_NS[1:-1]})
    for i, piece in enumerate(pieces):
        if i:
            root.append(_spacer())
        for child in list(piece):
            root.append(child)
    _cache[tex] = root
    return root


def equation_element(num):
    el = to_omml(EQS[num])
    return etree.fromstring(etree.tostring(el))


if __name__ == "__main__":
    ok, bad = [], []
    for n in sorted(EQS):
        try:
            el = to_omml(EQS[n])
            xml = etree.tostring(el, encoding="unicode")
            ok.append((n, len(xml)))
            io.open("omml_check.txt", "a", encoding="utf-8").write("eq%d\n%s\n\n" % (n, xml[:400]))
        except Exception as exc:  # noqa: BLE001
            bad.append((n, repr(exc)[:160]))
    print("omml ok:", ok)
    if bad:
        print("FAILED:")
        for n, m in bad:
            print(" ", n, m)
