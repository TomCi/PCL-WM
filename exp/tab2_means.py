# -*- coding: utf-8 -*-
"""Two-seed means for the main comparison table and the integrator ablation across all domains."""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")

FAM = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM"]
DOMS = ["D1", "D2", "D3", "D9"]


def load(dom, model, seed):
    p = os.path.join(RES, "%s_%s_s%d.json" % (dom, model, seed))
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def mean_of(dom, model, key):
    r = [load(dom, model, s) for s in (0, 1)]
    r = [x for x in r if x]
    if not r:
        return None, None, None
    v = [x[key] for x in r]
    return statistics.mean(v), min(v), max(v)


print("=== main table: two-seed means (200-step error) ===")
for m in FAM:
    cells = []
    for d in DOMS:
        mu, lo, hi = mean_of(d, m, "nrmse_200")
        cells.append("%s %.3f [%.3f-%.3f]" % (d, mu, lo, hi))
    print("%-8s %s" % (m, "  ".join(cells)))

print()
print("=== main table: other columns, D1 and latencies ===")
for m in FAM:
    mu25, _, _ = mean_of("D1", m, "nrmse_25")
    mu100, _, _ = mean_of("D1", m, "nrmse_100")
    b1, l1, h1 = mean_of("D1", m, "beta")
    b2, l2, h2 = mean_of("D2", m, "beta")
    b3, l3, h3 = mean_of("D3", m, "beta")
    b9, l9, h9 = mean_of("D9", m, "beta")
    one, _, _ = mean_of("D1", m, "one_step")
    lat = load("D1", m, 0)["latency_ms"]
    par = load("D1", m, 0)["params"]
    print("%-8s @25 %.3f @100 %.3f beta1 %.2f[%.2f-%.2f] beta2 %.2f[%.2f-%.2f] beta3 %.2f beta9 %.2f one %.4f lat %.1f par %s"
          % (m, mu25, mu100, b1, l1, h1, b2, l2, h2, b3, b9, one, lat, "{:,}".format(par)))

print()
print("=== integrator ablation across domains (200-step) ===")
for d in DOMS:
    w, lo, hi = mean_of(d, "PCL-WM", "nrmse_200")
    wo, lo2, hi2 = mean_of(d, "abl-no-symplectic", "nrmse_200")
    print("%s  with %.3f [%.3f-%.3f]  without %.3f [%.3f-%.3f]  ratio %.2f"
          % (d, w, lo, hi, wo, lo2, hi2, w / wo))
