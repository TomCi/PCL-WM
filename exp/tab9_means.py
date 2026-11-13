# -*- coding: utf-8 -*-
"""Extra numbers needed by the text: @100 and beta means for the integrator ablation, plus latency."""
import json
import os
import statistics

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")


def mu(dom, model, key):
    v = []
    for s in (0, 1):
        p = os.path.join(RES, "%s_%s_s%d.json" % (dom, model, s))
        if os.path.exists(p):
            v.append(json.load(open(p, encoding="utf-8"))[key])
    return statistics.mean(v) if v else float("nan")


for d in ["D1", "D2", "D3", "D9"]:
    print(d, "nosym @25 %.3f @100 %.3f @200 %.3f beta %.2f | pcl beta %.2f"
          % (mu(d, "abl-no-symplectic", "nrmse_25"), mu(d, "abl-no-symplectic", "nrmse_100"),
             mu(d, "abl-no-symplectic", "nrmse_200"), mu(d, "abl-no-symplectic", "beta"),
             mu(d, "PCL-WM", "beta")))
for m in ["PCL-WM", "PI-Lat", "MLP-MD"]:
    print(m, "latency s0", json.load(open(os.path.join(RES, "D1_%s_s0.json" % m), encoding="utf-8"))["latency_ms"])
print("D2 best learned family vs PCL:", mu("D2", "MLP-MD", "nrmse_200"), mu("D2", "PCL-WM", "nrmse_200"))
print("D9 ratio", mu("D9", "abl-no-symplectic", "nrmse_200") / mu("D9", "PCL-WM", "nrmse_200"))
