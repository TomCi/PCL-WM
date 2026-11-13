# -*- coding: utf-8 -*-
"""Aggregate the two seeds of the data-efficiency table."""
import json

MODELS = [("MLP-MD", "mlpmd"), ("PI-Lat", "pilat"), ("PCL-WM", "pclwm"),
          ("PCL-WM w/o symplectic", "ablnosymplectic")]
s0 = {r["model"]: r for r in json.load(open("data_eff.json", encoding="utf-8"))}
S0KEY = {"PCL-WM w/o symplectic": "abl-no-symplectic"}

out = []
for label, slug in MODELS:
    a = s0[S0KEY.get(label, label)]
    b = json.load(open("data_eff_%s_s1.json" % slug, encoding="utf-8"))
    row = {"model": label}
    for key in ("nrmse_25", "nrmse_100", "nrmse_200", "beta", "nrmse_100_ood", "nrmse_200_ood", "beta_ood"):
        v0, v1 = a[key], b[key]
        row[key] = (v0 + v1) / 2
        row[key + "_s0"], row[key + "_s1"] = v0, v1
    out.append(row)

for r in out:
    print("%-24s ID200 %.3f (%.3f-%.3f)  beta %.3f  OOD100 %.3f  OOD200 %.3f (%.3f-%.3f)  betaOD %.3f"
          % (r["model"], r["nrmse_200"], r["nrmse_200_s0"], r["nrmse_200_s1"], r["beta"],
             r["nrmse_100_ood"], r["nrmse_200_ood"], r["nrmse_200_ood_s0"], r["nrmse_200_ood_s1"],
             r["beta_ood"]))

abl = next(r for r in out if r["model"].startswith("PCL-WM w/o"))
pi = next(r for r in out if r["model"] == "PI-Lat")
mlp = next(r for r in out if r["model"] == "MLP-MD")
print("\nOOD gain of the structure: %.1f%% against PI-Lat, %.1f%% against MLP"
      % (100 * (1 - abl["nrmse_200_ood"] / pi["nrmse_200_ood"]),
         100 * (1 - abl["nrmse_200_ood"] / mlp["nrmse_200_ood"])))
print("ID difference: %.3f vs %.3f" % (pi["nrmse_200"], abl["nrmse_200"]))
json.dump(out, open("data_eff_means.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved data_eff_means.json")
