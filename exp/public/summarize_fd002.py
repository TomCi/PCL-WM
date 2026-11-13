# -*- coding: utf-8 -*-
"""Summarise the FD002 runs (7 configurations x 2 seeds) into one csv."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "results_public2", "*.json"))):
    r = json.load(open(path, encoding="utf-8"))
    rows.append(r)

out = os.path.join(HERE, "results_public2_summary.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write("model,seed,nrmse_25,nrmse_50,nrmse_100,beta,one_step,final_loss,n_train\n")
    for r in sorted(rows, key=lambda x: (x["model"], x["seed"])):
        f.write("%s,%d,%.4f,%.4f,%.4f,%.3f,%.4f,%.3f,%d\n" % (
            r["model"], r["seed"], r["nrmse_25"], r["nrmse_50"], r["nrmse_100"],
            r["beta"], r["one_step"], r["final_loss"], r["n_train"]))
print("wrote %s with %d runs" % (out, len(rows)))

by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)
print("\nmodel                  @25      @50      @100     beta")
for m, rs in sorted(by_model.items()):
    mean = lambda k: sum(x[k] for x in rs) / len(rs)
    print("%-20s %.3f    %.3f    %.3f    %+.3f" % (m, mean("nrmse_25"), mean("nrmse_50"),
                                                  mean("nrmse_100"), mean("beta")))
