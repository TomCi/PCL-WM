# -*- coding: utf-8 -*-
import io
import json
import glob
import numpy as np

out = io.open(r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\rul_audit.txt", "w",
              encoding="utf-8")
rul = np.loadtxt(r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\data_cmapss\RUL_FD002.txt")
out.write("true RUL: n=%d min=%.0f q25=%.0f med=%.0f q75=%.0f max=%.0f\n"
          % (rul.size, rul.min(), np.percentile(rul, 25), np.median(rul),
             np.percentile(rul, 75), rul.max()))
for p in sorted(glob.glob(r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\results_rul\*.json")):
    d = json.load(io.open(p, encoding="utf-8"))
    preds = np.array(d["rul_predictions"], float)
    frac_cap = float((preds >= 160).mean())
    out.write("%-30s n=%d capped=%.2f  pred min=%d med=%d max=%d  unique=%d  "
              "median|e|=%.1f rmse=%.1f w20=%.3f\n"
              % (p.split("\\")[-1], preds.size, frac_cap, preds.min(), np.median(preds),
                 preds.max(), len(np.unique(preds)), d["rul_median_abs"], d["rul_rmse"],
                 d["rul_within20"]))
out.close()
print("ok")
