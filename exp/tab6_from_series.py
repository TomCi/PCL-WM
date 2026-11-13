# -*- coding: utf-8 -*-
"""Print the repeated-run summary used by Table 6 and Fig. 6."""
import json

rows = json.load(open(r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\exp\mpc_series_rep_D1.json", encoding="utf-8"))
rows.sort(key=lambda r: r["mpc_pos_rmse"])
for r in rows:
    rm = r["run_means"]
    print("%-24s mean %.4f  runs %s  curve_len %d  episodes %d"
          % (r["model"], r["mpc_pos_rmse"],
             " / ".join("%.4f" % v for v in rm),
             len(r["series"][0]), len(r["series"])))
