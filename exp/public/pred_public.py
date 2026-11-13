# -*- coding: utf-8 -*-
"""Roll out two trained models on held-out C-MAPSS engines and save the traces for Fig. 7."""
import json
import os

import numpy as np
import torch

import public_domain
import train

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_traces.json")
H = 100
T0 = 50

dom = public_domain.make()
d = train.build_data(dom, 68, 16, dom.steps, 0)
res = {"t0": T0, "H": H, "obs_idx": dom.obs_idx,
       "channels": ["T24", "T30", "T50", "Nf", "Ps30", "htBleed"]}

for name in ("PI-Lat", "PCL-WM"):
    _, model = train.train("D5", name, epochs=60, seed=0, n_train=68, steps=dom.steps, L=4,
                           Kmax=30, horizons=(25, 50, 100), dom=dom, return_model=True)
    model.eval()
    preds = []
    for e in range(4):
        w = d["yte"][e, T0 - 4 + 1:T0 + 1].unsqueeze(0)
        prev = d["yte"][e, T0]
        out = []
        with torch.enable_grad():
            z = model.encode(w)
            for _ in range(H):
                z = model.step(z, torch.zeros(1, 0, device=d["yte"].device,
                                               dtype=d["yte"].dtype)).detach()
                prev = prev + model.decode(z)[0].detach()
                out.append(prev.cpu().numpy())
        preds.append(np.stack(out))
    res[name] = np.stack(preds).tolist()

res["truth"] = d["yte"][:4, T0 + 1:T0 + 1 + H].cpu().numpy().tolist()
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f)
print("traces written", OUT)
