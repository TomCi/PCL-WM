# -*- coding: utf-8 -*-
"""Run the v2 experiment grid (partial observability, long-horizon training)."""
import json
import os
import time
import itertools

import train as trainer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")
os.makedirs(RES, exist_ok=True)

FAMILIES = ["MLP-MD", "GRU-LD", "NODE", "HNN", "PI-Lat", "PCL-WM"]
ABLATIONS = ["abl-no-symplectic", "abl-no-gauge", "abl-free-dissipation", "abl-no-cert"]
DOMS = ["D1", "D2", "D3", "D9"]
EPOCHS = 60
KMAX = 30

GRID = []
for dom, fam in itertools.product(DOMS, FAMILIES):
    GRID.append((dom, fam, 0, True, KMAX))
for ab in ABLATIONS:
    GRID.append(("D1", ab, 0, True, KMAX))
for fam in FAMILIES:
    GRID.append(("D1", fam, 1, True, KMAX))
GRID.append(("D1", "abl-no-curriculum", 0, False, KMAX))
GRID.append(("D1", "abl-dz16", 0, True, KMAX))
GRID.append(("D1", "abl-dz64", 0, True, KMAX))


def path_for(dom, model, seed):
    return os.path.join(RES, f"{dom}_{model}_s{seed}.json")


def run_one(dom, model, seed, curriculum, kmax):
    out = path_for(dom, model, seed)
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    t0 = time.time()
    m = trainer.train(dom, model, epochs=EPOCHS, seed=seed,
                      Kmax=kmax, curriculum=curriculum)
    m["wall_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return m


def main():
    for i, (dom, model, seed, cur, kmax) in enumerate(GRID, 1):
        print(f"[{i}/{len(GRID)}] {dom} {model} seed={seed} K={kmax}", flush=True)
        m = run_one(dom, model, seed, cur, kmax)
        print("    nrmse@25=%.4f @200=%.4f beta=%.3f one=%.4f (%.0fs)"
              % (m["nrmse_25"], m["nrmse_200"], m["beta"], m["one_step"], m["wall_s"]),
              flush=True)


if __name__ == "__main__":
    main()
