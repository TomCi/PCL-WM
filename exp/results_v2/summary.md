# PCL-WM experiments, v2 (partial observability)

Runs: 68 configurations; 60 epochs; K curriculum up to 30; window L=4; measured channels only.

## Table A. Accuracy, error growth and cost

| Model | D1@25 | D1@200 | D2@200 | D3@200 | D9@200 | $\beta$(D1) | $\beta$(D2) | latency (ms) | params |
|---|---|---|---|---|---|---|---|---|---|
| MLP-MD | 0.05 | 0.54 | 0.38 | 0.01 | 0.22 | 1.10 | 0.84 | 0.20 | 37382 |
| GRU-LD | 0.05 | 0.41 | 0.50 | 0.37 | 0.44 | 1.05 | 1.06 | 0.23 | 31974 |
| NODE | 0.05 | 0.57 | 0.38 | 0.01 | 0.22 | 1.17 | 0.83 | 0.97 | 37383 |
| HNN | 0.16 | 1.46 | 2.48 | 3.14 | 4.32 | 1.06 | 1.00 | 1.30 | 34059 |
| PI-Lat | 0.05 | 0.35 | 0.48 | 0.18 | 0.44 | 0.96 | 1.15 | 0.24 | 70598 |
| PCL-WM | 0.05 | 0.79 | 1.60 | 1.02 | 3.27 | 1.33 | 1.74 | 2.12 | 102911 |

## Table B. Ablations (D1)

| Configuration | NRMSE@25 | NRMSE@100 | NRMSE@200 | $\beta$ | one-step | params |
|---|---|---|---|---|---|---|
| PCL-WM | 0.046 | 0.167 | 0.793 | 1.33 | 0.0130 | 102911 |
| abl-no-symplectic | 0.045 | 0.147 | 0.343 | 0.96 | 0.0130 | 102911 |
| abl-no-gauge | 0.046 | 0.169 | 0.992 | 1.41 | 0.0130 | 102911 |
| abl-free-dissipation | 0.046 | 0.172 | 0.969 | 1.41 | 0.0130 | 102911 |
| abl-no-cert | 0.046 | 0.172 | 0.717 | 1.29 | 0.0130 | 102911 |
| abl-no-curriculum | 0.046 | 0.204 | 0.950 | 1.42 | 0.0130 | 102911 |
| abl-dz16 | 0.046 | 0.256 | 1.491 | 1.54 | 0.0130 | 94927 |
| abl-dz64 | 0.047 | 0.166 | 0.794 | 1.31 | 0.0130 | 119647 |

## Table C. Closed-loop MPC on D1 (position RMSE, normalised)

| Model | MPC RMSE |
|---|---|
| MLP-MD | 0.057 |
| PCL-WM | 0.068 |
