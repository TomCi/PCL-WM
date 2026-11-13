# PCL-WM: structured latent world models for engineering control

Code and aggregated results for

> Which physics priors actually help? A component-wise evaluation of structured latent world models
> for engineering control. Submitted to *Engineering Applications of Artificial Intelligence*.

Version 1.0, fixed for the submission. Every number in the manuscript comes out of the files here.

## Layout

```
exp/                 simulation domains, models, training loop, filters, MPC, aggregation
exp/public/          C-MAPSS pipeline (FD001 as reported, FD002/FD004 as checked and not reported)
exp/diagnostics/     scripts that produce the dataset-precondition numbers quoted in Section 4.1 and 5.9
exp/results_*/       aggregated json, one file per (model, seed) or per (grid, configuration)
paper/               manuscript source: build.py assembles the DOCX, figs.py draws Figures 1, 6 and 7
data/README.md       where to get the C-MAPSS records and the checksums they were read with
```

## Environment

Python 3.11 with `pip install -r requirements.txt`. The reported runs used torch 2.11.0 with CUDA 12.8 on
one RTX 5090 laptop GPU; every script falls back to CPU, only the wall-clock changes. Seeds are explicit
arguments, never global state.

## Reproducing the manuscript

Run from `exp/`. One entry point per artefact; each writes json into its own results directory and
never overwrites a finished run.

| Artefact | Command |
|---|---|
| Table 2, Table 3 (simulated domains) | `python run_grid_v2.py` then `python tab2_means.py`, `python tab4_means.py` |
| Table 4 (ablation) | `python run_grid_v2.py --abl` then `python tab4_means.py` |
| Table 5, Table 6 (data efficiency, closed loop) | `python data_eff.py`, `python mpc_rep.py`, `python tab5_means.py`, `python tab6_from_series.py` |
| Table 7 (invariants) | `python tab7_means.py` |
| Table 10, Table 11 (fleet, filters) | `python public/run_public.py`, `python ekf_ukf.py` |
| Figures 1, 6, 7 | `cd ../paper && python figs.py` |
| Figures 2 to 5 | `python analyze_v2.py`, `python fig_ablation_2seed.py`, `python fig_data_eff.py` |
| Graphical abstract | `cd ../paper && python graphical_abstract.py` |
| The DOCX itself | `cd ../paper && python build.py` |

`python draw_all.py` draws Figures 2 to 7 in the one order that keeps the two-seed ablation panel from
being overwritten by the single-seed one.

## What is not reproducible from this tree, and why

* **FD002 and FD004 are scored only on the degradation target of Section 5.9** (`exp/results_public4/`,
  two seeds per configuration, plus `exp/run_public_drift.py` and `exp/summarize_drift.py`). A
  reconstruction score is not reported on them: the sensors are a step function of the operating point,
  which changes almost every cycle, so 205 of 205 windows contain a switch, adjacent cycles share the
  operating point in 0.18 of cases, and the one-step error of the last-value predictor is sqrt(2). The
  per-condition-normalised grid that was run as the repair is in `exp/results_public3/`, and
  `exp/diagnostics/` reproduces every number in both paragraphs.
* **A standard remaining-life score on FD002 was attempted and is not reported as a result**
  (`exp/results_rul/`, `exp/public/run_public_rul.py`, `exp/public/rul_rate.py`). The rollout is turned
  into a remaining life by the cycles it needs to cross the training-end health threshold, but the rate
  it implies is off by a median factor of 2 to 22 against each record's own rate, so the construction
  measures the threshold rather than the model. The section keeps the accuracy-bound statement instead.
* The C-MAPSS records are downloaded, not bundled: see `data/README.md`.
* Monte Carlo seeds beyond the two reported per configuration are not included as files; the
  four-seed median quoted in Section 5.8 comes from `exp/results_public/`.

## License

MIT for the code in this archive. The C-MAPSS data keep their own terms and are not redistributed.
