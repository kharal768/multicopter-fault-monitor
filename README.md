# Quadrotor fault monitor: hybrid digital twin with calibrated alarm thresholds

Code and per-flight results for the manuscript

> M. Imran, Z. He, F. S. Zaidi and K. P. Tran, "Detecting controller-compensated faults in real quadrotor flights with a hybrid digital twin and calibrated alarm thresholds" (submitted to *Mechanical Systems and Signal Processing*).

The monitor combines a hybrid digital twin (identified rigid-body physics layer, label-free per-flight calibration, learned correction with manoeuvre-dependent noise), physics-based consistency checks between redundant sensors and group-wise MEWMA charts whose alarm thresholds are calibrated on held-out healthy flight time. It is evaluated on the real flights of the public RflyMAD dataset.

All commands below are run from the repository root.

## Contents

| Item | Contents |
|---|---|
| `*.py` | Conversion, monitoring, statistics and figure scripts (listed under Files) |
| `data/` | Flight-day groups, chronological flight order and motor-fault severity of the real flights |
| `results/` | Per-flight result files behind the tables and figures of the paper |
| `RUNS.md` | The command and settings that produced each result folder |

## Data

The analyses use the RflyMAD dataset (Le et al., 2025; https://doi.org/10.1177/02783649241305153). The paper uses the Kaggle copy, https://www.kaggle.com/datasets/xianglile/rflymad, linked from the dataset page https://rfly-openha.github.io/documents/4_resources/dataset.html. The flight data are not included here.

`data/` contains:
- `real_groups.csv`: flight-day group of each real flight, used for the grouped cross-validation.
- `real_sessions.csv`: case identifier, flight date and source path of each real flight, used by the day-wise recalibration option.
- `real_prev.csv`: preceding flight of the same day, used by the previous-flight calibration option.
- `motor_severity.csv`: efficiency loss and number of affected motors for each motor-fault flight.

## Quick check without the flight data

```
python paper_stats.py results/main data/motor_severity.csv stats_out
```

`stats_out/table_main.csv` then reproduces the main results table of the paper (for example, 60.9% detection [56.5, 65.4] for the proposed monitor and 32.3% for the data-driven twin with consistency checks).

## Requirements

Python 3.13 with the packages in `requirements.txt` (the results were produced with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5 and scikit-learn 1.9.1):

```
pip install -r requirements.txt
```

## Steps

1. Convert the raw PX4 logs into one 20 Hz CSV per flight. The converter reads the extracted folders or the zip files directly:

   ```
   python convert_ulog.py <RflyMAD folder or zip folder> <flight folder>
   ```

2. Run one method on the real flights (five-fold cross-validation by flight; add `FOLD_GROUPS=data/real_groups.csv` for cross-validation grouped by flight day):

   ```
   DOMAINS=3 RESET_POLICY=1 python run_one.py <flight folder> <results folder> hybrid_plus_consistency hybrid+consistency
   ```

   Methods: `twin`, `twin_arx`, `consistency`, `twin+consistency`, `hybrid`, `hybrid+consistency`, the `hybrid_*` ablation variants defined in `twin_baseline.py`, and the squared-input variants `twin_sq` and `twin+consistency_sq`. The result file is `<name>_results.csv` in the results folder.

3. Per-flight MEWMA baseline on raw signals:

   ```
   python demo_mewma.py <flight folder> <results folder>
   ```

4. Statistics and tables:

   ```
   python paper_stats.py <results folder> data/motor_severity.csv <statistics folder>
   ```

5. Figures:

   ```
   python paper_figures.py <results folder> data/motor_severity.csv <figure folder>
   python fig_example.py <flight folder> <figure folder>/fig_example.pdf
   ```

6. False-alarm mechanism (needs the `hybrid` and `hybrid_nohetero` results): `python fa_mechanism.py <results folder> <flight folder> <output csv>`

7. Timing: `python timing.py <flight folder>`

## Options

Environment variables select the variants reported in the paper. The defaults are the paper's settings.

| Variable | Default | Purpose |
|---|---|---|
| `DOMAINS`, `RESET_POLICY` | 321, 1 | Domains to run (3 real, 2 HIL, 1 SIL); `DOMAINS=3` selects the real flights. `RESET_POLICY=1` is the reset rule |
| `FOLD_GROUPS` | unset | Grouped cross-validation by flight day (`data/real_groups.csv`) |
| `CAUSAL` | 0 | `1` uses trailing averages and backward differences (online use) |
| `THRESH` | window | `tukey_raw` or `tukey_mewma`: per-sample threshold rule of Erdogan et al. (2026) |
| `LEARNER` | ridge | `hgb` uses gradient boosting for the learned layer |
| `DAYRECAL`, `DAY_SESSIONS` | 0, `data/real_sessions.csv` | Day-wise recalibration from the first k flights of each day |
| `CAL_PREV` | unset | Previous-flight calibration (`data/real_prev.csv`) |
| `CAL_STRESS` | unset | `1` moves the calibration window of faulty flights to the first 10 s after onset |
| `LIMIT` | empirical | `gpd` uses a generalised Pareto tail model for small false-alarm levels |
| `RESCALE` | unset | `1` rescales residuals per flight from the calibration window |
| `TEST_SCALE` | 1.0 | Factor applied to the monitored statistic of test flights (residual-scaling test) |
| `NO_BONF`, `NO_BONF_GROUP` | unset | Remove the Bonferroni split over all groups or for one group |
| `TRAIN_FRAC`, `SUB_SEED` | 1.0, 0 | Train and calibrate on a fraction of the training flights |
| `FOLD_SEED` | 0 | Seed for the fold assignment |
| `N_ENS`, `CLIP_LO`, `CLIP_HI` | 10, 0.2, 50 | Ensemble size and noise bounds |
| `CAL_END`, `CAL_SHRINK` | 14, 50 | Calibration window end (s) and shrinkage of the affine calibration |
| `MEWMA_LAMBDA`, `RUN_LENGTH`, `WIN_S` | 0.1, 3, 60 | MEWMA weight, alarm run length and window length |

## Paper items and how to reproduce them

| Paper item | How | Output |
|---|---|---|
| Table 2 (data) | step 1 | counts printed by `convert_ulog.py` |
| Table 3 (main comparison) | step 2 for each method, then step 4 | `table_main.csv` |
| Table 4 (grouped by flight day) | step 2 with `FOLD_GROUPS`, then step 4 | `table_main.csv` |
| Table 5 (ablation) | step 2 with `hybrid_physonly`, `hybrid_cal`, `hybrid_nohetero`, `hybrid`, `hybrid+consistency`, `hybrid_nophys`, `hybrid_nophys_cal` and `LEARNER=hgb` variants | `<name>_results.csv` |
| Table 6 (uncertainty calibration) | `python uq_check.py <flight folder> <statistics folder>` | `uq_calibration.csv` |
| Table 7 (fault isolation) | step 4 | `table_isolation.csv` |
| Section 4.6 (causal filtering) | step 2 with `CAUSAL=1` | `<name>_results.csv` |
| Section 4.7 (per-sample threshold rule) | step 2 with `THRESH=tukey_raw` (`RUN_LENGTH=1` or `3`) or `THRESH=tukey_mewma` | `<name>_results.csv` |
| Supplementary Table 2 (squared inputs) | step 2 with `twin_sq` and `twin+consistency_sq` | `<name>_results.csv` |
| Supplementary Table 4 (sensitivity) | step 2 with `MEWMA_LAMBDA`, `WIN_S`, `RUN_LENGTH`, `CAL_END`, `CAL_SHRINK`, `CLIP_LO`/`CLIP_HI` or `N_ENS`, then step 4 | `<name>_results.csv` |
| Supplementary Section S7 (residual scaling) | step 2 with `TEST_SCALE` and `FOLD_GROUPS` | `<name>_results.csv` |
| Figures 2, 3 and 5 | step 4, then step 5 | `fig_detection_by_fault.pdf`, `fig_motor_strength.pdf`, `fig_operating_curve.pdf` |
| Figure 4 (thrust-model example) | `fig_example.py` | `fig_example.pdf` |
| Figure 1 (monitor structure) | drawn in the manuscript source | none |

The exact settings of every stored result are in `RUNS.md`.

## Files

- `convert_ulog.py`: conversion of PX4 logs to the analysis format.
- `rflymad.py`: loading and labelling of the converted flights.
- `consistency.py`: physics-based sensor-consistency residuals.
- `physics_twin.py`: physics layer, per-flight calibration and learned layer of the hybrid twin.
- `twin_baseline.py`: evaluation protocol, MEWMA charts, thresholds and alarm rules; defines the compared methods.
- `run_one.py`: runs one method on selected domains.
- `demo_mewma.py`: per-flight MEWMA baseline.
- `uq_check.py`: calibration of the twin's predictive uncertainty.
- `paper_stats.py`, `paper_figures.py`, `fa_mechanism.py`, `fig_example.py`, `timing.py`: tables, figures and timing.

## Citation

Software: Imran, M., He, Z., Zaidi, F. S., & Tran, K. P. (2026). *Quadrotor fault monitor: hybrid digital twin with calibrated alarm thresholds* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23423553

Paper: Imran, M., He, Z., Zaidi, F. S., & Tran, K. P. Detecting controller-compensated faults in real quadrotor flights with a hybrid digital twin and calibrated alarm thresholds. Submitted to *Mechanical Systems and Signal Processing*.
