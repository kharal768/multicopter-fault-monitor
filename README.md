# Quadrotor fault monitor: hybrid digital twin with calibrated alarm thresholds

Code and results for the manuscript

> M. Imran, Z. He, F. S. Zaidi and K. P. Tran, "Detecting controller-compensated faults in real quadrotor flights with a hybrid digital twin and calibrated alarm thresholds" (submitted to *Mechanical Systems and Signal Processing*).

The monitor combines a hybrid digital twin (identified rigid-body physics layer, label-free per-flight calibration, learned correction with manoeuvre-dependent noise), physics-based consistency checks between redundant sensors and group-wise MEWMA charts whose alarm thresholds are calibrated on held-out healthy flight time. It is evaluated on the real flights of the public RflyMAD dataset.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Conversion, monitoring, statistics and figure scripts; see `code/README.md` for step-by-step use |
| `data/` | Flight-day groups, chronological flight order and motor-fault severity for the real flights |
| `results/` | Per-flight result files behind every table and figure in the paper |
| `RUNS.md` | The command and settings that produced each result folder |

The flight data are not included. Download RflyMAD from its authors (https://rfly-openha.github.io/documents/4_resources/dataset.html) or the Kaggle copy used in the paper (https://www.kaggle.com/datasets/xianglile/rflymad), and cite its data paper: Le et al. (2025), https://doi.org/10.1177/02783649241305153.

## Quick check

The statistics of the main comparison can be recomputed from the included results without the flight data:

```
cd code
python paper_stats.py ../results/main ../data/motor_severity.csv ../stats_out
```

`stats_out/table_main.csv` then reproduces the main results table (for example, 60.9% detection [56.5, 65.4] for the proposed monitor and 32.3% for the data-driven twin with consistency checks).

## Options used in the paper

All runs use `DOMAINS=3` (real flights) and `RESET_POLICY=1` (reset rule). Further environment variables select the variants reported in the paper, including `FOLD_GROUPS` (day-grouped cross-validation), `CAUSAL=1` (trailing averages and backward differences for online use), `THRESH=tukey_raw|tukey_mewma` (per-sample threshold rule of Erdogan et al., 2026) and `LEARNER=hgb` (gradient-boosting learned layer). `RUNS.md` lists every command.

## Environment

The results were produced with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5 and scikit-learn 1.9.1 (`code/requirements.txt`).
