# How the results were produced

All runs use the reset rule (RESET_POLICY=1, the default) and the real RflyMAD flights (DOMAINS=3), with five-fold cross-validation by flight unless FOLD_GROUPS is set (grouped by flight day, data/real_groups.csv).

Example command (from the code folder):

    env DOMAINS=3 RESET_POLICY=1 python run_one.py <flight CSV folder> <output folder> <name> hybrid+consistency

| Result folder | Settings (environment variables) |
|---|---|
| results/main | Random five-fold, all configurations (the main tables) |
| results/real_random_extra | Ablations and learners: hybrid_physonly, hybrid_cal, hybrid_nohetero, hybrid_nophys(+cal), hgb learners, seeds (FOLD_SEED=1,2) |
| results/grouped | FOLD_GROUPS=data/real_groups.csv, day-grouped validation |
| results/sensitivity | MEWMA_LAMBDA, RUN_LENGTH, WIN_S, N_ENS, CLIP_LO/CLIP_HI |
| results/round2_previous_flight_stress | CAL_PREV=data/real_prev.csv (previous-flight calibration), CAL_STRESS=1, CAL_SHRINK |
| results/final_reruns | CAL_END=10, LIMIT=gpd, RESCALE=1, DAYRECAL=1/2/3, NO_BONF_GROUP=twin, LEARNER=hgb |

Flight CSVs are not included. Rebuild them from the Flights20Hz1-4 archives with convert_ulog.py, into a folder named allflights.

## Day-grouped gradient boosting with physics

Result file: results/final_reruns/grp_hgb_hybrid_plus_consistency_results.csv (real flights, reset rule, 437 faults after excluding the five cases listed in the paper, 51 fault-free flights).

    env DOMAINS=3 RESET_POLICY=1 LEARNER=hgb FOLD_GROUPS=data/real_groups.csv python run_one.py <folder of Case_3*.csv> <output folder> grp_hgb_hybrid_plus_consistency hybrid+consistency

The same command without FOLD_GROUPS gives the random-fold figures in results/real_random_extra/hgb_hybrid_plus_consistency_results.csv. The grouped result for gradient boosting without physics is results/final_reruns/grp_hgb_nophys_pc_results.csv.

## Day-wise recalibration (DAYRECAL=1,2,3), corrected scoring
Command (per k): env DOMAINS=3 RESET_POLICY=1 DAYRECAL=k FOLD_GROUPS=data/real_groups.csv DAY_SESSIONS=data/real_sessions.csv python run_one.py <flight folder> <output folder> dayrecal<k>_fixed_results hybrid+consistency
Output: results/final_reruns/dayrecal<k>_fixed_results.csv. Calibration flights are scored by a twin refitted without them.
Compared with results/grouped/hybrid_plus_consistency_results.csv on the same flights (excluding the 5 cases). Each day's limit is set by 1, 2 or 3 calibration flights (median 1, 2 and 3 windows per group).

## Data efficiency (TRAIN_FRAC = 0.25, 0.10; random 5-fold, one draw SUB_SEED=0)
Command: env DOMAINS=3 RESET_POLICY=1 TRAIN_FRAC=f [LEARNER=hgb] python run_one.py <flight folder> <output folder> <name> <mode>
  ridge monitor: mode hybrid+consistency; gradient boosting without physics: LEARNER=hgb, mode hybrid_nophys_plus_consistency.
Output: results/final_reruns/sub{25,10}_{ridge,hgb_nophys}_results.csv. Fit and calibration use only the sampled training flights of each fold.

## ARX twin baseline (Table 3)
Command: env DOMAINS=3 RESET_POLICY=1 python run_one.py <flight folder> <output folder> twin_arx twin_arx
Output: results/main/twin_arx_results.csv (random five-fold, reset rule).

## Causal filters for online use (Section "Causal filtering for online use")
CAUSAL=1 replaces the centred moving averages by trailing averages and the central differences by backward differences in physics_twin.py and consistency.py; the twin is refitted and the thresholds recalibrated.
Commands:
  env DOMAINS=3 RESET_POLICY=1 CAUSAL=1 python run_one.py <flight folder> <output folder> causal_hybrid_plus_consistency hybrid+consistency
  env DOMAINS=3 RESET_POLICY=1 CAUSAL=1 python run_one.py <flight folder> <output folder> causal_twin_plus_consistency twin+consistency
Output: results/final_reruns/causal_hybrid_plus_consistency_results.csv and causal_twin_plus_consistency_results.csv.

## Per-sample threshold rule of Erdogan et al. (2026) (Section "Comparison with a per-sample threshold rule")
THRESH=tukey_raw: one common threshold, the largest absolute boxplot adjacent value of the standardised residuals over channels (out-of-sample healthy calibration data); RUN_LENGTH=1 (single exceedance) or 3. THRESH=tukey_mewma: per-group upper adjacent value of the MEWMA statistic. Default THRESH=window is the paper's calibration.
Commands:
  env DOMAINS=3 RESET_POLICY=1 THRESH=tukey_raw RUN_LENGTH=1 python run_one.py <flight folder> <output folder> erd_raw_run1 hybrid+consistency
  env DOMAINS=3 RESET_POLICY=1 THRESH=tukey_raw RUN_LENGTH=3 python run_one.py <flight folder> <output folder> erd_raw_run3 hybrid+consistency
  env DOMAINS=3 RESET_POLICY=1 THRESH=tukey_mewma python run_one.py <flight folder> <output folder> erd_mewma hybrid+consistency
Output: results/final_reruns/erd_raw_run1_results.csv, erd_raw_run3_results.csv, erd_mewma_results.csv. Run the tukey_raw variants one at a time (they hold all calibration residuals in memory).
