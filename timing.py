"""Measures the run time of the hybrid twin, its physics layer, the consistency checks and the
learned layer on one held-out fold of the real flights. The learned-layer fit uses random targets:
only its run time is measured here.

Usage: python timing.py <ProcessData folder>
"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_baseline as tb
import physics_twin as pt
from rflymad import load_case
from consistency import phys_residuals

FOLD_SEED, N_FOLDS = 0, 5
data = pd.concat([load_case(f) for f in sorted(glob.glob(os.path.join(sys.argv[1], "Case_3*.csv")))],
                 ignore_index=True)
prepped = {cid: tb.prep(g, False, True) for cid, g in data.groupby("case_id")}
ids = list(prepped)
cols = prepped[ids[0]][1]
ys = list(pt.HY)
pcs = ["phys_" + c for c in tb.PHYS if all(prepped[i][0]["phys_" + c].notna().mean() > 0.95 for i in ids)]
need = cols + ys + pcs

rng = np.random.default_rng(FOLD_SEED)
folds = [list(f) for f in np.array_split(rng.permutation(ids), N_FOLDS)]
test = folds[0]
train = [c for c in ids if c not in test]


def onset(g):
    return g.t[g.is_fault == 1].iloc[0] if g.is_fault.any() else np.inf


def healthy(c):
    g = prepped[c][0]
    return g, (g.t >= tb.T0) & (g.t < onset(g)) & g[need].notna().all(axis=1)


parts = [healthy(c) for c in train]
parts = [(g, m) for g, m in parts if m.sum() > 0]
print("train flights", len(parts), "healthy train samples", sum(int(m.sum()) for _, m in parts))

t0 = time.perf_counter()
hybrid = pt.HybridTwin(tb.HeteroTwin).fit(parts, cols)
print("hybrid fit s", round(time.perf_counter() - t0, 2))

t0 = time.perf_counter()
physics_only = pt.HybridTwin(tb.HeteroTwin, learned=False).fit(parts, cols)
print("physics-only fit s", round(time.perf_counter() - t0, 2))

test_parts = [healthy(c) for c in test]
test_samples = sum(len(g) for g, _ in test_parts)
allrows = lambda g: pd.Series(True, index=g.index)

t0 = time.perf_counter()
for g, _ in test_parts:
    hybrid.std_residual(g, allrows(g))
t_hybrid = time.perf_counter() - t0

t0 = time.perf_counter()
for g, _ in test_parts:
    physics_only._calibrated(g)
t_physics = time.perf_counter() - t0

raw = [data[data.case_id == c].reset_index(drop=True) for c in test]
t0 = time.perf_counter()
for g in raw:
    phys_residuals(g)
t_checks = time.perf_counter() - t0

print("test samples", test_samples,
      "| hybrid twin us/sample", round(1e6 * t_hybrid / test_samples, 1),
      "| physics layer us/sample", round(1e6 * t_physics / test_samples, 1),
      "| consistency checks us/sample", round(1e6 * t_checks / test_samples, 1),
      "| twin plus checks us/sample", round(1e6 * (t_hybrid + t_checks) / test_samples, 1))

need_phys = sorted(set(pt.PH_ALL) | set(pt.HY))
gd = pd.concat([g.loc[m, need_phys] for g, m in parts])
t0 = time.perf_counter()
for y in pt.HY:
    a = np.c_[gd[pt.REGRESSORS[y]].fillna(0).values, np.ones(len(gd))]
    np.linalg.solve(a.T @ a + 1e-3 * len(a) * np.eye(a.shape[1]), a.T @ gd[y].values)
print("physics least squares s", round(time.perf_counter() - t0, 3),
      "parameters per equation", [len(pt.REGRESSORS[y]) + 1 for y in pt.HY])

x_learned = np.vstack([g.loc[m, cols].values for g, m in parts])
targets = np.random.default_rng(0).standard_normal((len(x_learned), len(pt.HY)))
t0 = time.perf_counter()
tb.HeteroTwin().fit(x_learned, targets)
print("learned layer fit s", round(time.perf_counter() - t0, 2))
