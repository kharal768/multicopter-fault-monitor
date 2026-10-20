"""uq_check.py - is the twin's predictive uncertainty calibrated on held-out healthy flight?
Five-fold CV by flight on the real flights (same fold seed as the main runs).

Usage: python uq_check.py <ProcessData folder> <output folder> For every
held-out healthy sample: standardised residual e, aleatoric and epistemic variance.
Reports variance of e, central 95% coverage, epistemic share, and a reliability table
(predicted vs observed residual SD by decile of predicted SD)."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_baseline as tb
from physics_twin import HybridTwin, HY
from rflymad import load_case
DATA, OUT = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(DATA, "Case_3*.csv")))
data = pd.concat([load_case(f) for f in files], ignore_index=True)
prepped = {cid: tb.prep(g, False, True) for cid, g in data.groupby("case_id")}
ids = list(data.case_id.unique()); del data
excl = [c for c in ids if (lambda g: (g.fault_type.iloc[0] != "no_fault") and not g.is_fault.any())(prepped[c][0])]
ids = [c for c in ids if c not in excl]
cols = prepped[ids[0]][1]; ys = list(HY); need = cols + ys
rng = np.random.default_rng(0); folds = [list(f) for f in np.array_split(rng.permutation(ids), 5)]
def healthy(c):
    g = prepped[c][0]; on = g.t[g.is_fault == 1].iloc[0] if g.is_fault.any() else np.inf
    return g, (g.t >= tb.T0) & (g.t < on) & g[need].notna().all(axis=1)
out = {"hetero": [], "constant": []}
for fold in folds:
    parts = [healthy(c) for c in ids if c not in fold]; parts = [(g, m) for g, m in parts if m.sum() > 0]
    for name, cls in [("hetero", tb.HeteroTwin), ("constant", tb.Twin)]:
        tw = HybridTwin(cls).fit(parts, cols)
        for c in fold:
            g, m = healthy(c)
            if m.sum() < 20: continue
            r = (g[HY].values - tw._calibrated(g))[m.values]; X = g.loc[m, cols].values
            mu, ep = tw.learned.predict(X)
            al = tw.learned.noise_var(X) if name == "hetero" else np.broadcast_to(tw.learned.alea, r.shape)
            out[name].append(np.stack([r - mu, al, ep], -1))
    print("fold done", flush=True)
rows, rel = [], []
for name, L in out.items():
    A = np.concatenate(L)                       # n x 6 x 3
    res, al, ep = A[..., 0], A[..., 1], A[..., 2]
    e = res / np.sqrt(al + ep)
    for j, y in enumerate(HY):
        rows.append(dict(model=name, output=y, var_e=np.var(e[:, j]), cover95=np.mean(np.abs(e[:, j]) < 1.96),
                         cover_iqr=np.mean(np.abs(e[:, j]) < 0.674),
                         ep_share_median=np.median(ep[:, j] / (al[:, j] + ep[:, j])),
                         ep_share_p99=np.percentile(ep[:, j] / (al[:, j] + ep[:, j]), 99)))
        if name == "hetero":
            sd = np.sqrt(al[:, j] + ep[:, j]); q = np.quantile(sd, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(q, sd, side="right") - 1, 0, 9)
            for k in range(10):
                s = b == k
                rel.append(dict(output=y, decile=k + 1, pred_sd=sd[s].mean(), obs_sd=np.sqrt(np.mean(res[s, j] ** 2))))
    # constant noise: observed SD by decile of the hetero model's predicted SD (does noise really vary?)
T = pd.DataFrame(rows); R = pd.DataFrame(rel)
os.makedirs(OUT, exist_ok=True)
T.to_csv(os.path.join(OUT, "uq_calibration.csv"), index=False); R.to_csv(os.path.join(OUT, "uq_reliability.csv"), index=False)
print(T.round(3).to_string(index=False))
R["ratio"] = R.obs_sd / R.pred_sd
print(R.groupby("decile")[["ratio"]].median().round(2).T.to_string())
print("Spearman pred vs obs per output:", R.groupby("output").apply(lambda d: d.pred_sd.corr(d.obs_sd, method="spearman")).round(2).to_dict())
print("obs SD ratio top/bottom decile:", R.groupby("output").apply(lambda d: d.obs_sd.iloc[-1] / d.obs_sd.iloc[0]).round(1).to_dict())
