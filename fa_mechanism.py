"""Where do the false alarms of the constant-noise twin occur? Manoeuvre intensity
(1-s mean of |3-sample change| of the four attitude/thrust commands) in the 5 s before each
false alarm, as a percentile of all healthy samples of the real flights.

Usage: python fa_mechanism.py <results folder> <ProcessData folder> <output csv>
The results folder must contain hybrid_nohetero_results.csv and hybrid_results.csv."""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rflymad import load_case
F, DATA, OUT = sys.argv[1:4]
def res(k):
    d = pd.read_csv(f"{F}/{k}_results.csv", dtype={"case_id": str}); d = d[d.domain == "Real"]
    return d[~((d.fault_type != "no_fault") & d.fault_onset_s.isna())].set_index("case_id")
C = res("hybrid_nohetero"); H = res("hybrid")
U = ["u_roll", "u_pitch", "u_yaw", "u_thrust"]
inten, at = {}, []
for f in sorted(glob.glob(os.path.join(DATA, "Case_3*.csv"))):
    g = load_case(f); cid = g.case_id.iloc[0]
    if cid not in C.index: continue
    a = sum((g[c] - g[c].shift(3)).abs() for c in U).rolling(20, min_periods=1).mean().values
    on = C.loc[cid, "fault_onset_s"]; on = np.inf if pd.isna(on) else on
    h = (g.t.values >= 3) & (g.t.values < on)
    inten[cid] = (g.t.values, a, h)
allv = np.concatenate([a[h] for t, a, h in inten.values()]); allv = allv[np.isfinite(allv)]
def pct(x): return 100 * (allv < x).mean()
rows = []
for name, R in [("constant", C), ("heteroscedastic", H)]:
    for cid, r in R[R.false_alarm_before_onset.astype(bool)].iterrows():
        t, a, h = inten[cid]; tf = r.false_alarm_time_s
        w = (t >= tf - 5) & (t <= tf)
        rows.append(dict(model=name, case_id=cid, fault_free=r.fault_type == "no_fault",
                         only_this=not bool((H if name == "constant" else C).loc[cid, "false_alarm_before_onset"]),
                         pct_max5s=pct(np.nanmax(a[w])), status=r.status, group=r.false_alarm_group))
D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
print(D.groupby(["model", "only_this"]).pct_max5s.describe().round(1).to_string())
# reference: max over random 5-s healthy windows
rng = np.random.default_rng(0); ref = []
for t, a, h in inten.values():
    idx = np.where(h)[0]
    if len(idx) > 100:
        for i in rng.choice(idx[100:], 3): ref.append(pct(np.nanmax(a[i - 100:i + 1])))
print("random 5-s windows: median pct", np.median(ref).round(1), "IQR", np.percentile(ref, [25, 75]).round(1))
