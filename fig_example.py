"""fig_example.py - residual traces of one real flight in which all four motors lose 6% efficiency:
the controller raises the thrust command, the data-driven twin follows it, the hybrid twin's
vertical specific-force residual shifts. Twins trained on all other real flights (healthy samples).

Usage: python fig_example.py <ProcessData folder> <output pdf> [case_id]
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_baseline as tb
from physics_twin import HybridTwin, HY
from rflymad import load_case
from scipy.signal import lfilter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA, OUT = sys.argv[1], sys.argv[2]
CID = sys.argv[3] if len(sys.argv) > 3 else "3200000003"
files = sorted(glob.glob(os.path.join(DATA, "Case_3*.csv")))
pH, pT = {}, {}
for f in files:
    g = load_case(f); c = g.case_id.iloc[0]
    pH[c] = tb.prep(g, False, True); pT[c] = tb.prep(g, False, False)
colsH = pH[CID][1]; colsT = pT[CID][1]
ysT = [y for y in tb.Y_BASE + tb.Y_VEL if all(y in pT[c][2] for c in pT)]
def healthy(g, need):
    on = g.t[g.is_fault == 1].iloc[0] if g.is_fault.any() else np.inf
    return (g.t >= tb.T0) & (g.t < on) & g[need].notna().all(axis=1)
train = [c for c in pH if c != CID]
parts = [(pH[c][0], healthy(pH[c][0], colsH + list(HY))) for c in train]
parts = [(g, m) for g, m in parts if m.sum() > 0]
hyb = HybridTwin(tb.HeteroTwin).fit(parts, colsH)
X = np.vstack([pT[c][0].loc[healthy(pT[c][0], colsT + ysT), colsT].values for c in train])
Y = np.vstack([pT[c][0].loc[healthy(pT[c][0], colsT + ysT), ysT].values for c in train])
twin = tb.Twin().fit(X, Y)
g = pH[CID][0]; m = (g.t >= tb.T0) & g[colsH + list(HY)].notna().all(axis=1)
eH, _ = hyb.std_residual(g, m); tH = g.t[m].values
gt = pT[CID][0]; mt = (gt.t >= tb.T0) & gt[colsT + ysT].notna().all(axis=1)
eT, _ = twin.std_residual(gt.loc[mt, colsT].values, gt.loc[mt, ysT].values); tT = gt.t[mt].values
ew = lambda x: lfilter([0.1], [1, -0.9], x)
on = g.t[g.is_fault == 1].iloc[0]
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)
pwm = g[["pwm1", "pwm2", "pwm3", "pwm4"]].mean(axis=1)
ax[0].plot(g.t, pwm, color="#555", lw=0.9)
ax[0].set_ylabel("Mean motor\ncommand ($\\mu$s)")
ax[1].plot(tT, ew(eT[:, ysT.index("acc_z")]), color="#2a78d6", lw=1.1, label="Data-driven twin, vertical specific force")
ax[1].plot(tH, ew(eH[:, HY.index("hy_fz")]), color="#eb6834", lw=1.1, label="Hybrid twin, vertical specific force")
ax[1].axhline(0, color="#999", lw=0.6)
ax[1].set_ylabel("Smoothed standardised\nresidual (EWMA, $\\lambda$=0.1)"); ax[1].set_xlabel("Time (s)")
for a in ax:
    a.axvline(on, color="#c53030", ls=":", lw=1)
ax[0].text(on + 1, ax[0].get_ylim()[1], "fault injected", va="top", fontsize=8, color="#c53030")
ax[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT)
pre = (tH > on - 20) & (tH < on); post = (tH > on + 5) & (tH < on + 25)
print("hybrid fz mean e pre/post", eH[pre, HY.index('hy_fz')].mean().round(2), eH[post, HY.index('hy_fz')].mean().round(2))
pre = (tT > on - 20) & (tT < on); post = (tT > on + 5) & (tT < on + 25)
print("twin accz mean e pre/post", eT[pre, ysT.index('acc_z')].mean().round(2), eT[post, ysT.index('acc_z')].mean().round(2))
print("PWM pre/post", pwm[(g.t > on - 20) & (g.t < on)].mean().round(1), pwm[(g.t > on + 5) & (g.t < on + 25)].mean().round(1), "onset", on)
