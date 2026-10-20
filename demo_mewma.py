"""
demo_mewma.py - first fault-detection baseline on RflyMAD ProcessData.

Per case:
  1. Phase I  = healthy flight from 3 s to 70% of the fault onset time.
     Validation = remaining 30% of healthy flight (used only to count false alarms).
  2. Each signal is whitened with an AR(5) model fitted on Phase I
     (residual-chart idea for autocorrelated data).
  3. MEWMA on the residual vector (lambda = 0.1); the control limit is the
     99.5% quantile of the Phase I statistic.
  4. Alarm = 3 consecutive points above the limit. Detection delay = first
     alarm after onset minus onset.

This is a deliberately simple per-flight baseline, not the proposed twin.
Usage:  python demo_mewma.py <ProcessData folder> <output folder>
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rflymad import load_all, case_table

RAW = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
       "mag_x", "mag_y", "mag_z", "vel_x", "vel_y", "vel_z",
       "pwm1", "pwm2", "pwm3", "pwm4"]
DIFF = ["baro_alt", "pos_x", "pos_y", "pos_z"]   # slow signals: use increments
P, LAM, Q, RUN = 5, 0.1, 0.995, 3


def features(g):
    x = g[RAW].copy()
    for c in DIFF:
        x["d_" + c] = g[c].diff()
    x = x.dropna(axis=1, how="all")          # e.g. HIL has no position/velocity
    return x.iloc[1:].reset_index(drop=True), g.iloc[1:].reset_index(drop=True)


def ar_residuals(x_train, x_all, p=P):
    res = np.full(x_all.shape, np.nan)
    for j in range(x_all.shape[1]):
        tr, al = x_train[:, j], x_all[:, j]
        X = np.column_stack([tr[p - k - 1:len(tr) - k - 1] for k in range(p)] + [np.ones(len(tr) - p)])
        beta, *_ = np.linalg.lstsq(X, tr[p:], rcond=None)
        Xa = np.column_stack([al[p - k - 1:len(al) - k - 1] for k in range(p)] + [np.ones(len(al) - p)])
        res[p:, j] = al[p:] - Xa @ beta
    return res


def mewma(r, mu, cov, lam=LAM):
    z = np.zeros(r.shape[1]); out = np.full(len(r), np.nan)
    inv = np.linalg.pinv(cov * lam / (2 - lam))
    for i, ri in enumerate(r):
        if np.isnan(ri).any():
            continue
        z = lam * (ri - mu) + (1 - lam) * z
        out[i] = z @ inv @ z
    return out


def first_run(mask, run=RUN):
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        if c >= run:
            return i - run + 1
    return None


RESET = os.environ.get("RESET_POLICY", "1") == "1"   # same detection rule as twin_baseline.py


def detect_after_fault(exceed, post):
    """Absolute index of the first run of RUN exceedances that counts as a detection, or None (reset rule)."""
    c, armed = 0, True
    if not RESET:
        for i in range(len(exceed)):
            c = c + 1 if (exceed[i] and post[i]) else 0
            if c >= RUN:
                return i - RUN + 1
        return None
    for i in range(len(exceed)):
        if exceed[i]:
            c += 1
            if c >= RUN and armed:
                st = i - RUN + 1
                if post[st]:
                    return st
                armed = False
        else:
            c, armed = 0, True
    return None


def run_case(g):
    x, g = features(g)
    t, y = g.t.values, g.is_fault.values
    has_fault = bool(y.any())
    onset = t[np.argmax(y == 1)] if has_fault else t[-1] + 1e-3   # fault-free flight
    tr = (t >= 3) & (t < 0.7 * min(onset, t[-1]))
    va = (t >= 0.7 * min(onset, t[-1])) & (t < onset)
    sd = x[tr].std().replace(0, 1)
    xs = ((x - x[tr].mean()) / sd).values
    r = ar_residuals(xs[tr], xs)
    ok = tr & ~np.isnan(r).any(1)
    mu, cov = r[ok].mean(0), np.cov(r[ok].T)
    T2 = mewma(r, mu, cov)
    ucl = np.nanquantile(T2[ok], Q)
    fa = first_run(T2[va] > ucl) is not None
    after = (t >= onset) & (y == 1)          # count detections only while the fault is active
    k = detect_after_fault(T2 > ucl, after) if has_fault else None   # reset rule, as in twin_baseline.py
    delay = None if k is None else round(t[k] - onset, 2)
    return (dict(onset=onset if has_fault else None, delay_s=delay, false_alarm_before_onset=fa),
            (t, T2, ucl, onset if has_fault else None))


def main(src, out):
    os.makedirs(out, exist_ok=True)
    data = load_all(src)
    info = case_table(data)
    rows, curves = [], {}
    for cid, g in data.groupby("case_id"):
        res, curve = run_case(g.reset_index(drop=True))
        rows.append({"case_id": cid, **res}); curves[cid] = curve
    res = info.merge(pd.DataFrame(rows), on="case_id")
    res.to_csv(os.path.join(out, "mewma_baseline_results.csv"), index=False)

    # Figure: one example per domain for the same scenario (hover, motor fault)
    if not all(c in curves for c in ["1000000000", "2000000000", "3000000000"]):
        return res
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)
    for ax, cid, name in zip(axes, ["1000000000", "2000000000", "3000000000"],
                             ["SIL", "HIL", "Real flight"]):
        t, T2, ucl, onset = curves[cid]
        ax.semilogy(t, T2, lw=0.9, color="#2b6cb0", label="MEWMA statistic")
        ax.axhline(ucl, color="#c53030", lw=1, ls="--", label="control limit (99.5%)")
        ax.axvline(onset, color="#444", lw=1, ls=":", label="fault injected")
        ax.set_title(f"{name}: hover, motor fault (case {cid})", fontsize=10, loc="left")
        ax.set_ylabel("statistic (log)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mewma_example_motor_fault.png"), dpi=150)
    return res


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    pd.set_option("display.width", 200)
    print(r[["case_id", "domain", "status", "fault_type", "duration_s", "fault_onset_s",
             "delay_s", "false_alarm_before_onset"]].to_string(index=False))
