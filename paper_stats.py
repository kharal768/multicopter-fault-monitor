"""paper_stats.py - tables for the manuscript from the per-flight result files.

Bootstrap 95% CIs (2,000 resamples of flights), McNemar exact tests for paired
method comparisons, operating curves, ablation and fault-strength tables.
Usage: python paper_stats.py <results folder> <motor_severity.csv> <output folder>
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

B = 2000
METHODS = {"twin_inputs_only": "Data-driven twin", "consistency_only": "Consistency checks",
           "twin_plus_consistency": "Data-driven twin + consistency",
           "hybrid_physonly": "Physics layer only", "hybrid_cal": "Physics + calibration",
           "hybrid_nohetero": "Physics + calibration + learned (constant noise)",
           "hybrid": "Hybrid twin (full)", "hybrid_plus_consistency": "Hybrid twin + consistency"}


def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = np.random.default_rng(1).choice(x, (B, len(x)), replace=True).mean(1)   # fixed seed per interval: results do not depend on which methods run first
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(t, pct=True):
    m, lo, hi = t
    k = 100 if pct else 1
    return f"{m*k:.1f} [{lo*k:.1f}, {hi*k:.1f}]"


def mcnemar(a, b):
    """exact McNemar test on paired booleans a, b; returns (#a only, #b only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, binomtest(n01, n01 + n10, 0.5).pvalue


def load(folder):
    R = {}
    for k in METHODS:
        p = os.path.join(folder, f"{k}_results.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, dtype={"case_id": str})
            # flights labelled with a fault type but without a logged injection are excluded
            d = d[~((d.fault_type != "no_fault") & d.fault_onset_s.isna())]
            d["det"] = d.delay_s.notna()
            d = d.sort_values("case_id").reset_index(drop=True)   # fixed row order: bootstrap draws do not depend on file order
            R[k] = d
    return R


def main(folder, sev_path, out):
    os.makedirs(out, exist_ok=True)
    R = load(folder)
    rows = []
    for k, d in R.items():
        for dom, g in d.groupby("domain"):
            f = g[g.fault_onset_s.notna()]
            nf = g[g.fault_type == "no_fault"]
            hours = g.monitored_healthy_s.sum() / 3600 if "monitored_healthy_s" in g else np.nan
            rows.append(dict(method=METHODS[k], domain=dom, faulty=len(f),
                             detection=fmt(ci(f.det)),
                             median_delay_s=round(f.delay_s.median(), 2),
                             fa_before_onset=fmt(ci(g.false_alarm_before_onset)),
                             fa_fault_free=fmt(ci(nf.false_alarm_before_onset)) if len(nf) else "",
                             fa_per_hour=round(g.false_alarm_before_onset.sum() / hours, 2) if hours == hours else None))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "table_main.csv"), index=False)

    # paired tests on real flights against the proposed method
    if "hybrid_plus_consistency" in R:
        P = R["hybrid_plus_consistency"].set_index("case_id")
        tests = []
        for k, d in R.items():
            if k == "hybrid_plus_consistency":
                continue
            d = d.set_index("case_id")
            for dom in ["Real", "SIL", "HIL"]:
                ids = P.index[(P.domain == dom)].intersection(d.index)
                f = [i for i in ids if pd.notna(P.loc[i, "fault_onset_s"])]
                a, b = P.loc[f, "det"], d.loc[f, "det"]
                n01, n10, p = mcnemar(a, b)
                nf = [i for i in ids if P.loc[i, "fault_type"] == "no_fault"]
                m01, m10, pf = mcnemar(P.loc[nf, "false_alarm_before_onset"], d.loc[nf, "false_alarm_before_onset"]) if nf else (0, 0, np.nan)
                tests.append(dict(domain=dom, compared_with=METHODS[k],
                                  detected_only_by_proposed=n01, detected_only_by_other=n10, p_detection=p,
                                  ff_alarm_only_proposed=m01, ff_alarm_only_other=m10, p_fault_free_fa=pf))
        pd.DataFrame(tests).to_csv(os.path.join(out, "table_mcnemar.csv"), index=False)

    # detection by fault type (real), with CIs
    fr = []
    for k, d in R.items():
        f = d[(d.domain == "Real") & d.fault_onset_s.notna()]
        for ft, g in f.groupby("fault_type"):
            fr.append(dict(method=METHODS[k], fault_type=ft, n=len(g), detection=fmt(ci(g.det)),
                           median_delay_s=round(g.delay_s.median(), 2)))
    pd.DataFrame(fr).to_csv(os.path.join(out, "table_fault_type.csv"), index=False)

    # operating curves (real)
    oc = []
    for k, d in R.items():
        lv = sorted({c.split("_at_")[1] for c in d.columns if c.startswith("fa_at_")})
        g = d[d.domain == "Real"]
        f = g[g.fault_onset_s.notna()]; nf = g[g.fault_type == "no_fault"]
        for l in lv:
            oc.append(dict(method=METHODS[k], alpha=float(l),
                           detection=f[f"det_at_{l}"].astype(bool).mean(),
                           fa_fault_free=nf[f"fa_at_{l}"].astype(bool).mean(),
                           fa_all=g[f"fa_at_{l}"].astype(bool).mean()))
    pd.DataFrame(oc).to_csv(os.path.join(out, "table_operating_curve.csv"), index=False)

    # fault strength (real motor faults)
    sev = pd.read_csv(sev_path, dtype={"case_id": str})
    sev["loss_bin"] = pd.cut(sev.efficiency_loss, [0, 0.1, 0.2, 0.3], labels=["5-10%", "11-20%", "21-30%"])
    st = []
    for k, d in R.items():
        m = sev.merge(d[["case_id", "det"]], on="case_id")
        for col in ["loss_bin", "motors_affected"]:
            for v, g in m.groupby(col, observed=True):
                st.append(dict(method=METHODS[k], by=col, level=v, n=len(g), detection=fmt(ci(g.det))))
    pd.DataFrame(st).to_csv(os.path.join(out, "table_fault_strength.csv"), index=False)

    # isolation (proposed, real)
    if "hybrid_plus_consistency" in R:
        d = R["hybrid_plus_consistency"]; f = d[(d.domain == "Real") & d.fault_onset_s.notna()]
        pd.crosstab(f.fault_type, f.alarm_group.fillna("not detected")).to_csv(os.path.join(out, "table_isolation.csv"))
        nf = d[d.fault_type == "no_fault"]
        nf.groupby("status").false_alarm_before_onset.agg(["size", "mean"]).to_csv(os.path.join(out, "table_fault_free_by_status.csv"))
    return T


if __name__ == "__main__":
    T = main(*sys.argv[1:4])
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
