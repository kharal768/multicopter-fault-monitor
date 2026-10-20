"""paper_figures.py - manuscript figures (PDF, vector) from the per-flight result files.
Usage: python paper_figures.py <results folder> <motor_severity.csv> <figure folder>"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a96", "axes.labelcolor": "#222", "xtick.color": "#444",
                     "ytick.color": "#444", "font.family": "serif"})
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK = "#222"


def load(folder, k):
    d = pd.read_csv(os.path.join(folder, f"{k}_results.csv"), dtype={"case_id": str})
    d["det"] = d.delay_s.notna()
    return d


def operating_curve(folder, out):
    """Detection vs false alarms on fault-free real flights, over target levels alpha."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    specs = [("twin_plus_consistency", "Data-driven twin + consistency", C["blue"], "s"),
             ("hybrid", "Hybrid twin", C["aqua"], "^"),
             ("hybrid_plus_consistency", "Hybrid twin + consistency (proposed)", C["orange"], "o")]
    for k, lab, col, mk in specs:
        p = os.path.join(folder, f"{k}_results.csv")
        if not os.path.exists(p):
            continue
        d = load(folder, k); r = d[d.domain == "Real"]
        f = r[r.fault_onset_s.notna()]; nf = r[r.fault_type == "no_fault"]
        lv = sorted({c.split("_at_")[1] for c in d.columns if c.startswith("fa_at_")}, key=float)
        x = [nf[f"fa_at_{l}"].astype(bool).mean() * 100 for l in lv]
        y = [f[f"det_at_{l}"].astype(bool).mean() * 100 for l in lv]
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6, label=lab, markeredgecolor="white", markeredgewidth=1.2)
        if k == "hybrid_plus_consistency":
            offs = [None, None, (6, -9), (6, -9), (6, -9)]   # low levels sit in a crowded corner: explained in the caption
            for (xi, yi, l), o in zip(zip(x, y, lv), offs):
                if o is None:
                    continue
                ax.annotate(f"$\\alpha$={float(l):g}", (xi, yi), textcoords="offset points", xytext=o,
                            fontsize=8, color="#555")
    ax.set_xlabel("False alarms on fault-free real flights (%)")
    ax.set_ylabel("Real faults detected (%)")
    ax.set_xlim(0, 80); ax.set_ylim(0, 100)
    ax.grid(True, color="#ececea", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_operating_curve.pdf")); plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def _dot_ci(ax, x, p, lo, hi, col, mk, lab, horizontal=False):
    """Point estimate with 95% interval; white edge separates overlapping markers."""
    if horizontal:
        ax.hlines(x, lo, hi, color=col, lw=2, alpha=0.9, zorder=2)
        ax.plot(p, x, mk, color=col, ms=7, mec="white", mew=1.2, label=lab, zorder=3, ls="none")
    else:
        ax.vlines(x, lo, hi, color=col, lw=2, alpha=0.9, zorder=2)
        ax.plot(x, p, mk, color=col, ms=7, mec="white", mew=1.2, label=lab, zorder=3, ls="none")


def by_fault(folder, out):
    """Dot-and-interval chart: detection rate with 95% Wilson interval per fault type."""
    order = ["motor", "barometer", "GPS", "magnetometer", "accelerometer", "gyroscope"]
    names = ["Motor", "Barometer", "GPS", "Magnetometer", "Accelerometer", "Gyroscope"]
    specs = [("twin_plus_consistency", "Data-driven twin + consistency", C["blue"], "s"),
             ("hybrid", "Hybrid twin", C["aqua"], "^"),
             ("hybrid_plus_consistency", "Hybrid twin + consistency (proposed)", C["orange"], "o")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(order))[::-1].astype(float)
    for i, (k, lab, col, mk) in enumerate(specs):
        d = load(folder, k); f = d[(d.domain == "Real") & d.fault_onset_s.notna()]
        g = f.groupby("fault_type").det.agg(["sum", "size"]).reindex(order)
        p = 100 * g["sum"] / g["size"]
        ci = np.array([wilson(a, b) for a, b in zip(g["sum"], g["size"])])
        _dot_ci(ax, y + (1 - i) * 0.22, p.values, ci[:, 0], ci[:, 1], col, mk, lab, horizontal=True)
        n = g["size"].values
    ax.set_yticks(y, [f"{a} (n={int(m)})" for a, m in zip(names, n)])
    ax.set_xlabel("Real faults detected (%), with 95% interval"); ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color="#ececea"); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_detection_by_fault.pdf")); plt.close(fig)


def strength(folder, sev_path, out):
    """Detection of motor faults with 95% Wilson intervals across ordered strength levels."""
    sev = pd.read_csv(sev_path, dtype={"case_id": str})
    sev["loss_bin"] = pd.cut(sev.efficiency_loss, [0, 0.1, 0.2, 0.3], labels=["5–10%", "11–20%", "21–30%"])
    specs = [("twin_inputs_only", "Data-driven twin", C["blue"], "s"),
             ("hybrid_physonly", "Physics layer only", C["yellow"], "D"),
             ("hybrid", "Hybrid twin", C["aqua"], "^")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"width_ratios": [3, 4]})
    for ax, col, xl in [(axs[0], "loss_bin", "Efficiency loss of the most affected motor"),
                        (axs[1], "motors_affected", "Number of motors affected")]:
        for i, (k, lab, colr, mk) in enumerate(specs):
            d = load(folder, k)
            m = sev.merge(d[["case_id", "det"]], on="case_id")
            g = m.groupby(col, observed=True).det.agg(["sum", "size"])
            x = np.arange(len(g)) + (i - 1) * 0.12
            p = 100 * g["sum"] / g["size"]
            ci = np.array([wilson(a, b) for a, b in zip(g["sum"], g["size"])])
            ax.plot(x, p.values, color=colr, lw=1.2, alpha=0.6, zorder=1)
            _dot_ci(ax, x, p.values, ci[:, 0], ci[:, 1], colr, mk, lab)
            levels, ns = g.index, g["size"].values
        labs = [str(int(l)) if isinstance(l, (float, np.floating, int, np.integer)) else str(l) for l in levels]
        ax.set_xticks(np.arange(len(levels)), [f"{l}\n(n={n})" for l, n in zip(labs, ns)])
        ax.set_xlabel(xl); ax.set_ylim(0, 100); ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.yaxis.grid(True, color="#ececea"); ax.set_axisbelow(True)
    axs[0].set_ylabel("Motor faults detected (%)")
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_motor_strength.pdf")); plt.close(fig)


if __name__ == "__main__":
    folder, sev, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    operating_curve(folder, out); by_fault(folder, out); strength(folder, sev, out)
    print("figures written to", out)
