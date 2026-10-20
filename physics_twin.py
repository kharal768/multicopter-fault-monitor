"""
physics_twin.py - hybrid (grey-box) digital twin of a quadrotor for RflyMAD.

Three layers
  1. Physics layer (system identification). Rigid-body quadrotor equations, linear in
     their unknown parameters, identified by least squares on healthy training flights:
        specific force   f_b  = A_f s + C_f v_b + b_f          (thrust + linear drag)
        angular accel.   dw/dt = A_w s + D_w w + b_w           (motor torques + damping)
     s = squared normalised motor commands  u_i = (PWM_i - 1000)/1000  at lags 0..2
     (thrust ~ rotor speed^2 ~ u^2, with a short motor lag), v_b = velocity in the body
     frame, w = body rates.
  2. Per-flight calibration. Battery charge, payload and air density change thrust from
     flight to flight, so each output's physics prediction gets an affine correction
     y ~ a*y_phys + b fitted in the first airborne seconds of the flight (4-14 s, more
     than 1 m above the start; all RflyMAD faults start later than 14 s). The fit is
     shrunk towards a=1, b=0, so a short window cannot over-fit. No fault labels are used.
  3. Learned correction with uncertainty. A bootstrap ridge ensemble (twin_baseline.HeteroTwin)
     learns what the physics layer misses, from the generic inputs (commands, attitude,
     manoeuvre aggressiveness); its ensemble spread is the epistemic uncertainty, and a
     second model predicts input-dependent (heteroscedastic) aleatoric noise, so sharp
     manoeuvres widen the twin's confidence band instead of raising alarms.

The residuals of this twin are what the MEWMA charts monitor (twin_baseline.py, modes
'hybrid' and 'hybrid+consistency').
"""
import os
import numpy as np
import pandas as pd

RATE = 20.0
CAUSAL = os.environ.get("CAUSAL", "0") == "1"   # 1: trailing averages and backward differences (online use)

def _d(a):
    """time derivative: central differences, or backward differences if CAUSAL"""
    a = np.asarray(a)
    return np.diff(a, axis=0, prepend=a[:1]) * RATE if CAUSAL else np.gradient(a, 1.0 / RATE, axis=0)

import os
CAL_WINDOW = (4.0, float(os.environ.get("CAL_END", "14")))   # override only for sensitivity analysis
CAL_MIN_ALT = 1.0            # m above the start altitude
CAL_SHRINK = float(os.environ.get("CAL_SHRINK", "50"))   # ridge weight pulling (a, b) towards (1, 0)
CAL_SOURCE = {}              # case_id -> frame of the flight whose calibration window is used (previous-flight calibration)
HY = ["hy_fx", "hy_fy", "hy_fz", "hy_wdx", "hy_wdy", "hy_wdz"]


def _rot(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _smooth(a, k=3):
    return pd.DataFrame(a).rolling(k, center=not CAUSAL, min_periods=1).mean().values


def add_physics_columns(g):
    """Adds hybrid-twin outputs (hy_*), physics regressors (ph_*) and a calibration mask."""
    g = g.copy()
    u = np.clip((g[["pwm1", "pwm2", "pwm3", "pwm4"]].values - 1000.0) / 1000.0, 0, 1)
    s = u ** 2
    phys = {}
    for lag in range(3):
        sl = pd.DataFrame(s).shift(lag).values
        for i in range(4):
            phys[f"ph_s{i + 1}_l{lag}"] = sl[:, i]
    w = g[["gyro_x", "gyro_y", "gyro_z"]].values.astype(float)
    for i, a in enumerate("xyz"):
        phys[f"ph_w{a}"] = w[:, i]
    q = g[["q0", "q1", "q2", "q3"]].values.astype(float)
    v = g[["vel_x", "vel_y", "vel_z"]].values.astype(float)
    if np.isnan(q).any() or np.isnan(v).all():
        vb = np.zeros_like(w)
    else:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        vb = np.einsum("nji,nj->ni", _rot(q), np.nan_to_num(v))      # NED -> body
    for i, a in enumerate("xyz"):
        phys[f"ph_vb{a}"] = vb[:, i]
    for k, val in phys.items():
        g[k] = val
    f = _smooth(g[["acc_x", "acc_y", "acc_z"]].values.astype(float))
    wd = _smooth(_d(_smooth(w)))
    for i, a in enumerate("xyz"):
        g[f"hy_f{a}"] = f[:, i]
        g[f"hy_wd{a}"] = wd[:, i]
    t = g["t"].values
    alt = g["baro_alt"].values
    start = np.nanmedian(alt[t < 2.0]) if (t < 2.0).any() else alt[0]
    g["cal_window"] = (t >= CAL_WINDOW[0]) & (t < CAL_WINDOW[1]) & (alt - start > CAL_MIN_ALT)
    return g


PH_THRUST = [f"ph_s{i}_l{l}" for l in range(3) for i in range(1, 5)]
PH_RATES = ["ph_wx", "ph_wy", "ph_wz"]
PH_VEL = ["ph_vbx", "ph_vby", "ph_vbz"]
# physics regressors per output (structure of the equations of motion)
REGRESSORS = {
    "hy_fx": PH_THRUST + ["ph_vbx"], "hy_fy": PH_THRUST + ["ph_vby"],
    "hy_fz": PH_THRUST + ["ph_vbz"],
    "hy_wdx": PH_THRUST + PH_RATES, "hy_wdy": PH_THRUST + PH_RATES,
    "hy_wdz": PH_THRUST + PH_RATES,
}
PH_ALL = PH_THRUST + PH_RATES + PH_VEL


class HybridTwin:
    def __init__(self, twin_cls, alpha=1e-3, calibrate=True, learned=True, physics=True,
                 post_cal=False):
        """calibrate / learned / physics switch layers off for the ablation study.
        post_cal: per-flight affine calibration of the learned prediction (used for the
        data-driven twin with calibration, physics=False)."""
        self.twin_cls, self.alpha, self.physics, self.post_cal = twin_cls, alpha, physics, post_cal
        self.calibrate, self.learned_on = calibrate, learned

    def _phys_pred(self, g):
        out = np.empty((len(g), len(HY)))
        for j, y in enumerate(HY):
            A = np.c_[g[REGRESSORS[y]].fillna(0).values, np.ones(len(g))]
            out[:, j] = A @ self.theta[y]
        return out

    def _calibrated(self, g):
        """Physics prediction with the per-flight affine correction (a, b) per output."""
        yp = self._phys_pred(g)
        src = CAL_SOURCE.get(str(g["case_id"].iloc[0])) if CAL_SOURCE and "case_id" in g else None
        if src is not None:                      # coefficients from another (earlier) flight
            ypc, ymc = self._phys_pred(src), src[HY].values
            cw = src["cal_window"].values & ~np.isnan(ymc).any(1)
        else:
            ypc, ymc = yp, g[HY].values
            cw = g["cal_window"].values & ~np.isnan(ymc).any(1)
        if not self.calibrate or cw.sum() < 20:
            return yp
        out = np.empty_like(yp)
        for j in range(len(HY)):
            x, y = ypc[cw, j], ymc[cw, j]
            # minimise |y - a x - b|^2 + lam ((a-1)^2 + b^2)  ->  2x2 linear system
            lam = CAL_SHRINK * np.var(y) + 1e-9
            M = np.array([[x @ x + lam, x.sum()], [x.sum(), len(x) + lam]])
            rhs = np.array([x @ y + lam, y.sum()])
            a, b = np.linalg.solve(M, rhs)
            out[:, j] = a * yp[:, j] + b
        return out

    def fit(self, parts, cols):
        """parts: list of (flight frame g with physics columns, boolean healthy mask)."""
        need = sorted(set(PH_ALL) | set(HY))
        G = pd.concat([g.loc[m, need] for g, m in parts])     # only the columns needed
        self.theta = {}
        for y in HY:                                     # physics layer: least squares
            A = np.c_[G[REGRESSORS[y]].fillna(0).values, np.ones(len(G))]
            self.theta[y] = np.linalg.solve(A.T @ A + self.alpha * len(A) * np.eye(A.shape[1]),
                                            A.T @ G[y].values)
            if not self.physics:                         # ablation: no physics layer
                self.theta[y] = np.zeros_like(self.theta[y])
        X, R = [], []
        for g, m in parts:                               # learned layer on what physics misses
            r = g[HY].values - self._calibrated(g)
            X.append(g.loc[m, cols].values); R.append(r[m.values])
        self.cols = cols
        R = np.vstack(R)
        if self.learned_on:
            self.learned = self.twin_cls().fit(np.vstack(X), R)
        else:                                            # physics only: constant noise level
            self.res_var = R.var(0) + 1e-12
        return self

    def std_residual(self, g, m):
        r = g[HY].values - self._calibrated(g)
        if not self.learned_on:
            rr = r[m.values]
            return rr / np.sqrt(self.res_var), np.zeros_like(rr)
        if self.post_cal:
            X = g[self.cols].values
            ok = np.isfinite(X).all(1)
            mu = np.full(r.shape, np.nan); ep = np.full(r.shape, np.nan); al = np.full(r.shape, np.nan)
            mu[ok], ep[ok] = self.learned.predict(X[ok]); al[ok] = self.learned.noise_var(X[ok])
            cw = g["cal_window"].values & ok & ~np.isnan(r).any(1)
            if cw.sum() >= 20:
                for j in range(r.shape[1]):
                    x, y = mu[cw, j], r[cw, j]
                    lam = CAL_SHRINK * np.var(y) + 1e-9
                    M = np.array([[x @ x + lam, x.sum()], [x.sum(), len(x) + lam]])
                    a, b = np.linalg.solve(M, np.array([x @ y + lam, y.sum()]))
                    mu[:, j] = a * mu[:, j] + b
            mm = m.values
            return (r[mm] - mu[mm]) / np.sqrt(al[mm] + ep[mm]), (ep[mm] / (al[mm] + ep[mm]))
        return self.learned.std_residual(g.loc[m, self.cols].values, r[m.values])
