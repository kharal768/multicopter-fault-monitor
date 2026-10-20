"""
twin_baseline.py - a first data-driven digital twin with uncertainty, and
MEWMA monitoring of its residuals, on RflyMAD ProcessData.

Twin
  Predicts the vehicle response y(t) = [gyro x/y/z, accelerometer x/y/z,
  baro climb rate, (velocity x/y/z where measured)] from the control inputs:
  actuator commands and motor PWM over the last 0.25 s (lags 0..5 at 20 Hz)
  plus attitude. Optional: also the previous measured output (ARX variant).
  Model = (Poisson-)bootstrap ensemble of ridge regressions (10 members).

Uncertainty
  aleatoric  = residual variance of each output on training data
  epistemic  = variance across ensemble members at each time step
  standardised residual  e = (y - mean) / sqrt(aleatoric + epistemic)

Evaluation (by flight: leave-one-out up to 15 flights per domain, else 5-fold)
  Train on the healthy part (3 s .. fault onset, or the whole flight if fault-free)
  of the training flights. Control limits are set on healthy flight TIME: the limit is
  the value that about 5% of healthy 60-second stretches would exceed (3 consecutive
  points), using flights the twin did not see (inner 3-fold) when there are enough.
  This controls false alarms per minute of flight, whatever the flight length.
  False alarm = 3 consecutive points over the limit before the onset; its time and
  the residual group that caused it are recorded.

Methods compared
  twin_inputs_only        twin residuals only
  twin_arx                twin that also sees the previous measured output
  consistency_only        physics-based sensor-consistency residuals only (consistency.py)
  twin_plus_consistency   both residual sets, one chart per group
  hybrid                  physics-based (grey-box) twin with per-flight calibration and a
                          learned correction (physics_twin.py)
  hybrid_plus_consistency hybrid twin + consistency checks
  Residuals are split into groups (twin, mag, baro, kin_v, kin_p, gyro_q); each group
  has its own MEWMA chart and flight-level limit (Bonferroni-adjusted), an alarm in any group counts,
  and the group that alarms first is reported (fault isolation, column alarm_group).
  Consistency residuals are centred/scaled with the central 98% of training values
  (trimmed mean/std), so a few unusual training segments do not set the scale.

Usage: python twin_baseline.py <ProcessData folder> <output folder>
"""
import os
import sys
import numpy as np
from scipy.signal import lfilter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rflymad import load_all, case_table
from consistency import phys_residuals, CHANNELS as PHYS
from physics_twin import add_physics_columns, HybridTwin, HY

U = ["u_roll", "u_pitch", "u_yaw", "u_thrust", "pwm1", "pwm2", "pwm3", "pwm4"]
ATT = ["q0", "q1", "q2", "q3"]
Y_BASE = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "climb"]
Y_VEL = ["vel_x", "vel_y", "vel_z"]
LAGS, ALPHA, T0 = 5, 1.0, 3.0
N_ENS = int(os.environ.get("N_ENS", "10"))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "1.0"))   # fraction of each fold's training flights used for fit and calibration
CLIP_LO = float(os.environ.get("CLIP_LO", "0.2")); CLIP_HI = float(os.environ.get("CLIP_HI", "50"))
LEARNER = os.environ.get("LEARNER", "ridge")      # ridge | hgb (nonlinear learned layer)
FOLD_GROUPS = os.environ.get("FOLD_GROUPS")      # csv case_id,group: grouped cross-validation
CAL_PREV = os.environ.get("CAL_PREV")            # csv case_id,prev: calibrate on the previous flight of the day
CAL_STRESS = os.environ.get("CAL_STRESS") == "1"
LIMIT = os.environ.get("LIMIT", "empirical")    # empirical | gpd (peaks-over-threshold tail model)
RESCALE = os.environ.get("RESCALE") == "1"      # per-flight residual rescaling in the calibration window
# monitoring parameters (environment overrides are used only for the sensitivity analysis)
DAYRECAL = int(os.environ.get("DAYRECAL", "0"))   # day-wise recalibration: limits from the healthy windows of the first k flights of each day
DAY_TIME = {}
if DAYRECAL:
    _s = pd.read_csv(os.environ.get("DAY_SESSIONS", "data/real_sessions.csv"), dtype=str)
    _ts = pd.to_datetime(_s.source.str.extract(r"(\d{4}-\d{1,2}-\d{1,2}-\d{2}-\d{2}-\d{2})\.ulg$")[0], format="%Y-%m-%d-%H-%M-%S")
    DAY_TIME = dict(zip(_s.case_id, _ts))
LAM = float(os.environ.get("MEWMA_LAMBDA", "0.1"))
RUN = int(os.environ.get("RUN_LENGTH", "3"))
TEST_SCALE = float(os.environ.get("TEST_SCALE", "1.0"))
FLIGHT_FA = 0.05              # target share of healthy 60-s stretches with a false alarm
WIN_S = float(os.environ.get("WIN_S", "60"))
THRESH = os.environ.get("THRESH", "window")   # window (paper) | tukey_raw | tukey_mewma (Erdogan et al. 2026 boxplot rule)

def tukey_adjacent(x):
    """lower and upper boxplot adjacent values per column (most extreme data inside Q1-1.5IQR, Q3+1.5IQR)"""
    lo_out, hi_out = [], []
    for j in range(x.shape[1]):
        v = x[:, j]; v = v[np.isfinite(v)]
        q1, q3 = np.quantile(v, [0.25, 0.75]); iqr = q3 - q1
        lo_out.append(v[v >= q1 - 1.5 * iqr].min()); hi_out.append(v[v <= q3 + 1.5 * iqr].max())
    return np.array(lo_out), np.array(hi_out)

FA_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20]   # extra target levels for the operating curve
MAX_LOO, N_FOLDS = 15, 5      # leave-one-flight-out up to 15 flights, else 5-fold by flight


def prep(g, arx, hybrid=False):
    g = g.reset_index(drop=True).copy()
    if hybrid:
        g = add_physics_columns(g)
    phys = phys_residuals(g)
    for c in PHYS:
        g["phys_" + c] = phys[c].values
    g["climb"] = g["baro_alt"].diff() * 20.0         # m/s, positive up (20 Hz)
    ys = Y_BASE + [c for c in Y_VEL if g[c].notna().mean() > 0.95]
    cols = []
    for k in range(LAGS + 1):
        for c in U:
            g[f"{c}_l{k}"] = g[c].shift(k); cols.append(f"{c}_l{k}")
    cols += ATT
    for a in "xyz":                           # response-only AR regressors (baseline ar_attitude)
        for k in range(1, 6):
            g[f"gyro_{a}_ar{k}"] = g[f"gyro_{a}"].shift(k)
    if hybrid:                                # manoeuvre aggressiveness (for the noise model)
        for c in U:
            g[f"{c}_d1"] = (g[c] - g[c].shift(1)).abs()
            g[f"{c}_d3"] = (g[c] - g[c].shift(3)).abs()
            cols += [f"{c}_d1", f"{c}_d3"]
        g["speed"] = np.sqrt(g[["vel_x", "vel_y", "vel_z"]].pow(2).sum(axis=1, min_count=3))
        g["climb_abs"] = g["baro_alt"].diff().abs() * 20.0
        cols += ["speed", "climb_abs"]
    if arx:                                   # previous measured outputs (added per domain later)
        for c in Y_BASE + Y_VEL:
            g[f"{c}_prev"] = g[c].shift(1)
    # float32 halves memory for the full dataset (1,379 flights); t stays float64
    f64 = g.select_dtypes("float64").columns.drop("t", errors="ignore")
    g[f64] = g[f64].astype("float32")
    return g, cols, ys


class Twin:
    def __init__(self, n=N_ENS, alpha=ALPHA, seed=0):
        self.n, self.alpha, self.rng = n, alpha, np.random.default_rng(seed)

    def fit(self, X, Y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-9
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        Xs, Ys = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        A = np.c_[Xs, np.ones(len(Xs))]
        self.W = []
        for _ in range(self.n):                 # Poisson bootstrap: weights, no data copies
            w = self.rng.poisson(1.0, len(A)).astype(float)
            Aw = A * w[:, None]
            W = np.linalg.solve(A.T @ Aw + self.alpha * np.eye(A.shape[1]), Aw.T @ Ys)
            self.W.append(W)
        mu, ep = self.predict(X)
        self.alea = ((Y - mu) ** 2).mean(0)
        return self

    def predict(self, X):
        A = np.c_[(X - self.mx) / self.sx, np.ones(len(X))]
        P = np.stack([A @ W for W in self.W]) * self.sy + self.my
        return P.mean(0), P.var(0)

    def std_residual(self, X, Y):
        mu, ep = self.predict(X)
        return (Y - mu) / np.sqrt(self.alea + ep), ep / (self.alea + ep)


class HeteroTwin(Twin):
    """Twin with input-dependent (heteroscedastic) aleatoric noise: a second ridge model
    predicts log(residual^2) from the inputs, so the noise level rises during aggressive
    manoeuvres (large command changes, high speed) and residuals there count for less."""

    def fit(self, X, Y):
        super().fit(X, Y)
        mu, _ = self.predict(X)
        L = np.log((Y - mu) ** 2 + 1e-12)
        A = np.c_[(X - self.mx) / self.sx, np.ones(len(X))]
        self.V = np.linalg.solve(A.T @ A + 10 * self.alpha * np.eye(A.shape[1]), A.T @ L)
        self.vc = ((Y - mu) ** 2 / np.exp(A @ self.V)).mean(0)        # calibrate the level
        return self

    def noise_var(self, X):
        A = np.c_[(X - self.mx) / self.sx, np.ones(len(X))]
        v = np.exp(A @ self.V) * self.vc
        return np.clip(v, CLIP_LO * self.alea, CLIP_HI * self.alea)

    def std_residual(self, X, Y):
        mu, ep = self.predict(X)
        al = self.noise_var(X)
        return (Y - mu) / np.sqrt(al + ep), ep / (al + ep)


class HGBTwin:
    """Nonlinear learned layer: per output, a bag of 3 histogram gradient-boosting models
    (each on a different random subsample) gives the mean and the epistemic spread; a fourth
    model, fitted on residuals of held-out samples, predicts log squared residuals
    (heteroscedastic aleatoric noise), bounded like HeteroTwin."""
    N_BAG, N_SUB = 3, 100_000

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def _hgb(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             min_samples_leaf=50, early_stopping=False,
                                             random_state=int(self.rng.integers(1e9)))

    def fit(self, X, Y):
        n = len(X); idx = self.rng.permutation(n); half = n // 2
        A, B = idx[:half], idx[half:]
        self.models = []
        for _ in range(self.N_BAG):
            sub = self.rng.choice(A, min(self.N_SUB, len(A)), replace=False)
            self.models.append([self._hgb().fit(X[sub], Y[sub, j]) for j in range(Y.shape[1])])
        sb = self.rng.choice(B, min(self.N_SUB, len(B)), replace=False)
        mu, _ = self.predict(X[sb]); r2 = (Y[sb] - mu) ** 2
        self.alea = r2.mean(0)
        self.noise = [self._hgb().fit(X[sb], np.log(r2[:, j] + 1e-12)) for j in range(Y.shape[1])]
        self.vc = (r2 / self._raw_noise(X[sb])).mean(0)
        return self

    def predict(self, X):
        P = np.stack([np.stack([mj.predict(X) for mj in mm], 1) for mm in self.models])
        return P.mean(0), P.var(0)

    def _raw_noise(self, X):
        return np.exp(np.stack([nj.predict(X) for nj in self.noise], 1))

    def noise_var(self, X):
        return np.clip(self._raw_noise(X) * self.vc, CLIP_LO * self.alea, CLIP_HI * self.alea)

    def std_residual(self, X, Y):
        mu, ep = self.predict(X)
        al = self.noise_var(X)
        return (Y - mu) / np.sqrt(al + ep), ep / (al + ep)


def limit_quantile(SC, p):
    """Upper limit per group at exceedance probability p. 'gpd': generalised Pareto fit to the
    15% largest window scores (peaks over threshold), so that small p beyond the sample
    resolution is extrapolated instead of being set by the largest observed score."""
    if LIMIT != "gpd" or p >= 0.15:
        return np.quantile(SC, 1 - p, axis=0)
    from scipy.stats import genpareto
    out = []
    for j in range(SC.shape[1]):
        x = SC[:, j]; u = np.quantile(x, 0.85); y = x[x > u] - u
        if len(y) < 15:
            out.append(np.quantile(x, 1 - p)); continue
        c, _, sc = genpareto.fit(y, floc=0)
        out.append(u + genpareto.ppf(1 - p / 0.15, c, loc=0, scale=sc))
    return np.array(out)


def mewma(r, lam=LAM):
    """MEWMA statistic for standardised residuals (covariance ~ identity)."""
    z = lfilter([lam], [1, -(1 - lam)], r, axis=0)
    return (z ** 2).sum(1) / (lam / (2 - lam))


def first_run(mask, run=RUN):
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        if c >= run:
            return i - run + 1
    return None


RESET = os.environ.get("RESET_POLICY", "1") == "1"   # 1: monitor re-arms after each alarm clears (paper default)


def detect_after_fault(exceed, post):
    """Absolute index of the start of the first run of RUN exceedances that counts as a detection, or None.
    RESET=1 (paper default): an alarm (a run of RUN exceedances) disarms the monitor until the statistic falls
    back below the limit, so an alarm already active at fault onset does not count, and detection needs a new
    alarm after re-arming. RESET=0 (previous behaviour): only samples in the fault window are counted, so an
    alarm carried over from before onset is credited as a detection at onset."""
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


def _group(col):
    """phys_mag_x -> 'mag', phys_kin_vx -> 'kin_v', phys_baro_rate -> 'baro', ..."""
    c = col.replace("phys_", "")
    for g in ("mag", "baro", "kin_v", "kin_p", "gyro_q"):
        if c.startswith(g):
            return g
    return c


def evaluate(data, mode):
    """mode: 'twin', 'twin_arx', 'consistency', 'twin+consistency',
    'hybrid' or 'hybrid+consistency' (physics_twin.HybridTwin instead of the black-box twin)."""
    arx = mode == "twin_arx"
    ar_att = mode.startswith("ar_attitude")       # response-only VAR baseline (Dutta et al. style)
    hybrid = mode.startswith("hybrid")
    physics = not mode.startswith("hybrid_nophys")   # learned layer without physics layer
    post_cal = mode.startswith("hybrid_nophys_cal")  # data-driven twin with per-flight calibration
    # ablation variants: hybrid_physonly, hybrid_cal, hybrid_nohetero (each optionally +consistency)
    calibrate = not mode.startswith("hybrid_physonly")
    learned = not (mode.startswith("hybrid_physonly") or mode.startswith("hybrid_cal"))
    hetero = not mode.startswith("hybrid_nohetero")
    use_twin, use_phys = mode != "consistency", "consistency" in mode
    rows, curves = [], {}
    prepped = {cid: prep(g, arx, hybrid) for cid, g in data.groupby("case_id")}
    for dom, ids in data.groupby("domain").case_id.unique().items():
        # fold assignment seed, restarted for every domain so that a domain gives the same result alone or with others
        rng = np.random.default_rng(int(os.environ.get("FOLD_SEED", "0")))
        ids = list(ids)
        grp_of = None
        if FOLD_GROUPS:
            gmap = pd.read_csv(FOLD_GROUPS, dtype=str).set_index("case_id")["group"]
            if all(c in gmap.index for c in ids):
                grp_of = {c: gmap[c] for c in ids}

        def gsplit(cases, k):
            """split cases into k folds; whole groups (flight days) stay together."""
            if grp_of is None:
                return [list(f) for f in np.array_split(rng.permutation(cases), k)]
            gs = pd.Series([grp_of[c] for c in cases], index=cases)
            order = rng.permutation(gs.unique()); sizes = gs.value_counts()
            order = sorted(order, key=lambda x: -sizes[x])          # largest first, greedy balance
            bins = [[] for _ in range(k)]; load = np.zeros(k)
            for gname in order:
                j = int(np.argmin(load)); bins[j] += list(gs.index[gs == gname]); load[j] += sizes[gname]
            return bins

        if len(ids) <= MAX_LOO:
            folds = [[c] for c in ids]
        elif grp_of is not None:
            folds = gsplit(ids, N_FOLDS)
        else:
            perm = rng.permutation(ids)
            folds = [list(f) for f in np.array_split(perm, N_FOLDS)]
        # outputs / consistency channels available in (almost) every flight of the domain
        _, cols0, _ = prepped[ids[0]]
        ys = [y for y in Y_BASE + Y_VEL if all(y in prepped[c][2] for c in ids)] if use_twin else []
        if hybrid:
            ys = list(HY)
        if ar_att:
            ys = ["gyro_x", "gyro_y", "gyro_z"]
        pcs = ["phys_" + c for c in PHYS
               if all(prepped[i][0]["phys_" + c].notna().mean() > 0.95 for i in ids)] if use_phys else []
        cols = (cols0 + ([f"{y}_prev" for y in ys] if arx else [])) if use_twin else []
        if ar_att:
            cols = [f"gyro_{a}_ar{k}" for a in "xyz" for k in range(1, 6)]
        need = cols + ys + pcs
        groups = (["twin"] * len(ys) if use_twin else []) + [_group(c) for c in pcs]
        names = list(dict.fromkeys(groups))
        gidx = [np.array([i for i, gname in enumerate(groups) if gname == n]) for n in names]

        def healthy(c):
            g = prepped[c][0]
            on = g.t[g.is_fault == 1].iloc[0] if g.is_fault.any() else np.inf
            return g, (g.t >= T0) & (g.t < on) & g[need].notna().all(axis=1)

        def fit(train):
            parts = [healthy(c) for c in train]
            parts = [(g, m) for g, m in parts if m.sum() > 0]
            if sum(int(m.sum()) for _, m in parts) < 50:
                return None
            X = np.vstack([g.loc[m, cols].values for g, m in parts])
            Y = np.vstack([g.loc[m, ys].values for g, m in parts])
            P = np.vstack([g.loc[m, pcs].values for g, m in parts])
            if hybrid:
                cls = HGBTwin if LEARNER == "hgb" else (HeteroTwin if hetero else Twin)
                twin = HybridTwin(cls, calibrate=calibrate and physics, learned=learned,
                                  physics=physics, post_cal=post_cal).fit(parts, cols)
            else:
                twin = Twin().fit(X, Y) if use_twin else None
            pm = ps = None
            if use_phys:                 # trimmed scaling: central 98% of training values
                lo, hi = np.percentile(P, [1, 99], axis=0)
                Pt = np.where((P >= lo) & (P <= hi), P, np.nan)
                pm, ps = np.nanmean(Pt, 0), np.nanstd(Pt, 0) + 1e-9
            return twin, pm, ps

        def resid(model, g, m, raw=False):
            twin, pm, ps = model
            parts, epi = [], np.zeros(int(m.sum()))
            if use_twin:
                if hybrid:
                    e, ep = twin.std_residual(g, m)
                else:
                    e, ep = twin.std_residual(g.loc[m, cols].values, g.loc[m, ys].values)
                parts.append(e); epi = ep.mean(1)
            if use_phys:
                parts.append((g.loc[m, pcs].values - pm) / ps)
            e = np.hstack(parts)
            if RESCALE and "cal_window" in g:      # label-free: centre and scale on the flight's own first seconds
                cw = g.loc[m, "cal_window"].values & np.isfinite(e).all(1)
                n, k = int(cw.sum()), 200.0           # shrinkage towards (0, 1) with weight of 200 samples
                if n >= 20:
                    mu = e[cw].sum(0) / (n + k)
                    sd = np.sqrt(((e[cw] - mu) ** 2).sum(0) + k) / np.sqrt(n + k)
                    e = (e - mu) / sd
            if raw:
                return e, epi
            if THRESH == "tukey_raw":   # per group: largest absolute standardised residual (common threshold applies channel-wise)
                return np.stack([np.nanmax(np.abs(e[:, ix]), 1) for ix in gidx], 1), epi
            return np.stack([mewma(e[:, ix]) for ix in gidx], 1), epi

        def flight_score(stats):
            """Scores of healthy data in windows of WIN_S seconds: per group, the max over
            the window of the 3-point running minimum (= the smallest limit that would NOT
            have raised a run-of-3 alarm). Windows make the limit independent of flight
            length: about FLIGHT_FA of healthy WIN_S-second stretches raise an alarm."""
            if len(stats) < RUN:
                return None
            rm = pd.DataFrame(stats).rolling(RUN).min().values
            w = int(WIN_S * 20)
            out = [np.nanmax(rm[i:i + w], 0) for i in range(0, len(rm), w)
                   if np.isfinite(rm[i:i + w]).any() and (len(rm[i:i + w]) >= w // 3 or i == 0)]
            return np.vstack(out) if out else None

        if CAL_PREV and hybrid:   # previous-flight calibration: each flight is calibrated on the previous flight of its day
            import physics_twin as _pt
            pmap = pd.read_csv(CAL_PREV, dtype=str).dropna().set_index("case_id")["prev"]
            _pt.CAL_SOURCE.clear()
            _pt.CAL_SOURCE.update({c: prepped[pmap[c]][0] for c in ids if c in pmap.index and pmap[c] in prepped})
        _sub_rng = np.random.default_rng(int(os.environ.get("SUB_SEED", "0")))
        for fold in folds:
            train = [c for c in ids if c not in fold]
            if TRAIN_FRAC < 1:   # data-efficiency test: train and calibrate on a random fraction of the training flights
                _k = max(1, int(round(TRAIN_FRAC * len(train))))
                train = [train[i] for i in sorted(_sub_rng.choice(len(train), _k, replace=False))]
            model = fit(train)
            if model is None:
                continue
            # control limits set on WHOLE healthy flights so that about FLIGHT_FA of
            # healthy flights would raise an alarm (Bonferroni over groups); scores come
            # from flights the twin did NOT see (inner 3-fold) when enough flights exist
            scores = []
            if len(train) >= 15:
                inner = gsplit(train, 3) if grp_of is not None else np.array_split(rng.permutation(train), 3)
                for k in range(3):
                    sub = fit([c for c in train if c not in inner[k]])
                    if sub is None:
                        continue
                    for c in inner[k]:
                        g, m = healthy(c)
                        if m.sum() >= RUN:
                            if THRESH == "tukey_raw":
                                scores.append(resid(sub, g, m, raw=True)[0].astype(np.float32)); continue
                            if THRESH == "tukey_mewma":
                                scores.append(resid(sub, g, m)[0]); continue
                            sc = flight_score(resid(sub, g, m)[0])
                            if sc is not None: scores.append(sc)
            else:
                for c in train:
                    g, m = healthy(c)
                    if m.sum() >= RUN:
                        sc = flight_score(resid(model, g, m)[0])
                        if sc is not None: scores.append(sc)
            SC = np.vstack(scores)
            KDIV = 1 if os.environ.get("NO_BONF") == "1" else len(names)   # NO_BONF=1: test without the Bonferroni split over groups
            if THRESH == "window":
                ucl = limit_quantile(SC, FLIGHT_FA / KDIV)
                ucl_lv = {lv: limit_quantile(SC, lv / KDIV) for lv in FA_LEVELS}
            if THRESH == "tukey_raw":      # one common symmetric threshold: largest absolute adjacent value over channels
                lo_a, hi_a = tukey_adjacent(SC)
                tau = float(np.max(np.maximum(np.abs(lo_a), np.abs(hi_a))))
                ucl = np.full(len(names), tau); ucl_lv = {lv: ucl for lv in FA_LEVELS}
                print(f"[{dom}] tukey_raw threshold {tau:.3f}", flush=True)
            elif THRESH == "tukey_mewma":  # per group: upper adjacent value of the MEWMA statistic
                ucl = tukey_adjacent(SC)[1]; ucl_lv = {lv: ucl for lv in FA_LEVELS}
                print(f"[{dom}] tukey_mewma thresholds " + " ".join(f"{n}={u:.2f}" for n, u in zip(names, ucl)), flush=True)
            NB = os.environ.get("NO_BONF_GROUP")   # test: drop the Bonferroni split for one group only
            if NB and NB in names:
                ucl = ucl.copy(); ucl[names.index(NB)] = limit_quantile(SC, FLIGHT_FA)[names.index(NB)]
            day_ucl = {}
            if DAYRECAL and grp_of is not None:
                for day in sorted(set(grp_of[c] for c in fold)):
                    dfl = sorted([c for c in ids if grp_of[c] == day], key=lambda c: DAY_TIME[c])
                    calib = dfl[:DAYRECAL]
                    sc_d = []
                    # score calibration flights with a twin that did not see them (in-sample residuals are too small)
                    sub = fit([c for c in train if c not in calib])
                    for c in (calib if sub is not None else []):
                        g, m = healthy(c)
                        if m.sum() >= RUN:
                            sc = flight_score(resid(sub, g, m)[0])
                            if sc is not None: sc_d.append(sc)
                    if sc_d:
                        _sd = np.vstack(sc_d)
                        day_ucl[day] = (limit_quantile(_sd, FLIGHT_FA / len(names)), calib, sum(len(x) for x in sc_d),
                                        {lv: limit_quantile(_sd, lv / len(names)) for lv in FA_LEVELS})
            for test in fold:
                g = prepped[test][0]
                m = (g.t >= T0) & g[need].notna().all(axis=1)
                if m.sum() < 20:
                    continue
                ucl_use, n_cal_windows, calib_ids = ucl, None, []
                ucl_lv_use = ucl_lv
                if DAYRECAL and grp_of is not None:
                    day = grp_of[test]
                    if day not in day_ucl or test in day_ucl[day][1]:
                        continue                  # no calibration for this day, or a calibration flight
                    ucl_use, calib_ids, n_cal_windows = day_ucl[day][0], day_ucl[day][1], day_ucl[day][2]
                    ucl_lv_use = day_ucl[day][3]
                if CAL_STRESS and hybrid and g.is_fault.any():
                    g = g.copy(); on0 = g.t[g.is_fault == 1].iloc[0]
                    g["cal_window"] = (g.t >= on0) & (g.t < on0 + 10.0)
                t = g.t[m].values; y = g.is_fault[m].values
                st, epi = resid(model, g, m)
                st = st * TEST_SCALE ** 2   # calibration-mismatch test: test residuals scaled by TEST_SCALE (MEWMA is quadratic)
                ratio = st / ucl_use                         # > 1: that group is out of control
                T2 = ratio.max(1)
                has_fault = bool(y.any())
                onset = t[np.argmax(y == 1)] if has_fault else np.inf
                pre, post = t < onset, (t >= onset) & (y == 1)   # detect only while fault active
                kf = first_run(T2[pre] > 1)
                k = detect_after_fault(T2 > 1, post) if has_fault else None      # absolute sample index
                grp = names[int(np.argmax(ratio[k + RUN - 1]))] if k is not None else None
                curve = {}
                for lv, u in ucl_lv_use.items():       # operating curve over target levels (day-wise limits when DAYRECAL)
                    T2l = (st / u).max(1)
                    curve[f"fa_at_{lv}"] = first_run(T2l[pre] > 1) is not None
                    curve[f"det_at_{lv}"] = (detect_after_fault(T2l > 1, post) is not None) if has_fault else None
                rows.append(dict(case_id=test, calib_flights=len(calib_ids), calib_windows=n_cal_windows, **curve,
                                 delay_s=None if k is None else round(t[k] - onset, 2),
                                 false_alarm_before_onset=kf is not None,
                                 false_alarm_time_s=None if kf is None else round(t[pre][kf], 1),
                                 false_alarm_group=None if kf is None else names[int(np.argmax(ratio[pre][kf + RUN - 1]))],
                                 monitored_healthy_s=round(float(pre.sum()) / 20.0, 1),
                                 alarm_group=grp,
                                 epistemic_share_pre=round(float(epi[pre].mean()), 3) if use_twin and pre.any() else None,
                                 epistemic_share_post=round(float(epi[post].mean()), 3) if use_twin and post.any() else None))
                curves[test] = (t, T2, 1.0, onset if has_fault else None)
    return pd.DataFrame(rows), curves


def summary(df):
    df = df.assign(detected=df.delay_s.notna(), faulty=df.fault_onset_s.notna())
    return df.groupby("domain").agg(flights=("case_id", "size"), faulty_flights=("faulty", "sum"),
                                    detected=("detected", "sum"),
                                    false_alarms=("false_alarm_before_onset", "sum"),
                                    median_delay_s=("delay_s", "median"))


def main(src, out):
    os.makedirs(out, exist_ok=True)
    data = load_all(src)
    info = case_table(data)
    results = {}
    for name, mode in [("twin_inputs_only", "twin"), ("twin_arx", "twin_arx"),
                       ("consistency_only", "consistency"),
                       ("twin_plus_consistency", "twin+consistency"),
                       ("hybrid", "hybrid"), ("hybrid_plus_consistency", "hybrid+consistency")]:
        r, curves = evaluate(data, mode)
        r = info.merge(r, on="case_id")
        r.to_csv(os.path.join(out, f"{name}_results.csv"), index=False)
        results[name] = (r, curves)
    comp, methods = [], [(k, v[0]) for k, v in results.items()]
    bpath = os.path.join(out, "mewma_baseline_results.csv")
    if os.path.exists(bpath):
        methods = [("per-flight MEWMA", pd.read_csv(bpath, dtype={"case_id": str}))] + methods
    for name, df in methods:
        s = summary(df); s.insert(0, "method", name); comp.append(s.reset_index())
    comp = pd.concat(comp)
    comp.to_csv(os.path.join(out, "comparison.csv"), index=False)

    # detection rate per fault type, all methods side by side
    rates = []
    for name, df in methods:
        f = df[df.fault_onset_s.notna()].assign(det=lambda d: d.delay_s.notna())
        rates.append(f.groupby(["domain", "fault_type"]).det.mean().rename(name))
    pd.concat(rates, axis=1).round(2).to_csv(os.path.join(out, "detection_rate_by_fault_and_method.csv"))

    # figure: manoeuvring real flight where the baseline false-alarmed
    r, curves = results["twin_inputs_only"]
    if not all(c in curves for c in ["3100000000", "3000000000"]):
        return comp, results
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2))
    for ax, cid, title in zip(axes, ["3100000000", "3000000000"],
                              ["Real flight, waypoint mission, motor fault",
                               "Real flight, hover, motor fault"]):
        t, T2, ucl, onset = curves[cid]
        ax.semilogy(t, T2, lw=0.9, color="#2b6cb0", label="MEWMA statistic / its control limit")
        ax.axhline(ucl, color="#c53030", ls="--", lw=1, label="control limit (= 1)")
        ax.axvline(onset, color="#444", ls=":", lw=1, label="fault injected")
        ax.set_title(f"{title} (case {cid})", fontsize=10, loc="left")
        ax.set_ylabel("statistic (log)")
    axes[0].legend(fontsize=8, loc="upper left"); axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "twin_residual_example.png"), dpi=150)
    return comp, results


if __name__ == "__main__":
    comp, results = main(sys.argv[1], sys.argv[2])
    pd.set_option("display.width", 200)
    print(comp.to_string(index=False))
    for k, (r, _) in results.items():
        print("\n", k)
        print(r[["case_id", "domain", "status", "fault_type", "delay_s", "false_alarm_before_onset",
                 "epistemic_share_pre", "epistemic_share_post"]].to_string(index=False))
