import os
"""
consistency.py - physics-based sensor-consistency residuals (analytical redundancy)
for RflyMAD flights. Each residual is ~0 when the sensors agree with each other and
with basic rigid-body kinematics, and moves away from 0 when one sensor is faulty.

Conventions (PX4): q = [w, x, y, z] rotates body (FRD) -> NED; accelerometer gives
specific force f (hover: f_z ~ -9.81); NED z points down.

Residual groups
  mag_*      magnetometer vs Earth field rotated into the body frame by the attitude
             (Earth field estimated per flight from its first healthy seconds)
  baro_alt   barometric altitude vs estimated altitude (-pos_z), offset removed
  baro_rate  barometric climb rate vs -vel_z
  kin_v*     velocity change vs IMU prediction   dv/dt  -  (R(q) f + g)
  kin_p*     position change vs velocity          dp/dt  -  v
  gyro_q*    gyroscope vs body rates implied by the attitude quaternion
"""
import numpy as np
import pandas as pd

G = 9.80665
RATE = 20.0
CAUSAL = os.environ.get("CAUSAL", "0") == "1"   # 1: trailing averages and backward differences (online use)

def _d(a):
    """time derivative: central differences, or backward differences if CAUSAL"""
    a = np.asarray(a)
    return np.diff(a, axis=0, prepend=a[:1]) * RATE if CAUSAL else np.gradient(a, 1.0 / RATE, axis=0)

REF_WINDOW = (3.0, 13.0)       # s, used to estimate per-flight constants (before any fault)
SMOOTH = 5                     # samples (0.25 s) moving average for derivatives

CHANNELS = (["mag_x", "mag_y", "mag_z", "baro_alt", "baro_rate"] +
            [f"kin_v{a}" for a in "xyz"] + [f"kin_p{a}" for a in "xyz"] +
            [f"gyro_q{a}" for a in "xyz"])


def _rot(q):
    """Rotation matrices (n,3,3), body -> NED, from quaternions (n,4) [w,x,y,z]."""
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _smooth(a, k=SMOOTH):
    return pd.DataFrame(a).rolling(k, center=not CAUSAL, min_periods=1).mean().values


def _deriv(a):
    return _smooth(_d(_smooth(a)))


def _continuous_quat(q):
    q = q.copy()
    for i in range(1, len(q)):                       # q and -q are the same rotation
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def phys_residuals(g):
    """Return a DataFrame of consistency residuals aligned with g (NaN where inputs are missing)."""
    t = g["t"].values
    out = pd.DataFrame(index=g.index, columns=CHANNELS, dtype=float)
    q = g[["q0", "q1", "q2", "q3"]].values.astype(float)
    if np.isnan(q).all():
        return out
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = _continuous_quat(np.nan_to_num(q, nan=0.0) + np.isnan(q) * np.array([1, 0, 0, 0]))
    R = _rot(q)
    ref = (t >= REF_WINDOW[0]) & (t < REF_WINDOW[1])
    if g["is_fault"].any():                           # never estimate constants from faulty data
        ref &= t < t[np.argmax(g["is_fault"].values == 1)]

    # magnetometer: body field vs R^T m_ned, m_ned estimated from the reference window
    mb = g[["mag_x", "mag_y", "mag_z"]].values.astype(float)
    if ref.sum() > 10 and not np.isnan(mb[ref]).all():
        m_ned = np.nanmean(np.einsum("nij,nj->ni", R[ref], mb[ref]), axis=0)
        out[["mag_x", "mag_y", "mag_z"]] = mb - np.einsum("nji,j->ni", R, m_ned)

    # barometer vs estimated altitude and climb rate
    pz, vz, ba = g["pos_z"].values, g["vel_z"].values, g["baro_alt"].values
    if not np.isnan(pz).all():
        d = ba - (-pz)
        out["baro_alt"] = d - np.nanmean(d[ref]) if ref.any() else d - np.nanmean(d)
    if not np.isnan(vz).all():
        out["baro_rate"] = _deriv(ba[:, None])[:, 0] - (-vz)

    # kinematics: dv/dt vs IMU-predicted acceleration, dp/dt vs v
    v = g[["vel_x", "vel_y", "vel_z"]].values.astype(float)
    f = g[["acc_x", "acc_y", "acc_z"]].values.astype(float)
    if not np.isnan(v).all():
        a_imu = _smooth(np.einsum("nij,nj->ni", R, f) + np.array([0, 0, G]))
        out[[f"kin_v{a}" for a in "xyz"]] = _deriv(v) - a_imu
        p = g[["pos_x", "pos_y", "pos_z"]].values.astype(float)
        out[[f"kin_p{a}" for a in "xyz"]] = _deriv(p) - _smooth(v)

    # gyroscope vs attitude-implied body rates: w = 2 vec(conj(q) * dq/dt)
    dq = _d(q)
    w0, x0, y0, z0 = q.T
    qc = np.stack([w0, -x0, -y0, -z0], 1)
    a1, b1, c1, d1 = qc.T
    a2, b2, c2, d2 = dq.T
    vec = np.stack([a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                    a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                    a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2], 1)
    w_att = _smooth(2 * vec)
    gyro = _smooth(g[["gyro_x", "gyro_y", "gyro_z"]].values.astype(float))
    out[[f"gyro_q{a}" for a in "xyz"]] = gyro - w_att
    return out
