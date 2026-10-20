"""
rflymad.py - loader for RflyMAD ProcessData CSV files.

Works on the 36-case ProcessData sample and on the full dataset once processed
the same way (one Case_XXXXXXXXXX.csv per flight, resampled to 20 Hz).

Case ID decoding (inferred from the sample files; check against the
RflyMAD documentation before publishing):
    digit 1      domain         1 = SIL, 2 = HIL, 3 = Real flight
    digit 2      flight status  0 = hover, 1 = waypoint, 2 = velocity,
                                3 = circling, 4 = acceleration, (5 = deceleration)
    digits 3-4   fault type     00 = motor, 05 = accelerometer, 06 = gyroscope,
                                07 = magnetometer, 08 = barometer, 09 = GPS
    last digits  repeat index

Note: HIL ProcessData files lack measured position and velocity
(pos_*, vel_* are NaN there); all other signals exist in every domain.

Fault label (is_fault):
    fault-injection command id differs from its idle value (0 in SIL/HIL,
    1500 in Real); injection id is 12345x, last digit = fault code
    (0 motor, 2 accelerometer, 3 gyroscope, 4 barometer, 5 magnetometer, 6 GPS).
    SIL/HIL also carry the simulator flag, kept as sim_fault_state.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

DOMAIN = {"1": "SIL", "2": "HIL", "3": "Real"}
STATUS = {"0": "hover", "1": "waypoint", "2": "velocity", "3": "circling",
          "4": "acceleration", "5": "deceleration"}
FAULT = {"00": "motor", "01": "propeller", "02": "low_voltage", "03": "wind_affect",
         "04": "load_lose", "05": "accelerometer", "06": "gyroscope",
         "07": "magnetometer", "08": "barometer", "09": "GPS", "10": "no_fault"}
# 01-04 and 10 are codes assigned by convert_ulog.py, not official RflyMAD codes

# Signals present in every domain (SIL, HIL, Real), renamed to short names
COMMON = {
    "_actuator_controls_0_0_control[0]": "u_roll",
    "_actuator_controls_0_0_control[1]": "u_pitch",
    "_actuator_controls_0_0_control[2]": "u_yaw",
    "_actuator_controls_0_0_control[3]": "u_thrust",
    "_actuator_outputs_0_output[0]": "pwm1",
    "_actuator_outputs_0_output[1]": "pwm2",
    "_actuator_outputs_0_output[2]": "pwm3",
    "_actuator_outputs_0_output[3]": "pwm4",
    "_sensor_combined_0_gyro_rad[0]": "gyro_x",
    "_sensor_combined_0_gyro_rad[1]": "gyro_y",
    "_sensor_combined_0_gyro_rad[2]": "gyro_z",
    "_sensor_combined_0_accelerometer_m_s2[0]": "acc_x",
    "_sensor_combined_0_accelerometer_m_s2[1]": "acc_y",
    "_sensor_combined_0_accelerometer_m_s2[2]": "acc_z",
    "_vehicle_air_data_0_baro_alt_meter": "baro_alt",
    "_vehicle_air_data_0_baro_pressure_pa": "baro_pa",
    "_vehicle_air_data_0_baro_temp_celcius": "baro_temp",
    "_vehicle_attitude_0_q[0]": "q0",
    "_vehicle_attitude_0_q[1]": "q1",
    "_vehicle_attitude_0_q[2]": "q2",
    "_vehicle_attitude_0_q[3]": "q3",
    "_vehicle_local_position_0_x": "pos_x",
    "_vehicle_local_position_0_y": "pos_y",
    "_vehicle_local_position_0_z": "pos_z",
    "_vehicle_local_position_0_vx": "vel_x",
    "_vehicle_local_position_0_vy": "vel_y",
    "_vehicle_local_position_0_vz": "vel_z",
    "_vehicle_magnetometer_0_magnetometer_ga[0]": "mag_x",
    "_vehicle_magnetometer_0_magnetometer_ga[1]": "mag_y",
    "_vehicle_magnetometer_0_magnetometer_ga[2]": "mag_z",
}


def decode_case(case_id: str) -> dict:
    """Turn '3309000000' into readable metadata."""
    return {
        "case_id": case_id,
        "domain": DOMAIN.get(case_id[0], "?"),
        "status": STATUS.get(case_id[1], "?"),
        "fault_type": FAULT.get(case_id[2:4], f"code_{case_id[2:4]}"),
        "repeat": int(case_id[4:]),
    }


def _fault_label(df: pd.DataFrame) -> np.ndarray:
    """0/1 label from the fault-injection command (same rule in all domains).

    In SIL/HIL the simulator's own 'UAVState_data_fault_state' switches
    about 0.3-0.4 s after the command, so the command is used as onset and
    the simulator flag is kept separately as 'sim_fault_state'.
    """
    idc = next(c for c in df.columns
               if c.endswith("send_data_id") or c == "_rfly_ctrl_lxl_0_id")
    idle = df[idc].iloc[0]
    return (df[idc].values != idle).astype(int)


def load_case(path: str) -> pd.DataFrame:
    """Load one Case_*.csv: t [s], common signals, is_fault label (0/1), metadata."""
    raw = pd.read_csv(path)
    case_id = os.path.basename(path)[5:-4]
    meta = decode_case(case_id)
    # HIL ProcessData has no measured local position/velocity (only setpoints),
    # so missing signals are kept as NaN columns.
    out = raw.reindex(columns=list(COMMON)).rename(columns=COMMON)
    out.insert(0, "t", (raw["Timestamp"] - raw["Timestamp"].iloc[0]) / 1e6)
    out["is_fault"] = _fault_label(raw)
    if "UAVState_data_fault_state" in raw:
        out["sim_fault_state"] = (raw["UAVState_data_fault_state"].values != 0).astype(int)
    for k, v in meta.items():
        out[k] = v
    return out


def load_all(folder: str) -> pd.DataFrame:
    """Load every Case_*.csv in a folder into one long frame."""
    files = sorted(glob.glob(os.path.join(folder, "Case_*.csv")))
    return pd.concat([load_case(f) for f in files], ignore_index=True)


def case_table(data: pd.DataFrame) -> pd.DataFrame:
    """One row per case: duration and fault onset/offset times."""
    rows = []
    for cid, g in data.groupby("case_id"):
        on = g.loc[g.is_fault == 1, "t"]
        rows.append({**decode_case(cid),
                     "n_samples": len(g),
                     "duration_s": round(g.t.iloc[-1], 1),
                     "fault_onset_s": round(on.iloc[0], 1) if len(on) else None,
                     "fault_end_s": round(on.iloc[-1], 1) if len(on) else None})
    return pd.DataFrame(rows)
