"""
convert_ulog.py - turn raw RflyMAD flights (PX4 .ulg logs) into one 20 Hz CSV
per flight, in the same column format as the official ProcessData files, so
rflymad.py / demo_mewma.py / twin_baseline.py run unchanged on the full dataset.

Works on extracted folders or directly on the downloaded .zip files (only the
.ulg logs are pulled out, one at a time, so disk use stays small).

Case ID written by this converter (same scheme as ProcessData, extended):
    digit 1      1 = SIL, 2 = HIL, 3 = Real
    digit 2      0 hover, 1 waypoint, 2 velocity, 3 circling, 4 acceleration, 5 deceleration
    digits 3-4   00 motor, 05 accelerometer, 06 gyroscope, 07 magnetometer, 08 barometer,
                 09 GPS  (as in ProcessData)
                 01 propeller, 02 low voltage, 03 wind, 04 load loss, 10 no fault
                 (codes chosen by this converter; not official RflyMAD codes)
    last 6       running index within that domain/status/fault group

Folder layouts: both the Beihang sample zips (SIL/hover/motor/...) and the
Kaggle copy (Real-Sensors/hover-GPS/..., HIL-Wind/acce-wind/..., Real-Motor/...,
Real-No_Fault/...) are recognised; the official 10-digit ID of SIL/HIL test
cases is kept in conversion_index.csv (column official_id).

Note: HIL logs DO contain measured position/velocity, so converted HIL files
have pos_*/vel_* (the official HIL ProcessData files do not).

Usage:
    python convert_ulog.py <input folder or folder of zips> <output folder>
"""
import os
import re
import sys
import zipfile
import tempfile
import numpy as np
import pandas as pd
from pyulog import ULog

RATE = 20.0
DOMAIN = {"SIL": "1", "HIL": "2", "Real": "3"}
STATUS = {"hover": "0", "waypoint": "1", "velocity": "2", "circling": "3",
          "acce": "4", "dece": "5"}
FAULT = {"motor": "00", "propeller": "01", "low_voltage": "02", "wind_affect": "03",
         "load_lose": "04", "accelerometer": "05", "gyroscope": "06",
         "magnetometer": "07", "barometer": "08", "GPS": "09", "no_fault": "10"}

# topic -> fields, using the ProcessData column names "_<topic>_0_<field>"
TOPICS = {
    "actuator_controls_0": [f"control[{i}]" for i in range(4)],
    "actuator_outputs": [f"output[{i}]" for i in range(4)],
    "sensor_combined": [f"gyro_rad[{i}]" for i in range(3)] + [f"accelerometer_m_s2[{i}]" for i in range(3)],
    "vehicle_air_data": ["baro_alt_meter", "baro_temp_celcius", "baro_pressure_pa"],
    "vehicle_attitude": [f"q[{i}]" for i in range(4)],
    "vehicle_local_position": ["x", "y", "z", "vx", "vy", "vz"],
    "vehicle_magnetometer": [f"magnetometer_ga[{i}]" for i in range(3)],
    "rfly_ctrl_lxl": ["id", "mode"] + [f"controls[{i}]" for i in range(6)],
}
HOLD = {"rfly_ctrl_lxl"}            # step signals: zero-order hold, not interpolation


def _topic(ulog, name):
    """Highest-rate instance of a topic (e.g. actuator_outputs has two in real logs)."""
    cands = [d for d in ulog.data_list if d.name == name]
    return max(cands, key=lambda d: len(d.data["timestamp"])) if cands else None


def ulog_to_frame(path):
    ulog = ULog(path, list(TOPICS))
    ctrl = _topic(ulog, "rfly_ctrl_lxl")
    if ctrl is None:
        raise ValueError("no rfly_ctrl_lxl topic (fault-injection channel) in log")
    t0, t1 = ctrl.data["timestamp"][0], ctrl.data["timestamp"][-1]   # offboard test window
    grid = np.arange(t0, t1, 1e6 / RATE)
    out = {"Index": np.arange(len(grid)), "Timestamp": grid}
    for name, fields in TOPICS.items():
        d = _topic(ulog, name)
        for f in fields:
            col = f"_{name}_0_{f}"
            if d is None or f not in d.data:
                out[col] = np.full(len(grid), np.nan); continue
            ts, v = d.data["timestamp"], d.data[f].astype(float)
            if name in HOLD:
                idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
                out[col] = v[idx]
            elif len(ts) > 2 * len(grid) * (ts[-1] - ts[0]) / max(t1 - t0, 1):
                # high-rate signal: average inside each 50 ms bin (anti-aliasing)
                b = np.floor((ts - grid[0]) / (1e6 / RATE) + 0.5).astype(int)
                ok = (b >= 0) & (b < len(grid))
                s = np.bincount(b[ok], weights=v[ok], minlength=len(grid))
                c = np.bincount(b[ok], minlength=len(grid))
                m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
                good = ~np.isnan(m)
                out[col] = np.interp(grid, grid[good], m[good]) if good.any() else m
            else:
                out[col] = np.interp(grid, ts, v, left=np.nan, right=np.nan)
    return pd.DataFrame(out)


SENSOR = {"acce": "accelerometer", "gyro": "gyroscope", "mag": "magnetometer",
          "baro": "barometer", "GPS": "GPS", "gps": "GPS", "wind": "wind_affect"}
CATEGORY = {"Motor": "motor", "No_Fault": "no_fault", "NoFault": "no_fault",
            "Wind": "wind_affect", "Propeller": "propeller", "Voltage": "low_voltage",
            "Load": "load_lose", "Sensors": None}


def parse_path(p):
    """Read domain / flight status / fault type from the folder names.

    Handles both layouts seen so far:
      sample zips : SIL/hover/motor/TestCase_3_1000000002/Log/x.ulg
      Kaggle copy : HIL-Wind/acce-wind/TestCase_.../Log/x.ulg
                    Real-Sensors/hover-GPS/100_1/log_.../x.ulg
                    Real-Motor/hover/10_1/...   Real-No_Fault/acce/451_1/...
    Returns (domain, status, fault, official_case_id or "").
    """
    parts = [x for x in re.split(r"[\\/]", p) if x]
    dom = st = fa = cat = None
    official = ""
    for x in parts:
        toks = x.split("-")
        if dom is None and toks[0] in DOMAIN:
            dom = toks[0]
            if len(toks) > 1:
                cat = "-".join(toks[1:])
            continue
        if dom is None:
            continue
        m = re.match(r"TestCase_\d+_(\d{10})$", x)
        if m:
            official = m.group(1); continue
        if st is None and toks[0] in STATUS:
            st = toks[0]
            if len(toks) > 1 and toks[1] in SENSOR:
                fa = SENSOR[toks[1]]
            continue
        if fa is None and x in FAULT:          # sample layout: .../hover/motor/...
            fa = x
    if fa is None and cat in CATEGORY:
        fa = CATEGORY[cat]
    return dom, st, fa, official


def iter_ulogs(src):
    """Yield (logical path, local file path, cleanup flag) for every .ulg.

    Skips duplicates (same file name and size), e.g. when a dataset holds both
    a zip and its extracted copy.
    """
    seen = set()
    for root, _, files in os.walk(src):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            if fn.endswith(".ulg"):
                key = (fn, os.path.getsize(full))
                if key in seen:
                    continue
                seen.add(key)
                yield full, full, False
            elif fn.endswith(".zip"):
                with zipfile.ZipFile(full) as z:
                    for info in sorted(z.infolist(), key=lambda i: i.filename):
                        m = info.filename
                        key = (os.path.basename(m), info.file_size)
                        if m.endswith(".ulg") and key not in seen:
                            seen.add(key)
                            tmp = tempfile.NamedTemporaryFile(suffix=".ulg", delete=False)
                            tmp.write(z.read(m)); tmp.close()
                            yield os.path.join(full, m), tmp.name, True


def convert(src, dst, domains=None, limit_per_group=None):
    """Convert every flight under src. Optional filters for a quick first run:
    domains = e.g. ["Real"]; limit_per_group = max flights per domain/status/fault."""
    os.makedirs(dst, exist_ok=True)
    counters, index = {}, []
    for logical, local, cleanup in iter_ulogs(src):
        dom, st, fa, official = parse_path(logical)
        try:
            if None in (dom, st, fa):
                raise ValueError("could not read domain/status/fault from path")
            key = DOMAIN[dom] + STATUS[st] + FAULT[fa]
            n = counters.get(key, 0)
            if (domains and dom not in domains) or (limit_per_group and n >= limit_per_group):
                continue
            counters[key] = n + 1
            cid = f"{key}{n:06d}"
            df = ulog_to_frame(local)
            df.to_csv(os.path.join(dst, f"Case_{cid}.csv"), index=False)
            cmd = df["_rfly_ctrl_lxl_0_id"]
            index.append(dict(case_id=cid, official_id=official, domain=dom, status=st,
                              fault_type=fa, n_samples=len(df),
                              fault_samples=int((cmd != cmd.iloc[0]).sum()),
                              fault_ids=" ".join(str(int(v)) for v in sorted(cmd.unique()) if v == v)[:60],
                              source=logical, error=""))
        except Exception as e:                              # keep going, log the failure
            index.append(dict(case_id="", official_id=official, domain=dom, status=st,
                              fault_type=fa, n_samples=0, fault_samples=0, fault_ids="",
                              source=logical, error=str(e)[:200]))
        finally:
            if cleanup:
                os.remove(local)
        if len(index) % 50 == 0:
            print(f"{len(index)} logs processed", flush=True)
    idx = pd.DataFrame(index, columns=["case_id", "official_id", "domain", "status", "fault_type",
                                       "n_samples", "fault_samples", "fault_ids", "source", "error"])
    if idx.empty:
        print("WARNING: no .ulg flight logs found under", src)
    idx.to_csv(os.path.join(dst, "conversion_index.csv"), index=False)
    return idx


if __name__ == "__main__":
    idx = convert(sys.argv[1], sys.argv[2])
    print(idx.groupby(["domain", "fault_type"]).size().unstack(fill_value=0))
    print("failed:", (idx.error != "").sum())
