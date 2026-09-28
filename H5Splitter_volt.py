# =====================================================================
# Split PEINCon "event mode" HDF5 into separate files per AO voltage.
# 19/09/2925
# =====================================================================

"""
Split PEINCon "event mode" HDF5 into separate files per AO voltage.

What it does:
 Loads the EVENTS HDF5 (structured dataset with event timestamps).
 Loads the AO HDF5 (piecewise-constant (timestamp_ms, ao_value_V) log).
 For each event, finds AO voltage at the event start time.
 Rounds to mV (configurable), groups events by voltage.
 Writes each group to its own HDF5 (same dtype), named with the voltage.

Usage:
 Edit USER SETTINGS below.
"""



# ---------------------------------------------------------------------

### USER Parameters ###

EVENT_H5_PATH = r"/Volumes/jcronk/project_cronk/Proteins/peptides/S4_3007_01.h5"     # main EVENT file
GROUP_STEP_V   = 0.001    # quantization step in V (0.001 V = 1 mV)
MIN_EVENTS     = 1        # skip writing files with fewer than this many events
COMPRESSION    = "gzip"
COMP_LEVEL     = 4

# ---------------------------------------------------------------------

# Imports
import os
import sys
import h5py
import numpy as np
from bisect import bisect_right
from datetime import datetime, timezone



def derive_ao_path(event_path: str) -> str:
    base, ext = os.path.splitext(os.path.abspath(event_path))
    if base.casefold().endswith("_events"):
        base = base[:-7]
    return base + "_AO" + ext if not base.endswith("_AO") else base + ext


def volts_to_safe_suffix(v: float) -> str:
    """Return suffix like 'V_pos200mV' for +0.200 V (no extra dots)."""
    mv = int(round(v * 1000.0))
    sign = "pos" if mv >= 0 else "neg"
    return f"{sign}{abs(mv)}mV"


class AOTrack:
    """Piecewise-constant AO timeline (times in SECONDS, values in VOLTS)."""
    def __init__(self, t_s, v_v, default_v=0.0):
        t = np.asarray(t_s, dtype=float)
        v = np.asarray(v_v, dtype=float)
        order = np.argsort(t, kind="stable")
        t = t[order]
        v = v[order]
        # de-duplicate consecutive duplicates (same time & value)
        if t.size:
            keep = np.ones_like(t, dtype=bool)
            keep[1:] = (t[1:] != t[:-1]) | (v[1:] != v[:-1])
            t = t[keep]
            v = v[keep]
        self.t = t
        self.v = v
        self.default = float(default_v)

    def value_at(self, t_s: float) -> float:
        if self.t.size == 0:
            return self.default
        idx = bisect_right(self.t, t_s) - 1
        return self.v[idx] if idx >= 0 else self.default


def load_ao(ao_h5_path: str) -> AOTrack:
    """
    Load AO timeline from '<events>_AO.h5'.
    IMPORTANT: Recording.py / AO_functions.py write AO timestamps in **milliseconds**.
    We convert to **seconds** here to match event timestamps.
    """
    if not os.path.exists(ao_h5_path):
        raise FileNotFoundError(f"AO file not found: {ao_h5_path}")
    with h5py.File(ao_h5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError("AO file missing dataset 'data'.")
        d = f["data"]
        if d.dtype.names is None or "timestamp" not in d.dtype.names or "ao_value" not in d.dtype.names:
            raise RuntimeError("AO '/data' must be a structured array with fields ('timestamp','ao_value').")
        arr = d[:]
    # Convert ms -> s
    t_s = np.asarray(arr["timestamp"], dtype=float) / 1000.0
    v_v = np.asarray(arr["ao_value"], dtype=float)
    return AOTrack(t_s, v_v, default_v=0.0)


def open_events_file(path: str):
    f = h5py.File(path, "r")
    if "events" not in f or not isinstance(f["events"], h5py.Group):
        f.close()
        raise RuntimeError("Event file does not contain group '/events' (expects event mode files).")
    return f, f["events"]


def list_events(ev_group: h5py.Group):
    """Return list of (name, timestamp_sec) tuples, sorted by name."""
    items = []
    for name, obj in ev_group.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        ts = obj.attrs.get("timestamp", None)
        if ts is None:
            raise RuntimeError(f"Event '{name}' missing 'timestamp' attribute.")
        items.append((name, float(ts)))  # ts in **seconds** per Recording.py
    items.sort(key=lambda x: x[0])
    return items


def ensure_outdir(event_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(event_path))
    out_dir = os.path.join(base_dir, "split_by_voltage")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_group(events_h5_path: str, events: list, source_h5: h5py.File, ev_group: h5py.Group, volts_label: float, out_dir: str):
    """Write a new H5 with these events (reindexed), preserving timestamps."""
    if len(events) < MIN_EVENTS:
        return None
    base, ext = os.path.splitext(os.path.basename(events_h5_path))
    safe_suffix = volts_to_safe_suffix(volts_label)  # e.g., V_neg800mV
    out_name = f"{base}_{safe_suffix}{ext}"
    out_path = os.path.join(out_dir, out_name)

    with h5py.File(out_path, "w") as g:
        for key, value in source_h5.attrs.items():
            g.attrs[key] = value
        g.attrs["source_file"] = os.path.abspath(events_h5_path)
        g.attrs["split_voltage_V"] = float(volts_label)
        # timezone-aware UTC (no deprecation warning)
        g.attrs["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        out_grp = g.create_group("events")
        for key, value in ev_group.attrs.items():
            out_grp.attrs[key] = value
        for idx, (name, ts) in enumerate(events):
            src = ev_group[name]
            data = src[()]  # event waveform
            ds_name = f"event_{idx:05d}"
            d = out_grp.create_dataset(ds_name, data=data, compression=COMPRESSION, compression_opts=COMP_LEVEL)
            d.attrs["timestamp"] = float(ts)           # seconds
            d.attrs["source_event_name"] = name

    return out_path


def quantize_volts(v: float, step_v: float) -> int:
    """Map a voltage to an integer bin by rounding to nearest multiple of step_v."""
    if step_v <= 0:
        # exact binning (not recommended due to float noise)
        return hash(round(v, 9))
    return int(round(v / step_v))


def bin_to_volts(bin_idx: int, step_v: float) -> float:
    if step_v <= 0:
        raise ValueError("Exact binning used; cannot reconstruct representative volts from bin index.")
    return float(bin_idx) * float(step_v)


def main():
    ao_path = derive_ao_path(EVENT_H5_PATH)
    print(f"Using AO file: {ao_path}")
    ao = load_ao(ao_path)  # timestamps now in seconds

    f_ev, ev_grp = open_events_file(EVENT_H5_PATH)
    try:
        ev_list = list_events(ev_grp)  # [(name, timestamp_s), ...]
        if not ev_list:
            print("No events found; nothing to split.")
            return

        # Classify events by quantized volts
        groups = {}  # bin_idx -> [(name, ts), ...]
        for name, ts in ev_list:
            v = ao.value_at(ts)                     # volts at event time (seconds)
            bin_idx = quantize_volts(v, GROUP_STEP_V)
            groups.setdefault(bin_idx, []).append((name, ts))

        # Write out
        outdir = ensure_outdir(EVENT_H5_PATH)
        written = []
        for bin_idx in sorted(groups.keys()):
            v_label = bin_to_volts(bin_idx, GROUP_STEP_V) if GROUP_STEP_V > 0 else np.mean(
                [ao.value_at(ts) for _, ts in groups[bin_idx]]
            )
            if len(groups[bin_idx]) < MIN_EVENTS:
                continue
            events_for_file = groups[bin_idx]
            path = write_group(EVENT_H5_PATH, events_for_file, f_ev, ev_grp, v_label, outdir)
            if path:
                written.append((v_label, len(events_for_file), path))

        print("\nSplit complete:")
        for v, cnt, path in sorted(written, key=lambda x: x[0]):
            print(f"  {v:+.3f} V : {cnt} events -> {path}")

    finally:
        f_ev.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
