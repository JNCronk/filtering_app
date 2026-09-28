"""Read raw events and split by the companion AO log without detecting segments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .H5Splitter_1000 import partition_event_names, recording_start_time
from .H5Splitter_volt import load_ao, volts_to_safe_suffix
from .dwell_t import EventPreview, PREVIEW_COUNT


@dataclass
class EventSource:
    filepath: str
    event_names: list[str]
    timestamps: dict[str, float | None]
    previews: list[EventPreview]
    time_origin: float
    recording_start_time_s: float | None = None
    sampling_rate_hz: float | None = None
    voltage_groups: dict[float, list[str]] = field(default_factory=dict)
    event_voltages_mV: dict[str, float] = field(default_factory=dict)
    ao_path: str | None = None
    voltage_message: str = "Looking for a companion AO file…"
    split_paths: list[str] = field(default_factory=list)


def timestamp_of(dataset) -> float | None:
    value = dataset.attrs.get("timestamp")
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def read_raw_previews(filepath: str, names: list[str]) -> list[EventPreview]:
    previews = []
    with h5py.File(filepath, "r") as h5:
        for name in names:
            ds = h5["events"][name]
            data = np.asarray(ds[...], dtype=float)
            if data.ndim != 1:
                raise ValueError(f"{name}: expected a one-dimensional current trace.")
            previews.append(EventPreview(name, timestamp_of(ds), data, 0.0, []))
    return previews


def inspect_file(filepath: str) -> EventSource:
    path = Path(filepath).expanduser().resolve()
    with h5py.File(path, "r") as h5:
        if "events" not in h5 or not isinstance(h5["events"], h5py.Group):
            raise ValueError("The H5 file must contain an 'events' group.")
        _, names = partition_event_names(h5["events"])
        if not names:
            raise ValueError("No recorded event datasets found.")
        timestamps = {name: timestamp_of(h5["events"][name]) for name in names}
        start_time = recording_start_time(h5, h5["events"])
        sample_rate = finite_attribute(h5, "sampling_rate_hz")
    companion = find_continuous_path(path)
    if companion is not None and (start_time is None or sample_rate is None):
        with h5py.File(companion, "r") as h5:
            if start_time is None:
                start_time = finite_attribute(h5, "recording_start_time_s")
            if sample_rate is None:
                sample_rate = finite_attribute(h5, "sampling_rate_hz")
    valid_times = [t for t in timestamps.values() if t is not None]
    origin = start_time if start_time is not None else min(valid_times, default=0.0)
    return EventSource(str(path), names, timestamps,
                       read_raw_previews(str(path), names[:PREVIEW_COUNT]), origin,
                       start_time, sample_rate)


def finite_attribute(h5: h5py.File, name: str) -> float | None:
    value = h5.attrs.get(name)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def recording_stem(path: Path) -> str:
    return path.stem[:-7] if path.stem.casefold().endswith("_events") else path.stem


def find_continuous_path(filepath: str | Path) -> Path | None:
    path = Path(filepath)
    if not path.stem.casefold().endswith("_events"):
        return None
    candidate = path.with_name(recording_stem(path) + path.suffix)
    return candidate if candidate.is_file() else None


def find_ao_path(filepath: str) -> Path | None:
    path = Path(filepath)
    # Match only the recording's companion, never an unrelated AO log.
    stem = (recording_stem(path) + "_AO").casefold()
    candidates = sorted(p for p in path.parent.iterdir()
                        if p.is_file() and p.stem.casefold() == stem
                        and p.suffix.casefold() in {".h5", ".hdf5"})
    same_extension = [p for p in candidates if p.suffix.casefold() == path.suffix.casefold()]
    return next(iter(same_extension or candidates), None)


def prepare_voltage_groups(source: EventSource) -> EventSource:
    """Classify events in memory; saving split recordings is a separate action."""
    source.voltage_groups.clear()
    source.event_voltages_mV.clear()
    ao_path = find_ao_path(source.filepath)
    if ao_path is None:
        source.voltage_message = "No companion AO file could be found. Enter a manual voltage (mV)."
        return source
    source.ao_path = str(ao_path)
    try:
        track = load_ao(str(ao_path))
        if not track.t.size or not np.all(np.isfinite(track.t)) or not np.all(np.isfinite(track.v)):
            raise ValueError("The AO timeline is empty or contains non-finite values.")
        if any(source.timestamps[name] is None for name in source.event_names):
            raise ValueError("Some events have no timestamp; voltage assignment is unavailable.")
        for name in source.event_names:
            # H5Splitter convention: event times are seconds, AO times are milliseconds;
            # default to 0 V before the first AO entry, then use the preceding value.
            voltage = float(round(track.value_at(source.timestamps[name]) * 1000))
            source.voltage_groups.setdefault(voltage, []).append(name)
            source.event_voltages_mV[name] = voltage
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        source.voltage_groups.clear()
        source.event_voltages_mV.clear()
        source.voltage_message = f"AO file found but could not be read: {exc} Enter a manual voltage."
        return source

    source.voltage_groups = dict(sorted(source.voltage_groups.items()))
    source.voltage_message = f"AO found: {ao_path.name}. Choose voltages below; files are saved only on request."
    return source


def write_voltage_splits(source: EventSource, voltages=None, output_dir=None) -> list[str]:
    selected = set(source.voltage_groups if voltages is None else voltages)
    if not selected or not selected.issubset(source.voltage_groups):
        raise ValueError("Select at least one available voltage to save.")
    output_dir = Path(output_dir) if output_dir is not None else Path(source.filepath).parent / "split_by_voltage"
    output_dir.mkdir(exist_ok=True)
    written = []
    with h5py.File(source.filepath, "r") as src:
        for voltage, names in source.voltage_groups.items():
            if voltage not in selected:
                continue
            stem = f"{Path(source.filepath).stem}_{volts_to_safe_suffix(voltage / 1000)}"
            path = output_dir / f"{stem}.h5"
            suffix = 2
            while True:
                try:
                    dst = h5py.File(path, "x")
                    break
                except FileExistsError:
                    path = output_dir / f"{stem}_{suffix}.h5"
                    suffix += 1
            try:
                with dst:
                    for key, value in src.attrs.items():
                        dst.attrs[key] = value
                    if source.recording_start_time_s is not None:
                        dst.attrs["recording_start_time_s"] = source.recording_start_time_s
                    if source.sampling_rate_hz is not None:
                        dst.attrs["sampling_rate_hz"] = source.sampling_rate_hz
                    dst.attrs["source_file"] = source.filepath
                    dst.attrs["split_voltage_V"] = voltage / 1000
                    group = dst.create_group("events", track_order=True)
                    for key, value in src["events"].attrs.items():
                        group.attrs[key] = value
                    for index, name in enumerate(names):
                        output_name = f"event_{index:05d}"
                        src.copy(src["events"][name], group, name=output_name)
                        group[output_name].attrs["source_event_name"] = name
                        group[output_name].attrs["voltage_mV"] = voltage
                written.append(str(path))
            except Exception:
                path.unlink(missing_ok=True)
                raise
    return written
