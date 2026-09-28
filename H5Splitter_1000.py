# =====================================================================
# Split a PEINCon "event mode" HDF5 into ordered 1,000-event chunks.
# =====================================================================

"""
Split a PEINCon "event mode" HDF5 into files containing at most 1,000
recorded events, in their original event-number order.

Recording-start time and sampling rate are stored as file attributes. Legacy
zero-length synthetic start datasets are recognised and omitted from output.

Usage:
    1. Edit EVENT_H5_PATH below and run this file, or
    2. Pass the input path on the command line:

       python H5Splitter_1000.py /path/to/events.h5
"""

# ---------------------------------------------------------------------

### USER PARAMETERS ###

EVENT_H5_PATH = r"/Volumes/jcronk/project_cronk/DNA/2000bp/1908.h5"
EVENTS_PER_FILE = 1000
OUTPUT_DIR_NAME = "split_by_1000_events"
COMPRESSION = "gzip"
COMP_LEVEL = 4

# ---------------------------------------------------------------------

import os
import re
import sys
from datetime import datetime, timezone

import h5py


def event_order_key(name: str) -> tuple:
    """Sort event names by their trailing event number when present."""
    match = re.search(r"(\d+)$", name)
    if match:
        prefix = name[: match.start()].lower()
        return (0, prefix, int(match.group(1)), name)
    return (1, name.lower(), 0, name)


def list_events(events_group: h5py.Group) -> list[str]:
    """Return dataset names in numeric event order."""
    names = [
        name
        for name, obj in events_group.items()
        if isinstance(obj, h5py.Dataset)
    ]
    return sorted(names, key=event_order_key)


def partition_event_names(events_group: h5py.Group) -> tuple[str | None, list[str]]:
    """Return an optional legacy empty anchor and all genuine event names."""
    names = list_events(events_group)
    if names and events_group[names[0]].size == 0:
        return names[0], names[1:]
    return None, names


def recording_start_time(source_h5: h5py.File, events_group: h5py.Group) -> float | None:
    """Read the start attribute, falling back to a legacy empty anchor."""
    value = source_h5.attrs.get("recording_start_time_s")
    if value is not None:
        value = float(value)
        if value == value and abs(value) != float("inf"):
            return value
    anchor, _ = partition_event_names(events_group)
    if anchor is not None:
        value = events_group[anchor].attrs.get("timestamp")
        if value is not None:
            value = float(value)
            if value == value and abs(value) != float("inf"):
                return value
    source_path = os.path.abspath(source_h5.filename)
    base, extension = os.path.splitext(source_path)
    if base.casefold().endswith("_events"):
        companion = base[:-7] + extension
        if os.path.isfile(companion):
            with h5py.File(companion, "r") as continuous_h5:
                value = continuous_h5.attrs.get("recording_start_time_s")
                if value is not None:
                    value = float(value)
                    if value == value and abs(value) != float("inf"):
                        return value
    return None


def ensure_output_dir(event_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(event_path))
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_attributes(source, destination) -> None:
    """Copy all HDF5 attributes from one object to another."""
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def copy_event(
    source_group: h5py.Group,
    output_group: h5py.Group,
    source_name: str,
    output_index: int,
) -> None:
    """Copy and reindex one event dataset while preserving its attributes."""
    source = source_group[source_name]
    output_name = f"event_{output_index:05d}"

    create_options = {}
    if source.shape != ():
        create_options = {
            "compression": COMPRESSION,
            "compression_opts": COMP_LEVEL,
        }

    destination = output_group.create_dataset(
        output_name,
        data=source[()],
        **create_options,
    )
    copy_attributes(source, destination)
    destination.attrs["source_event_name"] = source_name


def write_chunk(
    source_h5: h5py.File,
    source_group: h5py.Group,
    event_path: str,
    output_dir: str,
    chunk_events: list[str],
    chunk_number: int,
    first_event_number: int,
) -> str:
    """Write one ordered event chunk with recording metadata."""
    base, extension = os.path.splitext(os.path.basename(event_path))
    last_event_number = first_event_number + len(chunk_events) - 1
    output_name = (
        f"{base}_events_{first_event_number:06d}-"
        f"{last_event_number:06d}{extension}"
    )
    output_path = os.path.join(output_dir, output_name)

    with h5py.File(output_path, "w") as output_h5:
        copy_attributes(source_h5, output_h5)
        output_h5.attrs["source_file"] = os.path.abspath(event_path)
        output_h5.attrs["split_chunk_number"] = chunk_number
        output_h5.attrs["events_in_chunk"] = len(chunk_events)
        output_h5.attrs["events_per_file"] = EVENTS_PER_FILE
        start_time = recording_start_time(source_h5, source_group)
        if start_time is not None:
            output_h5.attrs["recording_start_time_s"] = start_time
        output_h5.attrs["created_utc"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        output_group = output_h5.create_group("events", track_order=True)
        copy_attributes(source_group, output_group)
        for output_index, source_name in enumerate(chunk_events):
            copy_event(source_group, output_group, source_name, output_index)

    return output_path


def split_file(event_path: str) -> list[str]:
    if EVENTS_PER_FILE <= 0:
        raise ValueError("EVENTS_PER_FILE must be greater than zero.")
    if not os.path.isfile(event_path):
        raise FileNotFoundError(f"Event file not found: {event_path}")

    output_dir = ensure_output_dir(event_path)
    written_paths = []

    with h5py.File(event_path, "r") as source_h5:
        if "events" not in source_h5 or not isinstance(
            source_h5["events"], h5py.Group
        ):
            raise RuntimeError(
                "Event file does not contain group '/events' "
                "(expects an event mode file)."
            )

        source_group = source_h5["events"]
        _, recorded_events = partition_event_names(source_group)
        if not recorded_events:
            return written_paths

        for offset in range(0, len(recorded_events), EVENTS_PER_FILE):
            chunk_events = recorded_events[offset : offset + EVENTS_PER_FILE]
            path = write_chunk(
                source_h5=source_h5,
                source_group=source_group,
                event_path=event_path,
                output_dir=output_dir,
                chunk_events=chunk_events,
                chunk_number=(offset // EVENTS_PER_FILE) + 1,
                first_event_number=offset + 1,
            )
            written_paths.append(path)

    return written_paths


def main() -> None:
    event_path = sys.argv[1] if len(sys.argv) > 1 else EVENT_H5_PATH
    written_paths = split_file(event_path)

    if not written_paths:
        print("No recorded events found; nothing was written.")
        return

    print(f"Split complete: wrote {len(written_paths)} file(s).")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR:", error)
        sys.exit(1)
