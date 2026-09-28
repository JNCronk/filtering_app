from __future__ import annotations

import os
from dataclasses import dataclass, field

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

from .H5Splitter_1000 import list_events


DEFAULT_H5 = ""
DEFAULT_SAMP_FREQ = 50.0
DEFAULT_BUFFER_TIME_MS = 1.0
DEFAULT_THRESHOLD_NA = 0.1
DEFAULT_RETURN_THRESHOLD_NA = 0.1
DEFAULT_MIN_SEPARATION_MS = 5.0
DEFAULT_VOLTAGE_MV = -1000.0
DEFAULT_END_AT_STD_PEAK = False
DEFAULT_END_STD_WINDOW_MS = 0.0
DEFAULT_END_STD_HEIGHT_FACTOR = 1.0
DEFAULT_END_STD_PROMINENCE = 0.0
DEFAULT_END_STD_PRE_CROSSING_BUFFER_FRACTION = 0.1
PREVIEW_COUNT = 12


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@dataclass
class Settings:
    filepath: str
    samp_freq: float
    buffer_time_ms: float
    threshold_nA: float
    return_threshold_nA: float
    min_separation_ms: float
    voltage_mV: float
    end_at_std_peak: bool = DEFAULT_END_AT_STD_PEAK
    end_std_window_ms: float = DEFAULT_END_STD_WINDOW_MS
    end_std_height_factor: float = DEFAULT_END_STD_HEIGHT_FACTOR
    end_std_prominence: float = DEFAULT_END_STD_PROMINENCE
    end_std_pre_crossing_buffer_fraction: float = DEFAULT_END_STD_PRE_CROSSING_BUFFER_FRACTION
    event_names: tuple[str, ...] | None = None
    event_voltages_mV: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectedSegment:
    start: int
    end: int
    direction: int
    baseline: float
    std_peak_index: int | None = None


@dataclass
class EventPreview:
    event_name: str
    timestamp: float | None
    data: np.ndarray
    baseline: float
    segments: list[DetectedSegment]


@dataclass
class DetectedEvent:
    event_name: str
    timestamp: float | None
    baseline: float
    segments: list[DetectedSegment]


@dataclass
class SegmentResult:
    event_name: str
    timestamp: float | None
    dwell_time_ms: float
    direction: int
    start: int
    end: int
    area_nA_ms: float
    delta_I_nA: float
    delta_I_rel: float
    resistance_MOhm: float
    voltage_mV: float = DEFAULT_VOLTAGE_MV


@dataclass(frozen=True)
class _SegmentNode:
    event_idx: int
    segment_idx: int


def compute_samples_per_buffer(samp_freq: float, buffer_time_ms: float) -> int:
    return _next_pow2(max(1, int(samp_freq * buffer_time_ms)))


def threshold_guide_value(baseline: float, direction: int, threshold_nA: float) -> float:
    """Return the plotted threshold level for the segment direction."""
    baseline_sign = 1.0 if baseline >= 0 else -1.0
    direction_sign = -baseline_sign if direction == 1 else baseline_sign
    return float(baseline + direction_sign * threshold_nA)


def _moving_std(values: np.ndarray, window_samples: int) -> tuple[np.ndarray, int]:
    if values.size < 2:
        return np.array([]), 0
    if values.size < window_samples:
        return np.array([float(np.std(values))]), values.size
    windows = sliding_window_view(values, window_shape=window_samples)
    return windows.std(axis=-1), window_samples


def _find_std_peak_for_segment_end(
    data: np.ndarray,
    baseline: float,
    start: int,
    threshold_cross_idx: int,
    settings: Settings,
) -> int | None:
    window_samples = max(2, int(round(settings.end_std_window_ms * settings.samp_freq)))

    std_start = 0
    std_end = len(data) - 1
    std_input = data[std_start : std_end + 1] - baseline
    std_trace, eff_window = _moving_std(std_input, window_samples)
    if std_trace.size == 0:
        return None

    threshold = get_segment_end_std_threshold(data, baseline, start, settings)
    if threshold <= 0:
        threshold = float(0.5 * np.max(std_trace)) if std_trace.size else 0.0
    peak_indices, _ = find_peaks(
        std_trace,
        height=threshold if threshold > 0 else None,
        prominence=settings.end_std_prominence,
    )
    if peak_indices.size == 0 and std_trace.size == 1 and std_trace[0] > threshold:
        peak_indices = np.array([0], dtype=int)
    if peak_indices.size == 0:
        return None

    center_offset = max(0.0, (eff_window - 1) / 2.0)
    peak_sample_positions = std_start + center_offset + peak_indices.astype(float)
    pre_crossing_allowance = int(round(
        settings.end_std_pre_crossing_buffer_fraction
        * compute_samples_per_buffer(settings.samp_freq, settings.buffer_time_ms)
    ))
    earliest_allowed = max(
        start,
        threshold_cross_idx - pre_crossing_allowance,
    )
    valid_positions = peak_sample_positions[peak_sample_positions >= earliest_allowed]
    if valid_positions.size == 0:
        return None
    closest_idx = int(np.argmin(np.abs(valid_positions - threshold_cross_idx)))
    peak_sample = int(round(float(valid_positions[closest_idx])))
    return max(start, min(len(data) - 1, peak_sample))


def get_segment_end_std_overlay(
    data: np.ndarray,
    segment: DetectedSegment,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, float]:
    window_samples = max(2, int(round(settings.end_std_window_ms * settings.samp_freq)))
    std_start = 0
    std_end = len(data) - 1
    std_input = data[std_start : std_end + 1] - segment.baseline
    std_trace, eff_window = _moving_std(std_input, window_samples)
    if std_trace.size == 0:
        return np.array([]), np.array([]), get_segment_end_std_threshold(data, segment.baseline, segment.start, settings)
    center_offset = max(0.0, (eff_window - 1) / 2.0)
    std_indices = np.arange(std_trace.size, dtype=float) + std_start + center_offset
    std_time_ms = std_indices / settings.samp_freq
    std_threshold = get_segment_end_std_threshold(data, segment.baseline, segment.start, settings)
    return std_time_ms, std_trace, std_threshold


def get_segment_end_std_threshold(
    data: np.ndarray,
    baseline: float,
    start: int,
    settings: Settings,
) -> float:
    samples_per_buffer = compute_samples_per_buffer(settings.samp_freq, settings.buffer_time_ms)
    noise_start = max(0, start - samples_per_buffer)
    noise_end = start
    noise_region = data[noise_start:noise_end] - baseline
    if noise_region.size < 2:
        fallback = data[: min(samples_per_buffer, data.size)] - baseline
        noise_region = fallback if fallback.size else np.array([0.0])
    threshold = float(np.std(noise_region) * settings.end_std_height_factor)
    if threshold <= 0:
        window_samples = max(2, int(round(settings.end_std_window_ms * settings.samp_freq)))
        std_trace, _ = _moving_std(data - baseline, window_samples)
        threshold = float(0.5 * np.max(std_trace)) if std_trace.size else 0.0
    return threshold


def _segment_metric_key(
    segment: DetectedSegment,
    data: np.ndarray,
    samp_freq: float,
) -> tuple[float, float, float]:
    segment_data = data[segment.start : segment.end + 1]
    dt_ms = 1.0 / samp_freq
    dwell_time_ms = (segment.end - segment.start + 1) / samp_freq
    area_nA_ms = float(trapezoid(np.abs(segment_data - segment.baseline), dx=dt_ms))
    delta_I = segment_data - segment.baseline
    delta_I_nA = float(np.average(delta_I))
    delta_I_rel = float(delta_I_nA / segment.baseline) if segment.baseline != 0 else 0.0
    return dwell_time_ms, area_nA_ms, delta_I_rel


def _edge_clearance(segment: DetectedSegment, event_length: int) -> int:
    left_clearance = segment.start
    right_clearance = max(0, event_length - 1 - segment.end)
    return min(left_clearance, right_clearance)


def _deduplicate_adjacent_segments(
    detected_events: list[DetectedEvent],
    event_data_map: dict[str, np.ndarray],
    samp_freq: float,
    adjacent_pairs: set[tuple[str, str]] | None = None,
) -> list[DetectedEvent]:
    if len(detected_events) < 2:
        return detected_events

    parents: dict[_SegmentNode, _SegmentNode] = {}

    def make_set(node: _SegmentNode) -> None:
        parents.setdefault(node, node)

    def find(node: _SegmentNode) -> _SegmentNode:
        parent = parents[node]
        if parent != node:
            parents[node] = find(parent)
        return parents[node]

    def union(a: _SegmentNode, b: _SegmentNode) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    metric_maps: list[dict[tuple[float, float, float], list[_SegmentNode]]] = []
    for event_idx, event in enumerate(detected_events):
        event_data = event_data_map.get(event.event_name)
        event_metric_map: dict[tuple[float, float, float], list[_SegmentNode]] = {}
        if event_data is None:
            metric_maps.append(event_metric_map)
            continue
        for segment_idx, segment in enumerate(event.segments):
            node = _SegmentNode(event_idx=event_idx, segment_idx=segment_idx)
            make_set(node)
            key = _segment_metric_key(segment, event_data, samp_freq)
            event_metric_map.setdefault(key, []).append(node)
        metric_maps.append(event_metric_map)

    for event_idx in range(len(detected_events) - 1):
        pair = (detected_events[event_idx].event_name, detected_events[event_idx + 1].event_name)
        if adjacent_pairs is not None and pair not in adjacent_pairs:
            continue
        current_map = metric_maps[event_idx]
        next_map = metric_maps[event_idx + 1]
        shared_keys = current_map.keys() & next_map.keys()
        for key in shared_keys:
            for current_node in current_map[key]:
                for next_node in next_map[key]:
                    union(current_node, next_node)

    groups: dict[_SegmentNode, list[_SegmentNode]] = {}
    for node in parents:
        groups.setdefault(find(node), []).append(node)

    keep_by_event: dict[int, set[int]] = {}
    for members in groups.values():
        if len(members) == 1:
            node = members[0]
            keep_by_event.setdefault(node.event_idx, set()).add(node.segment_idx)
            continue

        best_node: _SegmentNode | None = None
        best_score: tuple[int, float, int] | None = None
        for node in members:
            event = detected_events[node.event_idx]
            event_data = event_data_map.get(event.event_name)
            if event_data is None:
                continue
            segment = event.segments[node.segment_idx]
            clearance = _edge_clearance(segment, event_data.size)
            dwell_time_ms = (segment.end - segment.start + 1) / samp_freq
            score = (clearance, dwell_time_ms, -node.event_idx)
            if best_score is None or score > best_score:
                best_score = score
                best_node = node

        if best_node is not None:
            keep_by_event.setdefault(best_node.event_idx, set()).add(best_node.segment_idx)

    filtered_events: list[DetectedEvent] = []
    for event_idx, event in enumerate(detected_events):
        keep_indices = keep_by_event.get(event_idx, set())
        kept_segments = [
            segment for segment_idx, segment in enumerate(event.segments)
            if segment_idx in keep_indices
        ]
        filtered_events.append(
            DetectedEvent(
                event_name=event.event_name,
                timestamp=event.timestamp,
                baseline=event.baseline,
                segments=kept_segments,
            )
        )
    return filtered_events


def analyze_event(
    data: np.ndarray,
    settings: Settings,
) -> tuple[float, list[DetectedSegment]]:
    samples_per_buffer = compute_samples_per_buffer(settings.samp_freq, settings.buffer_time_ms)
    smoothed_data = gaussian_filter1d(data, sigma=settings.samp_freq / 10)
    segments: list[DetectedSegment] = []

    if data.size == 0:
        return 0.0, segments

    min_separation_samples = max(0, int(round(settings.min_separation_ms * settings.samp_freq)))
    search_start = 0
    max_anchor = max(0, data.size - 2 * samples_per_buffer)

    baseline = float(np.mean(data[: min(samples_per_buffer, data.size)]))
    found_any = False

    while search_start <= max_anchor:
        found_in_pass = False

        for anchor in range(search_start, max_anchor + 1):
            baseline_slice = data[anchor : anchor + samples_per_buffer]
            if baseline_slice.size == 0:
                continue
            baseline = float(np.mean(baseline_slice))

            probe_start = anchor + samples_per_buffer
            probe_end = min(probe_start + samples_per_buffer, data.size)
            probe = smoothed_data[probe_start:probe_end]
            if probe.size == 0:
                continue

            above = np.abs(probe - baseline) > settings.threshold_nA
            if not np.any(above):
                continue

            start = probe_start + int(np.argmax(above))
            end = start
            recovery_idx = None
            start_delta = float(smoothed_data[start] - baseline)
            outward_sign = 1.0 if start_delta >= 0 else -1.0
            return_armed = (start_delta * outward_sign) > settings.return_threshold_nA

            for idx in range(start + 1, data.size):
                delta = float(smoothed_data[idx] - baseline)
                signed_delta = delta * outward_sign
                if not return_armed:
                    end = idx
                    if signed_delta > settings.return_threshold_nA:
                        return_armed = True
                    continue
                if signed_delta > settings.return_threshold_nA:
                    end = idx
                    continue
                recovery_idx = idx
                break

            if recovery_idx is None:
                search_start = data.size
                found_in_pass = True
                break

            direction = 1 if abs(smoothed_data[start]) < abs(baseline) else -1
            std_peak_index = None
            if settings.end_at_std_peak:
                std_peak_index = _find_std_peak_for_segment_end(
                    data=data,
                    baseline=baseline,
                    start=start,
                    threshold_cross_idx=recovery_idx,
                    settings=settings,
                )
                if std_peak_index is None:
                    search_start = data.size
                    found_in_pass = True
                    break
                end = std_peak_index
            segments.append(
                DetectedSegment(
                    start=start,
                    end=end,
                    direction=direction,
                    baseline=baseline,
                    std_peak_index=std_peak_index,
                )
            )
            found_any = True
            found_in_pass = True
            search_start = max(end + min_separation_samples, start + 1)
            break

        if not found_in_pass:
            break

    if not found_any and search_start > max_anchor:
        short_slice = data[: min(samples_per_buffer, data.size)]
        baseline = float(np.mean(short_slice)) if short_slice.size else 0.0

    return baseline, segments


def list_event_names(filepath: str) -> tuple[str | None, list[str]]:
    with h5py.File(filepath, "r") as h5:
        if "events" not in h5:
            raise ValueError("No 'events' group found in file.")
        event_names = list_events(h5["events"])
    if not event_names:
        raise ValueError("No events datasets found in the 'events' group.")
    first_event = event_names[0]
    return first_event, event_names[1:]


def load_previews_from_events(
    settings: Settings,
    detected_events: list[DetectedEvent],
    event_names: list[str] | None = None,
) -> list[EventPreview]:
    wanted = set(event_names) if event_names is not None else None
    previews: list[EventPreview] = []

    with h5py.File(settings.filepath, "r") as h5:
        events_grp = h5["events"]
        for detected in detected_events:
            if wanted is not None and detected.event_name not in wanted:
                continue
            if detected.event_name not in events_grp:
                continue
            data = events_grp[detected.event_name][...]
            previews.append(
                EventPreview(
                    event_name=detected.event_name,
                    timestamp=detected.timestamp,
                    data=data,
                    baseline=detected.baseline,
                    segments=detected.segments,
                )
            )
    return previews


def get_processed_dir() -> str:
    filtering_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(filtering_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


def iter_detected_events(settings: Settings) -> list[DetectedEvent]:
    detected: list[DetectedEvent] = []
    event_data_map: dict[str, np.ndarray] = {}
    with h5py.File(settings.filepath, "r") as h5:
        if "events" not in h5:
            raise ValueError("No 'events' group found in file.")

        events_grp = h5["events"]
        event_names = list_events(events_grp)
        wanted = set(settings.event_names) if settings.event_names is not None else None
        for idx, event_name in enumerate(event_names):
            if idx == 0:
                continue

            if wanted is not None and event_name not in wanted:
                continue

            event_dset = events_grp[event_name]
            timestamp = event_dset.attrs.get("timestamp", None)
            data = event_dset[...]
            event_data_map[event_name] = data
            baseline, segments = analyze_event(data=data, settings=settings)
            detected.append(
                DetectedEvent(
                    event_name=event_name,
                    timestamp=timestamp,
                    baseline=baseline,
                    segments=segments,
                )
            )
    adjacent_pairs = {
        (left, right) for left, right in zip(event_names[1:], event_names[2:])
        if settings.event_voltages_mV.get(left, settings.voltage_mV)
        == settings.event_voltages_mV.get(right, settings.voltage_mV)
    }
    return _deduplicate_adjacent_segments(detected, event_data_map, settings.samp_freq, adjacent_pairs)


def collect_segment_results(settings: Settings, detected_events: list[DetectedEvent] | None = None) -> list[SegmentResult]:
    if detected_events is None:
        detected_events = iter_detected_events(settings)
    results: list[SegmentResult] = []

    with h5py.File(settings.filepath, "r") as h5:
        if "events" not in h5:
            raise ValueError("No 'events' group found in file.")

        events_grp = h5["events"]
        for event in detected_events:
            if event.event_name not in events_grp:
                continue

            data = events_grp[event.event_name][...]
            for segment_idx, segment_info in enumerate(event.segments):
                segment_name = event.event_name if segment_idx == 0 else f"{event.event_name}_{segment_idx}"
                segment = data[segment_info.start : segment_info.end + 1]
                dt_ms = 1.0 / settings.samp_freq
                area_abs = float(trapezoid(np.abs(segment - segment_info.baseline), dx=dt_ms))
                delta_I = segment - segment_info.baseline
                av_delta_I = float(np.average(delta_I))
                av_delta_I_rel = float(av_delta_I / segment_info.baseline) if segment_info.baseline != 0 else 0.0
                voltage_mV = settings.event_voltages_mV.get(event.event_name, settings.voltage_mV)
                resistance_MOhm = (
                    float(voltage_mV / segment_info.baseline) if segment_info.baseline != 0 else float("nan")
                )
                dwell_time_ms = (segment_info.end - segment_info.start + 1) / settings.samp_freq
                results.append(
                    SegmentResult(
                        event_name=segment_name,
                        timestamp=event.timestamp,
                        dwell_time_ms=dwell_time_ms,
                        direction=segment_info.direction,
                        start=segment_info.start,
                        end=segment_info.end,
                        area_nA_ms=area_abs,
                        delta_I_nA=av_delta_I,
                        delta_I_rel=av_delta_I_rel,
                        resistance_MOhm=resistance_MOhm,
                        voltage_mV=voltage_mV,
                    )
                )

    return results
