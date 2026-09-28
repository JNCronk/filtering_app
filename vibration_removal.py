from __future__ import annotations

import copy
from dataclasses import dataclass

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

from .dwell_t import DetectedEvent, DetectedSegment, EventPreview, Settings, compute_samples_per_buffer
from pyqtgraph.Qt import QtCore, QtWidgets
from .plots import TraceGrid, add_std_overlay
from .qt_helpers import TaskRunner, button, number_field


DEFAULT_STD_WINDOW_MS = 0.0
DEFAULT_STD_CONTEXT_MS = 1.0
DEFAULT_STD_HEIGHT_FACTOR = 1.0
DEFAULT_STD_PROMINENCE = 0.0
MAX_REMOVED_PLOTS = 12

BG_COLOR = "#f5f7fb"
PANEL_COLOR = "#ffffff"
FIELD_COLOR = "#ffffff"
TEXT_COLOR = "#18212b"
MUTED_COLOR = "#5f6b7a"


@dataclass
class VibrationSettings:
    std_window_ms: float = DEFAULT_STD_WINDOW_MS
    std_context_ms: float = DEFAULT_STD_CONTEXT_MS
    std_height_factor: float = DEFAULT_STD_HEIGHT_FACTOR
    std_prominence: float = DEFAULT_STD_PROMINENCE


@dataclass
class RemovedSegmentPreview:
    event_name: str
    segment_label: str
    timestamp: float | None
    data_full: np.ndarray
    baseline: float
    start: int
    end: int
    std_time_ms: np.ndarray
    std_trace: np.ndarray
    std_threshold: float


def _moving_std(values: np.ndarray, window_samples: int) -> tuple[np.ndarray, int]:
    if values.size < 2:
        return np.array([]), 0
    if values.size < window_samples:
        return np.array([float(np.std(values))]), values.size
    windows = sliding_window_view(values, window_shape=window_samples)
    return windows.std(axis=-1), window_samples


def _has_std_peak(std_trace: np.ndarray, noise_std: float, height_factor: float, prominence: float) -> tuple[bool, float]:
    if std_trace.size == 0:
        return False, float(noise_std * height_factor)
    threshold = float(noise_std * height_factor)
    if threshold <= 0:
        threshold = float(0.5 * np.max(std_trace)) if std_trace.size else 0.0
    if std_trace.size == 1:
        return bool(std_trace[0] > threshold), threshold
    peaks, _ = find_peaks(std_trace, height=threshold if threshold > 0 else None, prominence=prominence)
    return peaks.size > 0, threshold


def evaluate_vibration_segments(
    settings: Settings,
    detected_events: list[DetectedEvent],
    vibration_settings: VibrationSettings,
) -> tuple[list[DetectedEvent], list[RemovedSegmentPreview]]:
    filtered_events: list[DetectedEvent] = []
    removed: list[RemovedSegmentPreview] = []
    window_samples = max(2, int(round(vibration_settings.std_window_ms * settings.samp_freq)))
    ctx_samples = max(0, int(round(vibration_settings.std_context_ms * settings.samp_freq)))
    samples_per_buffer = compute_samples_per_buffer(settings.samp_freq, settings.buffer_time_ms)

    with h5py.File(settings.filepath, "r") as h5:
        if "events" not in h5:
            raise ValueError("No 'events' group found in provided HDF5.")
        events_grp = h5["events"]

        for event in detected_events:
            if event.event_name not in events_grp:
                continue
            data = events_grp[event.event_name][...]
            kept_segments: list[DetectedSegment] = []

            for segment_idx, segment in enumerate(event.segments):
                noise_start = max(0, segment.start - samples_per_buffer)
                noise_end = segment.start
                noise_region = data[noise_start:noise_end] - segment.baseline
                if noise_region.size < 2:
                    fallback = data[: min(samples_per_buffer, data.size)] - segment.baseline
                    noise_region = fallback if fallback.size else np.array([0.0])
                noise_std = float(np.std(noise_region))

                std_start = max(0, segment.start - ctx_samples)
                std_end = min(len(data) - 1, segment.end + ctx_samples)
                std_input = data[std_start:std_end + 1] - segment.baseline
                std_trace, eff_window = _moving_std(std_input, window_samples)
                has_peak, threshold = _has_std_peak(
                    std_trace,
                    noise_std=noise_std,
                    height_factor=vibration_settings.std_height_factor,
                    prominence=vibration_settings.std_prominence,
                )

                if has_peak:
                    kept_segments.append(segment)
                    continue

                if std_trace.size:
                    center_offset = max(0.0, (eff_window - 1) / 2.0)
                    std_indices = np.arange(std_trace.size) + std_start + center_offset
                    std_time = std_indices / settings.samp_freq
                else:
                    std_time = np.array([])

                label = event.event_name if segment_idx == 0 else f"{event.event_name}_{segment_idx}"
                removed.append(
                    RemovedSegmentPreview(
                        event_name=event.event_name,
                        segment_label=label,
                        timestamp=event.timestamp,
                        data_full=data,
                        baseline=segment.baseline,
                        start=segment.start,
                        end=segment.end,
                        std_time_ms=std_time,
                        std_trace=std_trace,
                        std_threshold=threshold,
                    )
                )

            filtered_events.append(
                DetectedEvent(
                    event_name=event.event_name,
                    timestamp=event.timestamp,
                    baseline=event.baseline,
                    segments=kept_segments,
                )
            )

    return filtered_events, removed


class VibrationRemovalWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings, source_events, apply_callback):
        super().__init__(parent)
        self.settings = settings
        self.source_events = copy.deepcopy(source_events)
        self.apply_callback = apply_callback
        self.current_events = None
        self.removed_previews = []
        self.rng = np.random.default_rng()
        self.setWindowTitle('Vibration removal')
        self.resize(1400, 900)
        self.runner = TaskRunner(self)
        self.runner.busyChanged.connect(self._busy_changed)
        self.runner.failed.connect(self._failed)
        layout = QtWidgets.QHBoxLayout(self)
        sidebar = QtWidgets.QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setFixedWidth(255)
        self.controls = QtWidgets.QWidget()
        sidebar.setWidget(self.controls)
        controls = QtWidgets.QVBoxLayout(self.controls)
        self.fields = {}
        for key, label, default in [
            ('std_window_ms', 'Window (ms)', DEFAULT_STD_WINDOW_MS),
            ('std_context_ms', 'Context (ms)', DEFAULT_STD_CONTEXT_MS),
            ('std_height_factor', 'Height factor', DEFAULT_STD_HEIGHT_FACTOR),
            ('std_prominence', 'Prominence', DEFAULT_STD_PROMINENCE),
        ]:
            controls.addWidget(QtWidgets.QLabel(label))
            field = number_field(default)
            self.fields[key] = field
            field.valueChanged.connect(self._invalidate)
            controls.addWidget(field)
        button('Preview removal', self.refresh_preview, controls)
        button('Randomise examples', self.render_removed_previews, controls)
        self.confirm_button = button('Confirm', self.confirm, controls)
        self.status = QtWidgets.QLabel('Preview changes, then confirm to apply. Closing leaves the working list unchanged.')
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch()
        layout.addWidget(sidebar)
        self.grid = TraceGrid(rows=4)
        layout.addWidget(self.grid, 1)
        self.confirm_button.setEnabled(False)
        QtCore.QTimer.singleShot(0, self.refresh_preview)

    def _invalidate(self, *_):
        self.current_events = None
        self.confirm_button.setEnabled(False)
        self.status.setText('Settings changed. Preview removal again before confirming.')

    def refresh_preview(self):
        options = VibrationSettings(**{key: field.value() for key, field in self.fields.items()})
        self.status.setText('Evaluating vibration signatures…')

        def ready(result):
            self.current_events, self.removed_previews = result
            self.render_removed_previews()
            kept = sum(len(event.segments) for event in self.current_events)
            self.status.setText(f'{len(self.removed_previews)} segments would be removed; {kept} remain. Confirm to apply.')
            self.confirm_button.setEnabled(True)

        self.runner.start(lambda: evaluate_vibration_segments(self.settings, self.source_events, options), ready)

    def render_removed_previews(self):
        count = min(MAX_REMOVED_PLOTS, len(self.removed_previews))
        indices = sorted(self.rng.choice(len(self.removed_previews), size=count, replace=False))
        entries = [self.removed_previews[i] for i in indices]
        previews = [EventPreview(entry.segment_label, entry.timestamp, entry.data_full, entry.baseline,
                                  [DetectedSegment(entry.start, entry.end, 1, entry.baseline)]) for entry in entries]
        # End-peak overlays here use the vibration window's own settings below.
        from dataclasses import replace
        display_settings = replace(self.settings, end_at_std_peak=False)
        self.grid.display(previews, self.settings.samp_freq, display_settings)
        for plot, entry in zip(self.grid.plots, entries):
            add_std_overlay(plot, entry.data_full, entry.std_time_ms, entry.std_trace, entry.std_threshold)

    def _busy_changed(self, busy):
        self.controls.setEnabled(not busy)

    def _failed(self, message):
        self.current_events = None
        self.confirm_button.setEnabled(False)
        self.status.setText(message)
        QtWidgets.QMessageBox.warning(self, 'Vibration removal failed', message)

    def confirm(self):
        if self.current_events is not None and not self.runner.busy:
            self.apply_callback(copy.deepcopy(self.current_events), keep_window=False)
            self.accept()

    def reject(self):
        if not self.runner.busy:
            super().reject()

    def closeEvent(self, event):
        if self.runner.busy:
            event.ignore()
        else:
            event.accept()
