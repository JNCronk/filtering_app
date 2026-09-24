from __future__ import annotations

import copy
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import h5py
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

from .dwell_t import DetectedEvent, DetectedSegment, Settings, collect_segment_results, compute_samples_per_buffer


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


class VibrationRemovalWindow:
    def __init__(
        self,
        parent,
        settings: Settings,
        source_events: list[DetectedEvent],
        apply_callback,
    ) -> None:
        self.parent = parent
        self.settings = settings
        self.source_events = copy.deepcopy(source_events)
        self.apply_callback = apply_callback
        self.confirmed = False
        self.current_events = copy.deepcopy(source_events)
        self.removed_previews: list[RemovedSegmentPreview] = []
        self.original_events = copy.deepcopy(source_events)
        self.current_sample: list[RemovedSegmentPreview] = []
        self.rng = np.random.default_rng()

        self.window = tk.Toplevel(parent)
        self.window.title("Vibration Removal")
        self.window.geometry("1450x900")
        self.window.configure(bg=BG_COLOR)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.std_window_var = tk.StringVar(value=str(DEFAULT_STD_WINDOW_MS))
        self.std_context_var = tk.StringVar(value=str(DEFAULT_STD_CONTEXT_MS))
        self.std_height_var = tk.StringVar(value=str(DEFAULT_STD_HEIGHT_FACTOR))
        self.std_prominence_var = tk.StringVar(value=str(DEFAULT_STD_PROMINENCE))
        self.status_var = tk.StringVar(value="Set std options and preview vibration removal.")

        self._build_ui()
        self.refresh_preview()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.window, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Window (ms)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.std_window_var, width=10).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(controls, text="Context (ms)").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(controls, textvariable=self.std_context_var, width=10).grid(row=0, column=3, sticky="w", pady=4)
        ttk.Label(controls, text="Height factor").grid(row=0, column=4, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(controls, textvariable=self.std_height_var, width=10).grid(row=0, column=5, sticky="w", pady=4)
        ttk.Label(controls, text="Prominence").grid(row=0, column=6, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(controls, textvariable=self.std_prominence_var, width=10).grid(row=0, column=7, sticky="w", pady=4)
        ttk.Button(controls, text="Preview Removal", command=self.refresh_preview).grid(row=0, column=8, padx=(12, 0), pady=4)
        ttk.Button(controls, text="Randomise Selection", command=self.show_another_random).grid(row=0, column=9, padx=(8, 0), pady=4)
        ttk.Button(controls, text="Confirm", command=self.confirm).grid(row=0, column=10, padx=(8, 0), pady=4)

        ttk.Label(self.window, textvariable=self.status_var, padding=(12, 0, 12, 8)).pack(side=tk.TOP, fill=tk.X)

        self.figure = Figure(figsize=(14.2, 8.4), dpi=100, facecolor=BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas.get_tk_widget().configure(bg=BG_COLOR, highlightthickness=0)

    def read_std_settings(self) -> VibrationSettings:
        return VibrationSettings(
            std_window_ms=float(self.std_window_var.get()),
            std_context_ms=float(self.std_context_var.get()),
            std_height_factor=float(self.std_height_var.get()),
            std_prominence=float(self.std_prominence_var.get()),
        )

    def refresh_preview(self) -> None:
        try:
            vibration_settings = self.read_std_settings()
            filtered, removed = evaluate_vibration_segments(
                self.settings,
                self.source_events,
                vibration_settings,
            )
            self.current_events = filtered
            self.removed_previews = removed
            self.apply_callback(copy.deepcopy(filtered), keep_window=True)
            self.current_sample = self.select_removed_sample()
            self.render_removed_previews()
            kept_count = sum(len(event.segments) for event in filtered)
            self.status_var.set(
                f"Removed {len(removed)} segments; {kept_count} segments remain in the working list."
            )
        except Exception as exc:
            messagebox.showerror("Vibration removal failed", str(exc), parent=self.window)

    def select_removed_sample(self) -> list[RemovedSegmentPreview]:
        if not self.removed_previews:
            return []
        sample_size = min(MAX_REMOVED_PLOTS, len(self.removed_previews))
        idx = self.rng.choice(len(self.removed_previews), size=sample_size, replace=False)
        return [self.removed_previews[i] for i in sorted(idx)]

    def show_another_random(self) -> None:
        if not self.removed_previews:
            return
        self.current_sample = self.select_removed_sample()
        self.render_removed_previews()

    def render_removed_previews(self) -> None:
        self.figure.clear()
        sample = self.current_sample or self.removed_previews[:MAX_REMOVED_PLOTS]
        ncols = 3
        nrows = 4
        axes = self.figure.subplots(nrows, ncols, squeeze=False).ravel()

        for idx, (ax, entry) in enumerate(zip(axes, sample)):
            row = idx // ncols
            col = idx % ncols
            ax.set_facecolor(PANEL_COLOR)
            for spine in ax.spines.values():
                spine.set_color(MUTED_COLOR)
            ax.tick_params(colors=MUTED_COLOR, labelsize=7)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)

            time_ms = np.arange(len(entry.data_full)) / self.settings.samp_freq
            ax.plot(time_ms, entry.data_full, color="#c7d0d9", linewidth=0.8)
            ax.axhline(entry.baseline, color="#7ee787", linestyle="--", linewidth=0.8)
            ax.axvspan(entry.start / self.settings.samp_freq, entry.end / self.settings.samp_freq, color="#ffb86b", alpha=0.12)
            match = re.search(r"(\d+)$", entry.event_name)
            event_label = f"Event {int(match.group(1))}" if match else entry.segment_label
            ax.text(
                0.03,
                0.95,
                event_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=TEXT_COLOR,
                fontsize=8,
                fontweight="bold",
                bbox={"facecolor": BG_COLOR, "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
            )
            ax.set_xlabel("Time (ms)" if row == nrows - 1 else "")
            ax.set_ylabel("Current (nA)" if col == 0 else "")

            ax2 = ax.twinx()
            for spine in ax2.spines.values():
                spine.set_color(MUTED_COLOR)
            ax2.tick_params(colors=MUTED_COLOR, labelsize=6)
            ax2.yaxis.label.set_color(TEXT_COLOR)
            ax2.yaxis.labelpad = 10
            std_top = max(
                float(np.max(entry.std_trace)) if entry.std_trace.size else 0.0,
                entry.std_threshold,
            )
            ax2.set_ylim(0.0, std_top * 1.9 if std_top > 0 else 1.0)
            if entry.std_trace.size:
                ax2.plot(entry.std_time_ms, entry.std_trace, color="#ff7b72", linewidth=0.9, alpha=0.7)
            if entry.std_threshold > 0:
                ax2.axhline(entry.std_threshold, color="#b388ff", linestyle="--", linewidth=0.8)
            ax2.set_ylabel("Std (nA)" if col == ncols - 1 else "")

        for ax in axes[len(sample):]:
            ax.axis("off")

        self.figure.subplots_adjust(left=0.05, right=0.94, top=0.96, bottom=0.06, wspace=0.34, hspace=0.38)
        self.canvas.draw()

    def confirm(self) -> None:
        self.confirmed = True
        self.apply_callback(copy.deepcopy(self.current_events), keep_window=False)
        self.window.destroy()

    def on_close(self) -> None:
        if not self.confirmed:
            self.apply_callback(copy.deepcopy(self.original_events), keep_window=False)
        self.window.destroy()
