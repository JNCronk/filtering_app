from __future__ import annotations

import copy
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .dwell_t import (
    DetectedEvent,
    SegmentResult,
    DEFAULT_BUFFER_TIME_MS,
    DEFAULT_END_AT_STD_PEAK,
    DEFAULT_END_STD_HEIGHT_FACTOR,
    DEFAULT_END_STD_PROMINENCE,
    DEFAULT_END_STD_PRE_CROSSING_BUFFER_FRACTION,
    DEFAULT_END_STD_WINDOW_MS,
    DEFAULT_H5,
    DEFAULT_MIN_SEPARATION_MS,
    DEFAULT_RETURN_THRESHOLD_NA,
    DEFAULT_SAMP_FREQ,
    DEFAULT_THRESHOLD_NA,
    DEFAULT_VOLTAGE_MV,
    PREVIEW_COUNT,
    EventPreview,
    Settings,
    collect_segment_results,
    get_segment_end_std_overlay,
    iter_detected_events,
    list_event_names,
    load_previews_from_events,
    threshold_guide_value,
)
from .segment_filter import SegmentFilterWindow
from .vibration_removal import VibrationRemovalWindow


BG_COLOR = "#f5f7fb"
PANEL_COLOR = "#ffffff"
FIELD_COLOR = "#ffffff"
TEXT_COLOR = "#18212b"
MUTED_COLOR = "#5f6b7a"
TOOLBAR_COLOR = "#e8edf5"
TRACE_COLOR = "#4a5564"
BASELINE_COLOR = "#7a828c"
BEGIN_THRESHOLD_COLOR = "#2da44e"
RETURN_THRESHOLD_COLOR = "#d73a49"
POS_HIGHLIGHT_COLOR = "#ef6c63"
NEG_HIGHLIGHT_COLOR = "#4a9eff"
STD_PEAK_COLOR = "#ff7b72"
LINE_EXTENSION_MS = 1
AXIS_LABEL_FONTSIZE = 8
TICK_LABELSIZE = 7
EVENT_LABEL_FONTSIZE = 8
SEGMENT_LABEL_FONTSIZE = 7.5
HIST_TITLE_FONTSIZE = 8
HIST_BAR_COLOR = "#49C957"
FULL_TRACE_XPAD_MS = 2.0
FULL_TRACE_YPAD_FRAC = 0.05


class FilteringToolbar(NavigationToolbar2Tk):
    toolitems = tuple(
        item for item in NavigationToolbar2Tk.toolitems
        if item[0] != "Subplots"
    )


class DwellTApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Event Filtering")
        self.root.geometry("1500x950")
        self.root.configure(bg=BG_COLOR)

        self.rng = np.random.default_rng()
        self.cached_filepath = ""
        self.first_event_name: str | None = None
        self.available_event_names: list[str] = []
        self.current_preview_names: list[str] = []
        self.cached_settings_key: tuple | None = None
        self.cached_detected_events: list[DetectedEvent] | None = None
        self.cached_segment_results: list[SegmentResult] = []
        self.active_detected_events: list[DetectedEvent] = []
        self.active_segment_results: list[SegmentResult] = []

        self.filepath_var = tk.StringVar(value=DEFAULT_H5)
        self.samp_freq_var = tk.StringVar(value=str(DEFAULT_SAMP_FREQ))
        self.buffer_time_ms_var = tk.StringVar(value=str(DEFAULT_BUFFER_TIME_MS))
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD_NA))
        self.return_threshold_var = tk.StringVar(value=str(DEFAULT_RETURN_THRESHOLD_NA))
        self.min_separation_ms_var = tk.StringVar(value=str(DEFAULT_MIN_SEPARATION_MS))
        self.voltage_mV_var = tk.StringVar(value=str(DEFAULT_VOLTAGE_MV))
        self.end_at_std_peak_var = tk.BooleanVar(value=DEFAULT_END_AT_STD_PEAK)
        self.end_std_window_var = tk.StringVar(value=str(DEFAULT_END_STD_WINDOW_MS))
        self.end_std_height_var = tk.StringVar(value=str(DEFAULT_END_STD_HEIGHT_FACTOR))
        self.end_std_prominence_var = tk.StringVar(value=str(DEFAULT_END_STD_PROMINENCE))
        self.end_std_pre_crossing_buffer_fraction_var = tk.StringVar(
            value=str(DEFAULT_END_STD_PRE_CROSSING_BUFFER_FRACTION)
        )
        self.status_var = tk.StringVar(value="Choose an H5 file, then load a preview.")

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure(
            "TButton",
            background=FIELD_COLOR,
            foreground=TEXT_COLOR,
            borderwidth=1,
            focusthickness=0,
            padding=6,
            relief="solid",
        )
        style.map(
            "TButton",
            background=[("active", "#e2e8f0"), ("pressed", "#d7deea")],
            foreground=[("active", TEXT_COLOR), ("pressed", TEXT_COLOR)],
        )
        style.configure(
            "TEntry",
            fieldbackground=FIELD_COLOR,
            foreground=TEXT_COLOR,
            insertcolor=TEXT_COLOR,
            bordercolor="#c7d1dd",
        )

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        row0 = ttk.Frame(controls)
        row0.pack(side=tk.TOP, anchor="w", fill=tk.X, pady=4)
        ttk.Label(row0, text="H5 file").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(row0, textvariable=self.filepath_var, width=100).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row0, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        row1 = ttk.Frame(controls)
        row1.pack(side=tk.TOP, anchor="w", pady=4)
        acquisition_specs = [
            ("Sampling (kHz)", self.samp_freq_var),
            ("Buffer (ms)", self.buffer_time_ms_var),
            ("Voltage (mV)", self.voltage_mV_var),
        ]
        for idx, (label_text, variable) in enumerate(acquisition_specs):
            if idx > 0:
                ttk.Label(row1, text="").pack(side=tk.LEFT, padx=(10, 0))
            ttk.Label(row1, text=label_text).pack(side=tk.LEFT, padx=(0, 8))
            ttk.Entry(row1, textvariable=variable, width=10).pack(side=tk.LEFT)

        row2 = ttk.Frame(controls)
        row2.pack(side=tk.TOP, anchor="w", pady=4)
        threshold_specs = [
            ("Threshold (nA)", self.threshold_var),
            ("Return thr. (nA)", self.return_threshold_var),
            ("Min gap (ms)", self.min_separation_ms_var),
        ]
        for idx, (label_text, variable) in enumerate(threshold_specs):
            if idx > 0:
                ttk.Label(row2, text="").pack(side=tk.LEFT, padx=(10, 0))
            ttk.Label(row2, text=label_text).pack(side=tk.LEFT, padx=(0, 8))
            ttk.Entry(row2, textvariable=variable, width=10).pack(side=tk.LEFT)

        end_group = ttk.Frame(controls)
        end_group.pack(side=tk.TOP, anchor="w", pady=4)
        ttk.Checkbutton(end_group, text="End at std peak", variable=self.end_at_std_peak_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(end_group, text="Std window (ms)").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(end_group, textvariable=self.end_std_window_var, width=8).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(end_group, text="Height factor").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(end_group, textvariable=self.end_std_height_var, width=8).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(end_group, text="Prominence").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(end_group, textvariable=self.end_std_prominence_var, width=8).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(end_group, text="Pre-cross buffer frac").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(
            end_group,
            textvariable=self.end_std_pre_crossing_buffer_fraction_var,
            width=8,
        ).pack(side=tk.LEFT)

        action_row = ttk.Frame(controls)
        action_row.pack(side=tk.TOP, anchor="w", pady=(8, 4))

        ttk.Button(action_row, text="Load / Reload Preview", command=self.reload_preview).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(action_row, text="Randomise Selection", command=self.random_preview).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(action_row, text="Vibration Removal", command=self.vibration_removal).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(action_row, text="Confirm Threshold", command=self.confirm_threshold).pack(
            side=tk.LEFT
        )

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 0, 12, 8)).pack(side=tk.TOP, fill=tk.X)

        self.figure = Figure(figsize=(16, 9.4), dpi=100, facecolor=BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        toolbar_frame = tk.Frame(self.root, bg=TOOLBAR_COLOR, padx=6, pady=4)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = FilteringToolbar(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.config(background=TOOLBAR_COLOR, borderwidth=0)
        self.toolbar._message_label.configure(background=TOOLBAR_COLOR, foreground=TEXT_COLOR)
        for child in self.toolbar.winfo_children():
            widget_class = child.winfo_class()
            if widget_class == "Frame":
                child.configure(background=TOOLBAR_COLOR, borderwidth=0, highlightthickness=0, relief="flat")
            elif widget_class == "Label":
                child.configure(background=TOOLBAR_COLOR, foreground=TEXT_COLOR, borderwidth=0, highlightthickness=0)
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas.get_tk_widget().configure(bg=BG_COLOR, highlightthickness=0)

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose event H5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if path:
            self.filepath_var.set(path)

    def read_settings(self) -> Settings:
        filepath = self.filepath_var.get().strip()
        if not filepath:
            raise ValueError("Please provide an H5 file path.")
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist: {filepath}")
        pre_crossing_buffer_fraction = float(self.end_std_pre_crossing_buffer_fraction_var.get())
        if pre_crossing_buffer_fraction < 0:
            raise ValueError("Pre-cross buffer fraction must be 0 or greater.")

        return Settings(
            filepath=filepath,
            samp_freq=float(self.samp_freq_var.get()),
            buffer_time_ms=float(self.buffer_time_ms_var.get()),
            threshold_nA=float(self.threshold_var.get()),
            return_threshold_nA=float(self.return_threshold_var.get()),
            min_separation_ms=float(self.min_separation_ms_var.get()),
            voltage_mV=float(self.voltage_mV_var.get()),
            end_at_std_peak=bool(self.end_at_std_peak_var.get()),
            end_std_window_ms=float(self.end_std_window_var.get()),
            end_std_height_factor=float(self.end_std_height_var.get()),
            end_std_prominence=float(self.end_std_prominence_var.get()),
            end_std_pre_crossing_buffer_fraction=pre_crossing_buffer_fraction,
        )

    def ensure_event_cache(self, settings: Settings) -> None:
        if settings.filepath == self.cached_filepath and self.available_event_names:
            return

        self.first_event_name, self.available_event_names = list_event_names(settings.filepath)
        self.cached_filepath = settings.filepath
        self.current_preview_names = []

        if not self.available_event_names:
            raise ValueError("The file only contains the synthetic start event; there is nothing to preview.")

    def settings_cache_key(self, settings: Settings) -> tuple:
        return (
            settings.filepath,
            settings.samp_freq,
            settings.buffer_time_ms,
            settings.threshold_nA,
            settings.return_threshold_nA,
            settings.min_separation_ms,
            settings.voltage_mV,
            settings.end_at_std_peak,
            settings.end_std_window_ms,
            settings.end_std_height_factor,
            settings.end_std_prominence,
            settings.end_std_pre_crossing_buffer_fraction,
        )

    def get_cached_results(self, settings: Settings) -> tuple[list, list[SegmentResult]]:
        cache_key = self.settings_cache_key(settings)
        if self.cached_settings_key != cache_key or self.cached_detected_events is None:
            self.cached_detected_events = iter_detected_events(settings)
            self.cached_segment_results = collect_segment_results(
                settings,
                detected_events=self.cached_detected_events,
            )
            self.cached_settings_key = cache_key
        return self.cached_detected_events, self.cached_segment_results

    def reset_active_results(self, settings: Settings) -> None:
        detected_events, _ = self.get_cached_results(settings)
        self.active_detected_events = copy.deepcopy(detected_events)
        self.active_segment_results = collect_segment_results(
            settings,
            detected_events=self.active_detected_events,
        )

    def render_active_previews(self, settings: Settings) -> None:
        previews = load_previews_from_events(
            settings,
            self.active_detected_events,
            event_names=self.current_preview_names,
        )
        self.render_previews(previews, settings, self.active_segment_results)

    def choose_random_preview_names(self) -> list[str]:
        sample_size = min(PREVIEW_COUNT, len(self.available_event_names))
        if sample_size == 0:
            return []
        idx = self.rng.choice(len(self.available_event_names), size=sample_size, replace=False)
        return [self.available_event_names[i] for i in sorted(idx)]

    def reload_preview(self) -> None:
        try:
            settings = self.read_settings()
            self.ensure_event_cache(settings)
            self.reset_active_results(settings)
            if not self.current_preview_names:
                self.current_preview_names = self.choose_random_preview_names()
            self.render_active_previews(settings)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))

    def random_preview(self) -> None:
        try:
            settings = self.read_settings()
            self.ensure_event_cache(settings)
            if not self.active_detected_events:
                self.reset_active_results(settings)
            self.current_preview_names = self.choose_random_preview_names()
            self.render_active_previews(settings)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))

    def vibration_removal(self) -> None:
        try:
            settings = self.read_settings()
            self.ensure_event_cache(settings)
            if not self.active_detected_events:
                self.reset_active_results(settings)
            VibrationRemovalWindow(
                parent=self.root,
                settings=settings,
                source_events=copy.deepcopy(self.active_detected_events),
                apply_callback=lambda events, keep_window: self.apply_vibration_results(settings, events),
            )
        except Exception as exc:
            messagebox.showerror("Vibration Removal", str(exc))

    def apply_vibration_results(self, settings: Settings, events: list[DetectedEvent]) -> None:
        self.active_detected_events = events
        self.active_segment_results = collect_segment_results(
            settings,
            detected_events=self.active_detected_events,
        )
        self.render_active_previews(settings)

    def _adaptive_bins(self, values: np.ndarray) -> int | str:
        if values.size < 2:
            return 1
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            return min(20, max(5, int(np.sqrt(values.size))))
        bin_width = 2 * iqr * np.power(values.size, -1 / 3)
        if bin_width <= 0:
            return "auto"
        data_range = values.max() - values.min()
        if data_range <= 0:
            return 1
        return max(1, int(np.ceil(data_range / bin_width)))

    def _robust_hist_values(self, values: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return finite, None
        if finite.size < 8:
            lo = float(finite.min())
            hi = float(finite.max())
        else:
            lo, hi = np.percentile(finite, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo = float(finite.min())
                hi = float(finite.max())
        if lo == hi:
            pad = abs(lo) * 0.05 if lo != 0 else 0.5
            lo -= pad
            hi += pad
        display = finite[(finite >= lo) & (finite <= hi)]
        if display.size == 0:
            display = finite
            lo = float(finite.min())
            hi = float(finite.max())
        return display, (float(lo), float(hi))

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_color(MUTED_COLOR)
        ax.tick_params(colors=MUTED_COLOR, labelsize=TICK_LABELSIZE)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
        ax.title.set_color(TEXT_COLOR)

    def _plot_std_overlay(self, ax, std_time_ms: np.ndarray, std_trace: np.ndarray, std_threshold: float) -> None:
        if std_trace.size == 0:
            return
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        if y_span <= 0:
            return
        std_top = max(float(np.max(std_trace)), std_threshold)
        if std_top <= 0:
            return
        overlay_base = y_min + 0.06 * y_span
        overlay_height = 0.24 * y_span
        scaled_std = overlay_base + (std_trace / std_top) * overlay_height
        ax.plot(
            std_time_ms,
            scaled_std,
            color=STD_PEAK_COLOR,
            alpha=0.7,
            linewidth=0.9,
            zorder=8,
        )
        if std_threshold > 0:
            scaled_threshold = overlay_base + (std_threshold / std_top) * overlay_height
            ax.axhline(
                scaled_threshold,
                color="#b388ff",
                alpha=0.75,
                linestyle="--",
                linewidth=0.8,
                zorder=9,
            )

    def _render_histograms(self, hist_axes: list, segment_results: list[SegmentResult]) -> None:
        metrics = [
            ("Dwell t (ms)", np.array([r.dwell_time_ms for r in segment_results], dtype=float)),
            ("EC (nA ms)", np.array([r.area_nA_ms for r in segment_results], dtype=float)),
            ("Relative  \u0394I", np.array([r.delta_I_rel for r in segment_results], dtype=float)),
            ("R (MOhm)", np.array([r.resistance_MOhm for r in segment_results], dtype=float)),
        ]

        for ax, (label, values) in zip(hist_axes, metrics):
            self._style_axis(ax)
            display, xlim = self._robust_hist_values(values)
            if display.size == 0:
                ax.text(
                    0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                color=MUTED_COLOR, fontsize=SEGMENT_LABEL_FONTSIZE
                )
                ax.set_title("")
                continue
            ax.hist(display, bins=self._adaptive_bins(display), color=HIST_BAR_COLOR, alpha=0.8, edgecolor="#eef2f7")
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.set_title("")
            ax.set_ylabel("Count")
            ax.set_xlabel(label)
            ax.xaxis.set_label_position("bottom")
            ax.tick_params(axis="x", labelbottom=True, labeltop=False, bottom=True, top=False)

        for ax in hist_axes:
            ax.xaxis.set_label_position("bottom")
            ax.tick_params(axis="x", labelbottom=True, labeltop=False, bottom=True, top=False)

    def render_previews(
        self,
        previews: list[EventPreview],
        settings: Settings,
        segment_results: list[SegmentResult],
    ) -> None:
        self.figure.clear()
        n_rows, n_cols = 4, 3
        gs = self.figure.add_gridspec(
            nrows=n_rows,
            ncols=4,
            width_ratios=[1.0, 1.0, 1.0, 1.5],
            wspace=0.24,
            hspace=0.30,
        )
        axes = [self.figure.add_subplot(gs[row, col]) for row in range(n_rows) for col in range(n_cols)]
        hist_axes = [self.figure.add_subplot(gs[row, 3]) for row in range(n_rows)]

        for idx, (ax, preview) in enumerate(zip(axes, previews)):
            row = idx // n_cols
            col = idx % n_cols
            self._style_axis(ax)

            time_ms = np.arange(len(preview.data)) / settings.samp_freq
            ax.plot(time_ms, preview.data, color=TRACE_COLOR, linewidth=0.8)
            match = re.search(r"(\d+)$", preview.event_name)
            event_label = f"Event {int(match.group(1))}" if match else preview.event_name

            if preview.segments:
                dwell_labels = []
                for segment_info in preview.segments:
                    colour = POS_HIGHLIGHT_COLOR if segment_info.direction == 1 else NEG_HIGHLIGHT_COLOR
                    line_start = max(0.0, (segment_info.start / settings.samp_freq) - LINE_EXTENSION_MS)
                    line_end = min(time_ms[-1], (segment_info.end / settings.samp_freq) + LINE_EXTENSION_MS)
                    ax.axvspan(
                        segment_info.start / settings.samp_freq,
                        segment_info.end / settings.samp_freq,
                        color=colour,
                        alpha=0.12,
                    )
                    ax.hlines(
                        segment_info.baseline,
                        line_start,
                        line_end,
                        colors=BASELINE_COLOR,
                        linestyles="--",
                        linewidth=1.0,
                    )
                    threshold_value = threshold_guide_value(
                        segment_info.baseline,
                        segment_info.direction,
                        settings.threshold_nA,
                    )
                    ax.hlines(
                        threshold_value,
                        line_start,
                        line_end,
                        colors=BEGIN_THRESHOLD_COLOR,
                        linestyles="--",
                        linewidth=0.9,
                    )
                    return_threshold_value = threshold_guide_value(
                        segment_info.baseline,
                        segment_info.direction,
                        settings.return_threshold_nA,
                    )
                    ax.hlines(
                        return_threshold_value,
                        line_start,
                        line_end,
                        colors=RETURN_THRESHOLD_COLOR,
                        linestyles="--",
                        linewidth=0.9,
                    )
                    dwell_ms = (segment_info.end - segment_info.start + 1) / settings.samp_freq
                    dwell_labels.append(f"{dwell_ms:.2f} ms")
                segment_label = ", ".join(dwell_labels)
            else:
                segment_label = "no event"

            if time_ms.size:
                ax.set_xlim(-FULL_TRACE_XPAD_MS, float(time_ms[-1] + FULL_TRACE_XPAD_MS))
                y_min = float(np.min(preview.data))
                y_max = float(np.max(preview.data))
                y_span = y_max - y_min
                y_pad = max(0.03, FULL_TRACE_YPAD_FRAC * y_span if y_span > 0 else 0.05 * max(abs(y_max), 1.0))
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            if settings.end_at_std_peak and preview.segments:
                overlay_segment = preview.segments[0]
                std_time_ms, std_trace, std_threshold = get_segment_end_std_overlay(
                    preview.data,
                    overlay_segment,
                    settings,
                )
                self._plot_std_overlay(ax, std_time_ms, std_trace, std_threshold)

            ax.text(
                0.03,
                0.95,
                event_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=TEXT_COLOR,
                fontsize=EVENT_LABEL_FONTSIZE,
                fontweight="bold",
                bbox={"facecolor": BG_COLOR, "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
                zorder=20,
            )
            ax.text(
                0.03,
                0.08,
                segment_label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color=TEXT_COLOR,
                fontsize=SEGMENT_LABEL_FONTSIZE,
                bbox={"facecolor": BG_COLOR, "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
                zorder=20,
            )
            ax.set_xlabel("Time (ms)" if row == n_rows - 1 else "")
            ax.set_ylabel("Current (nA)" if col == 0 else "")

        for ax in axes[len(previews):]:
            ax.axis("off")

        self._render_histograms(hist_axes, segment_results)
        self.figure.subplots_adjust(left=0.04, right=0.99, top=0.985, bottom=0.06)
        self.canvas.draw()

        first_note = ""
        if self.first_event_name:
            first_note = f" Synthetic start event '{self.first_event_name}' is excluded."
        self.status_var.set(
            f"Previewing {len(previews)} traces from {len(self.available_event_names)} events; "
            f"{len(segment_results)} detected segments in current threshold.{first_note}"
        )

    def confirm_threshold(self) -> None:
        try:
            settings = self.read_settings()
            self.ensure_event_cache(settings)
            self.status_var.set("Opening segment filtering stage...")
            self.root.update_idletasks()
            if not self.active_detected_events:
                self.reset_active_results(settings)
            SegmentFilterWindow(
                parent=self.root,
                settings=settings,
                detected_events=copy.deepcopy(self.active_detected_events),
                segment_results=copy.deepcopy(self.active_segment_results),
            )
        except Exception as exc:
            messagebox.showerror("Filtering stage", str(exc))


def main() -> None:
    root = tk.Tk()
    DwellTApp(root)
    root.mainloop()
