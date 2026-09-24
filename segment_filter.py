from __future__ import annotations

import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import h5py
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from .dwell_t import (
    DetectedEvent,
    SegmentResult,
    Settings,
    get_segment_end_std_overlay,
    threshold_guide_value,
)


BG_COLOR = "#f5f7fb"
PANEL_COLOR = "#ffffff"
FIELD_COLOR = "#ffffff"
TEXT_COLOR = "#18212b"
MUTED_COLOR = "#5f6b7a"
TRACE_COLOR = "#4a5564"
POS_COLOR = "#ef6c63"
NEG_COLOR = "#4a9eff"
SELECT_EDGE = "#18212b"


@dataclass
class SegmentTrace:
    result: SegmentResult
    data: np.ndarray
    baseline: float


class FilteringToolbar(NavigationToolbar2Tk):
    toolitems = tuple(item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots")


class SegmentFilterWindow:
    def __init__(
        self,
        parent,
        settings: Settings,
        detected_events: list[DetectedEvent],
        segment_results: list[SegmentResult],
    ) -> None:
        self.parent = parent
        self.settings = settings
        self.detected_events = detected_events
        self.segment_results = segment_results
        self.selected_indices: set[int] = set()
        self.last_selected_index: int | None = None
        self.selectors = []
        self.collections = []
        self.segment_traces = self._load_segment_traces()
        self.index_by_name = {row.event_name: idx for idx, row in enumerate(self.segment_results)}
        self.box_select_mode = tk.StringVar(value="add")
        self.base_event_names: dict[str, str] = {}
        self.segment_infos = {}
        for event in self.detected_events:
            for seg_idx, segment in enumerate(event.segments):
                seg_name = event.event_name if seg_idx == 0 else f"{event.event_name}_{seg_idx}"
                self.base_event_names[seg_name] = event.event_name
                self.segment_infos[seg_name] = segment

        self.window = tk.Toplevel(parent)
        self.window.title("Segment Filtering")
        self.window.geometry("1550x980")
        self.window.configure(bg=BG_COLOR)

        self.status_var = tk.StringVar(value=f"{len(self.segment_results)} segments loaded. Select points to keep.")
        self.scatter_axes = {}
        self.ax_trace = None

        self._build_ui()
        self.render()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.window, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls, text="Clear Selection", command=self.clear_selection).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Save Selected CSV", command=self.save_selected_csv).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="Save Selected H5", command=self.save_selected_h5).grid(row=0, column=2, padx=(0, 8))
        ttk.Label(controls, text="Box mode").grid(row=0, column=3, padx=(16, 6))
        ttk.Radiobutton(controls, text="Add", variable=self.box_select_mode, value="add").grid(row=0, column=4, padx=(0, 6))
        ttk.Radiobutton(controls, text="Remove", variable=self.box_select_mode, value="remove").grid(row=0, column=5, padx=(0, 6))

        ttk.Label(self.window, textvariable=self.status_var, padding=(12, 0, 12, 8)).pack(side=tk.TOP, fill=tk.X)

        self.figure = Figure(figsize=(15.2, 9.5), dpi=100, facecolor=BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)

        toolbar_frame = tk.Frame(self.window, bg="#e8edf5", padx=6, pady=4)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = FilteringToolbar(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.config(background="#e8edf5", borderwidth=0)
        self.toolbar._message_label.configure(background="#e8edf5", foreground=TEXT_COLOR)
        for child in self.toolbar.winfo_children():
            widget_class = child.winfo_class()
            if widget_class == "Frame":
                child.configure(background="#e8edf5", borderwidth=0, highlightthickness=0, relief="flat")
            elif widget_class == "Label":
                child.configure(background="#e8edf5", foreground=TEXT_COLOR, borderwidth=0, highlightthickness=0)
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas.get_tk_widget().configure(bg=BG_COLOR, highlightthickness=0)
        self.canvas.mpl_connect("pick_event", self.on_pick)

    def _load_segment_traces(self) -> list[SegmentTrace]:
        traces: list[SegmentTrace] = []
        segment_lookup: dict[str, tuple[str, int, float]] = {}
        for event in self.detected_events:
            for seg_idx, segment in enumerate(event.segments):
                seg_name = event.event_name if seg_idx == 0 else f"{event.event_name}_{seg_idx}"
                segment_lookup[seg_name] = (event.event_name, seg_idx, segment.baseline)

        with h5py.File(self.settings.filepath, "r") as h5:
            grp = h5["events"]
            for result in self.segment_results:
                base_event_name, _, baseline = segment_lookup[result.event_name]
                data = grp[base_event_name][...]
                traces.append(SegmentTrace(result=result, data=data, baseline=baseline))
        return traces

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_color(MUTED_COLOR)
        ax.tick_params(colors=MUTED_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
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
            color="#ff7b72",
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

    def _metric_arrays(self, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
        xs = np.array([getattr(row, x_key) for row in self.segment_results], dtype=float)
        ys = np.array([getattr(row, y_key) for row in self.segment_results], dtype=float)
        return xs, ys

    def _facecolors(self) -> np.ndarray:
        colors = []
        for idx, row in enumerate(self.segment_results):
            if idx in self.selected_indices:
                colors.append(POS_COLOR if row.direction == 1 else NEG_COLOR)
            else:
                colors.append("#c0c8d2")
        return np.array(colors, dtype=object)

    def _edgecolors(self) -> np.ndarray:
        colors = []
        for idx in range(len(self.segment_results)):
            colors.append(SELECT_EDGE if idx in self.selected_indices else "#c0c8d2")
        return np.array(colors, dtype=object)

    def render(self) -> None:
        self.figure.clear()
        self.collections.clear()
        self.selectors.clear()
        self.scatter_axes = {}

        gs = self.figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.30, wspace=0.22)
        ax_dt_ec = self.figure.add_subplot(gs[0, 0])
        ax_dt_rel = self.figure.add_subplot(gs[0, 1])
        ax_ec_rel = self.figure.add_subplot(gs[1, 0])
        ax_r_dt = self.figure.add_subplot(gs[1, 1])
        ax_trace = self.figure.add_subplot(gs[2, :])

        scatter_specs = [
            ("dt_ec", ax_dt_ec, "dwell_time_ms", "area_nA_ms", "Dwell Time (ms)", "EC (nA ms)"),
            ("dt_rel", ax_dt_rel, "dwell_time_ms", "delta_I_rel", "Dwell Time (ms)", "Relative  \u0394I"),
            ("ec_rel", ax_ec_rel, "area_nA_ms", "delta_I_rel", "EC (nA ms)", "Relative \u0394I"),
            ("r_dt", ax_r_dt, "resistance_MOhm", "dwell_time_ms", "R (MOhm)", "Dwell Time (ms)"),
        ]

        for axis_id, ax, x_key, y_key, xlabel, ylabel in scatter_specs:
            self._style_axis(ax)
            self.scatter_axes[axis_id] = ax
            xs, ys = self._metric_arrays(x_key, y_key)
            coll = ax.scatter(
                xs,
                ys,
                c=self._facecolors(),
                edgecolors=self._edgecolors(),
                linewidths=np.where(np.isin(np.arange(len(xs)), list(self.selected_indices)), 1.2, 0.7),
                alpha=0.9,
                s=28,
                picker=True,
            )
            coll._segment_indices = np.arange(len(xs))
            coll._axis_id = axis_id
            coll._x_key = x_key
            coll._y_key = y_key
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            self.collections.append(coll)
            selector = RectangleSelector(
                ax,
                lambda e0, e1, axis=ax, x=x_key, y=y_key: self.on_select(axis, x, y, e0, e1),
                useblit=False,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=False,
            )
            self.selectors.append(selector)

        self._style_axis(ax_trace)
        self.ax_trace = ax_trace
        ax_trace.set_xlabel("Time (ms)")
        ax_trace.set_ylabel("Current (nA)")
        self.update_trace_axis()
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.07)
        self.canvas.draw()
        self.update_status()

    def update_collections(self) -> None:
        facecolors = self._facecolors()
        edgecolors = self._edgecolors()
        linewidths = np.where(np.isin(np.arange(len(self.segment_results)), list(self.selected_indices)), 1.2, 0.7)
        for coll in self.collections:
            coll.set_facecolors(facecolors)
            coll.set_edgecolors(edgecolors)
            coll.set_linewidths(linewidths)

    def update_trace_axis(self) -> None:
        if self.ax_trace is None:
            return
        ax_trace = self.ax_trace
        ax_trace.clear()
        self._style_axis(ax_trace)
        ax_trace.set_xlabel("Time (ms)")
        ax_trace.set_ylabel("Current (nA)")
        if self.last_selected_index is not None:
            trace = self.segment_traces[self.last_selected_index]
            result = trace.result
            time_ms = np.arange(len(trace.data)) / self.settings.samp_freq
            ax_trace.plot(time_ms, trace.data, color=TRACE_COLOR, linewidth=0.9)
            ax_trace.axvspan(
                result.start / self.settings.samp_freq,
                result.end / self.settings.samp_freq,
                color=POS_COLOR if result.direction == 1 else NEG_COLOR,
                alpha=0.10,
            )
            ax_trace.axhline(trace.baseline, color="#8b949e", linestyle="--", linewidth=0.9)
            threshold_value = threshold_guide_value(
                trace.baseline,
                result.direction,
                self.settings.threshold_nA,
            )
            ax_trace.axhline(
                threshold_value,
                color="#7ee787",
                linestyle="--",
                linewidth=0.8,
            )
            return_threshold_value = threshold_guide_value(
                trace.baseline,
                result.direction,
                self.settings.return_threshold_nA,
            )
            ax_trace.axhline(
                return_threshold_value,
                color="#ff7b72",
                linestyle="--",
                linewidth=0.8,
            )
            segment_info = self.segment_infos.get(result.event_name)
            if self.settings.end_at_std_peak and segment_info is not None:
                std_time_ms, std_trace, std_threshold = get_segment_end_std_overlay(
                    trace.data,
                    segment_info,
                    self.settings,
                )
                self._plot_std_overlay(ax_trace, std_time_ms, std_trace, std_threshold)
            ax_trace.text(
                0.02,
                0.95,
                result.event_name,
                transform=ax_trace.transAxes,
                ha="left",
                va="top",
                color=TEXT_COLOR,
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": BG_COLOR, "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
                zorder=20,
            )
        else:
            ax_trace.text(
                0.5,
                0.5,
                "Select a point to view its trace",
                transform=ax_trace.transAxes,
                ha="center",
                va="center",
                color=MUTED_COLOR,
            )

    def refresh_selection_views(self) -> None:
        self.update_collections()
        self.update_trace_axis()
        self.canvas.draw_idle()
        self.update_status()

    def update_status(self) -> None:
        self.status_var.set(f"{len(self.selected_indices)} selected of {len(self.segment_results)} segments.")

    def on_pick(self, event) -> None:
        if not hasattr(event.artist, "_segment_indices"):
            return
        if len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.add(idx)
            self.last_selected_index = idx
        self.refresh_selection_views()

    def on_select(self, axis, x_key: str, y_key: str, eclick, erelease) -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        xs, ys = self._metric_arrays(x_key, y_key)
        mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
        indices = np.where(mask)[0]
        if indices.size == 0:
            return
        if self.box_select_mode.get() == "remove":
            for idx in indices.tolist():
                self.selected_indices.discard(idx)
            if self.last_selected_index in indices.tolist():
                self.last_selected_index = None
        else:
            self.selected_indices.update(indices.tolist())
            self.last_selected_index = int(indices[0])
        self.refresh_selection_views()

    def clear_selection(self) -> None:
        self.selected_indices.clear()
        self.last_selected_index = None
        self.refresh_selection_views()

    def _default_csv_path(self) -> str:
        base = os.path.splitext(os.path.basename(self.settings.filepath))[0]
        return os.path.join(os.path.dirname(self.settings.filepath), f"{base}_filtered.csv")

    def _default_h5_path(self) -> str:
        base = os.path.splitext(os.path.basename(self.settings.filepath))[0]
        return os.path.join(os.path.dirname(self.settings.filepath), f"{base}_filtered.h5")

    def save_selected_csv(self) -> None:
        if not self.selected_indices:
            messagebox.showinfo("Save Selected CSV", "No segments selected.", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save selected CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(self._default_csv_path()),
            initialdir=os.path.dirname(self._default_csv_path()),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        rows = [self.segment_results[idx] for idx in sorted(self.selected_indices)]
        import csv
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "event_name", "timestamp", "dwell_time_ms", "direction", "start", "end",
                "area_nA_ms", "delta_I_nA", "delta_I_rel", "resistance_MOhm"
            ])
            for row in rows:
                writer.writerow([
                    row.event_name, row.timestamp, row.dwell_time_ms, row.direction, row.start, row.end,
                    row.area_nA_ms, row.delta_I_nA, row.delta_I_rel, row.resistance_MOhm
                ])
        messagebox.showinfo("Save Selected CSV", f"Saved selected rows to:\n{path}", parent=self.window)

    def save_selected_h5(self) -> None:
        if not self.selected_indices:
            messagebox.showinfo("Save Selected H5", "No segments selected.", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save selected H5",
            defaultextension=".h5",
            initialfile=os.path.splitext(os.path.basename(self._default_h5_path()))[0],
            initialdir=os.path.dirname(self._default_h5_path()),
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return

        selected_rows = [self.segment_results[idx] for idx in sorted(self.selected_indices)]
        with h5py.File(self.settings.filepath, "r") as src, h5py.File(path, "w") as dst:
            src_events = src["events"]
            out_events = dst.create_group("events")

            event_names = sorted(src_events.keys())
            if event_names:
                start_name = event_names[0]
                start_ds = src_events[start_name]
                copied = out_events.create_dataset(start_name, data=start_ds[()])
                for key, value in start_ds.attrs.items():
                    copied.attrs[key] = value

            for row in selected_rows:
                src_name = self.base_event_names.get(row.event_name, row.event_name)
                if src_name not in src_events:
                    continue
                src_ds = src_events[src_name]
                copied = out_events.create_dataset(row.event_name, data=src_ds[()])
                for key, value in src_ds.attrs.items():
                    copied.attrs[key] = value
                copied.attrs["source_event_name"] = src_name
        messagebox.showinfo("Save Selected H5", f"Saved filtered events to:\n{path}", parent=self.window)
