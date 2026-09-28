"""PyQtGraph trace previews and the four analysis summaries."""
from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .dwell_t import get_segment_end_std_overlay, threshold_guide_value

METRICS = [
    ("dwell_time_ms", "Dwell time (ms)"),
    ("area_nA_ms", "EC (nA ms)"),
    ("delta_I_rel", "Relative ΔI"),
    ("resistance_MOhm", "Resistance (MΩ)"),
]
METRIC_COLORS = ["#149bff", "#00cfa0", "#bb55ff", "#ffac18"]
TRACE_COLOR = "#42627e"


def scatter_brush(color, alpha=160):
    color = pg.mkColor(color)
    color.setAlpha(alpha)
    return pg.mkBrush(color)


class ZeroAnchoredViewBox(pg.ViewBox):
    """Keep the count range at [0, span] during zoom, pan, and auto-range."""
    def __init__(self):
        super().__init__()
        self.setLimits(xMin=0)
        self.sigXRangeChanged.connect(self._pin_zero)

    def _pin_zero(self, view, limits):
        low, high = limits
        if low != 0:
            self.setXRange(0, max(high - low, 1e-9), padding=0)


def configure_plots():
    pg.setConfigOptions(background="w", foreground="#344456", antialias=True)


def style_plot(plot, xlabel="", ylabel=""):
    plot.setLabel("bottom", xlabel)
    plot.setLabel("left", ylabel)
    plot.showGrid(x=True, y=True, alpha=0.12)
    plot.hideButtons()
    for side in ("left", "bottom"):
        plot.getAxis(side).setPen(pg.mkPen("#c5cfda"))
        plot.getAxis(side).setTextPen(pg.mkPen("#566679"))


def robust_histogram(values):
    """Display the central 99% (0.5th–99.5th percentiles), with bounded bin counts.

    Bounds use observed ranks so small samples do not lose their extrema and
    at least 99% of finite observations remain visible, including tied values.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.array([]), np.array([]), 0
    # Rounding either tail outward guarantees at least 99% for every sample size.
    trim = int(np.floor(finite.size * 0.005))
    ordered = np.sort(finite)
    lo, hi = ordered[trim], ordered[-trim - 1]
    visible = finite[(finite >= lo) & (finite <= hi)]
    if lo == hi:
        pad = max(abs(float(lo)) * 0.02, 0.01)
        lo, hi = lo - pad, hi + pad
        bins = 1
    else:
        q25, q75 = np.percentile(visible, [25, 75])
        width = 2 * (q75 - q25) / np.cbrt(visible.size)
        bins = int(np.ceil((hi - lo) / width)) if width > 0 else int(np.sqrt(visible.size))
        bins = min(120, max(5, bins))
    counts, edges = np.histogram(visible, bins=bins, range=(lo, hi))
    return counts, edges, int(finite.size - visible.size)


def add_std_overlay(plot, data, times, values, threshold):
    finite = data[np.isfinite(data)]
    if not finite.size or not len(values):
        return
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 0.01)
    scale = max(float(np.max(values)), threshold, 1e-12)
    base, height = low + 0.04 * span, 0.24 * span
    plot.plot(times, base + values / scale * height, pen=pg.mkPen("#d99032", width=1))
    plot.addItem(pg.InfiniteLine(base + threshold / scale * height, angle=0,
                               pen=pg.mkPen("#a376be", style=QtCore.Qt.PenStyle.DashLine)))


def draw_trace(plot, preview, sampling, settings=None):
    plot.clear()
    style_plot(plot, "Time within event (ms)", "Current (nA)")
    title = html.escape(preview.event_name)
    if settings is not None:
        durations = [f"{(s.end - s.start + 1) / sampling:.2f} ms" for s in preview.segments]
        title += " · " + (", ".join(durations) if durations else "no segment")
    plot.setTitle(title, size="10pt")
    curve = plot.plot(np.arange(len(preview.data)) / sampling, preview.data,
                      pen=pg.mkPen(TRACE_COLOR, width=1), connect="finite")
    curve.setDownsampling(auto=True, method="peak")
    curve.setClipToView(True)
    if settings is not None:
        for segment in preview.segments:
            color = (234, 123, 92, 40) if segment.direction == 1 else (66, 144, 217, 40)
            start, end = segment.start / sampling, segment.end / sampling
            region = pg.LinearRegionItem((start, end), movable=False, brush=color,
                                         pen=pg.mkPen(None))
            region.setZValue(-5)
            plot.addItem(region)
            for value, color in [
                (segment.baseline, "#87939f"),
                (threshold_guide_value(segment.baseline, segment.direction, settings.threshold_nA), "#259c71"),
                (threshold_guide_value(segment.baseline, segment.direction, settings.return_threshold_nA), "#d16c76"),
            ]:
                plot.plot([max(0, start - 1), end + 1], [value, value],
                          pen=pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine))
        if settings.end_at_std_peak and preview.segments:
            times, values, threshold = get_segment_end_std_overlay(preview.data, preview.segments[0], settings)
            add_std_overlay(plot, preview.data, times, values, threshold)
    plot.enableAutoRange()


class TraceGrid(pg.GraphicsLayoutWidget):
    def __init__(self, rows=2, columns=3, parent=None):
        super().__init__(parent)
        self.plots = [self.addPlot(row=i // columns, col=i % columns)
                      for i in range(rows * columns)]
        for plot in self.plots:
            style_plot(plot, "Time within event (ms)", "Current (nA)")

    def display(self, previews, sampling, settings=None):
        for index, plot in enumerate(self.plots):
            if index < len(previews):
                draw_trace(plot, previews[index], sampling, settings)
            else:
                plot.clear()
                plot.setTitle("No event")


class SelectionViewBox(pg.ViewBox):
    rectangleSelected = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.selection_mode = 'Add'
        self.selection_box = QtWidgets.QGraphicsRectItem()
        self.selection_box.setPen(pg.mkPen('#387fa7', width=1))
        self.selection_box.setBrush(pg.mkBrush(56, 127, 167, 40))
        self.addItem(self.selection_box, ignoreBounds=True)
        self.selection_box.hide()

    def mouseDragEvent(self, event, axis=None):
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self.selection_mode == 'Pan':
            super().mouseDragEvent(event, axis=axis)
            return
        event.accept()
        start = self.mapToView(event.buttonDownPos())
        end = self.mapToView(event.pos())
        rect = QtCore.QRectF(start, end).normalized()
        self.selection_box.setRect(rect)
        self.selection_box.show()
        if event.isFinish():
            self.selection_box.hide()
            self.rectangleSelected.emit(rect)


class SummaryPlots(pg.GraphicsLayoutWidget):
    eventsSelected = QtCore.Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.histograms, self.timelines, self.curves = [], [], []
        self.point_names = []
        self.previewed_names = set()
        self.selected_segments = None
        self.histogram_data = []
        self.point_segments = []
        self.result_names = []
        for i, (_, label) in enumerate(METRICS):
            histogram = self.addPlot(row=i, col=0, viewBox=ZeroAnchoredViewBox())
            view = SelectionViewBox()
            timeline = self.addPlot(row=i, col=1, viewBox=view)
            style_plot(histogram, "Count", label)
            # A count grid clips the edge tick label in PyQtGraph; keep value-grid lines.
            histogram.showGrid(x=False, y=True, alpha=0.12)
            style_plot(timeline, "Event time since recording start (s)", label)
            histogram.setLimits(xMin=0)
            histogram.getAxis('bottom').setStyle(hideOverlappingLabels=False)
            histogram.setYLink(timeline)
            histogram.getAxis('left').setWidth(70)
            timeline.getAxis('left').setWidth(70)
            view.rectangleSelected.connect(lambda rect, index=i: self._box_selected(index, rect))
            self.histograms.append(histogram)
            self.timelines.append(timeline)
            if i:
                timeline.setXLink(self.timelines[0])
        self.display([], 0.0)

    @staticmethod
    def _additive(event=None):
        modifiers = event.modifiers() if event is not None else QtWidgets.QApplication.keyboardModifiers()
        return bool(modifiers & (QtCore.Qt.KeyboardModifier.ControlModifier |
                                 QtCore.Qt.KeyboardModifier.MetaModifier |
                                 QtCore.Qt.KeyboardModifier.ShiftModifier))

    def _points_clicked(self, scatter, points, event):
        if points:
            names = list(dict.fromkeys(point.data() for point in points))
            self.eventsSelected.emit(names, self._additive(event))

    def _box_selected(self, index, rect):
        curve = self.curves[index]
        x, y = curve.getData()
        if x is None:
            return
        mask = (x >= rect.left()) & (x <= rect.right()) & (y >= rect.top()) & (y <= rect.bottom())
        names = list(dict.fromkeys(self.point_names[index][i] for i in np.flatnonzero(mask)))
        if names:
            self.eventsSelected.emit(names, self._additive())

    def display(self, results, origin, event_names=None):
        mapping = event_names or {}
        names = [mapping.get(r.event_name, r.event_name) for r in results]
        self.curves, self.point_names, self.point_segments = [], [], []
        self.histogram_data = []
        self.result_names = [r.event_name for r in results]
        timestamps = np.array([r.timestamp if r.timestamp is not None else np.nan for r in results], dtype=float)
        for (key, label), color, histogram, timeline in zip(METRICS, METRIC_COLORS, self.histograms, self.timelines):
            histogram.clear()
            # Keep the ViewBox's selection rectangle when replacing plotted data.
            for item in list(timeline.items):
                timeline.removeItem(item)
            values = np.array([getattr(r, key) for r in results], dtype=float)
            counts, edges, excluded = robust_histogram(values)
            if counts.size:
                base = pg.BarGraphItem(x0=0, width=counts, y0=edges[:-1], y1=edges[1:],
                                       brush=color, pen=pg.mkPen('w', width=0.5))
                overlay = pg.BarGraphItem(x0=0, width=np.zeros_like(counts), y0=edges[:-1], y1=edges[1:],
                                          brush=color, pen=pg.mkPen('w', width=0.5))
                overlay.setZValue(1)
                histogram.addItem(base)
                histogram.addItem(overlay)
                self.histogram_data.append((base, overlay, values, edges, color))
                histogram.setXRange(0, max(float(counts.max()) * 1.1, 1), padding=0)
                timeline.setYRange(float(edges[0]), float(edges[-1]), padding=0.05)
                histogram.setToolTip(f"Central 99%: {excluded} values outside the initial view")
            else:
                self.histogram_data.append(None)
                histogram.setToolTip("No segment metrics to display")
                histogram.setXRange(0, 1, padding=0)
                timeline.setYRange(0, 1, padding=0)
            valid = np.flatnonzero(np.isfinite(timestamps) & np.isfinite(values))
            visible_names = [names[i] for i in valid]
            curve = timeline.plot(timestamps[valid] - origin, values[valid], pen=None,
                                  symbol='o', symbolSize=6, symbolBrush=scatter_brush(color), symbolPen=None,
                                  data=visible_names)
            curve.scatter.sigClicked.connect(self._points_clicked)
            self.curves.append(curve)
            self.point_names.append(visible_names)
            self.point_segments.append([self.result_names[i] for i in valid])
            missing = int(np.count_nonzero(np.isfinite(values) & ~np.isfinite(timestamps)))
            histogram.setTitle(None)
            timeline.setTitle(None)
            timeline.setToolTip(f"{missing} segments with missing timestamps are omitted")
            timeline.enableAutoRange(x=True, y=False)
        self.set_selection(self.selected_segments)

    def set_selection(self, names):
        """None shows all results normally; a set overlays only selected segments."""
        self.selected_segments = None if names is None else set(names)
        selected = np.array([name in (self.selected_segments or set()) for name in self.result_names], dtype=bool)
        for entry in self.histogram_data:
            if entry is None:
                continue
            base, overlay, values, edges, color = entry
            filtering = self.selected_segments is not None
            base.setOpts(brush="#cbd1d8" if filtering else color)
            overlay.setVisible(filtering)
            if filtering:
                counts, _ = np.histogram(values[selected & np.isfinite(values)], bins=edges)
                overlay.setOpts(width=counts)
        self.highlight_events(self.previewed_names)

    def highlight_events(self, names):
        self.previewed_names = set(names)
        for curve, point_names, segment_names, color in zip(self.curves, self.point_names, self.point_segments, METRIC_COLORS):
            base = pg.mkColor(color)
            light = pg.mkColor(tuple(round(channel + (255 - channel) * 0.5)
                                    for channel in base.getRgb()[:3]))
            brushes = []
            for event_name, segment_name in zip(point_names, segment_names):
                if self.selected_segments is not None and segment_name not in self.selected_segments:
                    brushes.append(scatter_brush('#aeb8c3', 100))
                else:
                    brushes.append(scatter_brush(light if event_name in self.previewed_names else base))
            curve.scatter.setBrush(brushes)
            curve.scatter.setSize([9 if name in self.previewed_names else 6 for name in point_names])


class SegmentPreviewWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings, previews):
        super().__init__(parent)
        self.setWindowTitle("Example segments")
        self.resize(1350, 850)
        layout = QtWidgets.QVBoxLayout(self)
        self.grid = TraceGrid(rows=4)
        layout.addWidget(self.grid, 1)
        self.grid.display(previews, settings.samp_freq, settings)
