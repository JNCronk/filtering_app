from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .H5Splitter_1000 import recording_start_time
from .dwell_t import EventPreview
from .plots import SelectionViewBox, draw_trace, scatter_brush, style_plot
from .qt_helpers import button, number_field


def write_selected_csv(path, rows):
    if not rows:
        raise ValueError('No segments selected.')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_selected_h5(path, settings, rows, base_event_names):
    if Path(path).resolve() == Path(settings.filepath).resolve():
        raise ValueError('Choose a different filename; the input file cannot be overwritten.')
    if not rows:
        raise ValueError('No segments selected.')
    # Build beside the destination and publish only after the complete export succeeds.
    descriptor, temporary = tempfile.mkstemp(prefix='.filtered-', suffix='.h5', dir=Path(path).parent)
    os.close(descriptor)
    try:
        with h5py.File(settings.filepath, 'r') as src, h5py.File(temporary, 'w') as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs['source_file'] = settings.filepath
            events = dst.create_group('events', track_order=True)
            for key, value in src['events'].attrs.items():
                events.attrs[key] = value
            start_time = recording_start_time(src, src['events'])
            if start_time is not None:
                dst.attrs['recording_start_time_s'] = start_time
            for index, row in enumerate(rows):
                name = base_event_names[row.event_name]
                output_name = f'event_{index:05d}'
                src.copy(src['events'][name], events, name=output_name)
                dataset = events[output_name]
                dataset.attrs['source_event_name'] = name
                dataset.attrs['source_segment_name'] = row.event_name
                for key, value in asdict(row).items():
                    if value is not None and key != 'event_name':
                        dataset.attrs[key] = value
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EventTimeRange(QtWidgets.QGroupBox):
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self, times):
        super().__init__('Event time range')
        layout = QtWidgets.QVBoxLayout(self)
        finite = times[np.isfinite(times)]
        self.available = bool(finite.size)
        self.bounds = (float(finite.min()), float(finite.max())) if self.available else (0.0, 1.0)
        lo, hi = self.bounds
        self.slider = pg.PlotWidget()
        self.slider.setFixedHeight(100)
        self.slider.setMouseEnabled(x=False, y=False)
        self.slider.setMenuEnabled(False)
        self.slider.hideButtons()
        self.slider.hideAxis('left')
        self.slider.setLabel('bottom', 'Event time (s)')
        self.slider.setToolTip('Seconds since recording start; drag either handle or move the whole interval.')
        self.slider.setYRange(0, 1, padding=0)
        pad = max((hi - lo) * 0.04, 0.001)
        self.slider.setXRange(lo - pad, hi + pad, padding=0)
        self.slider.plot([lo, hi], [0.5, 0.5], pen=pg.mkPen('#abb8c5', width=2))
        self.region = pg.LinearRegionItem(values=(lo, hi), bounds=self.bounds,
                                         brush=pg.mkBrush(20, 155, 255, 45), pen=pg.mkPen('#149bff', width=2))
        self.slider.addItem(self.region)
        layout.addWidget(self.slider)
        form = QtWidgets.QFormLayout()
        self.start = number_field(lo, minimum=lo, maximum=hi, decimals=6)
        self.end = number_field(hi, minimum=lo, maximum=hi, decimals=6)
        for field in (self.start, self.end):
            field.setMaximumWidth(16777215)
            field.setSingleStep(max((hi - lo) / 100, 0.001))
        form.addRow('From (s)', self.start)
        form.addRow('To (s)', self.end)
        layout.addLayout(form)
        self.reset_button = button('Full time range', lambda: self.region.setRegion(self.bounds), layout)
        self.start.valueChanged.connect(lambda value: self.region.setRegion((min(value, self.region.getRegion()[1]), self.region.getRegion()[1])))
        self.end.valueChanged.connect(lambda value: self.region.setRegion((self.region.getRegion()[0], max(value, self.region.getRegion()[0]))))
        self.region.sigRegionChanged.connect(self._changed)
        if not self.available:
            layout.addWidget(QtWidgets.QLabel('No timestamps: range unavailable.'))
            self.setEnabled(False)

    def _changed(self):
        lo, hi = self.region.getRegion()
        for field, value in ((self.start, lo), (self.end, hi)):
            field.blockSignals(True)
            field.setValue(value)
            field.blockSignals(False)
        self.rangeChanged.emit(lo, hi)

    def contains(self, times):
        if not self.available:
            return np.ones(len(times), dtype=bool)
        lo, hi = self.region.getRegion()
        return np.isfinite(times) & (times >= lo) & (times <= hi)


class SegmentFilterWindow(QtWidgets.QDialog):
    selectionChanged = QtCore.Signal(object)

    def __init__(self, parent, settings, detected_events, segment_results, time_origin=None):
        super().__init__(parent)
        self.settings = settings
        self.detected_events = detected_events
        self.segment_results = segment_results
        timestamps = np.array([row.timestamp if row.timestamp is not None else np.nan for row in segment_results], dtype=float)
        finite = timestamps[np.isfinite(timestamps)]
        if time_origin is None:
            source = getattr(parent, 'source', None)
            time_origin = source.time_origin if source else (float(finite.min()) if finite.size else 0.0)
        self.event_times = timestamps - time_origin
        self.selected_indices = set()
        self.last_selected_index = None
        self.base_event_names = {}
        self.segment_infos = {}
        for event in detected_events:
            for index, segment in enumerate(event.segments):
                name = event.event_name if index == 0 else f'{event.event_name}_{index}'
                self.base_event_names[name] = event.event_name
                self.segment_infos[name] = segment
        self.setWindowTitle(f'Segment filtering · {Path(settings.filepath).name}')
        self.resize(1400, 920)
        layout = QtWidgets.QHBoxLayout(self)
        sidebar = QtWidgets.QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setFixedWidth(245)
        self.controls = QtWidgets.QWidget()
        sidebar.setWidget(self.controls)
        controls = QtWidgets.QVBoxLayout(self.controls)
        self.time_range = EventTimeRange(self.event_times)
        controls.addWidget(self.time_range)
        button('Select all in range', self.select_all, controls)
        button('Clear selection', self.clear_selection, controls)
        button('Save selected CSV', self.save_selected_csv, controls)
        button('Save selected H5', self.save_selected_h5, controls)
        controls.addWidget(QtWidgets.QLabel('Drag mode'))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Add', 'Remove', 'Pan'])
        self.mode.currentTextChanged.connect(self.change_mode)
        controls.addWidget(self.mode)
        button('Reset view', self.reset_view, controls)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch()
        layout.addWidget(sidebar)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.collections = []
        self.views = []
        self.plots = []
        specs = [
            ('dwell_time_ms', 'area_nA_ms', 'Dwell time (ms)', 'EC (nA ms)'),
            ('dwell_time_ms', 'delta_I_rel', 'Dwell time (ms)', 'Relative ΔI'),
            ('area_nA_ms', 'delta_I_rel', 'EC (nA ms)', 'Relative ΔI'),
            ('resistance_MOhm', 'dwell_time_ms', 'Resistance (MΩ)', 'Dwell time (ms)'),
        ]
        for index, (xkey, ykey, xlabel, ylabel) in enumerate(specs):
            view = SelectionViewBox()
            plot = self.graphics.addPlot(row=index // 2, col=index % 2, viewBox=view)
            style_plot(plot, xlabel, ylabel)
            plot.getAxis('left').setWidth(75)
            xs = np.array([getattr(row, xkey) for row in segment_results])
            ys = np.array([getattr(row, ykey) for row in segment_results])
            indices = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
            scatter = pg.ScatterPlotItem(x=xs[indices], y=ys[indices], data=indices,
                                         size=7, brush=scatter_brush('#b7c5d0'), pen=None)
            scatter.sigClicked.connect(self.on_pick)
            plot.addItem(scatter)
            view.rectangleSelected.connect(lambda rect, x=xs, y=ys: self.on_select(rect, x, y))
            self.collections.append((scatter, indices))
            self.views.append(view)
            self.plots.append(plot)
        self.trace_plot = self.graphics.addPlot(row=2, col=0, colspan=2)
        style_plot(self.trace_plot, 'Time within event (ms)', 'Current (nA)')
        self.trace_plot.getAxis('left').setWidth(75)
        self.trace_plot.setTitle('Select a point to view its trace')
        self.time_range.rangeChanged.connect(self._time_range_changed)
        self.update_selection()

    def _time_range_changed(self, *_):
        self.update_selection()

    def change_mode(self, mode):
        for view in self.views:
            view.selection_mode = mode

    def reset_view(self):
        for plot in self.plots:
            plot.enableAutoRange()

    def on_pick(self, scatter, points, event):
        if not points:
            return
        index = int(points[0].data())
        if not self.time_range.contains(self.event_times)[index]:
            return
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            if self.last_selected_index == index:
                self.last_selected_index = None
        else:
            self.selected_indices.add(index)
            self.last_selected_index = index
        self.update_selection()

    def on_select(self, rect, xs, ys):
        indices = np.flatnonzero((xs >= rect.left()) & (xs <= rect.right()) &
                                 (ys >= rect.top()) & (ys <= rect.bottom()) & self.time_range.contains(self.event_times))
        if self.mode.currentText() == 'Remove':
            self.selected_indices.difference_update(indices.tolist())
            if self.last_selected_index not in self.selected_indices:
                self.last_selected_index = None
        else:
            self.selected_indices.update(indices.tolist())
            if len(indices):
                self.last_selected_index = int(indices[0])
        self.update_selection()

    def select_all(self):
        self.selected_indices = set(np.flatnonzero(self.time_range.contains(self.event_times)).tolist())
        self.last_selected_index = min(self.selected_indices) if self.selected_indices else None
        self.update_selection()

    def clear_selection(self):
        self.selected_indices.clear()
        self.last_selected_index = None
        self.update_selection()

    def update_selection(self):
        eligible = self.time_range.contains(self.event_times)
        self.selected_indices.intersection_update(np.flatnonzero(eligible).tolist())
        if self.last_selected_index not in self.selected_indices:
            self.last_selected_index = None
        for scatter, indices in self.collections:
            scatter.setBrush([scatter_brush(('#ff7655' if self.segment_results[i].direction == 1 else '#149bff')
                                            if i in self.selected_indices else '#aeb8c3',
                                            170 if eligible[i] else 40) for i in indices])
        self.trace_plot.clear()
        if self.last_selected_index is not None:
            row = self.segment_results[self.last_selected_index]
            name = self.base_event_names[row.event_name]
            segment = self.segment_infos[row.event_name]
            with h5py.File(self.settings.filepath, 'r') as h5:
                data = h5['events'][name][...]
            draw_trace(self.trace_plot, EventPreview(row.event_name, row.timestamp, data, segment.baseline, [segment]),
                       self.settings.samp_freq, self.settings)
        else:
            self.trace_plot.setTitle('Select a point to view its trace')
        missing = int(np.count_nonzero(~np.isfinite(self.event_times)))
        note = f' {missing} without timestamps cannot be selected within a time range.' if missing and self.time_range.available else ''
        self.status.setText(f'{len(self.selected_indices)} selected · {int(eligible.sum())} of {len(self.segment_results)} segments in range. Click to toggle; drag to add/remove; scroll to zoom.' + note)
        self.selectionChanged.emit({self.segment_results[i].event_name for i in self.selected_indices})

    def _save(self, extension):
        if not self.selected_indices:
            QtWidgets.QMessageBox.information(self, 'Save selection', 'No segments selected.')
            return
        source = Path(self.settings.filepath)
        default = str(source.with_name(source.stem + '_filtered.' + extension))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save selected segments', default,
                                                       f'{extension.upper()} files (*.{extension})')
        if not path:
            return
        if not Path(path).suffix:
            path += '.' + extension
        try:
            if Path(path).resolve() == source.resolve():
                raise ValueError('The input file cannot be overwritten.')
            rows = [self.segment_results[i] for i in sorted(self.selected_indices)]
            if extension == 'csv':
                write_selected_csv(path, rows)
            else:
                write_selected_h5(path, self.settings, rows, self.base_event_names)
            self.status.setText(f'Saved {len(rows)} selected segments to {path}')
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Save failed', str(exc))

    def save_selected_csv(self):
        self._save('csv')

    def save_selected_h5(self):
        self._save('h5')
