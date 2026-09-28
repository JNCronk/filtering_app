from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from . import dwell_t as analysis
from .event_source import inspect_file, prepare_voltage_groups, read_raw_previews, write_voltage_splits
from .plots import SegmentPreviewWindow, SummaryPlots, TraceGrid, configure_plots
from .qt_helpers import APP_STYLE, TaskRunner, button, number_field
from .segment_filter import SegmentFilterWindow
from .vibration_removal import VibrationRemovalWindow


def dropped_file(mime):
    if not mime.hasUrls():
        return None
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    return path if Path(path).suffix.lower() in {'.h5', '.hdf5'} else None


class DropArea(QtWidgets.QLabel):
    fileDropped = QtCore.Signal(str)

    def __init__(self):
        super().__init__('Drop an H5 file anywhere')
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(32)
        self.setMaximumHeight(44)
        self.setWordWrap(True)
        self.setStyleSheet('QLabel {border: 2px dashed #9cbbd2; border-radius: 10px; background: #edf5fb; color: #42627e; padding: 4px;}')

    def dragEnterEvent(self, event):
        if dropped_file(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = dropped_file(event.mimeData())
        if path:
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            self.fileDropped.emit(path)


class DwellTApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Event Filtering')
        self.setStyleSheet(APP_STYLE)
        self.resize(1530, 1000)
        self.setAcceptDrops(True)
        self.source = None
        self.raw_offset = 0
        self.preview_cache = []
        self.preview_order_names = []
        self.focus_names = None
        self.selected_segment_names = None
        self.filter_window = None
        self.active_settings = None
        self.active_detected_events = []
        self.active_segment_results = []
        self.preview_window = None
        self.child_windows = []
        self.rng = np.random.default_rng()
        self.runner = TaskRunner(self)
        self.runner.busyChanged.connect(self._busy_changed)
        self.runner.failed.connect(self._failed)
        configure_plots()
        self._build_ui()
        QtWidgets.QApplication.instance().installEventFilter(self)

    def _build_ui(self):
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(290)
        self.controls = QtWidgets.QWidget()
        left_scroll.setWidget(self.controls)
        left = QtWidgets.QVBoxLayout(self.controls)
        left.setSpacing(12)
        title = QtWidgets.QLabel('Event Filtering')
        title.setStyleSheet('font-size: 22px; font-weight: 600; color: #24415c;')
        left.addWidget(title)
        self.fields = {}
        self.file_box = QtWidgets.QGroupBox('File and acquisition')
        file_layout = QtWidgets.QVBoxLayout(self.file_box)
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.load_file)
        file_layout.addWidget(self.drop_area)
        self.path_field = QtWidgets.QLineEdit()
        self.path_field.setPlaceholderText('Or paste an H5 file path')
        self.path_field.setAcceptDrops(True)
        self.path_timer = QtCore.QTimer(self)
        self.path_timer.setSingleShot(True)
        self.path_timer.setInterval(500)
        self.path_timer.timeout.connect(self._load_pasted_path)
        self.path_field.textChanged.connect(lambda: self.path_timer.start())
        self.path_field.returnPressed.connect(self._load_pasted_path)
        file_layout.addWidget(self.path_field)
        self.browse_button = button('Browse…', self.browse_file, file_layout)
        acquisition = QtWidgets.QFormLayout()
        acquisition.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, default, minimum in [
            ('samp_freq', 'Sampling (kHz)', analysis.DEFAULT_SAMP_FREQ, 0.001),
            ('buffer_time_ms', 'Buffer (ms)', analysis.DEFAULT_BUFFER_TIME_MS, 0.0001),
        ]:
            field = number_field(default, minimum=minimum)
            self.fields[key] = field
            acquisition.addRow(label, field)
            field.valueChanged.connect(self.invalidate_analysis)
        self.fields['samp_freq'].valueChanged.connect(self._refresh_raw)
        file_layout.addLayout(acquisition)
        left.addWidget(self.file_box)
        voltage_box = self.voltage_box = QtWidgets.QGroupBox('Voltage')
        voltage_layout = QtWidgets.QVBoxLayout(voltage_box)
        self.ao_status = QtWidgets.QLabel('Drop a file to look for its companion AO log.')
        self.ao_status.setWordWrap(True)
        voltage_layout.addWidget(self.ao_status)
        self.voltage_list = QtWidgets.QListWidget()
        self.voltage_list.setMaximumHeight(145)
        self.voltage_list.hide()
        self.voltage_list.itemChanged.connect(self._voltage_changed)
        voltage_layout.addWidget(self.voltage_list)
        self.manual_voltage = QtWidgets.QCheckBox('Use manual voltage')
        self.manual_voltage.setChecked(True)
        voltage_layout.addWidget(self.manual_voltage)
        self.voltage_field = number_field(analysis.DEFAULT_VOLTAGE_MV, minimum=-1e6, maximum=1e6, decimals=2)
        self.voltage_field.setMaximumWidth(16777215)
        self.voltage_field.setSuffix(' mV')
        voltage_layout.addWidget(self.voltage_field)
        self.manual_voltage.toggled.connect(self.voltage_field.setEnabled)
        self.manual_voltage.toggled.connect(self.invalidate_analysis)
        self.voltage_field.valueChanged.connect(self.invalidate_analysis)
        self.save_voltages_button = button('Save selected voltages…', self.save_selected_voltages, voltage_layout)
        left.addWidget(voltage_box)
        settings_box = QtWidgets.QGroupBox('Detection settings')
        form = QtWidgets.QFormLayout(settings_box)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, default, minimum in [
            ('threshold_nA', 'Threshold (nA)', analysis.DEFAULT_THRESHOLD_NA, 0.0001),
            ('return_threshold_nA', 'Return threshold (nA)', analysis.DEFAULT_RETURN_THRESHOLD_NA, 0.0),
            ('min_separation_ms', 'Min gap (ms)', analysis.DEFAULT_MIN_SEPARATION_MS, 0.0),
        ]:
            field = number_field(default, minimum=minimum)
            self.fields[key] = field
            form.addRow(label, field)
            field.valueChanged.connect(self.invalidate_analysis)
        left.addWidget(settings_box)
        self.std_box = QtWidgets.QGroupBox('End at std peak')
        self.std_box.setCheckable(True)
        self.std_box.setChecked(analysis.DEFAULT_END_AT_STD_PEAK)
        std_form = QtWidgets.QFormLayout(self.std_box)
        std_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, default in [
            ('end_std_window_ms', 'Std window (ms)', analysis.DEFAULT_END_STD_WINDOW_MS),
            ('end_std_height_factor', 'Height factor', analysis.DEFAULT_END_STD_HEIGHT_FACTOR),
            ('end_std_prominence', 'Prominence', analysis.DEFAULT_END_STD_PROMINENCE),
            ('end_std_pre_crossing_buffer_fraction', 'Pre-cross fraction', analysis.DEFAULT_END_STD_PRE_CROSSING_BUFFER_FRACTION),
        ]:
            field = number_field(default)
            self.fields[key] = field
            std_form.addRow(label, field)
            field.valueChanged.connect(self.invalidate_analysis)
        self.std_box.toggled.connect(self.invalidate_analysis)
        left.addWidget(self.std_box)
        self.preview_controls = QtWidgets.QGroupBox('Event previews')
        preview_controls = QtWidgets.QVBoxLayout(self.preview_controls)
        preview_controls.addWidget(QtWidgets.QLabel('Preview order'))
        self.preview_order = QtWidgets.QComboBox()
        self.preview_order.addItems(['Chronological', 'Random'])
        self.preview_order.currentTextChanged.connect(self._preview_order_changed)
        preview_controls.addWidget(self.preview_order)
        self.all_events_button = button('All events', self.show_all_events, preview_controls)
        self.randomize_button = button('Randomise', self.randomize_previews, preview_controls)
        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = button('Previous', lambda: self.change_raw_page(-6), navigation)
        self.next_button = button('Next', lambda: self.change_raw_page(6), navigation)
        preview_controls.addLayout(navigation)
        left.addWidget(self.preview_controls)
        self.analyze_button = button('Analyze selected voltages', lambda: self.run_analysis(force=True), left)
        self.analyze_button.setStyleSheet('background: #277d94; color: white; font-weight: 600; padding: 9px;')
        self.preview_button = button('Preview segments…', lambda: self.run_analysis(self.show_segment_preview), left)
        self.vibration_button = button('Vibration removal…', self.vibration_removal, left)
        self.filter_button = button('Filter / save segments…', self.confirm_threshold, left)
        left.addStretch()
        splitter.addWidget(left_scroll)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QtWidgets.QWidget()
        right_scroll.setWidget(right)
        right_layout = QtWidgets.QVBoxLayout(right)
        raw_header = QtWidgets.QHBoxLayout()
        self.raw_label = QtWidgets.QLabel('Raw events · drop a file to preview')
        self.raw_label.setStyleSheet('font-size: 16px; font-weight: 600;')
        raw_header.addWidget(self.raw_label, 1)
        right_layout.addLayout(raw_header)
        self.raw_grid = TraceGrid(rows=2)
        self.raw_grid.setMinimumHeight(420)
        right_layout.addWidget(self.raw_grid)
        self.summary_label = QtWidgets.QLabel('Segment metrics · run analysis when ready')
        self.summary_label.setStyleSheet('font-size: 16px; font-weight: 600;')
        right_layout.addWidget(self.summary_label)
        self.recording_start_label = QtWidgets.QLabel('Recording started — s after acquisition.')
        self.recording_start_label.setWordWrap(True)
        self.recording_start_label.setToolTip('The recording_start_time_s file attribute, in seconds after acquisition start.')
        right_layout.addWidget(self.recording_start_label)
        explanation = QtWidgets.QLabel('Click or box-select time points to preview events; Shift/⌘/Ctrl adds events. Lighter dots are previewed. Each pair shares a value axis, initially showing the central 99%.')
        explanation.setWordWrap(True)
        right_layout.addWidget(explanation)
        self.summary = SummaryPlots()
        self.summary.eventsSelected.connect(self.preview_plot_events)
        self.summary.setMinimumHeight(820)
        right_layout.addWidget(self.summary)
        splitter.addWidget(right_scroll)
        splitter.setSizes([360, 1170])
        splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage('Drop an event H5 file from Finder, paste a path, or choose Browse.')
        self._update_actions()

    def eventFilter(self, watched, event):
        if (isinstance(watched, QtWidgets.QWidget) and
                (watched is self or self.isAncestorOf(watched)) and
                event.type() in (QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.DragMove, QtCore.QEvent.Type.Drop)):
            path = dropped_file(event.mimeData())
            if path and not self.runner.busy:
                event.setDropAction(QtCore.Qt.DropAction.CopyAction)
                event.accept()
                if event.type() == QtCore.QEvent.Type.Drop:
                    self.load_file(path)
                return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event):
        if not self.runner.busy and dropped_file(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = dropped_file(event.mimeData())
        if path and not self.runner.busy:
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            self.load_file(path)

    def _load_pasted_path(self):
        path = self.path_field.text().strip().strip("\"'")
        url = QtCore.QUrl(path)
        if url.isLocalFile():
            path = url.toLocalFile()
        if Path(path).expanduser().is_file() and (not self.source or str(Path(path).expanduser().resolve()) != self.source.filepath):
            self.load_file(path)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose event H5 file', '', 'HDF5 files (*.h5 *.hdf5)')
        if path:
            self.load_file(path)

    def load_file(self, path):
        if self.runner.busy:
            return
        self.path_timer.stop()
        self.source = None
        self.recording_start_label.setText('Recording started — s after acquisition.')
        self.raw_offset = 0
        self.preview_cache = []
        self.preview_order_names = []
        self.focus_names = None
        self.invalidate_analysis()
        self.raw_grid.display([], self.fields['samp_freq'].value())
        self.raw_label.setText('Loading raw events…')
        self.voltage_list.blockSignals(True)
        self.voltage_list.clear()
        self.voltage_list.hide()
        self.voltage_list.blockSignals(False)
        self.path_field.blockSignals(True)
        self.path_field.setText(str(path))
        self.path_field.setToolTip(str(path))
        self.path_field.blockSignals(False)
        self.ao_status.setText('Looking for a companion AO file…')
        self.statusBar().showMessage('Loading raw events; no segment detection is running…')
        self.runner.start(lambda: inspect_file(path), self._file_loaded)

    def _file_loaded(self, source):
        self.source = source
        if source.sampling_rate_hz is not None and source.sampling_rate_hz > 0:
            field = self.fields['samp_freq']
            field.blockSignals(True)
            field.setValue(source.sampling_rate_hz / 1000.0)
            field.blockSignals(False)
        if source.recording_start_time_s is None:
            valid_times = [value for value in source.timestamps.values() if value is not None]
            first_time = f'{min(valid_times):.6f}' if valid_times else 'unknown'
            self.recording_start_label.setText(
                f'Recording-start attribute unavailable. First event {first_time} s after acquisition started.'
            )
        else:
            self.recording_start_label.setText(
                f'Recording started {source.recording_start_time_s:.6f} s after acquisition.'
            )
        self.preview_order_names = list(source.event_names)
        self._display_raw(source.previews)
        self.statusBar().showMessage('Raw events loaded. Checking AO voltages…')
        self.runner.start(lambda: prepare_voltage_groups(source), self._voltage_ready)

    def _voltage_ready(self, source):
        self.source = source
        self.ao_status.setText(source.voltage_message)
        self.voltage_list.blockSignals(True)
        for voltage, names in source.voltage_groups.items():
            item = QtWidgets.QListWidgetItem(f'{voltage:+g} mV · {len(names)} events')
            item.setData(QtCore.Qt.ItemDataRole.UserRole, voltage)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.voltage_list.addItem(item)
        self.voltage_list.blockSignals(False)
        self.manual_voltage.setChecked(not bool(source.voltage_groups))
        self.voltage_list.setVisible(bool(source.voltage_groups))
        self.manual_voltage.setEnabled(bool(source.voltage_groups))
        self.voltage_field.setEnabled(self.manual_voltage.isChecked())
        self.statusBar().showMessage(f'{len(source.event_names)} raw events loaded. No segment detection has run.')
        self._rebuild_preview_order()
        self._refresh_raw()
        self._update_actions()

    def selected_voltages(self):
        return {self.voltage_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.voltage_list.count())
                if self.voltage_list.item(i).checkState() == QtCore.Qt.CheckState.Checked}

    def save_selected_voltages(self):
        if self.runner.busy or not self.source or not self.selected_voltages():
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Save voltage files in', str(Path(self.source.filepath).parent))
        if not directory:
            return
        source, voltages = self.source, self.selected_voltages()
        self.statusBar().showMessage('Saving selected voltage recordings…')
        def ready(paths):
            source.split_paths.extend(paths)
            self.statusBar().showMessage(f'Saved {len(paths)} voltage files in {directory}')
        self.runner.start(lambda: write_voltage_splits(source, voltages, directory), ready)

    def selected_names(self):
        if not self.source:
            return []
        if not self.source.voltage_groups:
            return self.source.event_names
        voltages = {self.voltage_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                    for i in range(self.voltage_list.count())
                    if self.voltage_list.item(i).checkState() == QtCore.Qt.CheckState.Checked}
        return [name for name in self.source.event_names if self.source.event_voltages_mV[name] in voltages]

    def _voltage_changed(self, *_):
        self.focus_names = None
        self._rebuild_preview_order()
        self.invalidate_analysis()
        self._refresh_raw()

    def _rebuild_preview_order(self):
        available = self.selected_names()
        available_set = set(available)
        names = available if self.focus_names is None else [name for name in self.focus_names if name in available_set]
        if self.preview_order.currentText() == 'Random':
            names = list(self.rng.permutation(names))
        elif self.source:
            names = sorted(names, key=lambda name: (self.source.timestamps[name] is None,
                                                    self.source.timestamps[name] or 0.0))
        self.preview_order_names = names
        self.raw_offset = 0

    def _preview_order_changed(self, *_):
        self._rebuild_preview_order()
        self._refresh_raw()

    def randomize_previews(self):
        if self.runner.busy:
            return
        self.preview_order.blockSignals(True)
        self.preview_order.setCurrentText('Random')
        self.preview_order.blockSignals(False)
        self._preview_order_changed()

    def show_all_events(self):
        if self.runner.busy:
            return
        self.focus_names = None
        self._preview_order_changed()

    def preview_plot_events(self, names, additive=False):
        if self.runner.busy or not self.source:
            return
        available = set(self.selected_names())
        selected = (self.focus_names or []) if additive else []
        self.focus_names = list(dict.fromkeys([*selected, *(name for name in names if name in available)]))
        self._rebuild_preview_order()
        self._refresh_raw()

    def change_raw_page(self, step):
        self.raw_offset = max(0, self.raw_offset + step)
        self._refresh_raw()

    def _refresh_raw(self, *_):
        if not self.source or self.runner.busy:
            return
        names = self.preview_order_names[self.raw_offset:self.raw_offset + 12]
        path = self.source.filepath
        self.runner.start(lambda: read_raw_previews(path, names), self._display_raw)

    def _display_raw(self, previews):
        self.preview_cache = previews
        self.raw_grid.display(previews[:6], self.fields['samp_freq'].value())
        total = len(self.preview_order_names)
        scope = 'selected events' if self.focus_names is not None else 'events'
        self.raw_label.setText(f'Raw {scope} · {self.raw_offset + 1 if previews else 0}–{self.raw_offset + min(6, len(previews))} of {total}')
        self._render_example()
        self._sync_preview_highlights()
        self._update_actions()

    def _example_previews(self):
        events = {event.event_name: event for event in self.active_detected_events}
        previews = []
        for raw in self.preview_cache:
            event = events.get(raw.event_name)
            segments = []
            if event:
                for index, segment in enumerate(event.segments):
                    name = event.event_name if index == 0 else f'{event.event_name}_{index}'
                    if self.selected_segment_names is None or name in self.selected_segment_names:
                        segments.append(segment)
            previews.append(replace(raw, baseline=event.baseline if event else 0.0, segments=segments))
        return previews

    def _render_example(self):
        if self.preview_window is not None:
            self.preview_window.grid.display(self._example_previews(), self.fields['samp_freq'].value(), self.active_settings)

    def _sync_preview_highlights(self, *_):
        count = 12 if self.preview_window is not None and self.preview_window.isVisible() else 6
        self.summary.highlight_events([preview.event_name for preview in self.preview_cache[:count]])

    def _reset_filter(self):
        if self.filter_window is not None:
            self.filter_window.close()
            self.filter_window = None
        self.selected_segment_names = None
        self.summary.set_selection(None)

    def read_settings(self):
        if not self.source:
            raise ValueError('Drop or choose an event file first.')
        names = self.selected_names()
        if not names:
            raise ValueError('Select at least one voltage containing recorded events.')
        return analysis.Settings(
            filepath=self.source.filepath, voltage_mV=self.voltage_field.value(),
            event_names=tuple(names),
            event_voltages_mV={} if self.manual_voltage.isChecked() else dict(self.source.event_voltages_mV),
            end_at_std_peak=self.std_box.isChecked(),
            **{key: field.value() for key, field in self.fields.items()},
        )

    def invalidate_analysis(self, *_):
        self._reset_filter()
        self.active_settings = None
        self.active_detected_events = []
        self.active_segment_results = []
        if hasattr(self, 'summary'):
            self.summary.display([], self.source.time_origin if self.source else 0)
            self.summary_label.setText('Segment metrics · run analysis with the current settings')
            self._render_example()
            self._sync_preview_highlights()
            self._update_actions()

    def run_analysis(self, after=None, force=False):
        if self.runner.busy:
            return
        try:
            settings = self.read_settings()
        except ValueError as exc:
            self._failed(str(exc))
            return
        if not force and settings == self.active_settings:
            if after:
                after()
            return
        self._reset_filter()
        self.statusBar().showMessage(f'Detecting segments in {len(settings.event_names)} events…')

        def analyze():
            events = analysis.iter_detected_events(settings)
            return events, analysis.collect_segment_results(settings, events)

        def ready(result):
            self.active_settings = settings
            self.active_detected_events, self.active_segment_results = result
            self._display_results()
            if after:
                after()

        self.runner.start(analyze, ready)

    def _display_results(self):
        mapping = {event.event_name if i == 0 else f'{event.event_name}_{i}': event.event_name
                   for event in self.active_detected_events for i, _ in enumerate(event.segments)}
        self.summary.display(self.active_segment_results, self.source.time_origin, mapping)
        self._render_example()
        self._sync_preview_highlights()
        self.summary_label.setText(f'Segment metrics · {len(self.active_segment_results)} segments')
        self.statusBar().showMessage(f'Analysis complete: {len(self.active_segment_results)} segments in {len(self.active_detected_events)} events.')
        self._update_actions()

    def show_segment_preview(self):
        if not self.active_settings or self.runner.busy:
            return
        if self.preview_window is None:
            self.preview_window = SegmentPreviewWindow(self, self.active_settings,
                                                       self._example_previews())
            self.preview_window.finished.connect(self._sync_preview_highlights)
        self._render_example()
        self.preview_window.show()
        self.preview_window.raise_()
        self._sync_preview_highlights()

    def vibration_removal(self):
        if not self.active_settings:
            return
        settings = copy.deepcopy(self.active_settings)

        def apply(events, keep_window=False):
            def ready(results):
                self.active_detected_events = events
                self.active_segment_results = results
                self._reset_filter()
                self._display_results()
            self.statusBar().showMessage('Updating metrics after vibration removal…')
            self.runner.start(lambda: analysis.collect_segment_results(settings, events), ready)

        window = VibrationRemovalWindow(self, settings, self.active_detected_events, apply)
        window.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.child_windows.append(window)
        window.show()

    def confirm_threshold(self):
        if self.active_settings:
            self._reset_filter()
            window = SegmentFilterWindow(self, copy.deepcopy(self.active_settings),
                                         copy.deepcopy(self.active_detected_events),
                                         copy.deepcopy(self.active_segment_results))
            self.filter_window = window
            self.selected_segment_names = set()
            window.selectionChanged.connect(lambda names: self._filter_selection_changed(window, names))
            window.finished.connect(lambda _: self._filter_closed(window))
            self.summary.set_selection(set())
            self._render_example()
            self.child_windows.append(window)
            window.show()

    def _filter_selection_changed(self, window, names):
        if window is self.filter_window:
            self.selected_segment_names = set(names)
            self.summary.set_selection(self.selected_segment_names)
            self._render_example()

    def _filter_closed(self, window):
        if window is self.filter_window:
            self.filter_window = None
            self.selected_segment_names = None
            self.summary.set_selection(None)
            self._render_example()

    def _update_actions(self):
        busy = self.runner.busy
        available = bool(self.selected_names())
        self.analyze_button.setEnabled(available and not busy)
        self.preview_button.setEnabled(available and not busy)
        self.vibration_button.setEnabled(self.active_settings is not None and bool(self.active_segment_results) and not busy)
        self.filter_button.setEnabled(self.active_settings is not None and bool(self.active_segment_results) and not busy)
        self.save_voltages_button.setEnabled(bool(self.source and self.source.voltage_groups and self.selected_voltages()) and not busy)
        self.previous_button.setEnabled(self.raw_offset > 0 and not busy)
        self.next_button.setEnabled(self.raw_offset + 6 < len(self.preview_order_names) and not busy)

    def _busy_changed(self, busy):
        self.controls.setEnabled(not busy)
        self.preview_order.setEnabled(not busy)
        self.all_events_button.setEnabled(not busy)
        self.randomize_button.setEnabled(not busy)
        self._update_actions()

    def _failed(self, message):
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.warning(self, 'Event Filtering', message)
        self._update_actions()

    def closeEvent(self, event):
        if self.runner.busy or any(getattr(w, 'runner', None) and w.runner.busy for w in self.child_windows):
            self.statusBar().showMessage('Please wait for the current operation to finish before closing.')
            event.ignore()
        else:
            event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Event Filtering')
    app.setStyleSheet(APP_STYLE)
    window = DwellTApp()
    window.show()
    if len(sys.argv) > 1:
        QtCore.QTimer.singleShot(0, lambda: window.load_file(sys.argv[1]))
    sys.exit(app.exec())
