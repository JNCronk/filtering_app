import csv
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from filtering_app import dwell_t as analysis
from filtering_app.event_source import inspect_file, prepare_voltage_groups, write_voltage_splits
from filtering_app.plots import robust_histogram
from filtering_app.segment_filter import SegmentFilterWindow, write_selected_csv, write_selected_h5
from filtering_app.ui import DwellTApp, dropped_file
from filtering_app.vibration_removal import VibrationRemovalWindow


def wait_for(app, condition, timeout=15):
    deadline = time.monotonic() + timeout
    while not condition():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail('Background operation timed out')
        time.sleep(0.005)
    app.processEvents()


def settings(path, **kwargs):
    return analysis.Settings(str(path), 50, 1, 0.1, 0.1, 5, -1000, **kwargs)


def test_raw_voltage_splits_and_metadata(recording):
    with patch.object(analysis, 'analyze_event', side_effect=AssertionError('Must not detect on load')):
        source = inspect_file(str(recording))
        assert source.event_names == [f'event_{i}' for i in range(13)]
        assert all(not preview.segments for preview in source.previews)
        prepare_voltage_groups(source)
    assert source.voltage_groups[-500] == [f'event_{i}' for i in range(7)]
    assert source.voltage_groups[-1000] == [f'event_{i}' for i in range(7, 13)]
    assert source.split_paths == []
    assert not (recording.parent / 'split_by_voltage').exists()
    source.split_paths = write_voltage_splits(source)
    assert len(source.split_paths) == 2
    for path in source.split_paths:
        with h5py.File(path, 'r') as h5:
            assert len(h5['events']) in (6, 7)
            assert h5.attrs['instrument'] == 'test recorder'
            assert h5.attrs['recording_start_time_s'] == 100
            assert h5['events'].attrs['units'] == 'nA'
            assert h5['events/event_00000'].attrs['calibration'] == 'preserve me'
    previous = {p: Path(p).read_bytes() for p in source.split_paths}
    again = prepare_voltage_groups(inspect_file(str(recording)))
    again.split_paths = write_voltage_splits(again, {-500})
    assert len(again.split_paths) == 1
    assert set(again.split_paths).isdisjoint(previous)
    assert all(Path(p).read_bytes() == data for p, data in previous.items())


def test_legacy_empty_start_event_is_excluded(tmp_path):
    path = tmp_path / 'legacy.h5'
    with h5py.File(path, 'w') as h5:
        events = h5.create_group('events')
        start = events.create_dataset('event_00000', data=np.array([], dtype=float))
        start.attrs['timestamp'] = 25.0
        event = events.create_dataset('event_00001', data=np.ones(32))
        event.attrs['timestamp'] = 26.0

    source = inspect_file(str(path))

    assert source.event_names == ['event_00001']
    assert source.recording_start_time_s == 25.0
    assert source.time_origin == 25.0


def test_missing_and_invalid_ao(recording):
    ao = recording.with_name(recording.stem.removesuffix('_Events') + '_AO.h5')
    ao.unlink()
    source = prepare_voltage_groups(inspect_file(str(recording)))
    assert 'could be found' in source.voltage_message
    assert not source.voltage_groups
    with h5py.File(ao, 'w') as h5:
        h5.create_dataset('wrong', data=[1])
    source = prepare_voltage_groups(inspect_file(str(recording)))
    assert 'could not be read' in source.voltage_message
    assert not source.voltage_groups


def test_missing_timestamps_and_split_write_failure(recording, monkeypatch):
    from filtering_app import event_source
    with patch.object(event_source, 'write_voltage_splits', side_effect=AssertionError('Must not write on load')):
        source = prepare_voltage_groups(inspect_file(str(recording)))
    assert len(source.voltage_groups) == 2
    assert 'saved only on request' in source.voltage_message
    with h5py.File(recording, 'a') as h5:
        del h5['events/event_1'].attrs['timestamp']
    source = prepare_voltage_groups(inspect_file(str(recording)))
    assert not source.voltage_groups
    assert 'no timestamp' in source.voltage_message
    assert 'manual voltage' in source.voltage_message


def test_selected_voltage_metrics_and_empty_results(recording):
    source = prepare_voltage_groups(inspect_file(str(recording)))
    options = settings(recording, event_names=tuple(source.voltage_groups[-500]),
                       event_voltages_mV=source.event_voltages_mV)
    events = analysis.iter_detected_events(options)
    assert [event.event_name for event in events] == source.voltage_groups[-500]
    results = analysis.collect_segment_results(options, events)
    assert len(results) == 7
    assert all(row.voltage_mV == -500 and row.resistance_MOhm == 250 for row in results)
    manual = analysis.collect_segment_results(replace(options, event_voltages_mV={}, voltage_mV=-200), events)
    assert all(row.resistance_MOhm == 100 for row in manual)
    with patch.object(analysis, 'iter_detected_events', side_effect=AssertionError('Do not rerun')):
        assert analysis.collect_segment_results(options, []) == []


def test_dedup_does_not_cross_voltage_boundaries_or_excluded_events(recording):
    with h5py.File(recording, 'a') as h5:
        data = h5['events/event_1'][...]
        for i in (2, 3):
            h5[f'events/event_{i}'][...] = data
    selected = ('event_1', 'event_2')
    options = settings(recording, event_names=selected, event_voltages_mV={'event_1': -500, 'event_2': -1000})
    assert sum(len(e.segments) for e in analysis.iter_detected_events(options)) == 2
    options = replace(options, event_voltages_mV={})
    assert sum(len(e.segments) for e in analysis.iter_detected_events(options)) == 1
    options = replace(options, event_names=('event_1', 'event_3'))
    assert sum(len(e.segments) for e in analysis.iter_detected_events(options)) == 2


def test_histogram_limits_keep_data_and_ignore_outliers():
    values = np.r_[np.arange(998.), -1e10, 1e10, np.nan, np.inf]
    original = values.copy()
    counts, edges, excluded = robust_histogram(values)
    assert counts.sum() == 990
    assert excluded == 10
    assert edges.min() >= 0 and edges.max() < 1000
    np.testing.assert_array_equal(values, original)
    counts, edges, excluded = robust_histogram(np.r_[np.ones(999), 1e10])
    assert counts.sum() == 999 and excluded == 1
    assert edges[-1] < 2
    assert robust_histogram([1, 2, 3])[0].sum() == 3
    assert robust_histogram([np.nan, np.inf])[0].size == 0


def test_exports_preserve_voltage_and_source(recording, tmp_path):
    options = settings(recording)
    events = analysis.iter_detected_events(options)
    rows = analysis.collect_segment_results(options, events)[:2]
    names = {row.event_name: row.event_name for row in rows}
    write_selected_csv(tmp_path / 'selected.csv', rows)
    with open(tmp_path / 'selected.csv') as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2 and float(csv_rows[0]['voltage_mV']) == -1000
    output = tmp_path / 'selected.h5'
    write_selected_h5(output, options, rows, names)
    with h5py.File(output, 'r') as h5:
        assert len(h5['events']) == 2
        assert h5.attrs['recording_start_time_s'] == 100
        assert h5['events/event_00000'].attrs['source_event_name'] == rows[0].event_name
        assert h5['events/event_00000'].attrs['voltage_mV'] == -1000
        assert h5['events/event_00000'].attrs['start'] == rows[0].start
    with pytest.raises(ValueError, match='overwritten'):
        write_selected_h5(recording, options, rows, names)


def test_qt_drop_preview_analysis_and_selection(app, recording, monkeypatch):
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda parent, title, message: errors.append(message))
    window = DwellTApp()
    window.show()
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(recording))])
    assert dropped_file(mime) == str(recording)
    with patch.object(analysis, 'analyze_event', side_effect=AssertionError('Detection on drop')):
        enter = QtGui.QDragEnterEvent(QtCore.QPoint(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                                      QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
        app.sendEvent(window.drop_area, enter)
        assert enter.isAccepted()
        drop = QtGui.QDropEvent(QtCore.QPointF(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                                QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
        app.sendEvent(window.drop_area, drop)
        assert drop.isAccepted()
        wait_for(app, lambda: window.source is not None and not window.runner.busy)
    assert not errors
    assert window.active_settings is None
    assert len(window.raw_grid.plots[0].listDataItems()[0].yData) == 600
    assert window.voltage_list.count() == 2
    assert not window.manual_voltage.isChecked()
    window.voltage_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    wait_for(app, lambda: not window.runner.busy)
    assert window.selected_names() == [f'event_{i}' for i in range(7)]
    window.run_analysis(window.show_segment_preview)
    wait_for(app, lambda: window.preview_window is not None and not window.runner.busy)
    assert len(window.active_segment_results) == 7
    assert not errors
    assert window.preview_window.isVisible()
    assert len(window.summary.timelines[0].listDataItems()[0].xData) == 7
    np.testing.assert_array_equal(window.summary.timelines[0].listDataItems()[0].xData, np.arange(7))
    window.grab().save('/tmp/filtering-ui-analysis.png')
    window.preview_window.grab().save('/tmp/filtering-segments.png')
    filtered = SegmentFilterWindow(window, window.active_settings, window.active_detected_events, window.active_segment_results)
    filtered.show()
    filtered.select_all()
    assert len(filtered.selected_indices) == 7
    filtered.clear_selection()
    xs = np.array([r.dwell_time_ms for r in window.active_segment_results])
    ys = np.array([r.area_nA_ms for r in window.active_segment_results])
    filtered.on_select(QtCore.QRectF(0, 0, 100, 100), xs, ys)
    assert len(filtered.selected_indices) == 7
    filtered.mode.setCurrentText('Remove')
    filtered.on_select(QtCore.QRectF(0, 0, 100, 100), xs, ys)
    assert not filtered.selected_indices
    filtered.close()
    applied = []
    vibration = VibrationRemovalWindow(window, window.active_settings, window.active_detected_events,
                                       lambda events, **kwargs: applied.append(events))
    vibration.show()
    wait_for(app, lambda: vibration.current_events is not None and not vibration.runner.busy)
    assert not applied
    vibration.close()
    assert not applied
    vibration = VibrationRemovalWindow(window, window.active_settings, window.active_detected_events,
                                       lambda events, **kwargs: applied.append(events))
    vibration.show()
    wait_for(app, lambda: vibration.current_events is not None and not vibration.runner.busy)
    vibration.confirm()
    assert len(applied) == 1
    window.active_detected_events = []
    window.active_segment_results = []
    window.run_analysis(force=True)
    wait_for(app, lambda: not window.runner.busy)
    assert len(window.active_segment_results) == 7
    window.fields['threshold_nA'].setValue(0.2)
    assert window.active_settings is None and not window.filter_button.isEnabled()
    window.voltage_list.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    wait_for(app, lambda: not window.runner.busy)
    assert not window.analyze_button.isEnabled()
    assert not window.selected_names()
    with pytest.raises(ValueError, match='Select at least one voltage'):
        window.read_settings()
    assert not errors
    window.close()


def test_qt_missing_ao_manual(app, recording, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: pytest.fail(str(args[-1])))
    recording.with_name(recording.stem.removesuffix('_Events') + '_AO.h5').unlink()
    window = DwellTApp()
    window.path_field.setText(str(recording))
    window._load_pasted_path()
    wait_for(app, lambda: window.source is not None and not window.runner.busy)
    assert window.manual_voltage.isChecked() and window.voltage_field.isEnabled()
    assert 'could be found' in window.ao_status.text()
    window.voltage_field.setValue(-200)
    window.run_analysis()
    wait_for(app, lambda: not window.runner.busy)
    assert all(row.resistance_MOhm == 100 for row in window.active_segment_results)
    window.close()
