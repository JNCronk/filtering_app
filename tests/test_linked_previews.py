from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, QT_LIB

from filtering_app.plots import METRIC_COLORS, SummaryPlots
from filtering_app.ui import DwellTApp
from test_workflow import wait_for


def load(app, recording, monkeypatch):
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda parent, title, message: errors.append(message))
    window = DwellTApp()
    window.show()
    window.load_file(str(recording))
    wait_for(app, lambda: window.source is not None and not window.runner.busy)
    assert not errors
    return window, errors


def test_horizontal_histograms_linked_axes_colors_and_real_click(app):
    summary = SummaryPlots()
    summary.resize(1200, 950)
    rows = [SimpleNamespace(event_name=f'event_{i}', timestamp=float(i), dwell_time_ms=float(i + 1),
                            area_nA_ms=float(i + 2), delta_I_rel=float(i + 3), resistance_MOhm=float(i + 4))
            for i in range(3)]
    summary.display(rows, 0)
    summary.show()
    app.processEvents()
    selected = []
    summary.eventsSelected.connect(lambda names, additive: selected.append((names, additive)))
    for index, (hist, timeplot, color) in enumerate(zip(summary.histograms, summary.timelines, METRIC_COLORS)):
        bars = next(item for item in hist.items if isinstance(item, pg.BarGraphItem))
        assert bars.opts['x0'] == 0
        assert 'y0' in bars.opts and 'y1' in bars.opts
        assert pg.mkBrush(bars.opts['brush']).color().name() == color
        assert summary.curves[index].scatter.points()[0].brush().color().name() == color
        hist.setXRange(-20, 10, padding=0)
        assert hist.viewRange()[0][0] >= 0
        timeplot.setYRange(1, 12, padding=0)
        app.processEvents()
        np.testing.assert_allclose(hist.viewRange()[1], timeplot.viewRange()[1])
    curve = summary.curves[0]
    scene_point = summary.timelines[0].vb.mapViewToScene(QtCore.QPointF(curve.xData[1], curve.yData[1]))
    pixel = summary.mapFromScene(scene_point)
    qt_test = import_module(f'{QT_LIB}.QtTest').QTest
    qt_test.mouseClick(summary.viewport(), QtCore.Qt.MouseButton.LeftButton,
                       QtCore.Qt.KeyboardModifier.NoModifier, pixel)
    assert selected == [(['event_1'], False)]
    summary._box_selected(0, QtCore.QRectF(-0.1, 0, 1.2, 4))
    assert selected[-1] == (['event_0', 'event_1'], False)
    summary.highlight_events(['event_1'])
    for curve, color in zip(summary.curves, METRIC_COLORS):
        spots = curve.scatter.points()
        assert spots[1].brush().color().lightness() > pg.mkColor(color).lightness()
        assert spots[0].brush().color().name() == color
    summary.close()


def test_linked_browsing_keeps_segmentless_events_and_updates_highlights(app, recording, monkeypatch):
    with h5py.File(recording, 'a') as h5:
        h5['events/event_2'][...] = np.full(600, -2.)
        # Verify chronological order follows timestamps, not dataset number.
        h5['events/event_12'].attrs['timestamp'] = 100.5
    window, errors = load(app, recording, monkeypatch)
    assert window.file_box.isAncestorOf(window.fields['samp_freq'])
    assert window.file_box.isAncestorOf(window.fields['buffer_time_ms'])
    assert window.voltage_box.geometry().top() > window.file_box.geometry().bottom()
    assert not (recording.parent / 'split_by_voltage').exists()
    window.run_analysis(window.show_segment_preview)
    wait_for(app, lambda: not window.runner.busy and window.preview_window is not None)
    example = window.preview_window
    assert window.preview_cache[0].event_name == 'event_12'
    previews = window._example_previews()
    assert len(previews) == 12
    assert next(p for p in previews if p.event_name == 'event_2').segments == []
    assert set(window.summary.previewed_names) == {p.event_name for p in previews}
    example.close()
    assert window.summary.previewed_names == {p.event_name for p in previews[:6]}
    window.show_segment_preview()
    assert window.preview_window is example
    # Selecting an event changes both preview windows, without rerunning detection.
    original_results = window.active_segment_results
    window.summary.eventsSelected.emit(['event_9'], False)
    wait_for(app, lambda: not window.runner.busy)
    assert [p.event_name for p in window.preview_cache] == ['event_9']
    assert window.active_segment_results is original_results
    assert window.summary.previewed_names == {'event_9'}
    window.summary.eventsSelected.emit(['event_3'], True)
    wait_for(app, lambda: not window.runner.busy)
    assert [p.event_name for p in window.preview_cache] == ['event_3', 'event_9']
    window.show_all_events()
    wait_for(app, lambda: not window.runner.busy)
    assert len(window.preview_cache) == 12
    window.randomize_button.click()
    wait_for(app, lambda: not window.runner.busy)
    assert window.preview_order.currentText() == 'Random'
    assert set(p.event_name for p in window.preview_cache) == set(window.source.event_names)
    window.preview_order.setCurrentText('Chronological')
    wait_for(app, lambda: not window.runner.busy)
    assert window.preview_cache[0].event_name == 'event_12'
    # Filtering changes only segment highlights, never the event population.
    before = [p.event_name for p in window.preview_cache]
    window.confirm_threshold()
    filtering = window.filter_window
    assert all(not p.segments for p in window._example_previews())
    filtering.selected_indices = {0}
    filtering.update_selection()
    assert sum(len(p.segments) for p in window._example_previews()) == 1
    assert [p.event_name for p in window.preview_cache] == before
    filtering.clear_selection()
    assert all(not p.segments for p in window._example_previews())
    filtering.select_all()
    assert sum(len(p.segments) for p in window._example_previews()) == len(window.active_segment_results)
    filtering.grab().save('/tmp/filtering-linked-filter.png')
    filtering.close()
    # Confirming vibration removal preserves the example window and its event list.
    window.vibration_removal()
    vibration = window.child_windows[-1]
    wait_for(app, lambda: vibration.current_events is not None and not vibration.runner.busy)
    vibration.grab().save('/tmp/filtering-linked-vibration.png')
    vibration.current_events = [replace(event, segments=[]) for event in vibration.current_events]
    vibration.confirm()
    wait_for(app, lambda: not window.runner.busy)
    assert window.preview_window is example and example.isVisible()
    assert [p.event_name for p in window.preview_cache] == before
    assert all(not p.segments for p in window._example_previews())
    assert not errors
    window.grab().save('/tmp/filtering-linked-main.png')
    example.grab().save('/tmp/filtering-linked-examples.png')
    window.close()


def test_save_only_checked_voltages_and_report_errors(app, recording, monkeypatch, tmp_path):
    window, errors = load(app, recording, monkeypatch)
    window.voltage_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    wait_for(app, lambda: not window.runner.busy)
    destination = tmp_path / 'exports'
    destination.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', lambda *args: str(destination))
    window.save_voltages_button.click()
    wait_for(app, lambda: not window.runner.busy)
    files = list(destination.glob('*.h5'))
    assert len(files) == 1
    with h5py.File(files[0], 'r') as h5:
        assert h5.attrs['split_voltage_V'] == -0.5
        assert len(h5['events']) == 7
    assert not errors
    from filtering_app import ui
    def fail(*args):
        raise PermissionError('read-only directory')
    monkeypatch.setattr(ui, 'write_voltage_splits', fail)
    window.save_selected_voltages()
    wait_for(app, lambda: not window.runner.busy)
    assert errors == ['read-only directory']
    assert window.selected_names() == [f'event_{i}' for i in range(1, 7)]
    window.close()


@pytest.mark.parametrize('target', ['path', 'plot'])
def test_drop_on_path_and_plot(app, recording, monkeypatch, target):
    window = DwellTApp()
    window.show()
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: errors.append(args[-1]))
    widget = window.path_field if target == 'path' else window.raw_grid.viewport()
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(recording))])
    enter = QtGui.QDragEnterEvent(QtCore.QPoint(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                                  QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    app.sendEvent(widget, enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(QtCore.QPointF(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    app.sendEvent(widget, drop)
    assert drop.isAccepted()
    wait_for(app, lambda: window.source is not None and not window.runner.busy)
    assert Path(window.source.filepath) == recording
    assert not errors
    window.close()
