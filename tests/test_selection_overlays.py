from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from filtering_app.plots import METRIC_COLORS, SummaryPlots
from filtering_app.segment_filter import EventTimeRange
from test_linked_previews import load
from test_workflow import wait_for


def test_count_axis_stays_at_zero_for_pan_zoom_and_auto_range(app):
    summary = SummaryPlots()
    summary.resize(1200, 900)
    summary.show()
    app.processEvents()
    for histogram in summary.histograms:
        view = histogram.getViewBox()
        histogram.setXRange(20, 30, padding=0)
        np.testing.assert_allclose(view.viewRange()[0], [0, 10])
        view.translateBy(x=100)
        np.testing.assert_allclose(view.viewRange()[0], [0, 10])
        view.scaleBy(x=0.5, center=pg.Point(8, 0))
        np.testing.assert_allclose(view.viewRange()[0], [0, 5])
        view.scaleBy(x=4, center=pg.Point(2, 0))
        np.testing.assert_allclose(view.viewRange()[0], [0, 20])
        histogram.enableAutoRange(x=True)
        app.processEvents()
        assert view.viewRange()[0][0] == 0
    summary.close()


def test_selected_histogram_uses_fixed_bins_and_exact_segments(app):
    summary = SummaryPlots()
    names = ['event_1', 'event_1_1', 'event_2']
    rows = [SimpleNamespace(event_name=name, timestamp=i + 1., dwell_time_ms=i + 1., area_nA_ms=i + 2.,
                            delta_I_rel=i / 10., resistance_MOhm=i + 3.) for i, name in enumerate(names)]
    summary.display(rows, 0, {'event_1_1': 'event_1'})
    summary.highlight_events(['event_1'])
    originals = [entry[0].opts['width'].copy() for entry in summary.histogram_data]
    summary.set_selection(set())
    for hist, timeline, entry in zip(summary.histograms, summary.timelines, summary.histogram_data):
        base, overlay, values, edges, color = entry
        assert not hist.titleLabel.isVisible() and not timeline.titleLabel.isVisible()
        assert pg.mkBrush(base.opts['brush']).color().name() == '#cbd1d8'
        assert overlay.opts['width'].sum() == 0
    summary.set_selection({'event_1_1'})
    for index, (curve, entry) in enumerate(zip(summary.curves, summary.histogram_data)):
        base, overlay, values, edges, color = entry
        np.testing.assert_array_equal(base.opts['width'], originals[index])
        np.testing.assert_array_equal(overlay.opts['width'], np.histogram(values[[1]], bins=edges)[0])
        assert overlay.opts['width'].sum() == 1
        spots = curve.scatter.points()
        assert spots[0].brush().color().name() == '#aeb8c3'
        assert spots[1].brush().color().name() != '#aeb8c3'
        for spot in spots:
            assert 0 < spot.brush().color().alpha() < 255
    summary.set_selection(None)
    for entry, color in zip(summary.histogram_data, METRIC_COLORS):
        assert pg.mkBrush(entry[0].opts['brush']).color().name() == color
        assert not entry[1].isVisible()
    summary.close()


def test_time_slider_gates_selection_and_updates_main_overlays(app, recording, monkeypatch):
    window, errors = load(app, recording, monkeypatch)
    window.run_analysis(window.show_segment_preview)
    wait_for(app, lambda: not window.runner.busy and window.preview_window is not None)
    assert not window.preview_window.findChildren(QtWidgets.QPushButton)
    for control in (window.preview_order, window.randomize_button, window.all_events_button,
                    window.next_button, window.previous_button):
        assert window.controls.isAncestorOf(control)
    window.confirm_threshold()
    filtering = window.filter_window
    assert window.summary.selected_segments == set()
    filtering.select_all()
    assert len(filtering.selected_indices) == 13
    # The window uses seconds since the same recording start as the main plots.
    np.testing.assert_array_equal(filtering.event_times, np.arange(13))
    filtering.time_range.region.setRegion((3, 6))
    assert filtering.selected_indices == {3, 4, 5, 6}
    for entry in window.summary.histogram_data:
        assert entry[1].opts['width'].sum() == 4
    assert window.selected_segment_names == {'event_3', 'event_4', 'event_5', 'event_6'}
    filtering.clear_selection()
    filtering.on_pick(None, [SimpleNamespace(data=lambda: 0)], None)
    assert not filtering.selected_indices
    filtering.on_pick(None, [SimpleNamespace(data=lambda: 3)], None)
    assert filtering.selected_indices == {3}
    xs = np.array([row.dwell_time_ms for row in filtering.segment_results])
    ys = np.array([row.area_nA_ms for row in filtering.segment_results])
    filtering.on_select(QtCore.QRectF(0, 0, 1000, 1000), xs, ys)
    assert filtering.selected_indices == {3, 4, 5, 6}
    # Both numeric bounds and slider handles update the same selection constraint.
    filtering.time_range.start.setValue(5)
    assert filtering.selected_indices == {5, 6}
    filtering.time_range.end.setValue(5)
    assert filtering.selected_indices == {5}
    filtering.time_range.reset_button.click()
    assert filtering.selected_indices == {5}  # Widening does not silently select more.
    filtering.select_all()
    assert len(filtering.selected_indices) == 13
    for scatter, _ in filtering.collections:
        assert all(0 < spot.brush().color().alpha() < 255 for spot in scatter.points())
    filtering.time_range.region.setRegion((3, 6))
    app.processEvents()
    window.grab().save('/tmp/filtering-overlay-main.png')
    window.summary.grab().save('/tmp/filtering-overlay-summary.png')
    filtering.grab().save('/tmp/filtering-time-range.png')
    filtering.close()
    assert window.filter_window is None
    assert window.summary.selected_segments is None
    assert all(not entry[1].isVisible() for entry in window.summary.histogram_data)
    assert not errors
    window.close()


def test_time_range_missing_and_constant_timestamps(app):
    times = np.array([1., 2., np.nan, np.inf])
    control = EventTimeRange(times)
    np.testing.assert_array_equal(control.contains(times), [True, True, False, False])
    control.region.setRegion((2., 2.))
    np.testing.assert_array_equal(control.contains(times), [False, True, False, False])
    constant = EventTimeRange(np.array([5., 5.]))
    assert constant.contains(np.array([5., 5.])).all()
    missing = EventTimeRange(np.array([np.nan]))
    assert not missing.isEnabled()
    assert not missing.available
    control.close()
    constant.close()
    missing.close()
