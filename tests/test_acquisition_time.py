import csv

import h5py
import numpy as np

from filtering_app.segment_filter import write_selected_csv
from test_linked_previews import load
from test_workflow import wait_for


def test_start_timestamp_reference_preserves_recording_relative_times(app, recording, monkeypatch, tmp_path):
    window, errors = load(app, recording, monkeypatch)
    assert window.source.time_origin == 100
    assert window.recording_start_label.text() == 'Recording started 100.000000 s after acquisition.'
    original_timestamps = dict(window.source.timestamps)
    original_voltages = dict(window.source.event_voltages_mV)
    window.run_analysis(window.show_segment_preview)
    wait_for(app, lambda: not window.runner.busy and window.preview_window is not None)
    results = window.active_segment_results
    for curve in window.summary.curves:
        np.testing.assert_allclose(curve.xData, np.arange(1, 13))
    window.confirm_threshold()
    filtering = window.filter_window
    filtering.time_range.region.setRegion((3, 6))
    filtering.select_all()
    assert filtering.selected_indices == {2, 3, 4, 5}
    np.testing.assert_allclose(filtering.event_times, np.arange(1, 13))
    for curve in window.summary.curves:
        np.testing.assert_allclose(curve.xData, filtering.event_times)
    assert window.summary.selected_segments == {'event_3', 'event_4', 'event_5', 'event_6'}
    assert window.source.timestamps == original_timestamps
    assert window.source.event_voltages_mV == original_voltages
    output = tmp_path / 'selected.csv'
    write_selected_csv(output, [results[2]])
    with open(output) as handle:
        assert float(next(csv.DictReader(handle))['timestamp']) == 103
    with h5py.File(recording, 'r') as h5:
        assert h5['events/event_0'].attrs['timestamp'] == 100
    assert not errors
    window.close()


def test_reference_labels_missing_start_timestamp_after_reload(app, recording, monkeypatch):
    window, errors = load(app, recording, monkeypatch)
    with h5py.File(recording, 'a') as h5:
        del h5['events/event_0'].attrs['timestamp']
    window.load_file(str(recording))
    wait_for(app, lambda: window.source is not None and not window.runner.busy)
    assert window.source.time_origin == 101
    assert window.recording_start_label.text() == 'No synthetic event found. First event 101.000000 s after acquisition started.'
    assert not errors
    window.close()
