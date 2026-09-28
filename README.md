# Filtering App

This app provides a GUI workflow for:

- detecting segments in event-centered H5 files
- previewing traces and summary histograms
- optionally removing segments without vibration signatures
- filtering segments in linked scatter plots
- saving the selected results as CSV and H5

## Launch

The interface uses Qt and PyQtGraph. Install dependencies in your Python environment, then launch from this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

You can also open a recording immediately:

```bash
python main.py "/path/to/recording.h5"
```

`requirements.txt` installs PySide6. An existing environment with PyQtGraph and PyQt5/PyQt6 also works through PyQtGraph's Qt compatibility layer. Tkinter and Matplotlib are no longer used.

## Input and voltage selection

Drag one `.h5` or `.hdf5` event file from Finder anywhere onto the app (including the path field and plots), use **Browse**, or paste its path. Local file URLs and paths containing spaces are supported. Loading runs in the background and immediately displays unprocessed event traces, before segment detection. Use **Previous** and **Next** to browse six raw events at a time.

The input must contain an `events` group of one-dimensional current datasets. Dataset names are sorted in numeric event order. The first dataset is treated as the synthetic start event and excluded from previews and analysis.

The app looks beside the input for a matching `<recording>_AO.h5` or `.hdf5` file (case-insensitive). Its `data` dataset must have `timestamp` and `ao_value` fields, as in `H5Splitter_volt.py`:

- AO timestamps are milliseconds; event `timestamp` attributes are seconds.
- Each event receives the last AO value at or before its timestamp, rounded to 1 mV. Before the first AO entry, the splitter convention is 0 V.
- Voltage grouping happens in memory; loading a recording creates no new files.
- **Save selected voltages…** lets you choose a destination folder and writes one H5 per checked voltage, with the synthetic start event, reindexed event names, source names, timestamps, and original metadata.
- Existing split files are preserved; repeated saves get numbered output filenames. Saving uses the recorded AO groups even when manual voltage is enabled for resistance calculations.
- Check the voltages to include in raw browsing and downstream analysis. All are initially checked. Resistance uses each event's assigned voltage.
- **Use manual voltage** overrides the AO-derived voltage for resistance calculations; voltage checkboxes still determine which events to include.

If no matching AO file is found, a message explains this and the manual voltage field is enabled. An invalid/empty AO log or missing event timestamps also falls back to manual voltage. If an explicit voltage save fails, the message explains the failure and voltage selection remains available in memory. The input recording is never modified.

## Main window

A scrollable panel down the left contains file loading, voltage selection, detection settings, and actions. Sampling frequency and buffer duration are in the file/acquisition section. The filtering and vibration windows also have controls down their left sides. The right side contains raw event previews and four pairs of plots:

- dwell time: histogram and event-time plot
- EC: histogram and event-time plot
- relative ΔI: histogram and event-time plot
- resistance: histogram and event-time plot

Segment metrics remain empty until **Analyze selected voltages** or **Preview segments…** runs detection. Changing detection settings or selected voltages clears previous results so filtering cannot use stale settings. The raw traces remain unprocessed.

### Parameters

- `Sampling (kHz)`: sampling frequency used to convert samples into milliseconds
- `Buffer (ms)`: sliding baseline window size used during segment detection
- `Threshold (nA)`: current deviation required to mark a segment
- `Return thr. (nA)`: current deviation below which a segment is considered returned
- `Min gap (ms)`: minimum recovery gap between successive segments in the same event
- `Voltage (mV)`: used to calculate segment resistance
- `End at std peak`: when enabled, ends each segment at a nearby moving-std peak instead of the return-threshold crossing
- `Std window (ms)`: sliding window size used to compute the moving standard deviation
- `Height factor`: std peak threshold factor relative to pre-event noise
- `Prominence`: required prominence for std peaks
- `Pre-cross buffer frac`: fraction of one buffer that a std peak may occur before the return-threshold crossing

## Segment Detection Logic

For each event trace, the app:

1. Smooths the current trace with a Gaussian filter.
2. Slides a one-buffer baseline window through the trace.
3. Checks whether the following buffer exceeds the threshold relative to that local baseline.
4. Starts a segment at the first threshold-exceeding sample in the smoothed trace.
5. Extends the segment until the signal returns within threshold.
6. Rejects the segment if it never returns within threshold before the trace ends.
7. Continues searching for later segments if they are separated by at least `Min gap (ms)`.

Each segment keeps its own local baseline.

## Duplicate Removal

After the initial pass over all events, the app removes simple adjacent-event duplicates.

A segment is treated as a duplicate if it has exactly the same:

- dwell time
- EC
- relative delta I

as a segment in the previous or following event. Comparisons use original event adjacency and do not cross different assigned voltages.

When duplicates are found, the app keeps the copy that is furthest from the beginning and end of its event file. This favors the least edge-clipped segment.

Duplicate segments are removed before they enter:

- the preview list
- vibration removal
- scatter-based filtering
- CSV saving
- filtered H5 saving

## Analysis and segment previews

1. Drop or choose the event H5 file and inspect its raw traces.
2. Select voltage groups, or enter a manual voltage when no AO log is available.
3. Adjust detection settings, then click **Analyze selected voltages**.
4. Inspect the four histograms and the matching plots against event time.
5. Click **Preview segments…** to open a separate window of up to 12 example events, including events without detected segments. This button also runs analysis first if needed.

The segment window highlights accepted segments, draws baseline and threshold guides, and labels dwell times. It stays open and updates when vibration removal is confirmed or detection is rerun. Preview controls live in the main window’s left **Event previews** panel; the example window contains only plots. Analysis and vibration evaluation run in worker threads so plots remain responsive.

### Linked browsing

- Click points on any metric-versus-time plot, or drag a selection rectangle, to show those events in the raw preview and the open example window. Shift, Command, or Control adds to an existing time-plot selection.
- Previewed events have larger, lighter dots in every time plot. Multiple segments from the same event share that preview highlight.
- **Preview order**, in the left **Event previews** panel, switches between chronological order (by event timestamp) and random order. This setting drives both preview windows. **Previous** and **Next** browse more events; **All events** clears the time-plot restriction within the checked voltages.
- Selecting segments in the filtering window changes only which segments are highlighted in the example window. It does not remove events from the preview population. Events without detected or selected segments remain available and show their full raw traces.
- Opening the filtering window starts with no selected segments. **Select all in range** highlights every detected segment within the time interval in the displayed examples.

**Analyze selected voltages** always reruns detection, resetting any earlier vibration removal. **Preview segments…** reuses the current working results when the settings are unchanged.

### Histograms and event time

Histograms are horizontal: count is on x and the physical parameter is on y. Each histogram shares its y range and parameter color with the adjacent time plot. The count axis is pinned at zero: panning cannot move its origin, while zooming changes the upper limit. Histogram and time-plot titles are hidden; axis labels identify each metric. The four parameters use distinct bright colors, and scatter points are translucent.

Histograms show the central 99% of finite values, trimming up to 0.5% from each tail using observed ranks. Small samples and ties may retain more than 99%. Constant-valued data gets a padded range. Adaptive bin counts are capped at 120. Each histogram’s tooltip reports how many values are outside its initial displayed range.

These display limits do not remove segments or change selection/export data. Time plots retain all finite metric values with valid timestamps and share the time axis. Each histogram/time pair initially uses the central-99% value range; pan or zoom the shared y axis to inspect time-plot outliers. The x coordinate is the event timestamp minus the synthetic start timestamp (or the earliest available timestamp if the start timestamp is missing), in seconds. Events without timestamps remain in the full histograms; time-plot tooltips report their omitted count.

## Vibration Removal

Click **Vibration removal…** to open a second window that removes segments with no accepted standard-deviation peak.

The vibration window includes these parameters:

- `std_window_ms`: sliding window size used to compute the moving standard deviation
- `std_context_ms`: extra context around the segment used during the std analysis
- `std_height_factor`: std peak threshold factor relative to noise
- `std_prominence`: required prominence for std peaks

### Vibration Removal Workflow

1. Adjust the std parameters.
2. Click `Preview Removal`.
3. Review the removed segments shown in the window.
4. Optionally click **Randomise examples** to inspect another subset of removed segments.
5. Click `Confirm` to keep the vibration-filtered working list.

Vibration previews are tentative. Only **Confirm** changes the working list; closing the window leaves it unchanged. Changing vibration parameters requires a new preview before confirming.

## Scatter-Based Segment Filtering

Click **Filter / save segments…** in the main window to open the scatter-filter stage.

This stage shows linked scatter plots of:

- dwell time vs EC
- dwell time vs relative delta I
- EC vs relative delta I
- resistance vs dwell time

It also shows the currently selected segment trace below the scatter plots.

### Event time range

The left panel has a two-handle **Event time range** slider and precise **From**/**To** inputs, in seconds since recording start. Drag either handle or the shaded interval to change the window. Clicks, box selection, and **Select all in range** select only segments whose event timestamps fall inside the inclusive interval. Narrowing the interval removes existing selections outside it; widening it does not automatically add selections. **Full time range** resets the interval. Out-of-range points are faded further.

When some timestamps are missing, those segments cannot be selected within the time range. If no timestamps are available, the range control is disabled and selection remains available without a time restriction.

### Main-plot selection overlays

While segment filtering is open, the main histograms and time points show the full dataset in gray. Selected segments are overlaid in each parameter’s original bright color. Selected histograms use exactly the same bins as the full histogram, and selecting points does not reset the plot ranges. In time plots, only selected segment points regain their parameter color; previewed selected points retain their lighter shade. Closing the filtering window restores the normal full-data colors.

### Selection

You can select segments by:

- clicking on points
- dragging a box around points

Selections are linked across all scatter plots.

The box selector supports two modes:

- `Add`: add enclosed points to the current selection
- `Remove`: remove enclosed points from the current selection

Choose **Pan** to pan instead of selecting. Scroll to zoom, use the PyQtGraph context menu for plot options, or click **Reset view**. **Select all in range** includes all results inside the current event-time interval, including metric outliers.

## Saving

From the scatter-filter window:

- `Save Selected CSV` writes the selected segments to CSV
- `Save Selected H5` writes a filtered H5 containing only the selected segments/events

### CSV Columns

The saved CSV includes:

- `event_name`
- `timestamp`
- `dwell_time_ms`
- `direction`
- `start`
- `end`
- `area_nA_ms`
- `delta_I_nA`
- `delta_I_rel`
- `resistance_MOhm`
- `voltage_mV`

The filtered H5 contains the synthetic start event and one full source trace per selected segment. Datasets are reindexed and preserve original attributes, source event/segment names, segment bounds, metrics, and applied voltage. Selecting the input file as the export destination is rejected.

## Verification

Run the synthetic-file and offscreen Qt integration tests with:

```bash
python -m pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest -q tests
```
