# Filtering App

This app provides a GUI workflow for:

- detecting segments in event-centered H5 files
- previewing traces and summary histograms
- optionally removing segments without vibration signatures
- filtering segments in linked scatter plots
- saving the selected results as CSV and H5

## Launch

First create the Conda environment using the [repository setup guide](../README.md#setup-with-conda).
Then, from the project root:

```bash
conda activate protein-data-analysis
python filtering_app/main.py
```

## Input File

The app expects an H5 file with an `events` group.

- The first event is treated as the synthetic start event and is excluded from preview and filtering.
- Each later event is scanned for one or more segments.

## Main Window

The main window contains:

- H5 file selector
- thresholding parameters
- 12 preview traces
- histograms of current segment metrics

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

as a segment in the previous or following event.

When duplicates are found, the app keeps the copy that is furthest from the beginning and end of its event file. This favors the least edge-clipped segment.

Duplicate segments are removed before they enter:

- the preview list
- vibration removal
- scatter-based filtering
- CSV saving
- filtered H5 saving

## Preview Window Workflow

1. Choose the H5 file.
2. Enter the thresholding parameters.
3. Click `Load / Reload Preview`.

This will:

- detect segments in all events
- remove adjacent duplicates
- build the working list
- display 12 example traces and 4 histograms

### Preview Controls

- `Load / Reload Preview`: reruns detection using the current settings
- `Show Another Random 12`: shows a different random subset from the current working list without recomputing everything
- `Vibration Removal`: opens the vibration-removal window
- `Confirm Threshold`: opens the scatter-based segment filtering window

### Trace Display

Each preview trace shows:

- the full saved event record
- highlighted detected segments
- segment-specific baseline and threshold guides
- dwell times overlaid on the plot
- `no event` if no valid segment remains in that event

### Histograms

The histograms summarize the current working list using adaptive binning:

- `Dwell t (ms)`
- `EC (nA ms)`
- `rel delta I`
- `R (MOhm)`

## Vibration Removal

Click `Vibration Removal` to open a second window that removes segments with no accepted standard-deviation peak.

The vibration window includes these parameters:

- `std_window_ms`: sliding window size used to compute the moving standard deviation
- `std_context_ms`: extra context around the segment used during the std analysis
- `std_height_factor`: std peak threshold factor relative to noise
- `std_prominence`: required prominence for std peaks

### Vibration Removal Workflow

1. Adjust the std parameters.
2. Click `Preview Removal`.
3. Review the removed segments shown in the window.
4. Optionally click `Show Another Random 12` to inspect another subset of removed segments.
5. Click `Confirm` to keep the vibration-filtered working list.

If the vibration-removal window is closed without confirming, the previous working list is restored.

## Scatter-Based Segment Filtering

Click `Confirm Threshold` in the main window to open the scatter-filter stage.

This stage shows linked scatter plots of:

- dwell time vs EC
- dwell time vs relative delta I
- EC vs relative delta I
- resistance vs dwell time

It also shows the currently selected segment trace below the scatter plots.

### Selection

You can select segments by:

- clicking on points
- dragging a box around points

Selections are linked across all scatter plots.

The box selector supports two modes:

- `Add`: add enclosed points to the current selection
- `Remove`: remove enclosed points from the current selection

The Matplotlib toolbar supports zoom, pan, and reset.

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

## Typical Workflow

1. Launch the app.
2. Choose the H5 file.
3. Set thresholding parameters.
4. Click `Load / Reload Preview`.
5. Inspect the preview traces and histograms.
6. Adjust thresholding settings if needed and reload.
7. Optionally run `Vibration Removal` and confirm the result.
8. Click `Confirm Threshold`.
9. Select the desired segment population in the scatter plots.
10. Save the selected CSV and/or filtered H5.
