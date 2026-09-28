import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pytest


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / 'recording with spaces.h5'
    with h5py.File(path, 'w') as h5:
        h5.attrs['instrument'] = 'test recorder'
        events = h5.create_group('events')
        events.attrs['units'] = 'nA'
        for i in range(13):
            # Non-zero baseline and a recovered pulse, with unique metrics.
            data = np.full(600, -2.0)
            data[150:220 + i] += 0.5 + i * 0.01
            ds = events.create_dataset(f'event_{i}', data=data)
            ds.attrs['timestamp'] = 100.0 + i
            ds.attrs['calibration'] = 'preserve me'
    with h5py.File(tmp_path / 'recording with spaces_AO.h5', 'w') as h5:
        h5.create_dataset('data', data=np.array([(100000., -0.5), (107000., -1.0)],
                                             dtype=[('timestamp', 'f8'), ('ao_value', 'f8')]))
    return path


@pytest.fixture(scope='session')
def app():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
