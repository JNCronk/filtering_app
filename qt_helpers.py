"""Small shared Qt widgets and background task support."""
from pyqtgraph.Qt import QtCore, QtWidgets

APP_STYLE = """
QGroupBox {border: 1px solid #d4dce4; border-radius: 6px; margin-top: 20px; padding-top: 8px;}
QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px;}
QPushButton {padding: 5px;}
QMainWindow {background: #f5f7fb;}
"""


class Worker(QtCore.QThread):
    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.function()
        except Exception as exc:
            self.error = str(exc)


class TaskRunner(QtCore.QObject):
    busyChanged = QtCore.Signal(bool)
    failed = QtCore.Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.worker = None
        self.callback = None

    @property
    def busy(self):
        return self.worker is not None

    def start(self, function, callback):
        if self.busy:
            return False
        self.callback = callback
        self.worker = Worker(function, self)
        self.worker.finished.connect(self._finished)
        self.busyChanged.emit(True)
        self.worker.start()
        return True

    def _finished(self):
        worker, callback = self.worker, self.callback
        self.worker = self.callback = None
        self.busyChanged.emit(False)
        if worker.error is not None:
            self.failed.emit(worker.error)
        else:
            try:
                callback(worker.result)
            except Exception as exc:
                self.failed.emit(str(exc))
        worker.deleteLater()


def number_field(value, minimum=0.0, maximum=1e9, decimals=4):
    field = QtWidgets.QDoubleSpinBox()
    field.setRange(minimum, maximum)
    field.setDecimals(decimals)
    field.setValue(value)
    field.setKeyboardTracking(False)
    field.setSingleStep(0.1)
    field.setMaximumWidth(115)
    return field


def button(text, callback, layout):
    widget = QtWidgets.QPushButton(text)
    widget.clicked.connect(callback)
    layout.addWidget(widget)
    return widget
