import sys
import time
import cv2

from PySide import QtCore
from PySide import QtGui

from Processors.WormTracker import Tracker
from interface import Microscope


class FrameDisplay(QtGui.QLabel):

    def __init__(self):
        super(FrameDisplay, self).__init__()

    @QtCore.Slot()
    def updateFrame(self, pixmap):
        self.setPixmap(pixmap)


class FrameThread(QtCore.QThread):
    '''Grabs and processes camera frames and signals when ready'''

    def __init__(self, microscope, log_suffix):
        super(FrameThread, self).__init__()
        self.camera = cv2.VideoCapture(0)

        self.tracker = Tracker(microscope, log_suffix)
        self.mutex = QtCore.QMutex()
        self.abort = False
        self.condition = QtCore.QWaitCondition()

    def update_parameters(self, params):
        self.mutex.lock()
        self.params = params
        self.mutex.unlock()

    def run(self):
        while not self.abort:
            self.mutex.lock()
            self.tracker.params = self.params
            self.mutex.unlock()

            ret, self.img = self.camera.read()
            #self.img = cv2.imread('tracking_fig01.jpg')
            #time.sleep(0.25)

            self.img = self.tracker.process_frame(self.img)

            self.mutex.lock()
            self.img_copy = self.img.copy()
            self.mutex.unlock()

            height, width, channels = self.img_copy.shape
            bpl = width*channels
            RGBimg = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2RGB)
            self.frame = QtGui.QImage(RGBimg.data, width, height, bpl,
                                      QtGui.QImage.Format_RGB888)
            self.emit(QtCore.SIGNAL("renderedImage(const QImage)"), self.frame)

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()

        self.wait()


class MicroscopeClient(QtGui.QWidget):
    '''Application for controlling the OpenLabTools Microscope'''

    newPixmap = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, serial_port, log_suffix):
        super(MicroscopeClient, self).__init__()

        self.microscope = Microscope(serial_port)
        self.microscope.set_ring_colour('FF0000')

        self.init_ui()

        self.params = {}
        self.thread = FrameThread(self.microscope, log_suffix)

        self.set_defaults()

        #self.newPixmap.connect(self.frame_display.updateFrame)
        self.connect(self, QtCore.SIGNAL("newPixmap(const QPixmap)"),
                     self.frame_display.updateFrame)

        #Start the frame processing thread
        self.connect(self.thread, QtCore.SIGNAL("renderedImage(const QImage)"),
                     self.update_frame)
        self.thread.start()

    def init_ui(self):
        '''Define UI Elements'''

        self.setWindowTitle('OpenLabTools Microscope')

        layout = QtGui.QHBoxLayout()

        self.frame_display = FrameDisplay()

        left_column = QtGui.QVBoxLayout()
        left_column.addWidget(self.frame_display)
        left_column.addWidget(self.init_stage_controls())
        left_column.addStretch(1)

        right_column = QtGui.QVBoxLayout()
        right_column.addWidget(self.init_frame_controls())

        layout.addLayout(left_column)
        layout.addLayout(right_column)
        self.setLayout(layout)

    def init_stage_controls(self):
        '''Define UI Elements for controlling the stage'''

        group_box = QtGui.QGroupBox('Stage Controls')

        hbox = QtGui.QHBoxLayout()

        grid = QtGui.QGridLayout()

        up_button = QtGui.QToolButton()
        up_button.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        down_button = QtGui.QToolButton()
        down_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        left_button = QtGui.QToolButton()
        left_button.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        right_button = QtGui.QToolButton()
        right_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)

        grid.addWidget(up_button, 0, 1)
        grid.addWidget(down_button, 2, 1)
        grid.addWidget(left_button, 1, 0)
        grid.addWidget(right_button, 1, 2)

        xy_group = QtGui.QGroupBox('XY Controls')
        xy_group.setLayout(grid)

        hbox.addWidget(xy_group)

        z_group = QtGui.QGroupBox('Z Control')

        z_up_button = QtGui.QToolButton()
        z_down_button = QtGui.QToolButton()

        z_up_button.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        z_down_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(z_up_button)
        vbox.addStretch(1)
        vbox.addWidget(z_down_button)

        z_group.setLayout(vbox)

        hbox.addWidget(z_group)

        hbox.addStretch(1)

        group_box.setLayout(hbox)

        return group_box

    def init_frame_controls(self):
        '''Define UI Elements for changing image processing parameters'''

        group_box = QtGui.QGroupBox('Frame Controls')

        vbox = QtGui.QVBoxLayout()

        #Thresholding Parameters

        thresh_group = QtGui.QGroupBox('Thresholding')

        thresh_form = QtGui.QFormLayout()

        self.simple_radio = QtGui.QRadioButton('Simple')
        self.simple_radio.toggled.connect(self.update_parameters)
        self.adapt_radio = QtGui.QRadioButton('Adaptive')
        self.adapt_radio.toggled.connect(self.update_parameters)

        thresh_mode = QtGui.QVBoxLayout()
        thresh_mode.addWidget(self.simple_radio)
        thresh_mode.addWidget(self.adapt_radio)

        thresh_mode_group = QtGui.QGroupBox()
        thresh_mode_group.setLayout(thresh_mode)

        thresh_form.addRow('Mode', thresh_mode_group)

        self.simple_thresh_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.simple_thresh_value.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Simple Threshold', self.simple_thresh_value)

        self.mean_radio = QtGui.QRadioButton('Mean')
        self.mean_radio.toggled.connect(self.update_parameters)
        self.gauss_radio = QtGui.QRadioButton('Gaussian')
        self.gauss_radio.toggled.connect(self.update_parameters)

        adapt_kernel = QtGui.QVBoxLayout()
        adapt_kernel.addWidget(self.mean_radio)
        adapt_kernel.addWidget(self.gauss_radio)
        adapt_kernel_group = QtGui.QGroupBox()
        adapt_kernel_group.setLayout(adapt_kernel)
        thresh_form.addRow('Adaptive Kernel', adapt_kernel_group)

        self.adapt_kernel_size = QtGui.QSpinBox()
        self.adapt_kernel_size.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Kernel Size', self.adapt_kernel_size)

        self.adapt_kernel_mean = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.adapt_kernel_mean.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Adaptive Mean', self.adapt_kernel_mean)

        self.morph_open = QtGui.QSpinBox()
        self.morph_open.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Opening Rounds', self.morph_open)

        self.morph_close = QtGui.QSpinBox()
        self.morph_close.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Closing Rounds', self.morph_close)

        self.morph_kernel = QtGui.QSpinBox()
        self.morph_kernel.valueChanged.connect(self.update_parameters)
        thresh_form.addRow('Morphological Kernel Size', self.morph_kernel)

        self.show_thresh = QtGui.QCheckBox()
        self.show_thresh.toggled.connect(self.update_parameters)
        thresh_form.addRow('Show Thresholded Image', self.show_thresh)

        thresh_group.setLayout(thresh_form)
        vbox.addWidget(thresh_group)

        #Tracking Parameters

        tracking_group = QtGui.QGroupBox('Tracking')

        tracking_form = QtGui.QFormLayout()

        self.enable_tracking = QtGui.QCheckBox()
        self.enable_tracking.toggled.connect(self.update_parameters)
        tracking_form.addRow('Enable Tracking', self.enable_tracking)

        self.step_size = QtGui.QSpinBox()
        self.step_size.valueChanged.connect(self.update_parameters)
        tracking_form.addRow('Step Size', self.step_size)

        self.step_interval = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.step_interval.valueChanged.connect(self.update_parameters)
        tracking_form.addRow('Step Interval', self.step_interval)

        self.margin = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.margin.valueChanged.connect(self.update_parameters)
        tracking_form.addRow('Centroid Margin', self.margin)

        tracking_group.setLayout(tracking_form)
        vbox.addWidget(tracking_group)

        skeleton_group = QtGui.QGroupBox('Skeletonisation')
        skeleton_form = QtGui.QFormLayout()

        self.enable_skeleton = QtGui.QCheckBox()
        self.enable_skeleton.toggled.connect(self.update_parameters)
        skeleton_form.addRow('Enable Skeletonisation', self.enable_skeleton)

        self.spline_smoothing = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.spline_smoothing.valueChanged.connect(self.update_parameters)
        skeleton_form.addRow('Spline Smoothing', self.spline_smoothing)

        self.centre_line = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.centre_line.valueChanged.connect(self.update_parameters)
        skeleton_form.addRow('Centre Line Points', self.centre_line)

        self.refine = QtGui.QCheckBox()
        self.refine.toggled.connect(self.update_parameters)
        skeleton_form.addRow('Refine Centreline', self.refine)

        self.record = QtGui.QCheckBox()
        self.record.toggled.connect(self.update_parameters)
        skeleton_form.addRow('Record', self.record)

        skeleton_group.setLayout(skeleton_form)
        vbox.addWidget(skeleton_group)

        vbox.addStretch(1)
        group_box.setLayout(vbox)

        return group_box

    def set_defaults(self):
        self.simple_radio.setChecked(True)
        self.simple_thresh_value.setRange(0, 255)
        self.simple_thresh_value.setValue(100)
        self.adapt_kernel_size.setValue(1)
        self.adapt_kernel_mean.setRange(0, 10)
        self.adapt_kernel_mean.setValue(5)
        self.mean_radio.setChecked(True)
        self.show_thresh.setChecked(False)

        self.enable_tracking.setChecked(False)
        self.step_size.setValue(1)
        self.step_interval.setRange(0, 1000)
        self.step_interval.setValue(500)
        self.margin.setRange(1, 200)
        self.margin.setValue(50)

        self.enable_skeleton.setChecked(False)
        self.spline_smoothing.setRange(0, 1000)
        self.spline_smoothing.setValue(100)
        self.centre_line.setRange(0, 600)
        self.centre_line.setValue(102)

        self.update_parameters()

    @QtCore.Slot()
    def update_parameters(self):

        #Grab parameters from UI Widgets
        self.params['adaptive'] = self.adapt_radio.isChecked()
        self.params['simple_value'] = self.simple_thresh_value.value()
        self.params['adaptive_gauss'] = self.gauss_radio.isChecked()
        self.params['adaptive_kernel'] = self.adapt_kernel_size.value()
        self.params['adaptive_m'] = self.adapt_kernel_mean.value()
        self.params['opening'] = self.morph_open.value()
        self.params['closing'] = self.morph_close.value()
        self.params['morph_kernel'] = self.morph_kernel.value()
        self.params['show_threshold'] = self.show_thresh.isChecked()

        self.params['tracking'] = self.enable_tracking.isChecked()
        self.params['step_size'] = self.step_size.value()
        self.params['step_interval'] = self.step_size.value()
        self.params['margin'] = self.margin.value()

        self.params['skeleton'] = self.enable_skeleton.isChecked()
        self.params['smoothing'] = self.spline_smoothing.value()
        self.params['centre_points'] = self.centre_line.value()
        self.params['refine'] = self.refine.isChecked()
        self.params['record'] = self.record.isChecked()

        #Disable thresholding params depending on mode
        if self.params['adaptive']:
            self.simple_thresh_value.setEnabled(False)
            self.mean_radio.setEnabled(True)
            self.gauss_radio.setEnabled(True)
            self.adapt_kernel_size.setEnabled(True)
            self.adapt_kernel_mean.setEnabled(True)
        else:
            self.simple_thresh_value.setEnabled(True)
            self.mean_radio.setEnabled(False)
            self.gauss_radio.setEnabled(False)
            self.adapt_kernel_size.setEnabled(False)
            self.adapt_kernel_mean.setEnabled(False)

        #Signal to transfer params to processing thread
        self.thread.update_parameters(self.params)

    @QtCore.Slot()
    def update_frame(self, frame):
        '''Resend signal from frame thread to frame widget'''
        self.pixmap = QtGui.QPixmap.fromImage(frame)
        self.emit(QtCore.SIGNAL("newPixmap(const QPixmap)"), self.pixmap)
        self.update()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    widget = MicroscopeClient(sys.argv[1], sys.argv[2])
    widget.show()
    r = app.exec_()
    widget.thread.stop()
    sys.exit(r)
