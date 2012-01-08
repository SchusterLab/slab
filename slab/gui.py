"""
gui.py

usage:
    - Create a UI in Qt Designer, and with pyuic convert the .ui to .py
        - Use the 'SlabWindow' template as a base to enable config_file menu items
          (! not yet fully functional !)
    - Create a class inheriting from SlabWindow and Ui_MainWindow (as provided by
    the designer file)
    - In the class' __init__ method
        - run self.setupSlabWindow()
        - register any plots (or actually any UI components to be
          dynamically updated) from the UI with the plot manager, i.e.
          self.plot_manager["myplot"] = self.myplot
        - For connecting parameter widgets there are three options
          - For "standard" widgets (i.e. SpinBox, Checkbox, ButtonGroup, LineEdit),
            register a widget with a name using self.register_widget(widget, "widgetName")
          - Automatically register parameter widgets by modifying the call to setupSlabWindow
            self.setupSlabWindow(autoparam=True)
            This will locate widgets named with the convention param_widgetName, and
            automatically call register_widget(widget, "widgetName")
          - For non-standard widgets find the appropriate signal to call self.set_param from, i.e
            self.connect(self.myparamSpinBox, SIGNAL("valueChanged(int)"),
                         lambda i: self.set_param("myparam", i))
    - Subclass DataThread to add run_script(). In the DataThread
      namespace, you will find self.instruments (an InstrumentManager),
      self.params (a dictionary of parameters), and self.plots (a PlotManager)
        - For any scripts/functions which you wish to execute in a non-blocking
          fashion, add a method to the DataThread subclass that executes this
          script using the provided resources
        - register the methods with start and stop buttons, i.e. after
          providing the "myscript" method:
          self.register_script("myscript", self.my_go_button, self.my_abort_button)
            - after registering all methods, call self.start_thread()

    A more complete (functional) example is given at the end of the file
"""
from instruments import InstrumentManager
from PyQt4.Qt import *
import PyQt4.Qwt5 as Qwt
import numpy as np
from time import sleep
import sys

class DataThread(QObject):
    def __init__(self, config_file=None):
        QObject.__init__(self)
        self.config_file = config_file
        self.instruments = InstrumentManager(self.config_file)
        self.params = {}
        self.plots = ManagerProxy("plotMethodCalled")
        self._aborted = False

    def load_config(config_file):
        self.instrument_manager.load_config_file(config_file)

    def run_on_instrument_manager(inst_name, method_name, args, kwargs):
        try: getattr(self.instrument_manager[inst_name], method_name)(*args, **kwargs)
        except Exception as e: self.emit("instrumentMethodFailed", (str(e),))

    def set_param(self, name, value):
        self.params[name] = value

    def save_experimental_settings(self):
        filename = str(QFileDialog.getSaveFileName())
        self.instruments.save_settings(filename)

    def save_all_settings(self):
        filename = str(QFileDialog.getSaveFileName())
        # save_settings does not currently (11/13) process params meaningfully
        self.instruments.save_settings(filename, params=self.params)

    def load_settings(self):
        filename = QFileDialog.getOpenFileName(filter="*.cfg")
        self.instruments.load_config_file(filename)

    def list_instruments(self):
        for key, inst in self.instruments.iteritems():
            self.msg(key + inst.name)

    def msg(self, s):
        self.emit(SIGNAL("msg"), s)

    def abort(self):
        self._aborted = True

    def aborted(self):
        if self._aborted:
            self._aborted = False
            return True
        else:
            return False

    def run_data_thread(self, method):
        self.run_script()
        self.emit(SIGNAL(method + "done"))

class ManagerProxy(QObject):
    def __init__(self, emit_name):
        QObject.__init__(self)
        self.emit_name = emit_name
    def __getitem__(self, item_name):
        class IProxy(object):
            def __getattr__(self2, method_name):
                return lambda *args, **kwargs: self.emit(SIGNAL(self.emit_name),
                item_name, method_name, args, kwargs)

        return IProxy()

class DictProxy(QObject):
    def __init__(self, emit_name):
        QObject.__init__(self)
        self.cache = {}
        self.emit_name = emit_name
    def __setitem__(self, name, value):
        self.cache[name] = value
        self.emit(SIGNAL(self.emit_name), name, value)
    def __getitem__(self, name):
        return self.cache[name]


class SlabWindow(QMainWindow):
    def __init__(self, DataThreadC, config_file=None):
        QMainWindow.__init__(self)
        self.instruments = ManagerProxy("instrumentMethodCalled")
        self.plot_manager = {}
        self.params = DictProxy("parameterChanged")
        self.param_widgets = []
        self.data_thread_obj = DataThreadC()
#        self.setup_commands()
        self.connect(self.instruments, SIGNAL("instrumentMethodCalled"),
                self.data_thread_obj.run_on_instrument_manager)
        self.connect(self.data_thread_obj, SIGNAL("instrumentMethodFailed"),
                self.msg)
        self.connect(self.data_thread_obj.plots, SIGNAL("plotMethodCalled"),
                self.run_on_plot_manager)
        self.connect(self.params, SIGNAL("parameterChanged"),
                self.data_thread_obj.set_param)
        self.connect(self.data_thread_obj, SIGNAL("msg"), self.msg)

    def register_script(self, method, go_button, abort_button=None):
        self.connect(go_button, SIGNAL("clicked()"),
                     lambda: go_button.setDisabled(True))
        self.connect(go_button, SIGNAL("clicked()"), lambda: self.emit(SIGNAL("method"), method))
        self.connect(self, SIGNAL("method"), self.data_thread_obj.run_data_thread)
        self.connect(self.data_thread_obj, SIGNAL(method + "done"),
                     lambda: go_button.setDisabled(False))
        if abort_button:
            self.connect(abort_button, SIGNAL("clicked()"),
                         self.data_thread_obj.abort, Qt.DirectConnection)

    def setupSlabWindow(self, autoparam=False):
        "Connect Ui components provided by the SlabWindow designer template"
        self.setupUi(self)

        self._data_thread = QThread()
        self.data_thread_obj.moveToThread(self._data_thread)

        try:
            self.connect(self.actionExperimental_Settings, SIGNAL("triggered()"),
                         self.data_thread_obj.save_experimental_settings)
            self.connect(self.actionExperimental_and_Instrument_Settings, SIGNAL("triggered()"),
                         self.data_thread_obj.save_all_settings)
            self.connect(self.actionLoad, SIGNAL("triggered()"),
                         self.data_thread_obj.load_settings)
#            self.connect(self.cmd_lineEdit, SIGNAL("returnPressed()"),
#                         self.process_cmd)
#            self.connect(self.cmd_lineEdit, SIGNAL("returnPressed()"),
#                         lambda: self.cmd_lineEdit.setText(""))
        except Exception as e:
            print "Could not connect menu actions", e

        if autoparam:
            for wname in dir(self):
                print wname
                if wname[:6] == "param_":
                    widget = getattr(self, wname)
                    if isinstance(widget, QWidget):
                        self.register_param(widget, wname[6:])

#    def setup_commands(self):
#        self.commands = {}
#
#        instrument_list = lambda: self.emit(SIGNAL("list_instruments"))
#        self.connect(self, SIGNAL("list_instruments"),
#                self.data_thread_obj.list_instruments)
#
#        self.commands["il"] = instrument_list
#
#    def process_cmd(self):
#        line = str(self.cmd_lineEdit.text()).split()
#        cmd = line[0]
#        args = tuple(line[1:])
#        try: apply(self.commands[cmd], args)
#        except Exception as e: self.msg("Command failed: " + repr(e))

    def run_on_plot_manager(self, plot_name, method_name, args, kwargs):
        getattr(self.plot_manager[plot_name], method_name)(*args, **kwargs)

    def msg(self, message):
        try:
            self.message_box.append(message)
        except:
            print "no msg box:", message

    def set_param(self, name, value):
#        self.msg(str(name) + " :: " + str(value))
        self.params[name] = value

    def start_thread(self):
        self.read_param_widgets()
        self._data_thread.start()

    def read_param_widgets(self):
        for widget, name in self.param_widgets:
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                self.set_param(name, widget.value)
            elif isinstance(widget, QLineEdit):
                self.set_param(name, widget.text())
            elif isinstance(widget, QButtonGroup):
                self.set_param(name, i)
            elif isinstance(widget, QCheckBox):
                self.set_param(name, i > 0)

    def register_param(self, widget, name):
        self.param_widgets.append((widget, name))
        if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
            self.connect(widget, SIGNAL("valueChanged(int)"),
                    lambda i: self.set_param(name, i))
        elif isinstance(widget, QLineEdit):
            self.connect(widget, SIGNAL("returnPressed()"),
                    lambda: self.set_param(name, widget.text()))
        elif isinstance(widget, QButtonGroup):
            self.connect(widget, SIGNAL("buttonClicked(int)"),
                    lambda i: self.set_param(name, i))
        elif isinstance(widget, QCheckBox):
            self.connect(widget, SIGNAL("stateChanged(int)"),
                    lambda i: self.set_param(name, i > 0))

# Example code

from test_ui import *

class test_DataThread(DataThread):
    def run_script(self):
        omega = self.params["rate"]
        for i in range(100):
            if (i % 10) == 0:
                self.msg(str(i))
            if self.aborted():
                self.msg("aborted")
                break
            xrng = np.linspace(0, 2 * np.pi, num = i)
            sleep(.1)
            self.plots["sine"].setData(xrng, np.sin(omega * xrng))
            self.plots["cosine"].setData(xrng, np.cos(omega * xrng))
            self.plots["plot"].replot()

class TestWin(SlabWindow, Ui_MainWindow):
    def __init__(self):
        SlabWindow.__init__(self, test_DataThread, config_file=None)
        self.setupSlabWindow(autoparam=True)
        self.register_script("run_script", self.go_button, self.abort_button)
#        self.connect(self.spinBox, SIGNAL("valueChanged(int)"),
#                lambda i: self.set_param("rate", i))
#        self.register_param(self.spinBox, "rate")
        self.start_thread()
        self.set_param("rate", 1)
        self.plot_manager["plot"] = self.qwtPlot
        sine_curve = Qwt.QwtPlotCurve("Sine")
        cosine_curve = Qwt.QwtPlotCurve("Cosine")
        sine_curve.attach(self.qwtPlot)
        cosine_curve.attach(self.qwtPlot)
        self.plot_manager["sine"] = sine_curve
        self.plot_manager["cosine"] = cosine_curve

def runWin(WinC, *args, **kwargs):
    app = QApplication([])
    win = WinC(*args, **kwargs)
    win.show()
    app.exec_()

if __name__ == "__main__":
    sys.exit(runWin(TestWin))
