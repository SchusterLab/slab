#!/usr/bin/python2.7
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
          self.plots["myplot"] = self.myplot
        - For connecting parameter widgets there are three options
          - For "standard" widgets (i.e. SpinBox, Checkbox, ButtonGroup, LineEdit),
            register a widget with a name using self.register_param(widget, "widgetName")
          - Automatically register parameter widgets by modifying the call to setupSlabWindow
            self.setupSlabWindow(autoparam=True)
            This will locate widgets named with the convention param_widgetName, and
            automatically call register_param(widget, "widgetName")
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
from dataanalysis import get_next_filename
from slab.widgets import SweepDialog
from PyQt4.Qt import *
import PyQt4.Qwt5 as Qwt
import numpy as np
from time import sleep
import sys, csv, os

DEBUG = True

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

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
    def __contains__(self, key):
        return key in self.cache

class DataThread(QObject):
    def __init__(self, config_file=None):
        QObject.__init__(self)
        self.config_file = config_file
        self.instruments = InstrumentManager(self.config_file)
        self.params = {}
        self.plots = ManagerProxy("plotMethodCalled")
        self._aborted = False

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

    def status(self, s):
        self.emit(SIGNAL("status"), s)

    def progress(self, val, tot):
        self.emit(SIGNAL("progress"), (val, tot))

    def abort(self):
        self._aborted = True

    def aborted(self):
        if self._aborted:
            self._aborted = False
            return True
        else:
            return False

    def run_sweep(self):
        params = Bunch(self.params)
        prange = range(params.sweep1_start, 
                params.sweep1_start, params.sweep1step)
        if params.sweep2_enabled:
            raise NotImplementedError
        actions = params.actions.splitlines()
        for p in prange:
            for action in actions:
                self.params[self.params.sweep1] = p
                getattr(self, action)()
                
            
    def run_data_thread(self, method):
        if DEBUG: print method, "running"
        getattr(self, method)()
        self.emit(SIGNAL(method + "done"))

class SlabWindow(QMainWindow):
    def __init__(self, DataThreadC, config_file=None):
        QMainWindow.__init__(self)
        self.instruments = ManagerProxy("instrumentMethodCalled")
        self.plots = {}
        self.params = DictProxy("parameterChanged")
        self.param_widgets = []
        self.registered_actions = []
        self.data_thread_obj = DataThreadC()

        self.connect(self.instruments, SIGNAL("instrumentMethodCalled"),
                self.data_thread_obj.run_on_instrument_manager)
        self.connect(self.data_thread_obj, SIGNAL("instrumentMethodFailed"),
                self.msg)
        self.connect(self.data_thread_obj.plots, SIGNAL("plotMethodCalled"),
                self.run_on_plot_manager)
        self.connect(self.params, SIGNAL("parameterChanged"),
                self.data_thread_obj.set_param)
        self.connect(self.params, SIGNAL("parameterChanged"),
                self.write_param_widget)

        self.connect(self.data_thread_obj, SIGNAL("msg"), self.msg)
        self.connect(self.data_thread_obj, SIGNAL("status"), self.status)
        self.connect(self.data_thread_obj, SIGNAL("progress"), self.progress)

    def register_script(self, method, go_button, abort_button=None):
        self.registered_actions.append(method)
        go_button.clicked.connect(lambda: go_button.setDisabled(True))
        # Emit / connect necessary for non-blocking
        go_button.clicked.connect(lambda: self.emit(SIGNAL(method), method))
        self.connect(self, SIGNAL(method), self.data_thread_obj.run_data_thread)
        self.connect(self.data_thread_obj, SIGNAL(method + "done"),
                     lambda: go_button.setDisabled(False))
        if abort_button:
            abort_button.clicked.connect(self.data_thread_obj.abort, 
                    Qt.DirectConnection)

    def setupSlabWindow(self, autoparam=False, prefix="param_"):
        "Connect Ui components provided by the SlabWindow designer template"
        self.setupUi(self)

        self._data_thread = QThread()
        self.data_thread_obj.moveToThread(self._data_thread)
        self.connect(self, SIGNAL("lastWindowClosed()"), self._data_thread.exit)

        try:
            self.actionExperimental_Settings.triggered.connect(
                self.data_thread_obj.save_experimental_settings)
            self.actionExperimental_and_InstrumentSettings.triggered.connect(
                self.data_thread_obj.save_all_settings)
            self.actionLoad.triggered.connect(
                self.data_thread_obj.load_settings)
            
        except Exception as e:
            print "Could not connect menu actions", e

        if autoparam:
            self.autoparam(self, prefix)

    def autoparam(self, container, prefix="param_"):
        for wname, widget in self_dict(container, prefix).items():
            if isinstance(widget, QWidget):
                if DEBUG: print "attached parameter", wname
                self.register_param(widget, wname)

    def run_on_plot_manager(self, plot_name, method_name, args, kwargs):
        getattr(self.plots[plot_name], method_name)(*args, **kwargs)

    def msg(self, message_):
        try:
            message = str(message_)
            self.message_box.append(message)
        except Exception as e:
            if DEBUG:
                print "no msg box:", e
                print message_

    def status(self, message):
        try:
            self.statusbar.showMessage(message)
        except Exception as e:
            if DEBUG:
                print "no status bar:", e
                print message

    def progress(self, pair):
        val, tot = pair
        self.progressBar.setMaximum(tot)
        self.progressBar.setValue(val)

    def set_param(self, name, value):
        if isinstance(value, QString):
            value = str(value)
        self.params[name] = value

    def start_thread(self):
        self.read_param_widgets()
        self._data_thread.start()

    widget_tools = {
            QSpinBox : {"read" : "value",
                        "write" : "setValue",
                        "change_signal" : "valueChanged"},
            QDoubleSpinBox : {"read" : "value",
                              "write" : "setValue",
                              "change_signal" : "valueChanged"},
            QLineEdit : {"read" : "text",
                         "write" : "setText",
                         "change_signal" : "textEdited"},
            QButtonGroup : {"read" : "checkedId",
                            "write" : "setValue",
                            "change_signal" : "valueChanged"},
            QCheckBox : {"read" : "isChecked",
                         "write" : "setValue",
                         "change_signal" : "valueChanged"},
            QComboBox : {"read" : "currentText",
                         "write" : 
                            lambda w, v: w.setIndex(w.findtext(v)),
                         "change_signal": 
                            lambda w, f: w.currentIndexChanged[str].connect(f)}
            }


    def read_param_widgets(self):
        for widget, name in self.param_widgets:
            for cls, clstools in self.widget_tools.items():
                if isinstance(widget, cls):
                    try: 
                        self.set_param(name, getattr(widget, clstools["read"])())
                    except:
                        self.set_param(clstools["read"](widget))
                    break

    def write_param_widget(self, name, value):
         widget, name =  lookup(self.param_widgets, name, 1)
         for cls, clstools in self.widget_tools.items():
             if isinstance(widget, cls):
                try:
                     getattr(widget, clstools["write"])(value)
                except:
                    self.clstools["write"](widget, value)
                break

    def register_param(self, widget, name):
        self.param_widgets.append((widget, name))
        set_fn = lambda i: self.set_param(name, i)
        for cls, clstools in self.widget_tools.items():
            if isinstance(widget, cls):
                try: 
                    getattr(widget, clstools["change_signal"]).connect(set_fn)
                except:
                    self.clstools["change_signal"](widget, set_fn)
                break


#    def read_param_widgets(self):
#        for widget, name in self.param_widgets:
#            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
#                self.set_param(name, widget.value())
#            elif isinstance(widget, QLineEdit):
#                self.set_param(name, widget.text())
#            elif isinstance(widget, QButtonGroup):
#                self.set_param(name, widget.checkedId())
#            elif isinstance(widget, QCheckBox):
#                self.set_param(name, widget.isChecked())
#            elif isinstance(widget, QComboBox):
#                self.set_param(name, widget.currentText())

#    def write_param_widget(self, name, value):
#        try:
#            widget, name = lookup(self.param_widgets, name, 1)
#            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
#                widget.setValue(value)
#            elif isinstance(widget, QLineEdit):
#                widget.setText(value)
#            elif isinstance(widget, QButtonGroup):
#                widget.setId(value)
#            elif isinstance(widget, QCheckBox):
#                widget.setChecked(Qt.Checked if value else Qt.Unchecked)
#            elif isintance(widget, QComboBox):
#                widget.set
#        except:
#            pass

#    def register_param(self, widget, name):
#        self.param_widgets.append((widget, name))
#        set_fn = lambda i: self.set_param(name, i)
#        if isinstance(widget, QSpinBox): 
#            widget.valueChanged.connect(set_fn)
#
#        elif isinstance(widget, QDoubleSpinBox):
#            widget.valueChanged.connect(set_fn)
#
#        elif isinstance(widget, QLineEdit):
#            widget.textEdited.connect(set_fn)
#
#        elif isinstance(widget, QButtonGroup):
#            widget.buttonClicked.connect(set_fn)
#
#        elif isinstance(widget, QCheckBox):
#            widget.stateChanged.connect(lambda i: self.set_param(name, i > 0))
#
#        elif isinstance(widget, QComboBox):
#            widget.currentIndexChanged[str].connect(set_fn)
#
#        else: print "could not match", name, "with widget"
        
    def save_time_series(self, *xs):
        datapath = self.params["datapath"]
        fname = os.path.join(datapath, get_next_filename(datapath, self.params["prefix"]))
        f = open(fname, 'w')
        csv.writer(f).writerows(zip(*xs))
        f.close()
        if DEBUG: print "saved", fname

    def add_instrument(self, iname):
        pass

    def add_sweep_dialog(self):
        self.sweep_dialog = sd = SweepDialog()
        self.autoparam("SweepDialog")
        try: self.actionStart_Sweep.triggered.connect(sd.exec_)
        except AttributeError: print "No actionStart_Sweep to connect"

        for name, widget in self.param_widgets:
            if isinstance(widget, QSpinBox):
                sd.param_sweep1.addItem(name)
                sd.param_sweep2.addItem(name)
        for name in self.registered_actions:
            sd.actions_comboBox.addItem(name)

        sd.addActionButton.clicked.connect(
            lambda: sd.param_actions.append(
                sd.actions_comboBox.currentText()))

        sd.accepted.connect(self.data_thread_obj.run_sweep)

def self_dict(obj, prefix="param_"):
    res = {}
    for name in dir(obj):
        if name.startswith(prefix):
            res[name[len(prefix):]] = getattr(obj, name)
    return res

def lookup(tuplelist, ident, idx=0):
    for t in tuplelist:
        if t[idx] == ident:
            return t

def runWin(WinC, *args, **kwargs):
    app = QApplication([])
    win = WinC(*args, **kwargs)
    win.show()
    app.connect(app, SIGNAL("lastWindowClosed()"), win, SIGNAL("lastWindowClosed()"))
    app.exec_()
################
# Example code #
################

if __name__ == "__main__":
    from test_ui import *
    from widgets import *
    import cProfile

    class test_DataThread(DataThread):
        def run_script(self):
            omega = self.params["rate"]
            for i in range(100):
                self.progress(i, 100)
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
            self.add_sweep_dialog()
            self.start_thread()

            self.plots["plot"] = self.qwtPlot
            sine_curve = Qwt.QwtPlotCurve("Sine")
            cosine_curve = Qwt.QwtPlotCurve("Cosine")
            sine_curve.attach(self.qwtPlot)
            cosine_curve.attach(self.qwtPlot)
            self.plots["sine"] = sine_curve
            self.plots["cosine"] = cosine_curve

    #cProfile.run("sys.exit(runWin(TestWin))")
    sys.exit(runWin(TestWin))
