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
#import PyQt4.Qwt5 as Qwt
from PyQt4 import uic
import numpy as np
#from time import sleep
import csv, os, inspect, time

DEBUG = False


class Bunch(object):
    def __init__(self, adict):
        try: self.__dict__.update(adict)
        except: self.__dict__.update(adict.cache)

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
    def set_nosignal(self, name, value):
        self.cache[name] = value
    def __setitem__(self, name, value):
        self.set_nosignal(name, value)
        self.emit(SIGNAL(self.emit_name+"Changed"), name, value)
    def update(self):
        self.emit(SIGNAL(self.emit_name+"Get"))
    def update_cache(self, db):
        self.cache = db
    def __getitem__(self, name):
        return self.cache[name]
    def __contains__(self, key):
        return key in self.cache
    def keys(self):                 #Added by DS not sure if this is right
        return self.cache.keys()

class DataThread(QObject):
    def __init__(self, config_file=None):
        QObject.__init__(self)
        self.config_file = config_file
        self.instruments = InstrumentManager(self.config_file)
        self.params = DictProxy("threadParameter")
        self.connect(self.params, SIGNAL("threadParameterChanged"),
                     lambda n,v: self.emit(SIGNAL("threadParameterChanged"), n, v))

        self.plots = ManagerProxy("plotMethodCalled")
        self.gui = ManagerProxy("guiMethodCalled")
        self._aborted = False

    def run_on_instrument_manager(inst_name, method_name, args, kwargs):
        try: getattr(self.instrument_manager[inst_name], method_name)(*args, **kwargs)
        except Exception as e: self.emit("instrumentMethodFailed", (str(e),))

    def set_param(self, name, value):
        self.params.set_nosignal(name, value)

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

    def get_local_params(self):
        self.emit(SIGNAL("localParams"), self.params)

    def msg(self, *args):
        self.emit(SIGNAL("msg"), *args)

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

    def run_sweep(self, whichStack):
        self.msg("Sweeping...")
        params = Bunch(self.params)
        if whichStack[params.sweep1] == 0:
            prange = range(params.sweep1_start_int,
                           params.sweep1_stop_int, params.sweep1_step_int)
        else:
            prange = np.arange(params.sweep1_start_float,
                           params.sweep1_stop_float, params.sweep1_step_float)
        self.msg(prange)
        if params.sweep2_enabled:
            raise NotImplementedError
        actions = params.actions.splitlines()
        for p in prange:
            self.msg(params.sweep1 + "=" + str(p))
            for action in actions:
                self.params[params.sweep1] = p
                try:
                    if self.params["abort_sweep"]:
                        self.params["abort_sweep"] = False
                        return
                except: pass
                getattr(self, action)()
        self.msg("Done")


    def run_data_thread(self, method, *args):
        if DEBUG:
            print method, "running"
            self.msg(method)
            self.msg(args)
        getattr(self, method)(*args)
        self.emit(SIGNAL(method + "done"))

class SlabWindow(QMainWindow):
    def __init__(self, DataThreadC, config_file=None):
        QMainWindow.__init__(self)
        self.instruments = ManagerProxy("instrumentMethodCalled")
        self.plots = {}
        self.gui = {}
        self.params = DictProxy("parameter")
        self.param_widgets = []
        self.registered_actions = []
        if config_file is not None:
            self.data_thread_obj = DataThreadC(config_file=config_file)
        else:
            self.data_thread_obj = DataThreadC()
            

        self.connect(self.instruments, SIGNAL("instrumentMethodCalled"),
                self.data_thread_obj.run_on_instrument_manager)
        self.connect(self.data_thread_obj, SIGNAL("instrumentMethodFailed"),
                self.msg)
        self.connect(self.data_thread_obj.plots, SIGNAL("plotMethodCalled"),
                self.run_on_plot_manager)
        self.connect(self.data_thread_obj.gui, SIGNAL("guiMethodCalled"),
                self.run_on_gui)
        self.connect(self.params, SIGNAL("parameterChanged"),
                self.data_thread_obj.set_param)
        self.connect(self.params, SIGNAL("parameterChanged"),
                self.write_param_widget)
        self.connect(self.data_thread_obj, SIGNAL("threadParameterChanged"),
                self.write_param_widget)
        self.connect(self.params, SIGNAL("parameterGet"),
                self.data_thread_obj.get_local_params)
        self.connect(self.data_thread_obj, SIGNAL("localParams"),
                self.params.update_cache)

        self.connect(self.data_thread_obj, SIGNAL("msg"), self.msg)
        self.connect(self.data_thread_obj, SIGNAL("status"), self.status)
        self.connect(self.data_thread_obj, SIGNAL("progress"), self.progress)

    
    def setupSlabWindow(self, autoparam=False, prefix="param_"):
        "Connect Ui components provided by the SlabWindow designer template"
        self.setupUi(self)

        self._data_thread = QThread()
        self.data_thread_obj.moveToThread(self._data_thread)
        self.connect(self, SIGNAL("lastWindowClosed()"), self._data_thread.exit)

        try:
            self.actionExperimental_Settings.triggered.connect(
                self.data_thread_obj.save_experimental_settings)
        except Exception as e:
                    print "Could not connect menu actions", e
        try:
            self.actionExperimental_and_Instrument_Settings.triggered.connect(
                    self.data_thread_obj.save_all_settings)

        except Exception as e:
            print "Could not connect menu actions", e

        try:
            self.actionLoad.triggered.connect(
                    self.data_thread_obj.load_settings)

        except Exception as e:
            print "Could not connect menu actions", e

        if autoparam:
            self.autoparam(self, prefix)

    def auto_register_gui(self, container=None, prefix=""):
        if container is None:
            container = self
        for wname, widget in self_dict(container, prefix).items():
            if not inspect.isroutine(widget):
                self.gui[wname] = widget

    def autoparam(self, container, prefix="param_"):
        for wname, widget in self_dict(container, prefix).items():
            if isinstance(widget, QWidget) or isinstance(widget, QAction):
                if DEBUG: print "attached parameter", wname
                self.register_param(widget, wname)

    def run_on_plot_manager(self, plot_name, method_name, args, kwargs):
        getattr(self.plots[plot_name], method_name)(*args, **kwargs)

    def run_on_gui(self, plot_name, method_name, args, kwargs):
        getattr(self.gui[plot_name], method_name)(*args, **kwargs)


    def set_param(self, name, value):
        if isinstance(value, QString):
            value = str(value)
        self.params[name] = value

    def start_thread(self):
        self.connect(self, SIGNAL("RunOnDataThread"), self.data_thread_obj.run_data_thread)
        self.read_param_widgets()
        self._data_thread.start()

    launcher_widget_tools = {
        QPushButton : "released",
        QDialog: "accepted",
        QAction: "triggered",
        QListWidget: "itemDoubleClicked",
        QTreeWidget: "itemDoubleClicked",
        QLineEdit: "returnPressed"
    }
    
    def register_script(self, method, launch_widget, abort_widget=None):
        """
        Connect a launcher widget to a method in the data thread, causing that method
        to run when the widget is activated.

        @param method: A string sharing the name of a method belonging to your data thread
        @param go_widget: A widget in this window, which must be an instance of either
          - QPushButton
          - QDialog
          - QAction
          - QListWidget
          - QTreeWidget

        In the case that the widget is a List/Tree Widget, the method will be invoked
        when an item is double-clicked, and the method will receive as
        it's first (and only) method the List/TreeWidgetItem clicked.
        """
        self.registered_actions.append(method)
        for WidgetC, sig_name in self.launcher_widget_tools.iteritems():
            if isinstance(launch_widget, WidgetC):
                getattr(launch_widget, sig_name).connect(
                    lambda *args:
                        self.emit(SIGNAL("RunOnDataThread"), method, *args))
                print "attached method", method
                break
        else:
            print "Did not attach", method, "could not identify widget", launch_widget

        if abort_widget is not None:
            for WidgetC, sig_name in self.launcher_widget_tools.iteritems():
                if isinstance(abort_widget, WidgetC):
                    getattr(abort_widget, sig_name).connect(self.data_thread_obj.abort, Qt.DirectConnection)

    widget_tools = {
            QSpinBox : {"read" : "value",
                        "write" : "setValue",
                        "change_signal" : "valueChanged"},
            QDoubleSpinBox : {"read" : "value",
                              "write" : "setValue",
                              "change_signal" : "valueChanged"},
            QTextEdit : {"read" : "toPlainText",
                         "write" : None,
                         "change_signal" :
                             lambda w, f: w.textChanged.connect(lambda: f(w.document().toPlainText()))},
            QLineEdit : {"read" : "text",
                         "write" : "setText",
                         "change_signal" : "textEdited"},
            QButtonGroup : {"read" : "checkedId",
                            "write" : "setId",
                            "change_signal" : "buttonClicked"},
            QCheckBox : {"read" : "isChecked",
                         "write" : "setChecked",
                         "change_signal" : "stateChanged"},
            QComboBox : {"read" : "currentText",
                         "write" :
                            lambda w, v: w.setCurrentIndex(w.findText(v)),
                         "change_signal":
                            lambda w, f: w.currentIndexChanged[str].connect(f)},
            QAction : {"read" : None,
                       "write" : None,
                       "change_signal":
                           lambda w, f: w.triggered.connect(lambda: f(True))},
            QTreeWidget : {"read" : "currentItem",
                           "write" : "setCurrentItem",
                           "change_signal":
                               lambda w, f:
                               w.currentItemChanged.connect(lambda cur, prev: f(cur))}
            }


    def read_param_widgets(self):
        for widget, name in self.param_widgets:
            for cls, clstools in self.widget_tools.items():
                if isinstance(widget, cls):
                    if clstools["read"] is not None:
                        if hasattr(clstools["read"], "__call__"):
                            self.set_param(clstools["read"](widget))
                        else:
                            self.set_param(name, getattr(widget, clstools["read"])())
                        break

    def write_param_widget(self, name, value):
         res = lookup(self.param_widgets, name, 1)
         if res is not None:
             widget, name = res
             for cls, clstools in self.widget_tools.items():
                 if isinstance(widget, cls):
                    if clstools["write"] is not None:
                        if hasattr(clstools["write"], "__call__"):
                            clstools["write"](widget, value)
                        else:
                            getattr(widget, clstools["write"])(value)
                        break

    def register_param(self, widget, name):
        self.param_widgets.append((widget, name))
        set_fn = lambda i: self.set_param(name, i)
        for cls, clstools in self.widget_tools.items():
            if isinstance(widget, cls):
                if hasattr(clstools["change_signal"], "__call__"):
                    clstools["change_signal"](widget, set_fn)
                else:
                    getattr(widget, clstools["change_signal"]).connect(set_fn)
                break
        else:
            print "Error: Could Not match parameter", name, "with class"

    def msg(self, *args):
        t = time.strftime("%d %H:%M:%S")
        message = t + " :: " + " ".join(map(str, args))
        try:
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

#    def save_time_series(self, *xs):
#        datapath = self.params["datapath"]
#        fname = os.path.join(datapath,
#                             get_next_filename(datapath, self.params["prefix"]))
#        f = open(fname, 'w')
#        csv.writer(f).writerows(zip(*xs))
#        f.close()
#        if DEBUG: print "saved", fname
#
#    def add_instrument(self, iname):
#        pass

    def add_sweep_dialog(self):
        self.sweep_dialog = sd = SweepDialog()

        try: self.actionStart_Sweep.triggered.connect(sd.exec_)
        except AttributeError: print "No actionStart_Sweep to connect"
        self.whichStack = {}
        for w, name in self.param_widgets:
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox):
                print "sweep enabled on", name
                sd.param_sweep1.addItem(name)
                sd.param_sweep2.addItem(name)
                if isinstance(w, QSpinBox):
                    self.whichStack[name] = 0
                else:
                    self.whichStack[name] = 1

        for name in self.registered_actions:
            sd.actions_comboBox.addItem(name)

        sd.addActionButton.clicked.connect(
            lambda: sd.param_actions.append(
                sd.actions_comboBox.currentText()))
        sd.clearActionsButton.clicked.connect(sd.param_actions.clear)
        sd.param_sweep1.currentIndexChanged[str].connect(
            lambda name:
                sd.sweep1_stackedWidget.setCurrentIndex(self.whichStack[str(name)]))
        sd.param_sweep2.currentIndexChanged[str].connect(
                    lambda name:
                        sd.sweep2_stackedWidget.setCurrentIndex(self.whichStack[str(name)]))

        sd.accepted.connect(lambda: self.emit(SIGNAL("run_sweep"), self.whichStack))
        self.data_thread_obj.connect(self, SIGNAL("run_sweep"), self.data_thread_obj.run_sweep)
        self.autoparam(self.sweep_dialog)

def compile_ui(prefix):
    uifilename = prefix + ".ui"
    pyfile = open(prefix + "_ui.py", 'w')
    uic.compileUi(uifilename, pyfile)
    pyfile.close()

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

