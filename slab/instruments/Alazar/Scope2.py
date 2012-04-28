from slab.gui import *
from slab.instruments import Alazar, AlazarConfig, AlazarConstants

DEBUG = False
if DEBUG:
    from slab import tic, toc
    import time
    from PyQt4 import uic
    uifile = open("ScopeWindow2_ui.py", 'w')
    uic.compileUi("ScopeWindow2.ui", uifile)
    uifile.close()
from ScopeWindow2_ui import *

from guiqwt.builder import make
import numpy as np
import sys

from slab.widgets import PlotWithTB

class ScopeDataThread(DataThread):
    def __init__(self):
        DataThread.__init__(self)
        self.settings = AlazarConfig()
        self.card = Alazar()
        self.trace_no = 0
        self.datapath = ""

    def apply_settings(self):
        self.settings.from_dict(self.params)
        self.settings.interpret_constants()
        self.card.configure(self.settings)

    def run_scope(self):
        if "timer" not in dir(self):
            self.timer = QTimer()
            self.timer.timeout.connect(self.acquire_trace)
        self.apply_settings()
        self.acquire_trace()

    def acquire_trace(self):
        self.timer.stop()
        self.trace_no += 1
        ex1, ex2 = (self.params["excise_start"], self.params["excise_end"])
        if not ex2: ex2 = 1
        debug = DEBUG and not self.trace_no % 20
        if debug: tic()
        times, ch1, ch2 = self.card.acquire_avg_data()
        if debug: self.msg("Acquire: %.3f" % toc())
        if ch1 is not None:
            self.plots["ch1"].setVisible(True)
            self.plots["ch1"].set_data(times, ch1[ex1:-ex2])
        else:
            self.plots["ch1"].setVisible(False)
        if ch2 is not None:
            self.plots["ch2"].setVisible(True)
            self.plots["ch2"].set_data(times, ch2[ex1:-ex2])
        else:
            self.plots["ch2"].setVisible(False)
        self.plots["plot"].replot()
        if self.params["autoscale"]:
            self.plots["plot"].do_autoscale()
        if self.params["autorun"] and not self.aborted():
            self.timer.start(1)

class ScopeWin(SlabWindow, Ui_ScopeWindow):
    def __init__(self):
        SlabWindow.__init__(self, ScopeDataThread)
        self.setupSlabWindow(autoparam=True)
        self.register_script("run_scope", self.runButton, self.stopButton)
        self.register_script("apply_settings", self.applyButton)
        self.datapathButton.clicked.connect(self.select_datapath)
        self.saveButton.clicked.connect(self.save_trace)
        self.add_sweep_dialog()
        self.sweepButton.clicked.connect(self.sweep_dialog.exec_)
        self.start_thread()      
        self.init_plots()

    def select_datapath(self):
        self.params["datapath"]=str(QFileDialog.getExistingDirectory(self,
                                    'Open Datapath',self.params["datapath"]))

    def save_trace(self):
        x, y1 = self.plots["ch1"].get_data()
        _, y2 = self.plots["ch2"].get_data()
        self.save_time_series(x, y1, y2)

        
    def init_plots(self):
        self.plots["plot"] = self.curvewidget.plot
        self.curvewidget.add_toolbar(self.toolBar)
        self.curvewidget.register_all_curve_tools()
        x = np.linspace(-5,5,1000)
        y1 = np.cos(x)
        y2 = np.sin(x)
        self.plots["ch1"] = make.mcurve(x, y1, label="Ch1")
        self.plots["plot"].add_item(self.plots["ch1"])
        self.plots["ch2"] = make.mcurve(x, y2, label="Ch2")
        self.plots["plot"].add_item(self.plots["ch2"])

if __name__ == "__main__":
    sys.exit(runWin(ScopeWin))
