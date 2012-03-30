from slab.gui import *
#from slab.instruments import Alazar, AlazarConfig
from ScopeWindow2_ui import *
from guiqwt.builder import make
import numpy as np
import sys

class ScopeDataThread(DataThread):
    def __init__(self):
        DataThread.__init__(self)
#        self.settings = AlazarConfig()
#        self.card = Alazar()
        self.timer = QTimer()
        self.timer.timeout.connect(self.acquire_trace)

    def run_scope(self):
        self.settings.from_dict(self.params)
        self.card.configure(self.settings)
        self.acquire_trace()

    def acquire_trace(self):
        self.timer.stop()
        time, ch1, ch2 = self.card.acquire_avg_data()
        if ch1:
            self.plots["ch1"].set_data(time, ch1)
        if ch2:
            self.plots["ch2"].set_data(time, ch2)
        self.plots["plot"].replot()
        if self.params["autoscale"]:
            self.plots["plot"].do_autoscale()
        if self.params["autorun"]:
            self.timer.start(1)

    def save_trace(self):
        pass

class ScopeWin(SlabWindow, Ui_ScopeWindow):
    def __init__(self):
        SlabWindow.__init__(self, ScopeDataThread)
        self.setupSlabWindow(autoparam=True)
        self.register_script("run_scope", self.runButton)
        self.register_script("save_trace", self.saveButton)
        self.start_thread()
        self.init_plots()

    def init_plots(self):
        self.plot_manager["plot"] = self.curvewidget.plot
        x = np.linspace(-5,5,1000)
        y1 = np.cos(x)
        y2 = np.sin(x)
        self.plot_manager["ch1"] = make.mcurve(x, y1, label="Ch1")
        self.plot_manager["plot"].add_item(self.plot_manager["ch1"])
        self.plot_manager["ch2"] = make.mcurve(x, y2, label="Ch2")
        self.plot_manager["plot"].add_item(self.plot_manager["ch2"])

if __name__ == "__main__":
    sys.exit(runWin(ScopeWin))
