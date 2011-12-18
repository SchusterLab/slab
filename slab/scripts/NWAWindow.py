# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:10:50 2011

@author: dave
"""
from slab.gui import *
from NWAWindow_ui import *
from guiqwt.builder import make

class nwa_DataThread(DataThread):
    def run_script(self):
        try:
            nwa=self.instruments['NWA']
        except:
            self.msg("NWA config not loaded!")
            return
        self.msg("Configuring NWA....")
        self.msg(str(nwa.get_id()))
        if self.params["centerspanstartstop"]:
            nwa.set_center_frequency(self.params['centerstart']*1e9)
            nwa.set_span(self.params['spanstop']*1e9)
        else:
            nwa.set_start_frequency(self.params['centerstart']*1e9)
            nwa.set_start_frequency(self.params['spanstop']*1e9)
        
        nwa.set_ifbw(self.params['ifbw'])
        nwa.set_power(self.params['power'])
        nwa.set_averages(self.params['avgs'])
        nwa.set_sweep_points(self.params['sweep_pts'])
        self.msg("Acquiring data...")
        freqs,mags,phases=nwa.take_one_averaged_trace()
        self.msg("Data acquisition complete.")
        self.plots["mag"].set_data(freqs,mags)
        self.plots["phase"].set_data(freqs,phases)
        self.plots["magplot"].replot()
        self.plots["phaseplot"].replot()

class NWAWin(SlabWindow, Ui_NWAWindow):
    def __init__(self):
        SlabWindow.__init__(self, nwa_DataThread, config_file=None)
        self.setupSlabWindow()
        self.register_script("run_script", self.go_button, self.abort_button)
        self.start_thread()

        #Connect controls
        self.connect(self.powerSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("power", i))
        self.connect(self.sweep_ptsSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("sweep_pts", i))
        self.connect(self.ifbwSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("ifbw", i))               
        self.connect(self.avgsSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("avgs", i)) 
        self.connect(self.centerstartSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("centerstart", i)) 
        self.connect(self.spanstopSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("spanstop", i)) 
        self.connect(self.centerspanstartstopCheckBox, SIGNAL("stateChanged(int)"),
                lambda i: self.set_param("centerspanstartstop", i)) 
        self.connect(self.resolutionSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("resolution", i)) 
        self.connect(self.centerspanstartstopCheckBox, SIGNAL("stateChanged(int)"),self.on_centerspanstartstopChanged)
        
        #Initialize Parameters
        self.set_param("power", -20)
        self.set_param("sweep_pts",1601)
        self.set_param("ifbw",1e3)
        self.set_param("avgs",1)
        self.set_param("centerstart",10.)
        self.set_param("spanstop",1.)
        self.set_param("centerspanstartstop",1)
        self.set_param("resolution", 100.)
        
        #Make some default data
        self.freqs=np.linspace(self.params['centerstart']-self.params['spanstop']/2.,self.params['centerstart']+self.params['spanstop']/2.,self.params['sweep_pts'])        
        self.mags=np.zeros(len(self.freqs)) 
        self.phases=np.zeros(len(self.freqs))
        
        #Initialize Magnitude plot"
        self.plot_manager["magplot"] = self.magplotwidget.plot        
        self.magplotwidget.add_toolbar(self.addToolBar("Curve"))
        self.magplotwidget.register_all_image_tools()
        self.plot_manager["magplot"].set_titles(title="Magnitude", xlabel="Frequency (GHz)", ylabel="S21")       
        self.plot_manager["mag"] = make.mcurve(self.freqs, self.mags,label='Magnitude') #Make Ch1 curve
        self.plot_manager["magplot"].add_item(self.plot_manager["mag"])

        #Initialize Phase plot
        self.plot_manager["phaseplot"] = self.phaseplotwidget.plot
        self.phaseplotwidget.add_toolbar(self.addToolBar("Curve"))
        self.phaseplotwidget.register_all_image_tools()
        self.plot_manager["phaseplot"].set_titles(title="Phase", xlabel="Frequency (GHz)", ylabel="Phase (deg)")
        self.plot_manager["phase"] = make.mcurve(self.freqs, self.phases,label='Phase') #Make Ch1 curve
        self.plot_manager["phaseplot"].add_item(self.plot_manager["phase"])

    def on_centerspanstartstopChanged(self,state):
        if state: 
            self.centerspanstartstopCheckBox.setText('Start/Stop')
            self.centerstartLabel.setText('Center')
            self.spanstopLabel.setText('Span')
        else:
            self.centerspanstartstopCheckBox.setText('Center/Span')
            self.centerstartLabel.setText('Start')
            self.spanstopLabel.setText('Stop')

        

if __name__ == "__main__":
    app = QApplication([])
    win = NWAWin()
    win.show()
    sys.exit(app.exec_())