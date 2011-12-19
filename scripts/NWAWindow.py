# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:10:50 2011

@author: dave
"""
from slab import *
from slab.gui import *
from NWAWindow_ui import *
from guiqwt.builder import make
import os

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
            nwa.set_span(self.params['spanstop']*1e6)
        else:
            nwa.set_start_frequency(self.params['centerstart']*1e9)
            nwa.set_stop_frequency(self.params['spanstop']*1e9)
        
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
        if self.params['save']:
            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
            np.savetxt(fname,transpose(array([freqs,mags,phases])),delimiter=',')

class NWAWin(SlabWindow, Ui_NWAWindow):
    def __init__(self):
        SlabWindow.__init__(self, nwa_DataThread, config_file=None)
        self.setupSlabWindow()
        self.register_script("run_script", self.go_button, self.abort_button)
        self.start_thread()

        #Connect controls
        self.connect(self.powerSpinBox, SIGNAL("valueChanged(double)"),
                lambda i: self.set_param("power", i))
        self.connect(self.sweep_ptsSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("sweep_pts", i))
        self.connect(self.ifbwSpinBox, SIGNAL("valueChanged(double)"),
                lambda i: self.set_param("ifbw", i))               
        self.connect(self.avgsSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("avgs", i)) 
        self.connect(self.centerstartSpinBox, SIGNAL("valueChanged(double)"),
                lambda i: self.set_param("centerstart", i)) 
        self.connect(self.spanstopSpinBox, SIGNAL("valueChanged(double)"),
                lambda i: self.set_param("spanstop", i)) 
        self.connect(self.centerspanstartstopCheckBox, SIGNAL("stateChanged(double)"),
                lambda i: self.set_param("centerspanstartstop", i)) 
        self.connect(self.resolutionSpinBox, SIGNAL("valueChanged(double)"),
                lambda i: self.set_param("resolution", i)) 
        self.connect(self.saveCheckBox, SIGNAL("stateChanged(int)"),
                lambda i: self.set_param("save", i)) 
        self.connect(self.centerspanstartstopCheckBox, SIGNAL("stateChanged(int)"),self.on_centerspanstartstopChanged)
        
        self.datapath='S:\\_Data\\'
        self.datapathButton.clicked.connect(self.selectDatapath)
        self.datapathLineEdit.textChanged.connect(self.update_filenumber)
        self.prefixLineEdit.textChanged.connect(self.update_filenumber)
        
        #Initialize Parameters
        self.set_param("power", -20)
        self.set_param("sweep_pts",1601)
        self.set_param("ifbw",1e3)
        self.set_param("avgs",1)
        self.set_param("centerstart",10.)
        self.set_param("spanstop",1000.)
        self.set_param("centerspanstartstop",1)
        self.set_param("resolution", 100.)
        self.set_param("save",0)
        self.set_param("datapath",self.datapath)
        self.set_param("prefix",'trace')
        self.set_param("filenumber",0)
        
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

    def selectDatapath(self):
        self.datapath=str(QFileDialog.getExistingDirectory(self,'Open Datapath',self.datapath))
        self.datapathLineEdit.setText(self.datapath)       
        
    def update_filenumber(self):
        filenumber=next_file_index(self.datapath,str(self.prefixLineEdit.text()))
        self.filenumberLabel.setText("%04d_" % filenumber)
        self.set_param("datapath",self.datapath)
        self.set_param("filenumber",filenumber)
        self.set_param("prefix",str(self.prefixLineEdit.text()))

    def on_centerspanstartstopChanged(self,state):
        if state: 
            self.centerspanstartstopCheckBox.setText('Start/Stop')
            self.centerstartLabel.setText('Center')
            self.spanstopLabel.setText('Span')

            self.spanstopSpinBox.setSuffix(' MHz')
            center=(self.centerstartSpinBox.value()+self.spanstopSpinBox.value())/2.
            span=(self.spanstopSpinBox.value()-self.centerstartSpinBox.value())*1e3
            self.centerstartSpinBox.setValue(center)
            self.spanstopSpinBox.setValue(span)
            self.set_param("centerstart",center)
            self.set_param("spanstop",span)
        else:
            self.centerspanstartstopCheckBox.setText('Center/Span')
            self.centerstartLabel.setText('Start')
            self.spanstopLabel.setText('Stop')

            self.spanstopSpinBox.setSuffix(' GHz')
            start=(self.centerstartSpinBox.value()-self.spanstopSpinBox.value()/1e3/2.)
            stop=(self.centerstartSpinBox.value()+self.spanstopSpinBox.value()/1e3/2.)
            self.centerstartSpinBox.setValue(start)
            self.spanstopSpinBox.setValue(stop)
            self.set_param("centerstart",start)
            self.set_param("spanstop",stop)
        

if __name__ == "__main__":
    app = QApplication([])
    win = NWAWin()
    win.show()
    sys.exit(app.exec_())