# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:53:17 2011

@author: Dave
"""


from slab import *
from slab.gui import *
from TSweep_ui import *
from guiqwt.builder import make
import os



class nwa_DataThread(DataThread):
    
    def run_script(self):
        try:
            nwa=self.instruments['NWA']
        except:
            self.msg("NWA config not loaded!")
            return
        try:
            fridge=self.instruments['FRIDGE']
        except:
            self.msg("Cryostat config not loaded!")
            return

        self.msg("Configuring NWA....")
        self.msg(str(nwa.get_id()))

        nwa.set_ifbw(self.params['ifbw'])
        nwa.set_power(self.params['power'])
        nwa.set_averages(self.params['avgs'])

        #calculate start and stop frequencies
        if self.params["centerspanstartstop"]:
            start=self.params['centerstart']*1e9-self.params['spanstop']*1e6/2.
            stop=self.params['centerstart']*1e9+self.params['spanstop']*1e6/2.
        else:
            start=self.params['centerstart']*1e9
            stop=self.params['spanstop']*1e9
            
        self.msg('Commencing normal Sweep...')
        #calculate start and stop frequencies       
        nwa.set_start_frequency(start)
        nwa.set_stop_frequency(stop)       
        nwa.set_sweep_points(self.params['sweep_pts'])

        self.msg("Acquiring data...")
        
        if self.params['save']:
            datapath=make_datapath(self.params['exptpath'],prefix)
               
        #im.save_settings(datapath,prefix,{'delay':0})
        
        print("Taking temperature data")
        count=0
        while(True):
            Temperature=fridge.get_temperature('MC RuO2')
            if not Temperature >0:
                Temperature=fridge.get_temperature('MC cernox')
            self.plots['temperatureLabel'].setText('Current T: %0.2f K' % Temperature )
            if self.params['save']:
                fname=os.path.join(datapath,"%04d_%s_%3.3f.csv" % (count,prefix,Temperature))
            else:
                fname=None
            self.msg("Count: %d\tTemperature: %3.3f" %(count,Temperature))
            freqs,mags,phases=nwa.take_one_averaged_trace(fname)
            
            # update plots etc.            
            self.plots["datacurve"].set_data(freqs/1e9,mags)
            self.plots["dataplot"].replot()
            
            if self.params['auto']:
                self.plots['tracenum'].setValue(count)


            time.sleep(self.params['delay'])
            count+=1
            if self.aborted():
                self.msg("aborted")
                break

class TSweepWin(SlabWindow, Ui_TSweepWindow):
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
        self.connect(self.saveCheckBox, SIGNAL("stateChanged(int)"),
                lambda i: self.set_param("save", i)) 
        self.connect(self.centerspanstartstopCheckBox, SIGNAL("stateChanged(int)"),self.on_centerspanstartstopChanged)

        self.connect(self.autoCheckBox, SIGNAL("stateChanged(int)"),
                lambda i: self.set_param("auto", i)) 
        self.connect(self.traceNumSpinBox, SIGNAL("valueChanged(int)"),
                lambda i: self.set_param("tracenum", i)) 

        
        self.exptpath='S:\\_Data\\'
        self.exptpathButton.clicked.connect(self.selectexptpath)
        self.exptpathLineEdit.textChanged.connect(self.update_filenumber)
        self.prefixLineEdit.textChanged.connect(self.update_filenumber)
        
        #Initialize Parameters
        self.set_param("power", -20)
        self.set_param("sweep_pts",1601)
        self.set_param("ifbw",1e3)
        self.set_param("avgs",1)
        self.set_param("centerstart",10.)
        self.set_param("spanstop",1000.)
        self.set_param("centerspanstartstop",1)
        self.set_param("save",0)
        self.set_param("exptpath",self.exptpath)
        self.set_param("prefix",'trace')
        self.set_param("filenumber",0)
        self.set_param("delay",0)
        
        self.set_param("tracenum",0)
        self.set_param("auto",True)

        #Proxy widgets into datathread
        self.plot_manager['tracenum']=self.traceNumSpinBox
        self.plot_manager['temperatureLabel']=self.temperatureLabel
        
        
        #Make some default data
        self.freqs=np.linspace(self.params['centerstart']-self.params['spanstop']/2.,self.params['centerstart']+self.params['spanstop']/2.,self.params['sweep_pts'])        
        self.mags=np.zeros(len(self.freqs)) 
        self.phases=np.zeros(len(self.freqs))
        
        #Initialize Plots
        self.plot_manager["dataplot"] = self.dataCurveWidget.plot        
        self.dataCurveWidget.add_toolbar(self.addToolBar("Curve"))
        self.dataCurveWidget.register_all_image_tools()
        self.plot_manager["dataplot"].set_titles(title="Data Curve", xlabel="Frequency (GHz)", ylabel="S21")       
        self.plot_manager["datacurve"] = make.mcurve(self.freqs, self.mags,label='Magnitude') #Make Ch1 curve
        self.plot_manager["dataplot"].add_item(self.plot_manager["datacurve"])


  
    def selectexptpath(self):
        self.exptpath=str(QFileDialog.getExistingDirectory(self,'Open Experiment Path',self.exptpath))
        self.exptpathLineEdit.setText(self.exptpath)       
        
    def update_filenumber(self):
        filenumber=next_file_index(self.exptpath,str(self.prefixLineEdit.text()))
        self.filenumberLabel.setText("%04d_" % filenumber)
        self.set_param("exptpath",self.exptpath)
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
    win = TSweepWin()
    win.show()
    sys.exit(app.exec_())