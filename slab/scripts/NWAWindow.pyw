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
    
    def do_normal_sweep(self,nwa,start,stop,sweep_pts):
        self.msg('Commencing normal Sweep...')
        #calculate start and stop frequencies       
        nwa.set_start_frequency(start)
        nwa.set_stop_frequency(stop)       
        nwa.set_sweep_points(sweep_pts)

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

    def do_segmented_sweep(self,nwa,start,stop,step):
        self.msg('Commencing segmented Sweep...')
        span=stop-start
        total_sweep_pts=span/step
        if total_sweep_pts<1600:
            print "Segmented sweep unnecessary"
        segments=np.ceil(total_sweep_pts/1600.)
        segspan=span/segments
        starts=start+segspan*np.arange(0,segments)
        stops=starts+segspan
        
        self.msg('Segments: %d\t Segment Span: %f MHz' % (segments,segspan/1e6))
              
        old_timeout=nwa.get_timeout()
        nwa.set_timeout(10000)
        nwa.set_trigger_average_mode(True)
        nwa.set_trigger_source('BUS')
        nwa.set_format('mlog')
    
        nwa.set_span(segspan)
        segs=[]
        if self.params['save']:
            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
            np.savetxt(fname,transpose(data),delimiter=',')
        for start,stop in zip(starts,stops):
            nwa.set_start_frequency(start)
            nwa.set_stop_frequency(stop)
    
            nwa.clear_averages()
            nwa.trigger_single()
            time.sleep(nwa.query_sleep)
            nwa.averaging_complete()    #Blocks!
            nwa.set_format('slog')
            
            seg_data=nwa.read_data()

            seg_data=seg_data.transpose()
            last=seg_data[-1]
            seg_data=seg_data[:-1].transpose()
            segs.append(seg_data)
            data=np.hstack(segs) 
            self.plots["mag"].set_data(data[0]/1e9,data[1])
            self.plots["phase"].set_data(data[0]/1e9,data[2])
            self.plots["magplot"].replot()
            self.plots["phaseplot"].replot()
            if self.params['save']:
                np.savetxt(fname,transpose(data),delimiter=',')
            if self.aborted():
                self.msg("aborted")
                return


            
        segs.append(np.array([last]).transpose())
        data=np.hstack(segs) 
        time.sleep(nwa.query_sleep)
        nwa.set_timeout(old_timeout)
        nwa.set_format('mlog')
        nwa.set_trigger_average_mode(False)
        nwa.set_trigger_source('INTERNAL')

        if self.params['save']:
            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
            np.savetxt(fname,transpose(data),delimiter=',')
            

    def run_script(self):
        try:
            nwa=self.instruments['NWA']
        except:
            self.msg("NWA config not loaded!")
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
            
        #Calculate resolution
        minstep=self.params['resolution']*1e6
        step=(stop-start)/self.params['sweep_pts']
        self.msg('start: %f GHz\tstop: %f GHz\tstep: %f MHz' % (start/1e9,stop/1e9,step/1e6))
        if step <= minstep:
            self.do_normal_sweep(nwa,start,stop,self.params['sweep_pts'])
        elif step > minstep:
            step=minstep
            if (stop-start)/minstep <= nwa.MAXSWEEPPTS:
                sweep_pts=(stop-start)/minstep 
                self.do_normal_sweep(nwa,start,stop,sweep_pts)
            else:
                self.do_segmented_sweep(nwa,start,stop,step)
        

class NWAWin(SlabWindow, Ui_NWAWindow):
    def __init__(self):
        SlabWindow.__init__(self, nwa_DataThread, config_file='instruments.cfg')
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