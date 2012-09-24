# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:10:50 2011

@author: dave
"""
from slab import *
from slab.gui import *
#from NWAWindow_ui import *
from guiqwt.builder import make
import os

import h5py
from guiqwt.qtdesigner import loadui
from slab.datamanagement import *
Ui_NWAWindow = loadui("c:\\_Lib\\python\\slab\\scripts\\NWAWindow.ui")
import inspect

class nwa_DataThread(DataThread):

    def open_datafile(self):
        try: 
            self.file = SlabFile(os.path.join(self.params['datapath'],self.filename))
        except Exception as e:
            self.msg("Could not open h5 file!")
            self.msg(e)
            return
        if 'settings' not in self.file:
            self.file.save_settings(dic=self.params)
        return self.file
      
#        if self.params['numberTraces']:
#            self.trace_no = get_next_trace_number(self.file)
#            self.plots['trace'].setText("%03d" % self.trace_no)
                       
#This should probably be moved into the Window class and probably we can make it 
#an automatic thing so we don't have to code it for each new script
    def save_defaults(self):  
        try:
            settings_file= SlabFile('c:\\_Lib\\python\\slab\\scripts\\NWAWindow_defaults.h5')
            settings_file.save_settings(self.params)
            settings_file.close()
        except Exception as e:
            self.msg("Could not open NWAWindow_defaults.h5!")
            self.msg(e)            
        return
            
        
        
    def do_normal_sweep(self,nwa,start,stop,sweep_pts):
        self.msg('Commencing normal Sweep...')
        #calculate start and stop frequencies      

        nwa.set_start_frequency(start)
        nwa.set_stop_frequency(stop)       
        nwa.set_sweep_points(sweep_pts)

        self.msg("Acquiring data...")
        freqs,mags,phases=nwa.take_one_averaged_trace()
        xrng, yrng = ((freqs[0],freqs[-1]),(start, stop))
        self.msg("Data acquisition complete.")
        self.plots["mag"].set_data(freqs/1e9,mags)
        self.plots["phase"].set_data(freqs/1e9,phases)
        self.plots["magplot"].replot()
        self.plots["phaseplot"].replot()   
        if self.params['autoscale']:
            self.plots["magplot"].do_autoscale()
            self.plots["phaseplot"].do_autoscale()
        self.msg(mags[1:5])
        self.msg(freqs[1:5])
        if self.params['save']:
            f=self.open_datafile()
            for n, d in [("mag", mags), ("phase", phases)]:
                if n not in f:
                    ds=f.create_dataset(n,shape=(1,len(d)),maxshape=(None,len(d)))

                    set_labels(ds, "Frequency (Hz)", "Response")
                else:
                    ds=f[n]
                    ds.resize((ds.shape[0]+1,ds.shape[1]))
                ds[ds.shape[0]-1,:]=d
            set_range(ds, start, stop,0,ds.shape[0]-1)
            f.close()
#            f = self.file[self.trace_no] if self.params["numberTraces"] else self.file
#            for n, d in [("mag", mags), ("phase", phases)]:
#                f[n] = d
#                set_range(f[n], start, stop)
#                set_labels(f[n], "Frequency (Hz)", "Response")
#            f.close()
#            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
#            np.savetxt(os.path.join(self.params['datapath'],fname),transpose(array([freqs,mags,phases])),delimiter=',')

    def do_segmented_sweep(self,nwa,start,stop,step):
        self.msg('Commencing segmented Sweep...')
        span=stop-start
        total_sweep_pts=span/step
        if total_sweep_pts<1600:
            print "Segmented sweep unnecessary"
        segments=np.ceil(total_sweep_pts/1600.)
        total_sweep_pts=segments*1600+1
        segspan=span/segments
        starts=start+segspan*np.arange(0,segments)
        stops=starts+segspan
        
        self.msg('Segments: %d\t Segment Span: %f MHz' % (segments,segspan/1e6))
              
        old_timeout=nwa.get_timeout()
        nwa.set_timeout(10000)
        nwa.set_trigger_average_mode(True)
        nwa.set_trigger_source('BUS')
    
        nwa.set_span(segspan)
        segs=[]
#        if self.params['save']:
#            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
        maxstop=0
        for start,stop in zip(starts,stops):
            maxstop=max(maxstop,stop)
            nwa.set_start_frequency(start)
            nwa.set_stop_frequency(stop)
            nwa.set_format('mlog')
    
            nwa.clear_averages()
            nwa.trigger_single()
            time.sleep(nwa.query_sleep)
            nwa.averaging_complete()    #Blocks!
            nwa.set_format('slog')
            
            seg_data=nwa.read_data()
    
            if stop<stops[-1]:
                seg_data=seg_data.transpose()
                seg_data=seg_data[:-1].transpose()
            segs.append(seg_data)
            data=np.hstack(segs) 

            self.plots["mag"].set_data(data[0]/1e9,data[1])
            self.plots["phase"].set_data(data[0]/1e9,data[2])
            self.plots["magplot"].replot()
            self.plots["phaseplot"].replot()
            if self.params['autoscale']:
                self.plots["magplot"].do_autoscale()
                self.plots["phaseplot"].do_autoscale()

#            if self.params['save']:
#                np.savetxt(os.path.join(self.params['datapath'],fname),transpose(data),delimiter=',')
            self.msg('data Length: %d' % len(data[1]))
            
            if self.params['save']:
                f=self.open_datafile()
                for n, d in [("mag", data[1]), ("phase", data[2])]:
                    if n not in f:
                        ds=f.create_dataset(n,shape=(1,len(d)),maxshape=(None,total_sweep_pts+1))
                        set_labels(ds, "Frequency (Hz)", "Response")
                    else:
                        ds=f[n]
                        if start == starts[0]:
                            ds.resize((ds.shape[0]+1,ds.shape[1]))
                            
                    if ds.shape[1]<len(d):
                        ds.resize((ds.shape[0],len(d)))
                    ds[ds.shape[0]-1,:len(d)]=d
                set_range(ds, starts[0], maxstop,0,ds.shape[0]-1)
                f.close()            
            if self.aborted():
                self.msg("aborted")
                return

        time.sleep(nwa.query_sleep)
        nwa.set_timeout(old_timeout)
        nwa.set_format('mlog')
        nwa.set_trigger_average_mode(False)
        nwa.set_trigger_source('INTERNAL')

#        if self.params['save']:
#            fname=get_next_filename(self.params['datapath'],self.params['prefix'],'.csv')
#            np.savetxt(os.path.join(self.params['datapath'],fname),transpose(data),delimiter=',')
        self.msg('Segmented scan complete.')
            

    def run_script(self):#self.attrs["_script"] = open(sys.argv[0], 'r').read()

        try:
            nwa=self.instruments['NWA']
        except Exception as E:
            self.msg("NWA config not loaded!")
            self.msg("loaded values:", self.instruments.keys())
            self.msg(E)
            return
        self.msg("Configuring NWA....")
        self.msg(str(nwa.get_id()))
        first=True
        nwa.set_output(state=True)
        if self.params['save']:                
            if self.params['filename'][-3:].lower() != '.h5': 
                self.params['filename']=self.params['filename']+'.h5'
        self.filename=get_next_filename(self.params['datapath'],self.params['filename'])

        while ((first or self.params['autorun']) and not self.aborted()):
            first=False
            #self.plots["self"].update_filenumber()
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
        SlabWindow.__init__(self, nwa_DataThread, config_file='c:\\_Lib\\python\\slab\\scripts\\instruments.cfg')
        self.setupSlabWindow(autoparam=True)
        self.register_script("run_script", self.go_button, self.abort_button)
        self.datapathButton.clicked.connect(self.selectDatapath)
        
        #self.connect(self.param_centerspanstartstop, SIGNAL("stateChanged(int)"),self.on_centerspanstartstopChanged)
        self.param_centerspanstartstop.stateChanged.connect(self.on_centerspanstartstopChanged)
        self.param_datapath.textChanged.connect(self.update_filenumber)
        self.param_filename.textChanged.connect(self.update_filenumber)
        self.go_button.clicked.connect(self.update_filenumber)
        self.go_button.clicked.connect(self.save_defaults)
        
        self.start_thread()       
        self.load_defaults()
        
        self.plots["self"] = self
        self.plots["trace"] = self.trace_label
        
        #Make some default data
        self.freqs=np.linspace(self.params['centerstart']-self.params['spanstop']/2.,self.params['centerstart']+self.params['spanstop']/2.,self.params['sweep_pts'])        
        self.mags=np.zeros(len(self.freqs)) 
        self.phases=np.zeros(len(self.freqs))
        
        #Initialize Magnitude plot"
        self.plots["magplot"] = self.magplotwidget.plot        
        self.magplotwidget.add_toolbar(self.addToolBar("Curve"))
        self.magplotwidget.register_all_image_tools()
        self.plots["magplot"].set_titles(title="Magnitude", xlabel="Frequency (GHz)", ylabel="S21")       
        self.plots["mag"] = make.mcurve(self.freqs, self.mags,label='Magnitude') #Make Ch1 curve
        self.plots["magplot"].add_item(self.plots["mag"])

        #Initialize Phase plot
        self.plots["phaseplot"] = self.phaseplotwidget.plot
        self.phaseplotwidget.add_toolbar(self.addToolBar("Curve"))
        self.phaseplotwidget.register_all_image_tools()
        self.plots["phaseplot"].set_titles(title="Phase", xlabel="Frequency (GHz)", ylabel="Phase (deg)")
        self.plots["phase"] = make.mcurve(self.freqs, self.phases,label='Phase') #Make Ch1 curve
        self.plots["phaseplot"].add_item(self.plots["phase"])
        


    def load_defaults(self):
        try: 
            settings_file= SlabFile('c:\\_Lib\\python\\slab\\scripts\\NWAWindow_defaults.h5','r')
        except Exception as e:
            self.msg("Could not open NWAWindow_defaults.h5!")
            self.msg(e)            
            return
        d=settings_file.load_settings('settings')
        for k in d.keys():
            self.set_param(k,d[k])
            #self.msg(k,': ',str(self.params[k]))
        settings_file.close()
          
    def save_defaults(self):  
        try:
            settings_file= SlabFile('c:\\_Lib\\python\\slab\\scripts\\NWAWindow_defaults.h5')
            settings_file.save_settings(self.params)
            settings_file.close()
        except Exception as e:
            self.msg("Could not open NWAWindow_defaults.h5!")
            self.msg(e)            
        return



    def selectFile(self):
        self.param_filename.setText(str(QFileDialog.getSaveFileName(self)))
        
    def selectDatapath(self):
        self.datapath=str(QFileDialog.getExistingDirectory(self,'Open Datapath',self.datapath))
        self.param_datapath.setText(self.datapath)       
        
    def update_filenumber(self):
        self.datapath=self.params['datapath']
        filenumber=next_file_index(self.datapath,str(self.param_filename.text()))
        self.trace_label.setText("%04d_" % filenumber)
        self.set_param("datapath",self.datapath)
        self.set_param("filenumber",filenumber)
        self.set_param("prefix",str(self.param_filename.text()))

    def on_centerspanstartstopChanged(self,state):
        if state: 
            self.param_centerspanstartstop.setText('Start/Stop')
            self.centerstartLabel.setText('Center')
            self.spanstopLabel.setText('Span')

            self.param_spanstop.setSuffix(' MHz')
            center=(self.param_centerstart.value()+self.param_spanstop.value())/2.
            span=(self.param_spanstop.value()-self.param_centerstart.value())*1e3
            self.param_centerstart.setValue(center)
            self.param_spanstop.setValue(span)
            self.set_param("centerstart",center)
            self.set_param("spanstop",span)
        else:
            self.param_centerspanstartstop.setText('Center/Span')
            self.centerstartLabel.setText('Start')
            self.spanstopLabel.setText('Stop')

            self.param_spanstop.setSuffix(' GHz')
            start=(self.param_centerstart.value()-self.param_spanstop.value()/1e3/2.)
            stop=(self.param_centerstart.value()+self.param_spanstop.value()/1e3/2.)
            self.param_centerstart.setValue(start)
            self.param_spanstop.setValue(stop)
            self.set_param("centerstart",start)
            self.set_param("spanstop",stop)
        

if __name__ == "__main__":
    #print inspect.stack()[-1][1]
    app = QApplication([])
    win = NWAWin()
    win.show()
    sys.exit(app.exec_())