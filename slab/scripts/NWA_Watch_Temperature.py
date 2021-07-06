# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:06:58 2011

@author: Dave
"""
import datetime
import json
from gui import *
from slablayout import FormWidget,FormTabWidget,FormComboWidget,fedit
from nwa import *
from cryostat import *
from dataanalysis import *
#import multiprocessing


#na,fridge,datapath,fileprefix,windows,powers,ifbws,avgs,timeout=10000,delay=0

nwaformdata=[('nwa_address','Network Analyzer IP','nwa.circuitqed.com'),
    ('fridge_address','Fridge IP', 'fridge.circuitqed.com'),
    ('datapath','Data Path','S:\\_Data\\110804 - Bing Cavity with Indium seal\\001 - test NWA Watch Temperature\\'),
    ('fileprefix','File Prefix','NWA_temp_sweep'),
    ('center','Center Frequency',10e9),
    ('span','Span',20e9),
    ('power','Power',-30.),
    ('ifbw','IF Bandwidth',100e3),
    ('avgs','Averages',16),
    ('timeout','Timeout',10000),
    ('delay','Delay between measurements',0)
    ]

class nwawatchtemperatureDataSet(DataSet):

    bgm1 = BeginGroup("Instrument Properties")
    fridgeaddress=StringItem("Fridge Address",default='fridge.circuitqed.com')   
    nwaaddress=StringItem("NWA Address",default='nwa.circuitqed.com')   
    egm1 = EndGroup("Instrument Properties")

    bgm2 = BeginGroup("File Properties")
    datapath = DirectoryItem("Directory", os.path.dirname(''))
    fileprefix = StringItem("File Prefix")
#    filenumber= IntItem("File #",default=1,min=0).set_pos(col=1)
    egm2 = EndGroup("File Properties")    

    bgm3=BeginGroup("Acquisition Properties")
    center = FloatItem("Center Frequency (Hz)",default=12.4458e9)
    span = FloatItem ("Span (Hz)",default=330e3)
    power = FloatItem("Power (dBm)",default=-20.)
    ifbw = FloatItem("IF Bandwidth (Hz)",default=1e3)
    avgs = IntItem("Averages",default=16)
    timeout = FloatItem("Timeout", default=10000)
    delay = FloatItem("Delay",default=0)
    egm3 = EndGroup("Acquisition Properties")
   


class NWAWindow(SlabWindow):
    
    def setupWidgets(self):
        #self.guidata=FormWidget(nwaformdata)
        self.guidata=DataSetEditGroupBox("Properties", nwawatchtemperatureDataSet,button_text='Go')        
        self.connect(self.guidata, SIGNAL("apply_button_clicked()"),
                     self.go)

        self.paramswidget.addWidget(self.guidata)
        #self.button=QPushButton("Apply",self.paramswidget)      
        #self.connect(self.button,SIGNAL("clicked()"),self.go)
       
        #Curve widgets
        x=np.linspace(-5,5,1000)        #Create some sample curves
        y1=np.cos(x)
        y2=np.sin(x)

        self.curvewidget = CurveWidget(self,title='Log Mag',xlabel='Frequency (GHz)',ylabel='Transmission, S21 (dB)')
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        
        self.plot = self.curvewidget.plot   
        self.ch1_plot = make.mcurve(x, y1, label="Magnitude") #Make Ch1 curve
        self.plot.add_item(self.ch1_plot)
        self.add_plot_updater("Mag","line_plot",self.plot,self.ch1_plot)    #Add plot autoupdater
        self.plotswidget.addWidget(self.curvewidget)    #Add the plot to the plots widget
        self.p=None

        self.curvewidget2 = CurveWidget(self,title='Phase',xlabel='Frequency (GHz)',ylabel='Phase (deg)')
        self.curvewidget2.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget2.register_all_image_tools()

        self.plot2 = self.curvewidget2.plot   
        self.ch2_plot = make.mcurve(x, y2, label="Phase") #Make Ch2 curve
        self.plot2.add_item(self.ch2_plot)
        self.plot2.add_item(make.legend('TR'))           #Add legend       

        self.add_plot_updater("Phase","line_plot",self.plot2,self.ch2_plot) 
        self.plotswidget.addWidget(self.curvewidget2)    #Add the plot to the plots widget


    def save_bookmark(self,filename):
        print(json.dumps(self.guidata.get()))

    def go(self):
        print("GO!")
        #d=self.guidata.get()
        #d={'fileprefix': 'NWA_temp_sweep', 'center': 10000000000.0, 'power': -30.0, 'ifbw': 1000.0, 'datapath': 'S:\\_DataH804 - Bing Cavity with Indium seal\\x01 - test NWA Watch Temperature\\', 'delay': 0, 'timeout': 10000, 'fridge_address': 'fridge.circuitqed.com', 'span': 20000000000.0, 'nwa_address': 'nwa.circuitqed.com', 'avgs': 16}
        #print d
        print(self.guidata.dataset.datapath)
            
        self.p = Process(target=acquire_data, args=(self.guidata.dataset,self.pipes))
        self.p.start()
        print("Process started")
        #print self.guidata.get()
        self.statusBar().showMessage("Acquiring Data", 5000)

#def nwa_watch_temperature_sweep(na,fridge,datapath,fileprefix,windows,powers,ifbws,avgs,timeout=10000,delay=0):
#    """nwa_watch_temperature_sweep monitors the temperature (via fridge) and tells the network analyzer (na) to watch certain windows 
#    at the specified powers, ifbws, and avgs specified
#    windows= [(center1,span1),(center2,span2), ...]
#    powers= [power1,power2,...]
#    ifbws=[ifbw1,ifbw2,...]
#    avgs=[avgs1,avgs2,...]"""

def acquire_data(params,pipes=None):
#    datapath=params['datapath']
#    fileprefix=params['fileprefix']
#    center=params['center']
#    span=params['span']
#    power=params['power']
#    ifbw=params['ifbw']
#    avgs=params['avgs']
#    timeout=params['timeout']
#    delay=params['delay']
    #datapath=params.datapath+'/'
    #fileprefix=params.fileprefix
    fileprefix= 'NWA_temp_sweep'
    datapath= 'S:\\_Data\\110804 - Bing Cavity with Indium seal\\002 - Bing Cavity on the way down\\'
    f=open(datapath+fileprefix+".cfg",'w')
    f.write('Getting parameters\n')
    nwaaddress=params.nwaaddress
    fridgeaddress=params.fridgeaddress
    center=params.center
    span=params.span
    power=params.power
    ifbw=params.ifbw
    avgs=params.avgs
    timeout=params.timeout
    delay=params.delay
    f.write('Got Parameters\n')



    f.write ("Acquiring data\n")
    f.close()
    f=open(datapath+fileprefix+".cfg",'a')
    if pipes!=None:    
        mag_pipe=pipes['Mag']
        phase_pipe=pipes['Phase']
        mag_pipe.send((zeros(100),zeros(100)))
    f.write( "Connecting to Network Analyzer\n")
    na=E5071(address=nwaaddress)
    f.write( "Connecting to Fridge\n")
    fridge=Triton(address=fridgeaddress)

    f.write( "Configuring Network Analyzer\n")
  #  f.write('datapath: %s\nfileprefix: %s\nCenter: %f\nSpan: %f\tPower: %f\tIFBW: %f\tavgs: %d' % (datapath,fileprefix,center,span,power,ifbw,avgs))
    f.close()
    f=open(datapath+fileprefix+".cfg",'a')
    na.set_trigger_average_mode(True)
    na.set_sweep_points()
    na.set_average_state(True)
    na.set_timeout(timeout)
    na.set_format('slog')
    na.set_trigger_source('BUS')
    na.set_center_frequency(center)
    na.set_span(span)
    na.set_power(power)
    na.set_ifbw(ifbw)
    na.set_averages(avgs)
    f.write("Configured Network Analyzer\n")

    f.write("Starting to acquire data\n")
    f.close()
    count=0
    Temperature=300
    while (Temperature>0.022):
        Temperature=fridge.get_temperature('MC RuO2')
        if not Temperature >0:
            Temperature=fridge.get_temperature('MC cernox')
        f=open(datapath+fileprefix+".cfg",'a')
        filename="%s%04d-%s-%3.3f.CSV" % (datapath,count,fileprefix.upper(),Temperature)
        f.write( "Trace: %d\tTemperature: %3.3f\tFilename: %s\n" % (count,Temperature,filename))
        f.close()
        na.trigger_single()
        na.averaging_complete()    #Blocks!
        na.save_file(filename)
        time.sleep(10)
        freqs,mags,phases = load_nwa_file(filename)
        if pipes!=None: mag_pipe.send((freqs/1e9,mags))
        if pipes!=None: phase_pipe.send((freqs/1e9,phases))
        time.sleep(delay)
        count+=1
            
        
if __name__ == "__main__":        
#    d=fedit(nwaformdata)
#    fedit()
#    print d
    #acquire_data(d)
#    acquire_data({'fileprefix': 'NWA_temp_sweep', 'center': 10000000000.0, 'power': -30.0, 'ifbw': 1000.0, 'datapath': 'S:\\_Data\\110804 - Bing Cavity with Indium seal\\001 - test NWA Watch Temperature\\', 'delay': 0, 'timeout': 10000, 'fridge_address': 'fridge.circuitqed.com', 'span': 20000000000.0, 'nwa_address': 'nwa.circuitqed.com', 'avgs': 16})
    app=guidata.qapplication()
    window=NWAWindow(title="NWA Watch Temperature Sweep",size=(1000,600))
    window.show()
    app.exec_()