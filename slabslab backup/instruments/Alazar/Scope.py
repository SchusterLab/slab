# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:29:46 2011

@author: Phil
"""
from gui import *
from instruments.Alazar.TestAlazarApi import *

class ScopeDataSet(DataSet):
    bgm1=BeginGroup("Acquisition Properties")
    samples= IntItem("Samples",default = 8192, min=0)
    records= IntItem("Records per buffer",default=4,min=0).set_pos(col=1)
    buffers= IntItem("Buffers per acquisition",default=1000,min=0)#.set_pos(col=2)
    averages= IntItem("Averages",default=1,min=1)
    egm1 = EndGroup("Acquisition Properties")
    
    bgm2 = BeginGroup("Horizontal Properties")    
    clock_type=ChoiceItem("Clock Type",[(0,'Fast AC'),(1,'Slow'),(3,'10 MHz PLL'),(4,'Internal')])
    clock_decimation=IntItem("Decimation",default=1,min=1).set_pos(col=1)
    egm2 = EndGroup("Horizontal Properties")    
    
    bgm3 = BeginGroup("Vertical Properties")    
    scale= ChoiceItem('Scale',[(0,'2 mV'),(1,'5 mV'),(2,'10 mV'),(3,'20 mV'),(4,'50 mV'),(5,'100 mV'),(5,'200 mV'),(6,'500 mV'),(7,'1 V'),(8,'2 V'),(9,'5 V'),(10,'10 V'),(11,'20 V')])
    coupling = ChoiceItem('Input Coupling', [(0,'DC'),('1','AC')]).set_pos(col=1)
    egm3 = EndGroup("Vertical Properties")
    
    bgm4 = BeginGroup("Trigger Properties")
    source= ChoiceItem('Source',[(0,"CH 1"),(1,'CH 2'),(2,'External'),(3,'Disabled')])    
    level = IntItem('Level (%)',default=50).set_pos(col=1)
    egm4 = EndGroup("Trigger Properties")
    
    bgm5 = BeginGroup("File Properties")
    fileprefix = FileSaveItem("File Prefix")
    filenumber= IntItem("File #",default=1,min=0).set_pos(col=1)
    egm5 = EndGroup("File Properties")    

class ScopeWindow(SlabWindow):
    
    def setupWidgets(self):
        self.guidata=DataSetEditGroupBox("Properties", ScopeDataSet,button_text='Go')
        #self.guidata.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.connect(self.guidata, SIGNAL("apply_button_clicked()"),
                     self.go)
        self.paramswidget.addWidget(self.guidata)
       
        #Oscilloscope widget
        self.curvewidget = CurveWidget(self,title='Oscilloscope',xlabel='Time (ns)',ylabel='Voltage (V)')
        self.plot = self.curvewidget.plot
        x=np.linspace(-5,5,1000)
        y1=np.cos(x)
        y2=np.sin(x)
        self.ch1_plot = make.mcurve(x, y1, label="Ch1")
        self.plot.add_item(self.ch1_plot)
        self.ch2_plot = make.mcurve(x, y2, label="Ch2")
        self.plot.add_item(self.ch2_plot)
        self.plot.add_item(make.legend('TR'))
        
        self.add_plot_updater("Ch1","line_plot",self.plot,self.ch1_plot)        
        self.add_plot_updater("Ch2","line_plot",self.plot,self.ch2_plot)        

        self.curvewidget.register_all_image_tools()
        self.plotswidget.addWidget(self.curvewidget)


    def go(self):
        print("GO!")
        p = Process(target=acquire_data, args=(self.guidata.dataset,self.pipes,))
        p.start()
        print("Process started")
        self.statusBar().showMessage("Acquiring Data", 5000)
        p.join()
 
 
def acquire_data(params,pipes=None):
    print("Acquiring data")
    if pipes!=None:    
        ch1_pipe=pipes['Ch1']
        ch2_pipe=pipes['Ch2']
    card = Card()
    x = linspace(-5, 5, card.samples)
    card.configure()
    for ii in range(4):
        card.acquire()
        y1 = card.cur_result[0]
        if pipes!=None:
            ch1_pipe.send((x,y1))
            #ch2_pipe.send((x,y2))
        
if __name__ == "__main__":        
    app=guidata.qapplication()
    window=ScopeWindow(title="Test Scope",size=(1000,600))
    window.show()
    app.exec_()