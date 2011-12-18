# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:06:58 2011

@author: Dave
"""
from gui import *
from slablayout import FormWidget,FormTabWidget,FormComboWidget
import datetime
import json
#from TestAlazarApi import *

ds=[('str', 'this is a string'),
                ('list', [0, '1', '3', '4']),
                ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
                           ('-.', 'DashDot'), ('-', 'Solid'),
                           ('steps', 'Steps'), (':', 'Dotted')]),
                ('float', 1.2),
                (None, 'Other:'),
                ('int', 12),
                ('font1', ('Arial', 10, False, True)),
                ('color', '#123409'),
                ('bool', True),
                ('date', datetime.date(2010, 10, 10)),
                ('datetime', datetime.datetime(2010, 10, 10)),
                ]

ds2=[('name','label','value'),
     ('str1','String 1','This is string 1'),
     ('str2','String 2','This is string 2'),
     ('list1','List 1',[0,'1','3','4']),
     ]

class ScopeWindow(SlabWindow):
    
    def setupWidgets(self):
        self.guidata=FormComboWidget([(ds2,'box 1','Params 1'),(ds2,'box 2', 'Params 2')])
        #self.connect(self.guidata, SIGNAL("apply_button_clicked()"),
        #             self.go)
        self.paramswidget.addWidget(self.guidata)
        self.button=QPushButton("Apply",self.paramswidget)
        self.connect(self.button,SIGNAL("clicked()"),self.go)
       
        #Oscilloscope widget
        self.curvewidget = CurveWidget(self,title='Oscilloscope',xlabel='Time (ns)',ylabel='Voltage (V)')
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        
        self.plot = self.curvewidget.plot   
        x=np.linspace(-5,5,1000)        #Create some sample curves
        y1=np.cos(x)
        y2=np.sin(x)
        self.ch1_plot = make.mcurve(x, y1, label="Ch1") #Make Ch1 curve
        self.plot.add_item(self.ch1_plot)
        self.ch2_plot = make.mcurve(x, y2, label="Ch2") #Make Ch2 curve
        self.plot.add_item(self.ch2_plot)
        self.plot.add_item(make.legend('TR'))           #Add legend
        
        self.add_plot_updater("Ch1","line_plot",self.plot,self.ch1_plot)    #Add plot autoupdater
        self.add_plot_updater("Ch2","line_plot",self.plot,self.ch2_plot) 

        self.plotswidget.addWidget(self.curvewidget)    #Add the plot to the plots widget

    def save_bookmark(self,filename):
        print json.dumps(self.guidata.get())

    def go(self):
        print "GO!"
        #p = Process(target=acquire_data, args=(self.guidata.dataset,self.pipes,))
        #p.start()
        print "Process started"
        print self.guidata.get()
        self.statusBar().showMessage("Acquiring Data", 5000)

def acquire_data(params,pipes=None):
    print "Acquiring data"
#    if pipes!=None:    
#        ch1_pipe=pipes['Ch1']
#        ch2_pipe=pipes['Ch2']
#    handle = get_handle()
#    samples = defaultConfig.postTriggerSamples + defaultConfig.preTriggerSamples
#    x = linspace(0, 1, samples)
#    MyAz.configureBoard(handle)
#    print "Configured Board"
#    for ii in xrange(params.buffers):
#        out_array = defaultConfig.output_array()
#        MyAz.acquireData(handle, C.byref(defaultConfig), out_array)
#        y1 = np.ctypeslib.as_array(out_array[0], (samples,))
#        if pipes!=None:
#            ch1_pipe.send((x,y1))
#            #ch2_pipe.send((x,y2))
        
if __name__ == "__main__":        
    app=guidata.qapplication()
    window=ScopeWindow(title="Test Scope",size=(1000,600))
    window.show()
    app.exec_()
#    ds=ScopeDataSet()