 # -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 18:30:23 2011

@author: dave
"""
#Todo
#Save button / autosave
#averaging mode
#fix array slicing math for multichannel
#clock reference
#fix buffers


import sys
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import QTimer

from guiqwt.builder import make

from slab import *
from ScopeWindow_ui import *
from numpy import arange,sin,cos
from slab.instruments import Alazar, AlazarConfig

class ScopeWindow(QMainWindow, Ui_ScopeWindow):
    def __init__(self, parent = None):
    
        QMainWindow.__init__(self, parent)
        self.setupUi(self)                

        self.datapath='S:\\_Data\\'
        self.ch1_initialized=False
        self.ch2_initialized=False
        self.time_pts=None
        self.ch1_pts=None
        self.ch2_pts=None
        self.scope_settings= AlazarConfig()
        
        self.card=Alazar(self.scope_settings.from_form(self))          
        
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        self.plot = self.curvewidget.plot   
        self.plot.set_titles(title="Oscilloscope", xlabel="Time", ylabel="Voltage")

        self.datapathButton.clicked.connect(self.selectDatapath)
        #self.runButton.clicked.connect(self.runScope)
        self.runButton.clicked.connect(self.config_and_run)
        self.datapathLineEdit.textChanged.connect(self.update_filenumber)
        self.prefixLineEdit.textChanged.connect(self.update_filenumber)
        
        self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
        self.ctimer.timeout.connect(self.runScope)

        
    def selectDatapath(self):
        self.datapath=str(QFileDialog.getExistingDirectory(self,'Open Datapath',self.datapath))
        self.datapathLineEdit.setText(self.datapath)
        
    def update_filenumber(self):
        filenumber=next_file_index(self.datapath,str(self.prefixLineEdit.text()))
        self.filenumberLabel.setText("%04d_" % filenumber)
        
    def update_plots(self):
        if (self.ch1_pts is not None):   
            if not self.ch1_initialized:
                self.ch1_initialized=True
                self.ch1_plot = make.mcurve(self.time_pts, self.ch1_pts,label='Ch1') #Make Ch1 curve
                self.plot.add_item(self.ch1_plot)
            
            self.update_lineplot(self.plot,self.ch1_plot,(self.time_pts,self.ch1_pts))

        if (self.ch2_pts is not None):   
            if not self.ch2_initialized:
                self.ch2_initialized=True
                self.ch2_plot = make.mcurve(self.time_pts, self.ch2_pts,label='Ch2') #Make Ch2 curve
                self.plot.add_item(self.ch2_plot)
            
            self.update_lineplot(self.plot,self.ch2_plot,(self.time_pts,self.ch2_pts))            
       
    def config_and_run(self):
        print self.scope_settings.get_dict()
        self.scope_settings.from_form(self)
        self.card.configure(self.scope_settings)
        self.runScope()
                
    def update_lineplot(self,plot,plot_item,data):
        plot_item.set_data(data[0],data[1])
        plot.replot()
        #self.plot.do_autoscale()
        
    def runScope(self):
        self.ctimer.stop()
        self.time_pts,self.ch1_pts,self.ch2_pts=self.card.acquire_avg_data()
        
        self.update_plots()
        if self.autoscaleCheckBox.isChecked():
            self.plot.do_autoscale()
        if self.autorunCheckBox.isChecked():
            self.ctimer.start(1)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScopeWindow()
    window.show()
    sys.exit(app.exec_())       