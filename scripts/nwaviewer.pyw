# -*- coding: utf-8 -*-
"""
Created on Sat Aug 06 11:57:27 2011

@author: dave
"""

import sys
import numpy as np

from PyQt4.QtGui import *

from guiqwt.builder import make

from slab import *
from nwaviewer_ui import *


#directory=r'C:\Users\dave\Documents\My Dropbox\UofC\code\004 - test temperature sweep\sweep data\\'

class MainWindow(QMainWindow, Ui_NWAViewerWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        self.plot = self.curvewidget.plot   

        self.curvewidget2.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget2.register_all_image_tools()
        self.freqs=None              
        
        self.directory=None
        
        self.fileButton.clicked.connect(self.selectFile)
        self.linearCheckBox.stateChanged.connect(self.update_plots)
        self.plotsInitialized=False
        
        
    def selectFile(self):
        if self.directory is None:
            startdir="S:\\_Data\\"
        else:
            startdir=self.directory
        filename=str(QFileDialog.getOpenFileName(self, 'Open NWA file',startdir))
        self.directory=os.path.split(filename)[0]
        print filename
        self.loadFile(filename)

    def loadFile(self,filename):
        self.fileLabel.setText(filename)
        self.freqs,self.mags,self.phases=load_nwa_file(filename)
       
        self.update_plots()

    def update_plots(self):
        if self.freqs is None: return
        if self.linearCheckBox.isChecked():
            mags=dBm_to_W(self.mags)
        else: mags=self.mags

        if not self.plotsInitialized:
            self.plotsInitialized=True
            self.ch1_plot = make.mcurve(self.freqs/1e9, mags,label='Magnitude') #Make Ch1 curve
            self.plot.add_item(self.ch1_plot)
            self.plot.set_titles(title="Magnitude", xlabel="Frequency, GHz", ylabel="Return Loss, dB")
    
            self.plot2 = self.curvewidget2.plot   
            self.plot2.set_titles(title="Phase", xlabel="Frequency, GHz", ylabel="Phase, deg")
            self.ch1_plot2 = make.mcurve(self.freqs/1e9, self.phases, label="Phase") #Make Ch1 curve
            self.plot2.add_item(self.ch1_plot2)

        self.update_lineplot(self.plot,self.ch1_plot,(self.freqs/1e9,mags))
        self.update_lineplot(self.plot2,self.ch1_plot2,(self.freqs/1e9,self.phases))
                
    def update_lineplot(self,plot,plot_item,data):
        plot_item.set_data(data[0],data[1])
        plot.replot()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    if len(sys.argv) >1:
        window.loadFile(sys.argv[1])
        f=open('C:\Users\Dave\Dropbox\UofC\code\log.txt','w')
        f.write(sys.argv[1])
        f.close()
    window.show()
    sys.exit(app.exec_())