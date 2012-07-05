# -*- coding: utf-8 -*-
"""
Created on Sat Aug 06 20:25:43 2011

@author: Dave
"""

import sys
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import QTimer

from guiqwt.builder import make

from slab import *
from nwadirectoryviewer_ui import *


#directory='C:\\Users\\Dave\\Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\'

class MainWindow(QMainWindow, Ui_NWADirectoryViewerWindow):
    def __init__(self, parent = None):
    
        QMainWindow.__init__(self, parent)
        self.setupUi(self)                
        self.imagewidget.add_toolbar(self.addToolBar("Image"))
        self.imagewidget.register_all_image_tools()
        self.plot = self.imagewidget.plot   
        self.plot.set_aspect_ratio(1.0, False)
        self.plot.set_axis_direction('left', False)
        self.plot_item=None
        self.plot.set_titles(title="Magnitude", xlabel="Frequency, GHz", ylabel="Trace #")
        self.data=None
        self.prefixes=None
        self.directory=None

        self.fileButton.clicked.connect(self.selectFile)
        self.linearCheckBox.stateChanged.connect(self.update_image)
        self.normalizeCheckBox.stateChanged.connect(self.update_image)
        self.watchCheckBox.stateChanged.connect(self.watchCheckBox_changed)
        #QtCore.QObject.connect(self.ctimer, QtCore.SIGNAL("timeout()"), self.update_plots)

        self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
        self.ctimer.timeout.connect(self.update_directory)

        
    def watchCheckBox_changed(self):
        if self.watchCheckBox.isChecked():
            if self.directory is not None:
                self.ctimer.start(100)
        else:
            self.ctimer.stop()

    def update_directory(self):
        if self.directory is None:
            return
        self.ctimer.stop()
        fnames=glob.glob(os.path.join(self.directory,"*.CSV"))
        fnames.sort()
        prefixes = [os.path.split(fname)[-1] for fname in fnames]
        #if len(prefixes)!=len(self.prefixes):
            #load_directory(self.directory)
        prefixes.sort()
        for prefix in prefixes:
            if prefix not in self.prefixes:
                print "added " + prefix
                self.prefixes.append(prefix)
                d=load_nwa_file(os.path.join(self.directory,prefix))
                self.data.append(d)
        self.update_image()
        if self.autoscaleCheckBox.isChecked():
            self.plot.do_autoscale()
        self.ctimer.start(100)
  
    
    def load_directory(self,directory=None):
        if directory is not None:
            print "Loading directory: "+ directory
            self.fileLabel.setText(directory)
            self.prefixes, self.data = load_nwa_dir(directory)
            print "Finished Loading"
            self.update_image()
            self.watchCheckBox_changed()
            
    def selectFile(self):
        if self.directory is None:
            startdir="S:\\_Data\\"
        else:
            startdir=self.directory
        self.directory=str(QFileDialog.getExistingDirectory(self,"Open Data Directory",startdir))
        if self.directory!='':
            self.load_directory(self.directory)
        else:
            self.directory=None
                
    def update_image(self):
        if self.data is None: return
        #print directory
        #print prefixes
        mag_image=[]
        for d in self.data:
            mags=d[1]
            if self.linearCheckBox.isChecked():
                mags=dBm_to_W(mags)
            if self.normalizeCheckBox.isChecked():
                mags=mags/max(mags)
            mag_image.append(mags)
        mag_image = array(mag_image)
        
        
        if self.plot_item is None:
            self.plot_item = make.xyimage(self.data[0][0]/1e9,range(len(self.data)),mag_image)
            self.plot.add_item(self.plot_item)
            self.plot_item.set_xy(self.data[0][0]/1e9,range(len(self.data)))
            #self.plot.setAxisScale(self.plot.xBottom, d[0][1]/1e9, d[0][-1]/1e9)
            self.plot.replot()
        else:
            self.plot_item.set_data(mag_image)
            self.plot_item.set_xy(self.data[0][0]/1e9,range(len(self.data)))
            self.plot.update_colormap_axis(self.plot_item)
            self.plot.replot()
            
if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    if len(sys.argv) >1:
        print sys.argv[1]
        window.load_directory(sys.argv[1])
    window.show()
    sys.exit(app.exec_())