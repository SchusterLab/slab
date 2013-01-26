# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:18:00 2013

@author: Dave
"""

import h5py
#import PyQt4.QtCore as Qt
import numpy as np
from guiqwt.image import *
from guiqwt.plot import ImageWidget, CurveWidget
from guiqwt.qtdesigner import loadui
from guiqwt.builder import make
from spyderlib.widgets.internalshell import InternalShell
from slab import gui
from collections import defaultdict, namedtuple
from copy import copy
import os, glob, sys,re,operator
import Pyro4
from slab.instruments import InstrumentManager
from PyQt4.QtCore import QAbstractTableModel,QVariant
from PyQt4.QtGui import QFont
from PyQt4.QtCore import *
from multiprocessing import Process
#import syntax
#UiClass = loadui(__file__.split(".")[0] + ".ui")
UiClass = loadui("InstrumentManager.ui")

class InstrumentManagerTableModel(QAbstractTableModel): 
    #based on http://www.saltycrane.com/blog/2007/12/pyqt-43-qtableview-qabstracttablemodel/
    def __init__(self, datain, headerdata, parent=None, *args): 
        """ datain: a list of lists
            headerdata: a list of strings
        """
        QAbstractTableModel.__init__(self, parent, *args) 
        self.arraydata = datain
        self.headerdata = headerdata
 
    def rowCount(self, parent): 
        return len(self.arraydata) 
 
    def columnCount(self, parent): 
        return len(self.arraydata[0]) 
 
    def data(self, index, role): 
        if not index.isValid(): 
            return QVariant() 
        elif role != Qt.DisplayRole: 
            return QVariant() 
        return QVariant(self.arraydata[index.row()][index.column()]) 

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[col])
        return QVariant()

    def sort(self, Ncol, order):
        """Sort table by given column number.
        """
        self.emit(SIGNAL("layoutAboutToBeChanged()"))
        self.arraydata = sorted(self.arraydata, key=operator.itemgetter(Ncol))        
        if order == Qt.DescendingOrder:
            self.arraydata.reverse()
        self.emit(SIGNAL("layoutChanged()"))


class InstrumentManagerThread(gui.DataThread):
    
    def start_server(self,):
        self.params['im']=InstrumentManager()
        print "start server"

class InstrumentManagerWindow(gui.SlabWindow, UiClass):
    def __init__(self, fname=None):
        gui.SlabWindow.__init__(self, InstrumentManagerThread)
        self.setupSlabWindow(autoparam=True)
        self.auto_register_gui()

#        self.register_param(self.datasets_treeWidget, "dataset")
        # Connect launchers

#        self.datapath_browse_pushButton.clicked.connect(self.select_datapath)
#        self.register_script("set_datapath", self.datapath_lineEdit)
        try:
            nameserver=Pyro4.locateNS()
        except:
            nameserver="No server found"
        self.set_param("nameserver", nameserver)
        
        # Setup Prompt
        message = "All of the instruments can be accessed by name"
        self.shell = InternalShell(self, message=message)
        self.shell.set_font(QFont("Consolas"))
        self.shell_dockWidget.setWidget(self.shell)
        self.gui["shell"] = self.shell

        self.start_thread()
        self.msg( self.params['nameserver'])
        self.createTable(r'C:\Users\Dave\Documents\instrument.cfg')
#        if fname is not None:
#            self.shell.exit_interpreter()
#            directory = os.path.dirname(fname)
#            nameitem = namedtuple('pseudoitem', ('filename',))(fname)
#            self.msg(fname)
#            self.set_param("datapath", directory)
#            self.start_script("set_datapath")
#            self.start_script("load_file", nameitem)

    def createTable(self,config_path):
        """Loads configuration file"""
        self.msg("Load Instruments.")        
        f = open(config_path, 'r')
        table=[]
        for line in f.readlines():
            #print line
            if line[0] != '#' and line[0] != '':
                table.append(line.split())
        f.close()
        self.tableModel=InstrumentManagerTableModel(table,['Name','Class','Address'],self)
        self.tableView.setModel(self.tableModel)        
        self.tableView.setSortingEnabled(True)
        
    def closeEvent(self, event):
        self.shell.exit_interpreter()
        event.accept()

#    def select_datapath(self):
#        path = str(Qt.QFileDialog.getExistingDirectory(
#                    self, 'Open Datapath',self.params["datapath"]))
#        if path:
#            self.params["datapath"] = path
#            self.emit(Qt.SIGNAL("RunOnDataThread"), "set_datapath")
#
    def restartInstrumentManager(self,event):
        def runIM(config_path=None, server=False, ns_address=None):
            im=InstrumentManager(config_path,server,ns_address)
        if self.im_process is not None:
            self.im_process.terminate()
        self.im_process=Process(target='runIM',name='IMServer',
                                kwargs={'config_path':r'C:\Users\Dave\Documents\instrument.cfg','server':False,'ns_address':'192.168.11.17'},daemon=True)


if __name__ == "__main__":
    #fname = "S:\\_Data\\120930 - EonHe - M005CHM3\\004_AutomatedFilling\\121017_M005CHM3-AutomatedFilling_003\\M005CHM3-AutomatedFilling.h5"
    fname = None
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    sys.exit(gui.runWin(InstrumentManagerWindow, fname=fname))
