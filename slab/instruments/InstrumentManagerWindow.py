# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:18:00 2013

@author: Dave
"""
#import numpy as np
#import h5py
#import PyQt4.QtCore as Qt
#from guiqwt.builder import make
#from guiqwt.image import *
#from guiqwt.plot import ImageWidget, CurveWidget
#from spyderlib.widgets.internalshell import InternalShell
#from collections import defaultdict, namedtuple
#from copy import copy
#from slab.instruments import InstrumentManager
#import syntax
#UiClass = loadui(__file__.split(".")[0] + ".ui")

from slab import gui
import os,sys,operator,Pyro4
Pyro4.config.HMAC_KEY = '6551d449b0564585a9d39c0bd327dcf1'
from guiqwt.qtdesigner import loadui
from spyderlib.widgets.externalshell.pythonshell  import ExternalPythonShell
from PyQt4.QtCore import QAbstractTableModel,QVariant
from PyQt4.QtGui import QFont,QFileDialog
from PyQt4.QtCore import *
from multiprocessing import Process

UiClass = loadui(os.path.join(os.path.dirname(__file__),"InstrumentManager.ui"))

class InstrumentManagerTableModel(QAbstractTableModel): 
    #based on http://www.saltycrane.com/blog/2007/12/pyqt-43-qtableview-qabstracttablemodel/
    def __init__(self, datain, headerdata, parent=None, *args): 
        """ datain: a list of lists
            headerdata:                                                                                                                                                                                                                                                                                                  a list of strings
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

class InstrumentManagerWindow(gui.SlabWindow, UiClass):
    def __init__(self, nameserver=None, filename=None):
        gui.SlabWindow.__init__(self)
        self.setupSlabWindow(autoparam=True)
        self.auto_register_gui()
        try:
            self.set_param("nameserver", str(Pyro4.locateNS()).split('@')[1].split(':')[0])
            self.msg('Found nameserver at: ' + self.params['nameserver'])
        except:
            self.msg("No name server found.")
            self.set_param("nameserver", '')
        self.read_param_widgets()

        if nameserver is not None:
            self.set_param('nameserver',nameserver)
        if filename is not None:
            self.set_param('filename',filename)

        self.servershell = ExternalPythonShell(self, fname=None, wdir=None, 
                 interact=True, debug=False, path=["C:\\_Lib\\python"], python_args='-i -m slab.instruments.instrumentmanager ',
                 arguments='-s -n %s -f %s' % (self.params['nameserver'],self.params['filename']))
        self.servershell_dockWidget.setWidget(self.servershell)

        self.clientshell = ExternalPythonShell(self, fname=None, wdir=None, 
                 interact=True, debug=False, path=["C:\\_Lib\\python"], python_args='-i -m slab.instruments.instrumentmanager ',
                 arguments='-n '+self.params['nameserver'])
        self.clientshell_dockWidget.setWidget(self.clientshell)
        #self.centralwidget.hide()
        #self.createTable(r'C:\Users\Dave\Documents\instrument.cfg')
        self.im_process=None
        self.start_pushButton.clicked.connect(self.startInstrumentManager)
        self.filename_pushButton.clicked.connect(self.selectFile)
        self.editInstrumentConfig_pushButton.clicked.connect(self.editInstrumentConfig)

    def createTable(self,config_path):
        """Loads configuration file"""
        self.msg("Load Instruments.")        
        f = open(config_path, 'r')
        self.table=[]
        for line in f.readlines():
            #print line
            if line[0] != '#' and line[0] != '':
                self.table.append(line.split())
        f.close()
        self.tableModel=InstrumentManagerTableModel(self.table,['Name','Class','Address'],self)
        self.tableView.setModel(self.tableModel)        
        self.tableView.setSortingEnabled(True)
        
    def closeEvent(self, event):
        #self.shell.exit_interpreter()
        if self.servershell.process is not None:
            self.servershell.process.kill()
        if self.servershell.process is not None:
            self.clientshell.process.kill()
        event.accept()

    def editInstrumentConfig(self):
        filename = self.params['filename']
        if filename == '':
            self.msg('Need a file to edit first!')
        else:
            os.startfile(filename)
        

#    def select_datapath(self):
#        path = str(Qt.QFileDialog.getExistingDirectory(
#                    self, 'Open Datapath',self.params["datapath"]))
#        if path:
#            self.params["datapath"] = path
#            self.emit(Qt.SIGNAL("RunOnDataThread"), "set_datapath")
#
    def startInstrumentManager(self,event):
        self.msg('Starting InstrumentManager')
        self.start_pushButton.setText("Restart Instrument Manager")
        if self.servershell.process is not None:
            self.servershell.process.kill()
        if self.clientshell.process is not None:
            self.clientshell.process.kill()

        self.servershell.arguments='-s'        
        self.clientshell.arguments=''        
        if self.params['nameserver'] !="":
            self.servershell.arguments+=" -n " + self.params['nameserver']
            self.clientshell.arguments+="-n " + self.params['nameserver']
        if self.params['filename'] !="":
            self.servershell.arguments+=" -f " + self.params['filename']
        else:
            self.msg('No config file specified.  Launching client only.')
        #self.msg('Server shell arguments: '+self.servershell.arguments)
        #self.msg('Client shell arguments: '+self.clientshell.arguments)
        if not (self.params['disableServer'] or self.params['filename']==''):
            self.servershell.start_shell()
        self.clientshell.start_shell()
        #self.param_disableServer.setEnabled(False)

        
    def selectFile(self):
        self.set_param('filename',str(QFileDialog.getOpenFileName(self)))

if __name__ == "__main__":
    sys.exit(gui.runWin(InstrumentManagerWindow))
