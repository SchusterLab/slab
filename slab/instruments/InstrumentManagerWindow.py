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
import sys,operator,Pyro4
from guiqwt.qtdesigner import loadui
from spyderlib.widgets.externalshell.pythonshell  import ExternalPythonShell
from PyQt4.QtCore import QAbstractTableModel,QVariant
from PyQt4.QtGui import QFont,QFileDialog
from PyQt4.QtCore import *
from multiprocessing import Process
UiClass = loadui("InstrumentManager.ui")

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


class InstrumentManagerThread(gui.DataThread):
    
    def start_server(self,):
        #self.params['im']=InstrumentManager()
        print "start server"

class InstrumentManagerWindow(gui.SlabWindow, UiClass):
    def __init__(self, fname=None):
        gui.SlabWindow.__init__(self)
        self.setupSlabWindow(autoparam=True)
        self.auto_register_gui()
        try:
            nameserver=Pyro4.locateNS()
            self.set_param("nameserver", nameserver)
        except:
            self.msg("No name server found")
            #self.set_param("nameserver", '192.168.11.17')
        self.read_param_widgets()

        self.msg('Loading Server Instrument Manager using nameserver at: %s' % (self.params['nameserver']))
        self.servershell = ExternalPythonShell(self, fname=None, wdir=None, commands=[],
                 interact=True, debug=False, path=[], python_args='-i -m slab.instruments.instrumentmanager ',
                 ipython=False, arguments='-s -n %s -f %s' % (self.params['nameserver'],self.params['filename']))
        self.servershell_dockWidget.setWidget(self.servershell)

        self.clientshell = ExternalPythonShell(self, fname=None, wdir=None, commands=[],
                 interact=True, debug=False, path=[], python_args='-i -m slab.instruments.instrumentmanager ',
                 ipython=False, arguments='-n '+self.params['nameserver'])
        self.clientshell_dockWidget.setWidget(self.clientshell)

        #self.start_thread()
        self.createTable(r'C:\Users\Dave\Documents\instrument.cfg')
        self.im_process=None
        self.start_pushButton.clicked.connect(self.startInstrumentManager)
        self.filename_pushButton.clicked.connect(self.selectFile)

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
        #self.shell.exit_interpreter()
        self.servershell.process.kill()
        self.clientshell.process.kill()
        event.accept()

#    def select_datapath(self):
#        path = str(Qt.QFileDialog.getExistingDirectory(
#                    self, 'Open Datapath',self.params["datapath"]))
#        if path:
#            self.params["datapath"] = path
#            self.emit(Qt.SIGNAL("RunOnDataThread"), "set_datapath")
#
    def startInstrumentManager(self,event):
        self.msg('Starting InstrumentManager')
        self.start_pushButton.setEnabled(False)
#        if self.im_process is not None:
#            self.im_process.terminate()
#            #self.shell.exit_interpreter()
#
#            self.shell.process.kill()
#            self.msg('Terminated Running Processes')
        self.servershell.arguments='-s -n %s -f %s' % (self.params['nameserver'],self.params['filename'])
        self.clientshell.arguments='-n '+self.params['nameserver']
        self.servershell.start_shell()
        self.clientshell.start_shell()

#        self.im_process=Process(target=runIM,name='IMServer',
#                                kwargs={'config_path':r'C:\Users\Dave\Documents\instrument.cfg','server':True,'ns_address':'192.168.11.17'})
#        self.im_process.daemon=True
#        self.im_process.start()
##        im=InstrumentManager(ns_address='192.168.11.17')
##        ns = { 'im':im}
##        self.shell.start_interpreter(namespace=ns)
#
#        self.shell.start_shell()
#        self.msg('Launched IM server process')
        
    def selectFile(self):
        self.param_filename.setText(str(QFileDialog.getOpenFileName(self)))
        self.read_param_widgets()

def runIM(config_path=None, server=False, ns_address=None):
    #im=InstrumentManager(config_path,server,ns_address)
    return


if __name__ == "__main__":
    #fname = "S:\\_Data\\120930 - EonHe - M005CHM3\\004_AutomatedFilling\\121017_M005CHM3-AutomatedFilling_003\\M005CHM3-AutomatedFilling.h5"
    fname = None
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    sys.exit(gui.runWin(InstrumentManagerWindow, fname=fname))
