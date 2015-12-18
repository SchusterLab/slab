# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:18:00 2013

@author: Dave
"""
from slab import gui
import os, sys, operator, Pyro4

Pyro4.config.HMAC_KEY = '6551d449b0564585a9d39c0bd327dcf1'
from PyQt4.QtGui import QFileDialog
from PyQt4.QtCore import *
from PyQt4 import uic
import time

UiClass = uic.loadUiType(os.path.join(os.path.dirname(__file__), "InstrumentManager.ui"))[0]


class InstrumentManagerTableModel(QAbstractTableModel):
    # based on http://www.saltycrane.com/blog/2007/12/pyqt-43-qtableview-qabstracttablemodel/
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
    def init_kernels(self, restartServer=False, restartClient=False):
        if self.params['nameserver'] == "":
            ns = "None"
        else:
            ns = r'"%s"' % self.params['nameserver']

        if restartServer:
            self.servershell.restart_kernel("reloading server", now=True)
            self.servershell.reset(clear=True)
            time.sleep(2)
            if not (self.params['disableServer'] or self.params['filename'] == ''):
                self.servershell.execute(source="from slab.instruments import InstrumentManager", hidden=True)
                self.servershell.execute(
                    source="im=InstrumentManager(config_path='%s', server=True, ns_address=%s)" % (
                        self.params['filename'], ns))
            else:
                self.msg('Launching client only.')

        if restartClient:
            self.clientshell.restart_kernel("reloading client", now=True)
            time.sleep(2)
            self.clientshell.execute(source="%pylab inline", hidden=True)
            self.clientshell.execute(source="from slab.instruments import InstrumentManager", hidden=True)
            self.clientshell.execute(source="im=InstrumentManager(ns_address=%s)" % (ns))

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
            self.set_param('nameserver', nameserver)
        if filename is not None:
            self.set_param('filename', filename)

        self.servershell = gui.IPythonWidget()
        self.servershell_dockWidget.setWidget(self.servershell)
        self.clientshell = gui.IPythonWidget()
        self.clientshell_dockWidget.setWidget(self.clientshell)

        self.im_process = None
        self.start_pushButton.clicked.connect(self.startInstrumentManagerServer)
        self.start_clientButton.clicked.connect(self.startInstrumentManagerClient)
        self.filename_pushButton.clicked.connect(self.selectFile)
        self.editInstrumentConfig_pushButton.clicked.connect(self.editInstrumentConfig)

    def createTable(self, config_path):
        """Loads configuration file"""
        self.msg("Load Instruments.")
        f = open(config_path, 'r')
        self.table = []
        for line in f.readlines():
            # print line
            if line[0] != '#' and line[0] != '':
                self.table.append(line.split())
        f.close()
        self.tableModel = InstrumentManagerTableModel(self.table, ['Name', 'Class', 'Address'], self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)

    def closeEvent(self, event):
        pass

    def editInstrumentConfig(self):
        filename = self.params['filename']
        if filename == '':
            self.msg('Need a file to edit first!')
        else:
            os.startfile(filename)

    def startInstrumentManagerServer(self, event):
        self.msg('Starting InstrumentManager Server')
        self.start_pushButton.setText("Restart Server")
        self.init_kernels(restartServer=True)

    def startInstrumentManagerClient(self, event):
        self.msg('Starting InstrumentManager Client')
        self.start_clientButton.setText("Restart Client")
        self.init_kernels(restartClient=True)

    def selectFile(self):
        self.set_param('filename', str(QFileDialog.getOpenFileName(self)))


if __name__ == "__main__":
    sys.exit(gui.runWin(InstrumentManagerWindow))
