# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Server.ui'
#
# Created: Wed Sep 28 16:53:52 2011
#      by: PyQt4 UI code generator 4.5.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_ServerWindow(object):
    def setupUi(self, ServerWindow):
        ServerWindow.setObjectName("ServerWindow")
        ServerWindow.resize(443, 371)
        self.centralwidget = QtGui.QWidget(ServerWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.curvewidget = CurveWidget(self.centralwidget)
        self.curvewidget.setGeometry(QtCore.QRect(30, 10, 400, 300))
        self.curvewidget.setOrientation(QtCore.Qt.Horizontal)
        self.curvewidget.setObjectName("curvewidget")
        ServerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(ServerWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 443, 22))
        self.menubar.setObjectName("menubar")
        ServerWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(ServerWindow)
        self.statusbar.setObjectName("statusbar")
        ServerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ServerWindow)
        QtCore.QMetaObject.connectSlotsByName(ServerWindow)

    def retranslateUi(self, ServerWindow):
        ServerWindow.setWindowTitle(QtGui.QApplication.translate("ServerWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))

from guiqwt.plot import CurveWidget
