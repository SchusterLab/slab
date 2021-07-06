# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nwaviewer.ui'
#
# Created: Sat Oct 15 12:22:15 2011
#      by: PyQt4 UI code generator 4.5.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_NWAViewerWindow(object):
    def setupUi(self, NWAViewerWindow):
        NWAViewerWindow.setObjectName("NWAViewerWindow")
        NWAViewerWindow.resize(946, 819)
        self.centralwidget = QtGui.QWidget(NWAViewerWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.fileLabel = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileLabel.sizePolicy().hasHeightForWidth())
        self.fileLabel.setSizePolicy(sizePolicy)
        self.fileLabel.setObjectName("fileLabel")
        self.horizontalLayout.addWidget(self.fileLabel)
        self.linearCheckBox = QtGui.QCheckBox(self.centralwidget)
        self.linearCheckBox.setObjectName("linearCheckBox")
        self.horizontalLayout.addWidget(self.linearCheckBox)
        self.fileButton = QtGui.QPushButton(self.centralwidget)
        self.fileButton.setObjectName("fileButton")
        self.horizontalLayout.addWidget(self.fileButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.curvewidget = CurveWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.curvewidget.sizePolicy().hasHeightForWidth())
        self.curvewidget.setSizePolicy(sizePolicy)
        self.curvewidget.setOrientation(QtCore.Qt.Horizontal)
        self.curvewidget.setObjectName("curvewidget")
        self.verticalLayout.addWidget(self.curvewidget)
        self.curvewidget2 = CurveWidget(self.centralwidget)
        self.curvewidget2.setOrientation(QtCore.Qt.Horizontal)
        self.curvewidget2.setObjectName("curvewidget2")
        self.verticalLayout.addWidget(self.curvewidget2)
        NWAViewerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(NWAViewerWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 22))
        self.menubar.setObjectName("menubar")
        NWAViewerWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(NWAViewerWindow)
        self.statusbar.setObjectName("statusbar")
        NWAViewerWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(NWAViewerWindow)
        self.toolBar.setObjectName("toolBar")
        NWAViewerWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(NWAViewerWindow)
        QtCore.QMetaObject.connectSlotsByName(NWAViewerWindow)

    def retranslateUi(self, NWAViewerWindow):
        NWAViewerWindow.setWindowTitle(QtGui.QApplication.translate("NWAViewerWindow", "Network Analyzer Viewer", None, QtGui.QApplication.UnicodeUTF8))
        self.fileLabel.setText(QtGui.QApplication.translate("NWAViewerWindow", "No File Loaded", None, QtGui.QApplication.UnicodeUTF8))
        self.linearCheckBox.setText(QtGui.QApplication.translate("NWAViewerWindow", "Linear", None, QtGui.QApplication.UnicodeUTF8))
        self.fileButton.setText(QtGui.QApplication.translate("NWAViewerWindow", "Open", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBar.setWindowTitle(QtGui.QApplication.translate("NWAViewerWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8))

from guiqwt.plot import CurveWidget
