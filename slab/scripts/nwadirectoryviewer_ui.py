# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nwadirectoryviewer.ui'
#
# Created: Thu Oct 20 20:25:02 2011
#      by: PyQt4 UI code generator 4.5.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_NWADirectoryViewerWindow(object):
    def setupUi(self, NWADirectoryViewerWindow):
        NWADirectoryViewerWindow.setObjectName("NWADirectoryViewerWindow")
        NWADirectoryViewerWindow.resize(800, 600)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(NWADirectoryViewerWindow.sizePolicy().hasHeightForWidth())
        NWADirectoryViewerWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(NWADirectoryViewerWindow)
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
        self.watchCheckBox = QtGui.QCheckBox(self.centralwidget)
        self.watchCheckBox.setChecked(True)
        self.watchCheckBox.setObjectName("watchCheckBox")
        self.horizontalLayout.addWidget(self.watchCheckBox)
        self.autoscaleCheckBox = QtGui.QCheckBox(self.centralwidget)
        self.autoscaleCheckBox.setChecked(True)
        self.autoscaleCheckBox.setObjectName("autoscaleCheckBox")
        self.horizontalLayout.addWidget(self.autoscaleCheckBox)
        self.linearCheckBox = QtGui.QCheckBox(self.centralwidget)
        self.linearCheckBox.setObjectName("linearCheckBox")
        self.horizontalLayout.addWidget(self.linearCheckBox)
        self.normalizeCheckBox = QtGui.QCheckBox(self.centralwidget)
        self.normalizeCheckBox.setObjectName("normalizeCheckBox")
        self.horizontalLayout.addWidget(self.normalizeCheckBox)
        self.fileButton = QtGui.QPushButton(self.centralwidget)
        self.fileButton.setObjectName("fileButton")
        self.horizontalLayout.addWidget(self.fileButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.imagewidget = ImageWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imagewidget.sizePolicy().hasHeightForWidth())
        self.imagewidget.setSizePolicy(sizePolicy)
        self.imagewidget.setOrientation(QtCore.Qt.Vertical)
        self.imagewidget.setObjectName("imagewidget")
        self.verticalLayout.addWidget(self.imagewidget)
        NWADirectoryViewerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(NWADirectoryViewerWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        NWADirectoryViewerWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(NWADirectoryViewerWindow)
        self.statusbar.setObjectName("statusbar")
        NWADirectoryViewerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(NWADirectoryViewerWindow)
        QtCore.QMetaObject.connectSlotsByName(NWADirectoryViewerWindow)

    def retranslateUi(self, NWADirectoryViewerWindow):
        NWADirectoryViewerWindow.setWindowTitle(QtGui.QApplication.translate("NWADirectoryViewerWindow", "Network Analyzer Directory Viewer ", None, QtGui.QApplication.UnicodeUTF8))
        self.watchCheckBox.setText(QtGui.QApplication.translate("NWADirectoryViewerWindow", "Watch Folder", None, QtGui.QApplication.UnicodeUTF8))
        self.autoscaleCheckBox.setText(QtGui.QApplication.translate("NWADirectoryViewerWindow", "autoscale", None, QtGui.QApplication.UnicodeUTF8))
        self.linearCheckBox.setText(QtGui.QApplication.translate("NWADirectoryViewerWindow", "Linear Scale", None, QtGui.QApplication.UnicodeUTF8))
        self.normalizeCheckBox.setText(QtGui.QApplication.translate("NWADirectoryViewerWindow", "Normalize", None, QtGui.QApplication.UnicodeUTF8))
        self.fileButton.setText(QtGui.QApplication.translate("NWADirectoryViewerWindow", "Open Directory", None, QtGui.QApplication.UnicodeUTF8))

from guiqwt.plot import ImageWidget
