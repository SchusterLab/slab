# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created: Thu Jun 14 17:12:43 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(839, 498)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 831, 451))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.param_rate = SlabSpinBox(self.verticalLayoutWidget)
        self.param_rate.setObjectName(_fromUtf8("param_rate"))
        self.horizontalLayout.addWidget(self.param_rate)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.qwtPlot = Qwt5.QwtPlot(self.verticalLayoutWidget)
        self.qwtPlot.setObjectName(_fromUtf8("qwtPlot"))
        self.verticalLayout.addWidget(self.qwtPlot)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.go_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.go_button.setObjectName(_fromUtf8("go_button"))
        self.horizontalLayout_2.addWidget(self.go_button)
        self.abort_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.abort_button.setObjectName(_fromUtf8("abort_button"))
        self.horizontalLayout_2.addWidget(self.abort_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.message_box = QtGui.QTextBrowser(self.verticalLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.message_box.sizePolicy().hasHeightForWidth())
        self.message_box.setSizePolicy(sizePolicy)
        self.message_box.setMaximumSize(QtCore.QSize(16777215, 75))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Lucida Grande"))
        self.message_box.setFont(font)
        self.message_box.setObjectName(_fromUtf8("message_box"))
        self.verticalLayout.addWidget(self.message_box)
        self.progressBar = QtGui.QProgressBar(self.verticalLayoutWidget)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 839, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuSave = QtGui.QMenu(self.menuFile)
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuInstruments = QtGui.QMenu(self.menubar)
        self.menuInstruments.setObjectName(_fromUtf8("menuInstruments"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setObjectName(_fromUtf8("menuView"))
        self.menuSweep = QtGui.QMenu(self.menubar)
        self.menuSweep.setObjectName(_fromUtf8("menuSweep"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionExperimental_Settings = QtGui.QAction(MainWindow)
        self.actionExperimental_Settings.setObjectName(_fromUtf8("actionExperimental_Settings"))
        self.actionExperimental_and_Instrument_Settings = QtGui.QAction(MainWindow)
        self.actionExperimental_and_Instrument_Settings.setObjectName(_fromUtf8("actionExperimental_and_Instrument_Settings"))
        self.actionLoad = QtGui.QAction(MainWindow)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionShow_Message_Box = QtGui.QAction(MainWindow)
        self.actionShow_Message_Box.setCheckable(True)
        self.actionShow_Message_Box.setChecked(True)
        self.actionShow_Message_Box.setObjectName(_fromUtf8("actionShow_Message_Box"))
        self.actionShow_Progress_Bar = QtGui.QAction(MainWindow)
        self.actionShow_Progress_Bar.setCheckable(True)
        self.actionShow_Progress_Bar.setChecked(True)
        self.actionShow_Progress_Bar.setObjectName(_fromUtf8("actionShow_Progress_Bar"))
        self.actionStart_Sweep = QtGui.QAction(MainWindow)
        self.actionStart_Sweep.setObjectName(_fromUtf8("actionStart_Sweep"))
        self.menuSave.addAction(self.actionExperimental_Settings)
        self.menuSave.addAction(self.actionExperimental_and_Instrument_Settings)
        self.menuFile.addAction(self.menuSave.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menuView.addAction(self.actionShow_Message_Box)
        self.menuView.addAction(self.actionShow_Progress_Bar)
        self.menuSweep.addAction(self.actionStart_Sweep)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuInstruments.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuSweep.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QObject.connect(self.actionShow_Message_Box, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.message_box.setVisible)
        QtCore.QObject.connect(self.actionShow_Progress_Bar, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.progressBar.setVisible)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Rate", None, QtGui.QApplication.UnicodeUTF8))
        self.go_button.setText(QtGui.QApplication.translate("MainWindow", "Go", None, QtGui.QApplication.UnicodeUTF8))
        self.abort_button.setText(QtGui.QApplication.translate("MainWindow", "Abort", None, QtGui.QApplication.UnicodeUTF8))
        self.message_box.setHtml(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuSave.setTitle(QtGui.QApplication.translate("MainWindow", "Save..", None, QtGui.QApplication.UnicodeUTF8))
        self.menuInstruments.setTitle(QtGui.QApplication.translate("MainWindow", "Instruments", None, QtGui.QApplication.UnicodeUTF8))
        self.menuView.setTitle(QtGui.QApplication.translate("MainWindow", "View", None, QtGui.QApplication.UnicodeUTF8))
        self.menuSweep.setTitle(QtGui.QApplication.translate("MainWindow", "Sweep", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_Settings.setText(QtGui.QApplication.translate("MainWindow", "Experimental Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_and_Instrument_Settings.setText(QtGui.QApplication.translate("MainWindow", "Experimental and Instrument Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setText(QtGui.QApplication.translate("MainWindow", "Load", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Message_Box.setText(QtGui.QApplication.translate("MainWindow", "Show Message Box", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Progress_Bar.setText(QtGui.QApplication.translate("MainWindow", "Show Progress Bar", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStart_Sweep.setText(QtGui.QApplication.translate("MainWindow", "Start Sweep", None, QtGui.QApplication.UnicodeUTF8))

from PyQt4 import Qwt5
from widgets import SlabSpinBox