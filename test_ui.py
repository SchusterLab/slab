# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created: Fri Dec  2 15:57:59 2011
#      by: PyQt4 UI code generator 4.8.6
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
        MainWindow.resize(862, 440)
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 851, 391))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Rate", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.spinBox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.spinBox.setMinimumSize(QtCore.QSize(100, 0))
        self.spinBox.setProperty("value", 1)
        self.spinBox.setObjectName(_fromUtf8("spinBox"))
        self.horizontalLayout.addWidget(self.spinBox)
        self.qwtPlot = Qwt5.QwtPlot(self.verticalLayoutWidget)
        self.qwtPlot.setObjectName(_fromUtf8("qwtPlot"))
        self.horizontalLayout.addWidget(self.qwtPlot)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.go_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.go_button.setText(QtGui.QApplication.translate("MainWindow", "Go", None, QtGui.QApplication.UnicodeUTF8))
        self.go_button.setObjectName(_fromUtf8("go_button"))
        self.horizontalLayout_2.addWidget(self.go_button)
        self.abort_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.abort_button.setText(QtGui.QApplication.translate("MainWindow", "Abort", None, QtGui.QApplication.UnicodeUTF8))
        self.abort_button.setObjectName(_fromUtf8("abort_button"))
        self.horizontalLayout_2.addWidget(self.abort_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.message_box = QtGui.QTextBrowser(self.verticalLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.message_box.sizePolicy().hasHeightForWidth())
        self.message_box.setSizePolicy(sizePolicy)
        self.message_box.setMaximumSize(QtCore.QSize(16777215, 80))
        self.message_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.message_box.setHtml(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8pt;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.message_box.setObjectName(_fromUtf8("message_box"))
        self.verticalLayout.addWidget(self.message_box)
        self.cmd_lineEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.cmd_lineEdit.setObjectName(_fromUtf8("cmd_lineEdit"))
        self.verticalLayout.addWidget(self.cmd_lineEdit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 862, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuSave = QtGui.QMenu(self.menuFile)
        self.menuSave.setTitle(QtGui.QApplication.translate("MainWindow", "Save..", None, QtGui.QApplication.UnicodeUTF8))
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuInstruments = QtGui.QMenu(self.menubar)
        self.menuInstruments.setTitle(QtGui.QApplication.translate("MainWindow", "Instruments", None, QtGui.QApplication.UnicodeUTF8))
        self.menuInstruments.setObjectName(_fromUtf8("menuInstruments"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionExperimental_Settings = QtGui.QAction(MainWindow)
        self.actionExperimental_Settings.setText(QtGui.QApplication.translate("MainWindow", "Experimental Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_Settings.setObjectName(_fromUtf8("actionExperimental_Settings"))
        self.actionExperimental_and_Instrument_Settings = QtGui.QAction(MainWindow)
        self.actionExperimental_and_Instrument_Settings.setText(QtGui.QApplication.translate("MainWindow", "Experimental and Instrument Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_and_Instrument_Settings.setObjectName(_fromUtf8("actionExperimental_and_Instrument_Settings"))
        self.actionLoad = QtGui.QAction(MainWindow)
        self.actionLoad.setText(QtGui.QApplication.translate("MainWindow", "Load", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.menuSave.addAction(self.actionExperimental_Settings)
        self.menuSave.addAction(self.actionExperimental_and_Instrument_Settings)
        self.menuFile.addAction(self.menuSave.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuInstruments.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        pass

from PyQt4 import Qwt5
