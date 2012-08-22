# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NWAWindow2.ui'
#
# Created: Tue Aug 21 19:52:49 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_NWAWindow(object):
    def setupUi(self, NWAWindow):
        NWAWindow.setObjectName(_fromUtf8("NWAWindow"))
        NWAWindow.resize(1084, 836)
        NWAWindow.setWindowTitle(QtGui.QApplication.translate("NWAWindow", "Network Analyzer Interface", None, QtGui.QApplication.UnicodeUTF8))
        self.centralwidget = QtGui.QWidget(NWAWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setTitle(QtGui.QApplication.translate("NWAWindow", "NWA Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setText(QtGui.QApplication.translate("NWAWindow", "Power", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.param_power = QtGui.QDoubleSpinBox(self.groupBox)
        self.param_power.setSuffix(QtGui.QApplication.translate("NWAWindow", " dBm", None, QtGui.QApplication.UnicodeUTF8))
        self.param_power.setMinimum(-100.0)
        self.param_power.setMaximum(10.0)
        self.param_power.setProperty("value", -20.0)
        self.param_power.setObjectName(_fromUtf8("param_power"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.param_power)
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setText(QtGui.QApplication.translate("NWAWindow", "Sweep pts", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.param_sweep_pts = QtGui.QSpinBox(self.groupBox)
        self.param_sweep_pts.setMaximum(1601)
        self.param_sweep_pts.setProperty("value", 1601)
        self.param_sweep_pts.setObjectName(_fromUtf8("param_sweep_pts"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.param_sweep_pts)
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setText(QtGui.QApplication.translate("NWAWindow", "IF BW", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_3)
        self.param_ifbw = QtGui.QDoubleSpinBox(self.groupBox)
        self.param_ifbw.setSuffix(QtGui.QApplication.translate("NWAWindow", " Hz", None, QtGui.QApplication.UnicodeUTF8))
        self.param_ifbw.setMinimum(10.0)
        self.param_ifbw.setMaximum(1000000.0)
        self.param_ifbw.setProperty("value", 1000.0)
        self.param_ifbw.setObjectName(_fromUtf8("param_ifbw"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.param_ifbw)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setText(QtGui.QApplication.translate("NWAWindow", "Avgs", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_4)
        self.param_avgs = QtGui.QSpinBox(self.groupBox)
        self.param_avgs.setMinimum(1)
        self.param_avgs.setObjectName(_fromUtf8("param_avgs"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.param_avgs)
        self.centerstartLabel = QtGui.QLabel(self.groupBox)
        self.centerstartLabel.setText(QtGui.QApplication.translate("NWAWindow", "Center", None, QtGui.QApplication.UnicodeUTF8))
        self.centerstartLabel.setObjectName(_fromUtf8("centerstartLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.centerstartLabel)
        self.param_centerstart = QtGui.QDoubleSpinBox(self.groupBox)
        self.param_centerstart.setSuffix(QtGui.QApplication.translate("NWAWindow", " GHz", None, QtGui.QApplication.UnicodeUTF8))
        self.param_centerstart.setDecimals(6)
        self.param_centerstart.setMaximum(20.0)
        self.param_centerstart.setSingleStep(0.1)
        self.param_centerstart.setProperty("value", 10.0)
        self.param_centerstart.setObjectName(_fromUtf8("param_centerstart"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.param_centerstart)
        self.spanstopLabel = QtGui.QLabel(self.groupBox)
        self.spanstopLabel.setText(QtGui.QApplication.translate("NWAWindow", "Span", None, QtGui.QApplication.UnicodeUTF8))
        self.spanstopLabel.setObjectName(_fromUtf8("spanstopLabel"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.spanstopLabel)
        self.param_spanstop = QtGui.QDoubleSpinBox(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.param_spanstop.sizePolicy().hasHeightForWidth())
        self.param_spanstop.setSizePolicy(sizePolicy)
        self.param_spanstop.setSuffix(QtGui.QApplication.translate("NWAWindow", " MHz", None, QtGui.QApplication.UnicodeUTF8))
        self.param_spanstop.setMaximum(20000.0)
        self.param_spanstop.setProperty("value", 1000.0)
        self.param_spanstop.setObjectName(_fromUtf8("param_spanstop"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.param_spanstop)
        self.label_5 = QtGui.QLabel(self.groupBox)
        self.label_5.setText(QtGui.QApplication.translate("NWAWindow", "Max Step", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_5)
        self.param_resolution = QtGui.QDoubleSpinBox(self.groupBox)
        self.param_resolution.setSuffix(QtGui.QApplication.translate("NWAWindow", " MHz", None, QtGui.QApplication.UnicodeUTF8))
        self.param_resolution.setDecimals(5)
        self.param_resolution.setMinimum(1e-05)
        self.param_resolution.setMaximum(999999999.0)
        self.param_resolution.setSingleStep(1e-05)
        self.param_resolution.setProperty("value", 100.0)
        self.param_resolution.setObjectName(_fromUtf8("param_resolution"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.param_resolution)
        self.verticalLayout_3.addLayout(self.formLayout)
        self.param_centerspanstartstop = QtGui.QCheckBox(self.groupBox)
        self.param_centerspanstartstop.setEnabled(False)
        self.param_centerspanstartstop.setText(QtGui.QApplication.translate("NWAWindow", "Start/Stop", None, QtGui.QApplication.UnicodeUTF8))
        self.param_centerspanstartstop.setChecked(True)
        self.param_centerspanstartstop.setObjectName(_fromUtf8("param_centerspanstartstop"))
        self.verticalLayout_3.addWidget(self.param_centerspanstartstop)
        self.param_autorun = QtGui.QCheckBox(self.groupBox)
        self.param_autorun.setText(QtGui.QApplication.translate("NWAWindow", "Autorun", None, QtGui.QApplication.UnicodeUTF8))
        self.param_autorun.setObjectName(_fromUtf8("param_autorun"))
        self.verticalLayout_3.addWidget(self.param_autorun)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.go_button = QtGui.QPushButton(self.groupBox)
        self.go_button.setText(QtGui.QApplication.translate("NWAWindow", "Go", None, QtGui.QApplication.UnicodeUTF8))
        self.go_button.setObjectName(_fromUtf8("go_button"))
        self.horizontalLayout_2.addWidget(self.go_button)
        self.abort_button = QtGui.QPushButton(self.groupBox)
        self.abort_button.setText(QtGui.QApplication.translate("NWAWindow", "Abort", None, QtGui.QApplication.UnicodeUTF8))
        self.abort_button.setObjectName(_fromUtf8("abort_button"))
        self.horizontalLayout_2.addWidget(self.abort_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.horizontalLayout_5.addWidget(self.groupBox)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.filenameButton = QtGui.QPushButton(self.centralwidget)
        self.filenameButton.setText(QtGui.QApplication.translate("NWAWindow", "H5 File", None, QtGui.QApplication.UnicodeUTF8))
        self.filenameButton.setObjectName(_fromUtf8("filenameButton"))
        self.horizontalLayout_3.addWidget(self.filenameButton)
        self.param_filename = QtGui.QLineEdit(self.centralwidget)
        self.param_filename.setText(QtGui.QApplication.translate("NWAWindow", "S:\\_Data\\", None, QtGui.QApplication.UnicodeUTF8))
        self.param_filename.setObjectName(_fromUtf8("param_filename"))
        self.horizontalLayout_3.addWidget(self.param_filename)
        self.param_save = QtGui.QCheckBox(self.centralwidget)
        self.param_save.setText(QtGui.QApplication.translate("NWAWindow", "Auto-Save", None, QtGui.QApplication.UnicodeUTF8))
        self.param_save.setObjectName(_fromUtf8("param_save"))
        self.horizontalLayout_3.addWidget(self.param_save)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setText(QtGui.QApplication.translate("NWAWindow", "Dataset Path", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout.addWidget(self.label_6)
        self.param_datasetPath = QtGui.QLineEdit(self.centralwidget)
        self.param_datasetPath.setText(QtGui.QApplication.translate("NWAWindow", "/", None, QtGui.QApplication.UnicodeUTF8))
        self.param_datasetPath.setObjectName(_fromUtf8("param_datasetPath"))
        self.horizontalLayout.addWidget(self.param_datasetPath)
        self.param_numberTraces = QtGui.QCheckBox(self.centralwidget)
        self.param_numberTraces.setText(QtGui.QApplication.translate("NWAWindow", "Number Traces", None, QtGui.QApplication.UnicodeUTF8))
        self.param_numberTraces.setObjectName(_fromUtf8("param_numberTraces"))
        self.horizontalLayout.addWidget(self.param_numberTraces)
        self.trace_label = QtGui.QLabel(self.centralwidget)
        self.trace_label.setText(QtGui.QApplication.translate("NWAWindow", "000", None, QtGui.QApplication.UnicodeUTF8))
        self.trace_label.setObjectName(_fromUtf8("trace_label"))
        self.horizontalLayout.addWidget(self.trace_label)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.magplotwidget = CurveWidget(self.centralwidget)
        self.magplotwidget.setOrientation(QtCore.Qt.Horizontal)
        self.magplotwidget.setObjectName(_fromUtf8("magplotwidget"))
        self.verticalLayout.addWidget(self.magplotwidget)
        self.phaseplotwidget = CurveWidget(self.centralwidget)
        self.phaseplotwidget.setOrientation(QtCore.Qt.Horizontal)
        self.phaseplotwidget.setObjectName(_fromUtf8("phaseplotwidget"))
        self.verticalLayout.addWidget(self.phaseplotwidget)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.message_box = QtGui.QTextBrowser(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.message_box.sizePolicy().hasHeightForWidth())
        self.message_box.setSizePolicy(sizePolicy)
        self.message_box.setMaximumSize(QtCore.QSize(16777215, 80))
        self.message_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.message_box.setHtml(QtGui.QApplication.translate("NWAWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.message_box.setObjectName(_fromUtf8("message_box"))
        self.verticalLayout_2.addWidget(self.message_box)
        self.cmd_lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.cmd_lineEdit.setObjectName(_fromUtf8("cmd_lineEdit"))
        self.verticalLayout_2.addWidget(self.cmd_lineEdit)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        NWAWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(NWAWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1084, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(QtGui.QApplication.translate("NWAWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuSave = QtGui.QMenu(self.menuFile)
        self.menuSave.setTitle(QtGui.QApplication.translate("NWAWindow", "Save..", None, QtGui.QApplication.UnicodeUTF8))
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuInstruments = QtGui.QMenu(self.menubar)
        self.menuInstruments.setTitle(QtGui.QApplication.translate("NWAWindow", "Instruments", None, QtGui.QApplication.UnicodeUTF8))
        self.menuInstruments.setObjectName(_fromUtf8("menuInstruments"))
        NWAWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(NWAWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        NWAWindow.setStatusBar(self.statusbar)
        self.actionExperimental_Settings = QtGui.QAction(NWAWindow)
        self.actionExperimental_Settings.setText(QtGui.QApplication.translate("NWAWindow", "Experimental Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_Settings.setObjectName(_fromUtf8("actionExperimental_Settings"))
        self.actionExperimental_and_Instrument_Settings = QtGui.QAction(NWAWindow)
        self.actionExperimental_and_Instrument_Settings.setText(QtGui.QApplication.translate("NWAWindow", "Experimental and Instrument Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExperimental_and_Instrument_Settings.setObjectName(_fromUtf8("actionExperimental_and_Instrument_Settings"))
        self.actionLoad = QtGui.QAction(NWAWindow)
        self.actionLoad.setText(QtGui.QApplication.translate("NWAWindow", "Load", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.menuSave.addAction(self.actionExperimental_Settings)
        self.menuSave.addAction(self.actionExperimental_and_Instrument_Settings)
        self.menuFile.addAction(self.menuSave.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuInstruments.menuAction())

        self.retranslateUi(NWAWindow)
        QtCore.QMetaObject.connectSlotsByName(NWAWindow)

    def retranslateUi(self, NWAWindow):
        pass

from guiqwt.plot import CurveWidget
