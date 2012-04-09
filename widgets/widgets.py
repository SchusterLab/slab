"""
widgets -- Customized PyQt widgets

A note on Qt Designer usage: In order to use these custom widgets in
Qt Designer, first put a place holder widget (if applicable, a member
of the base class of the widget being used), into your designed
gui. Right click on the element, and select 'Promote to...', If the
widget has not been previously promoted, write the widget name in the
promoted class name field (e.g. SlabSpinBox), 'slab.h' in the header
field (this will be appropriately translated by pyuic), and click
'add', then 'Promote'.
"""

from PyQt4.Qt import *
import guiqwt.plot
import math
import re

DEBUG = True
if DEBUG:
    from PyQt4 import uic
    uifile = open("S:\\_Lib\\python\\slab\\widgets\\SweepDialog_ui.py", 'w')
    uic.compileUi("S:\\_Lib\\python\\slab\\widgets\\SweepDialog.ui", uifile)
    uifile.close()
from SweepDialog_ui import Ui_SweepDialog
from AlazarWidget_ui import Ui_AlazarForm
#from slab import gui

class AlazarWidget(QWidget, Ui_AlazarForm):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)

class SweepDialog(QDialog, Ui_SweepDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

#    def run_sweep(self):
#        for p1 in range(p1start, p1end, p1step):
#            for p2 in range(p2start, p2end, p2step):
#                for action in actionslist:
#                    self.slab
#        self.slabwindow.msg(self.sweep_dialog.actions_comboBox.text())


class PlotWithTB(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent, Qt.Widget)
        self.plot = guiqwt.plot.CurveWidget()
        self.setCentralWidget(self.plot)
        self.toolbar = QToolBar(self)
        self.plot.add_toolbar(self.toolbar)
        self.plot.register_all_curve_tools()
    def __getattr__(self, item):
        return self.plot.__getattr__(item)
    

class SlabSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop('precision') if kwargs.has_key('precision') else 4
        QDoubleSpinBox.__init__(self, *args, **kwargs)
        self.maxv = 10 ** 3
        self.minv = 10 ** -2
        self.prefixes = { "n|nano" : 10 ** -9,
                          "u|micro" : 10 ** -6,
                          "milli" : 10 ** -3,
                          "k|kilo" : 10 ** 3,
                          "m|mega" : 10 ** 6,
                          "g|giga" : 10 ** 9,
                          "t|tera" : 10 ** 12,
                     }
        pfkeys = "".join([ key for key in self.prefixes ])
        self.RE = QRegExp(
            "^[\+-]?[0-9]+(\.[0-9]*)?(e[\+-]?[0-9]+|(" + "|".join(pfkeys) + "))?$",
            cs=Qt.CaseInsensitive)
        self.REValidator = QRegExpValidator(self.RE, self)
        #        self.lineEdit().setValidator(self.REValidator)

    def validate(self, text_, pos):
        return self.REValidator.validate(text_, pos)

    def valueFromText(self, text_):
        # Convert to python string for predictability
        text = str(text_).lower().replace(" ","")
        if QRegExp("^[\+-]?[0-9]+(\.[0-9]*)?$").exactMatch(text):
            return float(text)
        if QRegExp("^[\+-]?[0-9]+(\.[0-9]*)?e[\+-]?[0-9]+$").exactMatch(text):
            base, exp = text.split('e')
            return float(base) * (10 ** int(exp))
        for p in self.prefixes:
            if QRegExp("^[\+-]?[0-9]+(\.[0-9]*)?(" + p + ")$").exactMatch(text):
                base = float("".join([ c for c in text if c.isdigit() or c == '.']))
                return base * self.prefixes[p]
        return 0.


    def textFromValue(self, value):
        if value > self.maxv or value < self.minv:
            return QString(("%." + str(self.precision) + "e") % value)
        else:
            return QString(("%." + str(self.precision) + "f") % value)
