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

DEBUG = False
if DEBUG:
    from PyQt4 import uic
    uifile = open("S:\\_Lib\\python\\slab\\widgets\\SweepDialog_ui.py", 'w')
    uic.compileUi("S:\\_Lib\\python\\slab\\widgets\\SweepDialog.ui", uifile)
    uifile.close()
from SweepDialog_ui import Ui_SweepDialog
from AlazarWidget_ui import Ui_AlazarForm
import PyQt4.Qt as Qt
import guiqwt.plot
import guiqwt.builder

class AlazarWidget(QWidget, Ui_AlazarForm):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)

class SweepDialog(QDialog, Ui_SweepDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

class SlabLinePlot(guiqwt.plot.CurvePlot):
    def __init__(self, ncurves=1):
        guiqwt.plot.CurvePlot.__init__(self)
        self.curves = []
        for i in range(ncurves):
            c = guiqwt.builder.make([],[])
            self.curves.append(c)
            self.add_item(c)
    def update_curve(x, y, curve_n=0):
        self.curves[curve_n].set_data(x,y)
        
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
