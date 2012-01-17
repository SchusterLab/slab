"""
widgets -- Customized PyQt widgets
"""

from PyQt4.Qt import *
import math
import re

class SlabSpinBox(QDoubleSpinBox):
    def __init__(self, precision=4, *args, **kwargs):
        QDoubleSpinBox.__init__(self, *args, **kwargs)
        self.maxv = 10 ** 3
        self.minv = 10 ** -2
        self.precision = precision
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
                print base, base * self.prefixes[p]
                return base * self.prefixes[p]
        return 0.


    def textFromValue(self, value):
        if value > self.maxv or value < self.minv:
            return QString(("%." + str(self.precision) + "e") % value)
        else:
            return QString(("%." + str(self.precision) + "f") % value)
        
        



