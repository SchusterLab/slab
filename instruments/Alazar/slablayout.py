# -*- coding: utf-8 -*-
"""
formlayout
==========

Module creating PyQt4 form dialogs/layouts to edit various type of parameters


formlayout License Agreement (MIT License)
------------------------------------------

Copyright (c) 2009 Pierre Raybaut

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

# History:
# 1.0.7: added support for "Apply" button
# 1.0.6: code cleaning

__version__ = '1.0.9'
__license__ = __doc__

DEBUG = False

import sys
STDERR = sys.stderr

try:
    from PyQt4.QtGui import QFormLayout
except ImportError:
    raise ImportError("Warning: formlayout requires PyQt4 >v4.3")

from PyQt4.QtGui import (QWidget, QLineEdit, QComboBox, QLabel, QSpinBox, 
                         QDoubleSpinBox, QIcon, QStyle, QDialogButtonBox, 
                         QHBoxLayout, QVBoxLayout, QDialog, QColor, QPushButton, 
                         QCheckBox, QColorDialog, QPixmap, QTabWidget, 
                         QApplication, QStackedWidget, QDateEdit, QDateTimeEdit,
                         QFont, QFontComboBox, QFontDatabase, QGridLayout)

from PyQt4.QtCore import (Qt, SIGNAL, SLOT, QSize,
                          pyqtSignature, pyqtProperty)
QString = str

import datetime


class ColorButton(QPushButton):
    """
    Color choosing push button
    """
    __pyqtSignals__ = ("colorChanged(QColor)",)
    
    def __init__(self, parent=None):
        QPushButton.__init__(self, parent)
        self.setFixedSize(20, 20)
        self.setIconSize(QSize(12, 12))
        self.connect(self, SIGNAL("clicked()"), self.choose_color)
        self._color = QColor()
    
    def choose_color(self):
        rgba, valid = QColorDialog.getRgba(self._color.rgba(),
                                           self.parentWidget())
        if valid:
            color = QColor.fromRgba(rgba)
            self.set_color(color)
    
    def get_color(self):
        return self._color
    
    @pyqtSignature("QColor")
    def set_color(self, color):
        if color != self._color:
            self._color = color
            self.emit(SIGNAL("colorChanged(QColor)"), self._color)
            pixmap = QPixmap(self.iconSize())
            pixmap.fill(color)
            self.setIcon(QIcon(pixmap))
    
    color = pyqtProperty("QColor", get_color, set_color)


def text_to_qcolor(text):
    """
    Create a QColor from specified string
    Avoid warning from Qt when an invalid QColor is instantiated
    """
    color = QColor()
    if isinstance(text, QString):
        text = str(text)
    if not isinstance(text, str):
        return color
    if text.startswith('#') and len(text)==7:
        correct = '#0123456789abcdef'
        for char in text:
            if char.lower() not in correct:
                return color
    elif text not in list(QColor.colorNames()):
        return color
    color.setNamedColor(text)
    return color


class ColorLayout(QHBoxLayout):
    """Color-specialized QLineEdit layout"""
    def __init__(self, color, parent=None):
        QHBoxLayout.__init__(self)
        assert isinstance(color, QColor)
        self.lineedit = QLineEdit(color.name(), parent)
        self.connect(self.lineedit, SIGNAL("textChanged(QString)"),
                     self.update_color)
        self.addWidget(self.lineedit)
        self.colorbtn = ColorButton(parent)
        self.colorbtn.color = color
        self.connect(self.colorbtn, SIGNAL("colorChanged(QColor)"),
                     self.update_text)
        self.addWidget(self.colorbtn)

    def update_color(self, text):
        color = text_to_qcolor(text)
        if color.isValid():
            self.colorbtn.color = color

    def update_text(self, color):
        self.lineedit.setText(color.name())
        
    def text(self):
        return self.lineedit.text()
    
    
def font_is_installed(font):
    """Check if font is installed"""
    return [fam for fam in QFontDatabase().families() if str(fam)==font]

def tuple_to_qfont(tup):
    """
    Create a QFont from tuple:
        (family [string], size [int], italic [bool], bold [bool])
    """
    if not isinstance(tup, tuple) or len(tup) != 4 \
       or not font_is_installed(tup[0]) \
       or not isinstance(tup[1], int) \
       or not isinstance(tup[2], bool) \
       or not isinstance(tup[3], bool):
        return None
    font = QFont()
    family, size, italic, bold = tup
    font.setFamily(family)
    font.setPointSize(size)
    font.setItalic(italic)
    font.setBold(bold)
    return font

def qfont_to_tuple(font):
    return (str(font.family()), int(font.pointSize()),
            font.italic(), font.bold())


class FontLayout(QGridLayout):
    """Font selection"""
    def __init__(self, value, parent=None):
        QGridLayout.__init__(self)
        font = tuple_to_qfont(value)
        assert font is not None
        
        # Font family
        self.family = QFontComboBox(parent)
        self.family.setCurrentFont(font)
        self.addWidget(self.family, 0, 0, 1, -1)
        
        # Font size
        self.size = QComboBox(parent)
        self.size.setEditable(True)
        sizelist = list(range(6, 12)) + list(range(12, 30, 2)) + [36, 48, 72]
        size = font.pointSize()
        if size not in sizelist:
            sizelist.append(size)
            sizelist.sort()
        self.size.addItems([str(s) for s in sizelist])
        self.size.setCurrentIndex(sizelist.index(size))
        self.addWidget(self.size, 1, 0)
        
        # Italic or not
        self.italic = QCheckBox(self.tr("Italic"), parent)
        self.italic.setChecked(font.italic())
        self.addWidget(self.italic, 1, 1)
        
        # Bold or not
        self.bold = QCheckBox(self.tr("Bold"), parent)
        self.bold.setChecked(font.bold())
        self.addWidget(self.bold, 1, 2)
        
    def get_font(self):
        font = self.family.currentFont()
        font.setItalic(self.italic.isChecked())
        font.setBold(self.bold.isChecked())
        font.setPointSize(int(self.size.currentText()))
        return qfont_to_tuple(font)


class FormWidget(QWidget):
    def __init__(self, data, comment="", parent=None):
        super(FormWidget, self).__init__(parent)
        from copy import deepcopy
        self.data = deepcopy(data)
        self.widgets = []
        self.names=[]
        self.formlayout = QFormLayout(self)
        if comment:
            self.formlayout.addRow(QLabel(comment))
            self.formlayout.addRow(QLabel(" "))
        if DEBUG:
            print("\n"+("*"*80))
            print("DATA:", self.data)
            print("*"*80)
            print("COMMENT:", comment)
            print("*"*80)
        self.setup()

    def setup(self):
        for name,label, value in self.data:
            if DEBUG:
                print("value:", value)
            if not name is None:
                self.names.append(name)
            if name is None and label is None and value is None:
                # Separator: (None, None)
                self.formlayout.addRow(QLabel(" "), QLabel(" "))
                self.widgets.append(None)
                continue
            elif label is None:
                # Comment
                self.formlayout.addRow(QLabel(value))
                self.widgets.append(None)
                continue
            elif tuple_to_qfont(value) is not None:
                field = FontLayout(value, self)
            elif text_to_qcolor(value).isValid():
                field = ColorLayout(QColor(value), self)
            elif isinstance(value, str):
                field = QLineEdit(value, self)
            elif isinstance(value, (list, tuple)):
                selindex = value.pop(0)
                field = QComboBox(self)
                if isinstance(value[0], (list, tuple)):
                    keys = [ key for key, _val in value ]
                    value = [ val for _key, val in value ]
                else:
                    keys = value
                field.addItems(value)
                if selindex in value:
                    selindex = value.index(selindex)
                elif selindex in keys:
                    selindex = keys.index(selindex)
                elif not isinstance(selindex, int):
                    print("Warning: '%s' index is invalid (label: " \
                                    "%s, value: %s)" % (selindex, label, value), file=STDERR)
                    selindex = 0
                field.setCurrentIndex(selindex)
            elif isinstance(value, bool):
                field = QCheckBox(self)
                field.setCheckState(Qt.Checked if value else Qt.Unchecked)
            elif isinstance(value, float):
                field = QLineEdit(repr(value), self)
#                field = QDoubleSpinBox(self)
#                field.setRange(-1e9, 1e9)
#                field.setValue(value)
            elif isinstance(value, int):
                field = QSpinBox(self)
                field.setRange(-1e9, 1e9)
                field.setValue(value)
            elif isinstance(value, datetime.datetime):
                field = QDateTimeEdit(self)
                field.setDateTime(value)
            elif isinstance(value, datetime.date):
                field = QDateEdit(self)
                field.setDate(value)
            else:
                field = QLineEdit(repr(value), self)
            self.formlayout.addRow(label, field)
            self.widgets.append(field)
            
    def get(self):
        valuelist = []
        params={}
        for index, (name,label, value) in enumerate(self.data):
            field = self.widgets[index]
            if label is None:
                # Separator / Comment
                continue
            elif tuple_to_qfont(value) is not None:
                value = field.get_font()
            elif isinstance(value, str):
                value = str(field.text())
            elif isinstance(value, (list, tuple)):
                index = int(field.currentIndex())
                if isinstance(value[0], (list, tuple)):
                    value = value[index][0]
                else:
                    value = value[index]
            elif isinstance(value, bool):
                value = field.checkState() == Qt.Checked
            elif isinstance(value, float):
                value = float(field.text())
            elif isinstance(value, int):
                value = int(field.value())
            elif isinstance(value, datetime.datetime):
                value = field.dateTime().toPyDateTime()
            elif isinstance(value, datetime.date):
                value = field.date().toPyDate()
            else:
                value = eval(str(field.text()))
            valuelist.append(value)
            params[name]=value
        return params


class FormComboWidget(QWidget):
    def __init__(self, datalist, comment="", parent=None):
        super(FormComboWidget, self).__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.combobox = QComboBox()
        layout.addWidget(self.combobox)
        
        self.stackwidget = QStackedWidget(self)
        layout.addWidget(self.stackwidget)
        self.connect(self.combobox, SIGNAL("currentIndexChanged(int)"),
                     self.stackwidget, SLOT("setCurrentIndex(int)"))
        
        self.widgetlist = []
        for data, title, comment in datalist:
            self.combobox.addItem(title)
            widget = FormWidget(data, comment=comment, parent=self)
            self.stackwidget.addWidget(widget)
            self.widgetlist.append(widget)

    def get(self):
        return [ widget.get() for widget in self.widgetlist]
        

class FormTabWidget(QWidget):
    def __init__(self, datalist, comment="", parent=None):
        super(FormTabWidget, self).__init__(parent)
        layout = QVBoxLayout()
        self.tabwidget = QTabWidget()
        layout.addWidget(self.tabwidget)
        self.setLayout(layout)
        self.widgetlist = []
        for data, title, comment in datalist:
            if len(data[0])==3:
                widget = FormComboWidget(data, comment=comment, parent=self)
            else:
                widget = FormWidget(data, comment=comment, parent=self)
            index = self.tabwidget.addTab(widget, title)
            self.tabwidget.setTabToolTip(index, comment)
            self.widgetlist.append(widget)
            
    def get(self):
        return [ widget.get() for widget in self.widgetlist]


class FormDialog(QDialog):
    """Form Dialog"""
    def __init__(self, data, title="", comment="",
                 icon=None, parent=None, apply=None):
        super(FormDialog, self).__init__(parent)

        self.apply_callback = apply
        
        # Form
        if isinstance(data[0][0], (list, tuple)):
            self.formwidget = FormTabWidget(data, comment=comment,
                                            parent=self)
        elif len(data[0])==4:
            self.formwidget = FormComboWidget(data, comment=comment,
                                              parent=self)
        else:
            self.formwidget = FormWidget(data, comment=comment, 
                                         parent=self)
        layout = QVBoxLayout()
        layout.addWidget(self.formwidget)
        
        # Button box
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        if self.apply_callback is not None:
            apply_btn = bbox.addButton(QDialogButtonBox.Apply)
            self.connect(apply_btn, SIGNAL("clicked()"), self.apply)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        layout.addWidget(bbox)

        self.setLayout(layout)
        
        self.setWindowTitle(title)
        if not isinstance(icon, QIcon):
            icon = QWidget().style().standardIcon(QStyle.SP_MessageBoxQuestion)
        self.setWindowIcon(icon)
        
    def accept(self):
        self.data = self.formwidget.get()
        QDialog.accept(self)
        
    def reject(self):
        self.data = None
        QDialog.reject(self)
        
    def apply(self):
        self.apply_callback(self.formwidget.get())
        
    def get(self):
        """Return form result"""
        return self.data


def fedit(data, title="", comment="", icon=None, parent=None, apply=None):
    """
    Create form dialog and return result
    (if Cancel button is pressed, return None)
    
    data: datalist, datagroup
    title: string
    comment: string
    icon: QIcon instance
    parent: parent QWidget
    apply: apply callback (function)
    
    datalist: list/tuple of (field_name, field_value)
    datagroup: list/tuple of (datalist *or* datagroup, title, comment)
    
    -> one field for each member of a datalist
    -> one tab for each member of a top-level datagroup
    -> one page (of a multipage widget, each page can be selected with a combo
       box) for each member of a datagroup inside a datagroup
       
    Supported types for field_value:
      - int, float, str, unicode, bool
      - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)
                (automatically detected from a string)
      - list/tuple:
          * the first element will be the selected index (or value)
          * the other elements can be couples (key, value) or only values
    """
    
    # Create a QApplication instance if no instance currently exists
    # (e.g. if the module is used directly from the interpreter)
    if QApplication.startingUp():
        _app = QApplication([])
        
    dialog = FormDialog(data, title, comment, icon, parent, apply)
    if dialog.exec_():
        return dialog.get()



if __name__ == "__main__":

    def create_datalist_example():
        return [('str', 'this is a string'),
                ('list', [0, '1', '3', '4']),
                ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
                           ('-.', 'DashDot'), ('-', 'Solid'),
                           ('steps', 'Steps'), (':', 'Dotted')]),
                ('float', 1.2),
                (None, 'Other:'),
                ('int', 12),
                ('font', ('Arial', 10, False, True)),
                ('color', '#123409'),
                ('bool', True),
                ('date', datetime.date(2010, 10, 10)),
                ('datetime', datetime.datetime(2010, 10, 10)),
                ]
    def create_datalist_example2():
        return [('str', 'this is a string'),
                ('list', [0, '1', '3', '4']),
                ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
                           ('-.', 'DashDot'), ('-', 'Solid'),
                           ('steps', 'Steps'), (':', 'Dotted')]),
                ('float', 1.2),
                (None, 'Other:'),
                ('bool', True),
                ('date', datetime.date(2010, 10, 10)),
                ('datetime', datetime.datetime(2010, 10, 10)),
                ]        
    def create_datagroup_example():
        datalist = create_datalist_example()
        datalist2 = create_datalist_example2()
        return ((datalist, "Category 1", "Category 1 comment"),
                (datalist2, "Category 2", "Category 2 comment"),
                (datalist, "Category 3", "Category 3 comment"))
    
    #--------- datalist example
    datalist = create_datalist_example()
    def apply_test(data):
        print("data:", data)
    print("result:", fedit(datalist, title="Example",
                           comment="This is just an <b>example</b>.",
                           apply=apply_test))
    
    #--------- datagroup example
    datagroup = create_datagroup_example()
    print("result:", fedit(datagroup, "Global title"))
        
    #--------- datagroup inside a datagroup example
    datalist = create_datalist_example()
    datagroup = create_datagroup_example()
    print("result:", fedit(((datagroup, "Title 1", "Tab 1 comment"),
                            (datalist, "Title 2", "Tab 2 comment"),
                            (datalist, "Title 3", "Tab 3 comment")),
                            "Global title"))
