# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:33:23 2013

@author: slab
"""

from PyQt4 import QtGui, QtCore
from subprocess import Popen
import os

slab_dir = r'C:\_Lib\python\slab'


def python_cmd(relpath):
    return ['pythonw', os.path.join(slab_dir, relpath)]


def select_dir():
    dir = str(QtGui.QFileDialog().getExistingDirectory())
    os.chdir(dir)
    return []

scripts = {
    'HDF Viewer' : python_cmd(r'plotting\hdfiview.py'),
    'Script Plotter' : python_cmd(r'plotting\script_viewer.py'),
    'LivePlot' : python_cmd(r'..\liveplot\window.py'),
    'NWA Viewer' : python_cmd(r'scripts\NWAWindow.pyw'),
    'Instrument Manager' : python_cmd(r'instruments\instrumentmanager.py') + ['-g'],
    'File Server' : python_cmd('datamanagement.py'),
    'IPython Notebook' : lambda: select_dir() + ['ipython', 'notebook', '--pylab', 'inline', '--ip=*']
}


def create_popen_fn(args):
    env = os.environ
    env.update({'PYTHONPATH':r'C:\_Lib\python'})
    fn_type = type(lambda: 1)
    if isinstance(args, fn_type):
        return lambda: Popen(args(), env=env)
    return lambda: Popen(args, env=env)

if __name__ == "__main__":
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    widget = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(widget)
    for name, args in scripts.items():
        button = QtGui.QPushButton(name)
        button.clicked.connect(create_popen_fn(args))
        layout.addWidget(button)
    win.setCentralWidget(widget)
    win.show()
    app.exec_()