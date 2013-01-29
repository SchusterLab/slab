# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:33:23 2013

@author: slab
"""

from PyQt4 import QtGui, QtCore
from subprocess import Popen
import os

slab_dir = 'C:\\_Lib\\python\\slab'

scripts = {
    'HDF Viewer' : [os.path.join(slab_dir, 'plotting\\hdfiview.py')],
    'Script Plotter' : [os.path.join(slab_dir, 'plotting\\script_viewer.py')],
    'NWA Viewer' : [os.path.join(slab_dir, 'scripts\\NWAWindow.pyw')],
    'Instrument Manager' : [os.path.join(slab_dir, 'instruments\\instrumentmanager.py'), '-g'],
    'File Server' : [os.path.join(slab_dir, 'datamanagement.py')]
}

def create_popen_fn(args):
    env = os.environ
    env.update({'PYTHONPATH':'C:\\_Lib\\python'})
    return lambda: Popen(['pythonw'] + args, env=env)

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