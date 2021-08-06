# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:54:11 2013

@author: slab
"""

import PyQt4.Qt as qt
import slab.datamanagement as dm

if __name__ == "__main__":
    app = qt.QApplication([])
    win = qt.QMainWindow()
    server = dm.FileServer(gui=app)
    widget = qt.QWidget()
    layout = qt.QVBoxLayout(widget)
    class FileList(qt.QListWidget):
        def __init__(self):
            qt.QListWidget.__init__(self)
            self.file_position = {}
        def add_file(self, fname):
            row = self.count()
            self.insertItem(row, fname)
            self.file_position[fname] = row
        def remove_file(self, fname):
            self.takeItem(self.file_position[fname])
    filelist = FileList()
    app.connect(app, server.fileAddedSignal, filelist.add_file)
    app.connect(app, server.fileRemovedSignal, filelist.remove_file)
    app.connect(app, server.resetSignal, filelist.clear)
    
    reset_button = qt.QPushButton('Reset Server')
    reset_button.clicked.connect(server.restart)
    layout.addWidget(filelist)
    layout.addWidget(reset_button)
    win.setCentralWidget(widget)
    app.connect(app, qt.SIGNAL("lastWindowClosed()"), server.close)
    win.show()
    app.exec_()