# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:36:16 2011

@author: Dave
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:18:46 2011

@author: Dave
"""

#Imports

from multiprocessing import Process,Pipe
import os
from numpy import *
import time

import guidata

from guidata.dataset.datatypes import (DataSet, BeginTabGroup, EndTabGroup,
                                       BeginGroup, EndGroup, ObjectItem)
from guidata.dataset.dataitems import (FloatItem, IntItem, BoolItem, ChoiceItem,
                             MultipleChoiceItem, ImageChoiceItem, FilesOpenItem,
                             StringItem, TextItem, ColorItem, FileSaveItem,
                             FileOpenItem, DirectoryItem, FloatArrayItem)

from guidata.dataset.qtwidgets import DataSetEditLayout, DataSetShowLayout,DataSetEditGroupBox
from guidata.dataset.qtitemwidgets import DataSetWidget
from guidata.qthelpers import create_action, add_actions, get_std_icon

from guiqwt.pyplot import *
from guiqwt.plot import ImageWidget,CurveWidget
from guiqwt.builder import make

#from guiqwt.io import imagefile_to_array, IMAGE_LOAD_FILTERS
####

import numpy as np

from PyQt4.QtGui import (QMainWindow, QMessageBox, QSplitter, QListWidget,
                         QFileDialog,QSizePolicy,QPushButton)
from PyQt4.QtCore import QSize, QT_VERSION_STR, PYQT_VERSION_STR, Qt, SIGNAL,QTimer
from PyQt4 import QtCore
   
   
"""
SlabWindow is a default base class for making a GUI script
In order to use SlabWindow properly you must define:
    setup_widgets()
    load_bookmark(filename)
    save_bookmark(filename)
There are several useful variables/methods to make it easy to create the gui
    self.paramswidget (this a vertical qsplitter on the left)    
    self.plotswidget (this is a vertical qsplitter on the right)
    self.add_plot_updater (self,plot_name,plot_type,plot,plot_item) adds an item to the autoupdate table
    self.status has a reference to the status bar
"""
class SlabWindow(QMainWindow):
    def __init__(self, icon_fname=None,title="My application",size=(1024,768),status_msg="Status bar"):
        """Setup window parameters"""
        QMainWindow.__init__(self)  #Call super constructor
        if icon_fname is None:      #If no icon specified use default python icon
            self.setWindowIcon(get_icon('python.png'))
        else:
            self.setWindowIcon(get_icon(icon_fname))
            
        self.setWindowTitle(title)              #Set window title
        self.resize(QSize(size[0],size[1]))     #Set window size
        self.status = self.statusBar()          #Define status bar
        self.status.showMessage(status_msg, 5000)    #put default status message
        
        self.mainwidget= QSplitter()    #Setup central widget to be a horizontal QSplitter
        self.mainwidget.setContentsMargins(10, 10, 10, 10)
        self.mainwidget.setOrientation(Qt.Horizontal)
        self.mainwidget.toolbar=self.addToolBar("Image")
        self.setCentralWidget(self.mainwidget)   
        
        self.paramswidget= QSplitter()  #parameters on left
        self.paramswidget.setContentsMargins(10, 10, 10, 10)
        self.paramswidget.setOrientation(Qt.Vertical) 
        
        self.plotswidget= QSplitter()   # and plot widgets on right
        self.plotswidget.setContentsMargins(10, 10, 10, 10)
        self.plotswidget.setOrientation(Qt.Vertical) 
        
        self.mainwidget.addWidget(self.paramswidget)    #Add params widget
        self.mainwidget.addWidget(self.plotswidget)     #and plots widget to window

        self.setupUpdaters()
        self.setupMenus()
        self.setupWidgets()

    def setupUpdaters(self):
        self.plot_update_list={}    #Dictionary of plots with format {'plot_type':plot_type,'plot':plot,'plot_item':plot_item,'parent_conn':parent_conn,'child_conn':child_conn}
        self.pipes={}               #Dictionary of (child) pipes to pass to processor
        self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
        QtCore.QObject.connect(self.ctimer, QtCore.SIGNAL("timeout()"), self.update_plots)
        self.ctimer.start(0.1)

    def setupMenus(self):
                # File menu
        file_menu = self.menuBar().addMenu("File")
        prefix_action = create_action(self, "Prefix...",
                                   shortcut="Ctrl+N",
                                   icon=get_icon('filenew.png'),
                                   tip="Select File prefix",
                                   triggered=self.select_prefix_action)
        load_bookmark_action = create_action(self, "Load Bookmark...",
                                             shortcut="Ctrl+O",
                                             icon=get_icon('fileopen.png'),
                                             tip="Load Bookmark",
                                             triggered=self.load_bookmark_action)
        save_bookmark_action = create_action(self, "Save Bookmark...",
                                             shortcut="Ctrl+S",
                                             icon=get_icon('filesave.png'),
                                             tip="Save Bookmark",
                                             triggered=self.save_bookmark_action)
        quit_action = create_action(self, "Quit",
                                    shortcut="Ctrl+Q",
                                    icon=get_std_icon("DialogCloseButton"),
                                    tip="Quit application",
                                    triggered=self.close)
        add_actions(file_menu, (prefix_action, None,load_bookmark_action, save_bookmark_action,None, quit_action))
        
        # Help menu
        help_menu = self.menuBar().addMenu("?")
        about_action = create_action(self, "About...",
                                     icon=get_std_icon('MessageBoxInformation'))#,
#                                     triggered=self.about)
        add_actions(help_menu, (about_action,))

    def select_prefix_action(self):
        self.ctimer.stop()
        filename = QFileDialog.getSaveFileName(self, "Set Prefix", "")
        if filename:
            self.prefix=filename
        self.ctimer.start(0.1)

    def load_bookmark_action(self):
        self.ctimer.stop()
        filename = QFileDialog.getOpenFileName(self, "Load Bookmark", "")
        if filename: self.load_bookmark(filename)
        self.ctimer.start(0.1)

    def load_bookmark(self,filename):
        print filename

    def save_bookmark_action(self):
        self.ctimer.stop()
        filename = QFileDialog.getSaveFileName(self, "Save Bookmark", "")
        if filename: self.save_bookmark(filename)
        self.ctimer.start(0.1)

    def save_bookmark(self,filename):
        print filename


    def add_plot_updater (self,plot_name,plot_type,plot,plot_item):
        parent_conn,child_conn=Pipe()
        self.plot_update_list[plot_name]={'plot_type':plot_type,'plot':plot,'plot_item':plot_item,'parent_conn':parent_conn,'child_conn':child_conn}
        self.pipes[plot_name]=child_conn

    def update_plots(self):
        for name,update_data in self.plot_update_list.iteritems():
            if update_data['plot_type'] == "line_plot":            
                self.update_lineplot(update_data)
                
    def update_lineplot(self, update_data):
        pconn=update_data['parent_conn']        
        if pconn!=None:
            x=None
            while pconn.poll():
                x,y=pconn.recv()
            if x!=None: 
                #---Update curve
                update_data['plot_item'].set_data(x, y)
                update_data['plot'].replot()
                #---        

#class SlabScript:
#    
#    params=MyDataSet()    
#    
#    def show_gui(self):
#        app=guidata.qapplication()
#        #self.params.edit()
#        window=SlabWindow()
#        #print self.params
#        window.show()
#        app.exec_()
#    def setupMenus(self):
#                # File menu
#        file_menu = self.menuBar().addMenu("File")
#        new_action = create_action(self, "New...",
#                                   shortcut="Ctrl+N",
#                                   icon=get_icon('filenew.png'),
#                                   tip="Create a new image")#,
##                                   triggered=self.new_image)
#        open_action = create_action(self, "Open...",
#                                    shortcut="Ctrl+O",
#                                    icon=get_icon('fileopen.png'),
#                                    tip="Open an image")#,
##                                    triggered=self.open_image)
#        quit_action = create_action(self, "Quit",
#                                    shortcut="Ctrl+Q",
#                                    icon=get_std_icon("DialogCloseButton"),
#                                    tip="Quit application")#,
##                                    triggered=self.close)
#        add_actions(file_menu, (new_action, open_action, None, quit_action))
#        
#        # Help menu
#        help_menu = self.menuBar().addMenu("?")
#        about_action = create_action(self, "About...",
#                                     icon=get_std_icon('MessageBoxInformation'))#,
##                                     triggered=self.about)
#        add_actions(help_menu, (about_action,))    
    
if __name__ == "__main__":
    # Create QApplication
    script=SlabScript()
    script.show_gui()
    