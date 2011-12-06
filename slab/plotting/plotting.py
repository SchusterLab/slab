# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:00:29 2011

@author: Dave
"""

from multiprocessing import Process,Pipe

import sys,os,time,traceback
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import QSize, QT_VERSION_STR, PYQT_VERSION_STR, Qt, SIGNAL,QTimer
from PyQt4 import QtCore

from guiqwt.plot import CurveWidget, ImageWidget
from guiqwt.builder import make

#from slab import *
from Server_ui import *

LOGFILENAME='log.txt'
LOGENABLED=True



def make_server_window(pipe=None,xlabel=None,ylabel=None,title=None):
    try:
        write_log('Launching Server App')
        app = QApplication(sys.argv)
        window =ServerWindow(pipe,xlabel,ylabel,title)
        window.show()
        sys.exit(app.exec_())
    except:
        if LOGENABLED: log_error()
        else: pass

class FigureClient():
    def __init__(self,xlabel=None,ylabel=None,title=None):            
        self.parent,self.child=Pipe()
        self.process = Process(target=make_server_window, args=(self.child,xlabel,ylabel,title))
        self.process.start()

    def send(self,name,*args,**kw):
        self.parent.send((name,args,kw))

    def send_command(self,name):
        return lambda *args, **kw: self.send (name,*args,**kw)
        
    def __getattr__(self,name):
        if name not in ('send','send_command'):
            return self.send_command(name)



class test_proxy():
    def __getattr__(self,name):
        if name not in ('send','send_command'):
            return self.send_command(name)
        
    def send(self,name,*args,**kw):
        return (name,args,kw)

    def send_command(self,name):
        return lambda *args, **kw: self.send (name,*args,**kw)


def clear_log():
    try: os.remove(LOGFILENAME)    
    except: pass

def log_error(info=''):
    write_log("Error: "+ info + '\n'.join(traceback.format_exception(*sys.exc_info())))

def print_log():
    try:
        f=open(LOGFILENAME,'r')
        print f.read()
        f.close()
    except:
        print "Log file empty/non-existant."

def write_log(s):
    f=open(LOGFILENAME,'a')
    f.write(s+'\n')
    f.close()

class ServerWindow (QMainWindow, Ui_ServerWindow):

    def __init__(self, pipe = None,xlabel=None,ylabel=None,title=None,parent = None):
        try:
            if LOGENABLED: write_log('Initializing server window')
            QMainWindow.__init__(self, parent)
            self.setupUi(self)
            self.setWindowTitle(title) 
            self.pipe=pipe
            self.setup_plot(title, xlabel, ylabel)
            write_log ('Starting timer')
            self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
            QtCore.QObject.connect(self.ctimer, QtCore.SIGNAL("timeout()"), self.process_command )
#            self.connect()            
            self.ctimer.start(10)
            if LOGENABLED: write_log('timer started')
        except:
            if LOGENABLED: log_error()
            else: pass

    def setup_plot(self, title, xlabel, ylabel):
#        self.curvewidget = CurveWidget()
#        self.setCentralWidget(self.curvewidget)
        
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        self.plot = self.curvewidget.plot   
        x=np.linspace(-5,5,1000)        #Create some sample curves
        y1=np.cos(x)
        self.plot_item = make.mcurve(x, y1,label='Magnitude') #Make Ch1 curve
        self.plot.add_item(self.plot_item)
        self.plot.set_titles(title=title, xlabel=xlabel, ylabel=ylabel)

    def process_command(self):
        self.ctimer.stop()
        #write_log('process command')
        try:
            if self.pipe!=None:
                while self.pipe.poll():
                    cmd=self.pipe.recv()
                    #write_log('Received command: '+cmd[0])
                    ans=getattr(self,cmd[0])(*cmd[1],**cmd[2])
        except:
            if LOGENABLED: log_error('process command')
            else: pass
        self.ctimer.start(10)
            
    def blah(self,cmd):
        return 'blahblah'
        
    def update_plot(self, data):
        x, y = data
        self.plot_item.set_data(x, y)
        self.plot.replot()
        
class ImageFigureClient(FigureClient):
    def setup_plot(self):
        self.imagewidget = ImageWidget()
        self.plot = self.imagewidget.plot
        self.image = make.image(np.random.rand(100,100))
        self.plot.add_item(self.image)
    def update_plot(self, data):
        self.image.set_data(data)

def test():
    clear_log()
    #write_log('test')
    fig1=FigureClient(xlabel='Frequency',ylabel='Amplitude',title='Parabola')
    fig2=FigureClient(xlabel='Frequency',ylabel='Amplitude',title='Cubic')
    time.sleep(4)
    #client.parent.send(('blah',))
    x=np.linspace(-5,5,100)
    y=x**2
    yy=x**3
    x2=[]
    y2=[]
    x3=[]
    y3=[]
    for ii in range (100):
        x2.append(x[ii])
        y2.append(y[ii])
        x3.append(x[ii])
        y3.append(yy[ii])
#        client.parent.send(('update_plot',x2,y2))
        fig1.update_plot((x2,y2))
        fig2.update_plot((x3,y3))
        time.sleep(0.1)
    print_log()        


if __name__ == "__main__":
    test()

#    t=test_proxy()
#    b=t.blah('a','b','c')