#!/usr/bin/python

"""
script.py -- helpers for visualizing data-taking scripts

usage:
    from script import *
    
    def my_fig():
        persistent_args = ...
        figure = ...
        return (figure, args)
        # or just (return figure)
    
    def my_fig_updater(fig, args, data):
        # or just (my_fig_updater(fig, data))
        ...
        update_figure(args, data)
    
    ... 
    
    p1 = ScriptPlot("Plot 1", my_fig, my_fig_updater)
    p2 = ScriptPlot("Plot 2", my_other_fig, my_other_fig_updater)
    
    while True:
        data1, data2 = get_data()
        p1.send_data(data)
        p2.send_data(data)
    
"""

from multiprocessing import Process, Pipe
from time import sleep

import PyQt4.Qt as Qt
import PyQt4.Qwt5 as Qwt
import PyQt4.QtGui as QtGui

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import sys, os, traceback


# Log for thread debugging
LOGFILENAME = "C:\\Users\\phil\\Documents\\log.txt"
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


SPThread = None

class ScriptPlot(object):
    """
    User must provide:
        - A name (any identifier)
        - A function make_fig()
          which returns a matplotlib figure
        - A function update_fn(figure, data) 
          which updates the figures state to reflect new data

          e.g. for a line_plot
          fig = Figure
    """
    def __init__(self, name, make_fig, update_fn):
        global SPThread
        if not SPThread:
            SPThread = ScriptPlotThread()

        self.parent, child = Pipe()
        print self.parent, child
        SPThread.add_fig(name, child, make_fig, update_fn)
        self.name = name

    def send_data(self, data):
        print "sending"
        self.parent.send(data)


class ScriptPlotThread(Process):
    def __init__(self, sleep_time=100):
        Process.__init__(self)
        
        self.pnames = []
        self.pipes = {}
        self.make_figures = {}
        self.args = {}
        self.figures = {}
        self.plots = {}
        self.update_fns = {}

        self.sleep_time=sleep_time

    def add_fig(self, name, pipe, make_fig, update_fn):
        "(Probably?) only works before process has been started" 
        self.pnames.append(name)
        self.pipes[name] = pipe
        self.make_figures[name] = make_fig
        self.update_fns[name] = update_fn

    def run(self):
        self.logfile = open("C:\\Users\\phil\\Documents\\logfile", "w")
        self.app = Qt.QApplication([])
        self.win = Qt.QMainWindow()
        self.cwidget = Qt.QSplitter(Qt.Qt.Vertical)
        self.win.setCentralWidget(self.cwidget)

        for name in self.pnames:
            result = self.make_figures[name]()
            try:
                self.figures[name], self.args[name] = result
            except:
                self.figures[name] = result
            self.figures[name] = self.make_figures[name]()
            self.figures[name].suptitle(name)
            plot = FigureCanvas(self.figures[name])
            self.plots[name] = plot
            self.cwidget.addWidget(plot)
        
        self.timer = Qt.QTimer()
        self.app.connect(self.timer, Qt.SIGNAL("timeout()"), self.wake)
        self.timer.start(self.sleep_time)

        self.win.show()
        self.app.exec_()

    def wake(self):
        for name in self.pnames:
            pipe = self.pipes[name]
            if pipe.poll():
                figure, plot, update_fn =\
                    (self.figures[name], self.plots[name], self.update_fns[name])
                
                data = pipe.recv()
                try:
                    update_fn(figure, self.args[name], data)
                except:
                    update_fn(figure, data)
                plot.draw()
                
        self.timer.start(self.sleep_time)
        
def go():
    SPThread.start()

# Example usage
def simple_fig():
    f = Figure()
    axes = f.add_subplot(111)
    axes.hold(False)
    return f

def simple_fig_updater(figure, data):
    x, y = data
    axes = figure.get_axes()[0]
    axes.plot(x, y)
    
def image_fig_updater(figure, data):
    axes = figure.get_axes()[0]
    axes.imshow(data)
    
if __name__ == "__main__":
    p1 = ScriptPlot("test", simple_fig, simple_fig_updater)
    p2 = ScriptPlot("test2", simple_fig, image_fig_updater)    
    print "made"
    go()
    print "started"

    sleep(4)
    for i in range(100):
        sleep(.1)
        rng = np.arange(0, 0.1 * i, 0.1)
        p1.send_data((rng, np.sin(rng)))
        p2.send_data(np.random.rand(100,100))
        
    print "done"
