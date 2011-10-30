#!/usr/bin/python

"""
script.py -- helpers for visualizing data-taking scripts

goal:
    import script
    import matplotlib.pyplot as plt

    class myPlot(script.Plot):
        def __init__(self):
            self.plotfn = plt.colorbar
            # self.plotfns = []
            script.Plot.__init__(self)
        def update_plot(self, myData):
            pass

    plot = myPlot()
    plotwin = script.PlotWindow([plot])

    while(True):
        myData = getData()
        plot.sendData(myData)
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
            self.figures[name] = self.make_figures[name]()
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
    
if __name__ == "__main__":
    p1 = ScriptPlot("test", simple_fig, simple_fig_updater)
    p2 = ScriptPlot("test2", simple_fig, simple_fig_updater)    
    print "made"
    go()
    print "started"

    sleep(4)
    for i in range(100):
        sleep(.1)
        rng = np.arange(0, 0.1 * i, 0.1)
        p1.send_data((rng, np.sin(rng)))
        p2.send_data((rng, np.cos(rng)))
        
    print "done"
