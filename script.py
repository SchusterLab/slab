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
        - A matplotlib figure
        - A function update_fn(figure, data) 
          which updates the figures state to reflect new data

          e.g. for a line_plot
          fig = Figure
    """
    def __init__(self, name, fig, update_fn):
        global SPThread
        if not SPThread:
            SPThread = ScriptPlotThread()

        self.parent, child = Pipe()
        SPThread.add_fig(name, child, fig, update_fn)
        self.name = name

    def send_data(self, data):
        print "sending"
        self.parent.send((self.name, data))


class ScriptPlotThread(Process):
    def __init__(self, sleep_time=100):
        Process.__init__(self)

        self.app = Qt.QApplication([])
        self.win = Qt.QMainWindow()
        self.cwidget = Qt.QSplitter(Qt.Qt.Vertical)
        self.win.setCentralWidget(self.cwidget)

        self.pipes = []
        self.figures = {}
        self.update_fns = {}

        self.sleep_time=sleep_time

    def add_fig(self, name, pipe, fig, update_fn):
        plot = FigureCanvas(fig)

        self.pipes.append(pipe)
        self.figures[name] = fig
        self.update_fns[name] = update_fn

        self.cwidget.addWidget(plot)

    def run(self):
        self.timer = Qt.QTimer()
        self.app.connect(self.timer, Qt.SIGNAL("timeout()"), self.wake)
        self.timer.start(self.sleep_time)

        self.win.show()
        self.app.exec_()

    def wake(self):
        for pipe in self.pipes:
            if pipe.poll():
                name, data = pipe.recv()
                self.updatefns[name](self.plots[name], data)
                self.plots[name].draw()
        self.timer.start(self.sleep_time)

# Example usage
def simple_fig():
    f = Figure()
    f.add_subplot(111)
    return f


def simple_fig_updater(figure, data):
    figure.get_axes()[0].plot(data)


if __name__ == "__main__":
    sp = ScriptPlot("test", simple_fig(), simple_fig_updater)
    print "made"
    SPThread.start()
    print "started"

    for i in range(100):
        sleep(.1)
        sp.send_data(np.sin(np.arange(0, 0.1 * i, 0.1)))

    print "done"
