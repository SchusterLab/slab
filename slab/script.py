#!/usr/bin/python

"""
script.py -- helpers for visualizing data-taking scripts

usage:
    from script import *
    
    window_1 = ScriptPlotWin()
    window_2 = ScriptPlotWin()
    
    plot_conn_1 = window_1.add_linePlot()
    plot_conn_2, plot_conn_3 = window_1.add_multiLinePlot(2)
    iplot_conn = window_2.add_imagePlot()
    
    window_1.go()
    window_2.go()
    
    while True:
        x_data_1, y_data_1 = get_data()
        image_data = get_image_data()
        plot_conn_1.send((x_data_1, y_data_1)) # Data is a tuple of numpy arrays (x,y)
        iplot_conn.send(image_data)
        ...
    
"""

from multiprocessing import Process, Pipe
from time import sleep

import PyQt4.Qt as Qt
import PyQt4.Qwt5 as Qwt
import PyQt4.QtGui as QtGui
import guiqwt.plot, guiqwt.curve, guiqwt.builder
import guidata

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

def get_last_from_pipe(pipe):
    try:
        while True:
            data = pipe.recv()
    except:
        return data

class ScriptLinePlot(guiqwt.plot.CurveWidget):
    def __init__(self, n_items, title, cumulative):
        guiqwt.plot.CurveWidget.__init__(self, title=title)
#        self.register_all_curve_tools()
        plot = self.get_plot()
#        plot = guiqwt.plot.CurvePlot()
#        plot.set_title(title)
#        self.add_plot(plot)
        self.plot_items = []
        self.cumulative = cumulative
        if cumulative:
            self.datasets = []
            for i in range(n_items):
                self.datasets.append((np.array([]), np.array([])))

        for i in range(n_items):
            item = guiqwt.curve.CurveItem()
            self.plot_items.append(item)
            plot.add_item(item)
    
    def setup(self, thread):
        pass
  #      thread.add_guiqwt_manager(self.get_plot())
       # thread.manager.add_plot(self.get_plot())
        
    def update(self, data):        
        n, (x, y) = data        
        if self.cumulative:
            np.append(self.datasets[n][0], x)
            np.append(self.datasets[n][1], y)
            x, y = self.datasets[n]
            
        self.plot_items[n].set_data(x, y)
        self.get_plot().replot()
    
    def redraw(self):
        self.get_plot().replot()

class ScriptMPLPlot(FigureCanvas):
    def __init__(self, figure):
        pass

class multiPipe:
    def __init__(self, i, parent):
        self.parent = parent
        self.i = i
    def send(self, data):
        self.parent.send((self.i, data))
    def redraw(self):
        self.parent.send("redraw")

class ScriptPlotWin(object):
    """
    User must provide:
        - A name (any identifier)
        - A function make_fig() which returns either
              - matplotlib figure
              - (matplotlib figure, persistent arguments)
              - QWidget
              - (QWidget, persistent arguments)
        - A function update_fn which updates the state to reflect new data
          it can take arguments (as appropriate)
              - (figure, data)
              - (figure, args, data)
              - (widget, data)
              - (widget, args, data)

          e.g. for a line_plot
          fig = Figure
    """
    def __init__(self, grid_x=0, grid_y=0):
        self.SPThread = ScriptPlotThread(grid_x, grid_y)
            

    def add_linePlot(self, *args):
        return apply(self.add_multiLinePlot, (1,) + args)[0]
    
    def add_multiLinePlot(self, n, title=None, auto_redraw=True, cumulative=False):
        """
        n sets the number of lines to be included on the plot,
        returns a list of n "pipes", which are not actual pipes,
        but still "pipe.send(data)" in the same way
        """
        parent, child = Pipe()
        self.SPThread.add_plot(ScriptLinePlot, (n, title, cumulative), child, auto_redraw)
        return [ multiPipe(i, parent) for i in range(n) ] 
    
    def add_imagePlot(self, title=None):
        pass
    
    def add_customPlot(self, Plot, args):
        """
        Plot must be a subclass of QWidget which also provides an
        update(self, data) method
        """
        parent, child = Pipe() # mulitPipes?
        self.SPThread.add_plot(Plot, args, child)
        return parent

    def go(self):
        self.SPThread.start()


class ScriptPlotThread(Process):
    screenshotdir = "C:\\Users\\phil\\Desktop\\"
    def __init__(self, grid_x, grid_y, sleep_time=20):
        Process.__init__(self)
        
        self.pipes = []
        self.make_plot = []
        self.make_plot_args = []
        self.plot_redraw = []
        self.plots = []
        self.names = []
        
        if grid_x > 0:
            self.x_capped = True
            self.y_capped = False
            self.grid_x = grid_x
        elif grid_y > 0:
            self.x_capped = False
            self.y_capped = True
            self.grid_y = grid_y
        else:
            self.x_capped = False
            self.y_capped = False
        self.sleep_time=sleep_time
        self.manager = None

    def add_plot(self, PlotC, args, pipe, auto_redraw):
        self.make_plot.append(PlotC)
        self.make_plot_args.append(args)
        self.plot_redraw.append(auto_redraw)
        self.pipes.append(pipe)

    def run(self):
#        self.logfile = open("C:\\Users\\phil\\Documents\\logfile", "w")
        self.app = guidata.qapplication()
        self.win = Qt.QWidget()
        
        if self.x_capped or self.y_capped:
            self.layout = Qt.QGridLayout()
            cur_x, cur_y = (0,0)
        else:
            self.layout = Qt.QVBoxLayout()

        self.layout = Qt.QGridLayout()
        self.win.setLayout(self.layout)
       
       # Screenshot Button            
        self.ssn = 0 # Number of screenshots taken
        self.ss_button = Qt.QPushButton("Screenshot")
        self.layout.addWidget(self.ss_button)
        self.app.connect(self.ss_button, Qt.SIGNAL("clicked()"), self.screenshot)

        for PlotC, args in zip(self.make_plot, self.make_plot_args):
            plot = apply(PlotC, args)
            self.plots.append(plot)
            if self.x_capped or self.y_capped:
                self.layout.addWidget(plot, cur_x, cur_y)
                if self.x_capped:
                    next_cur_x = (cur_x + 1) % self.grid_x
                    if next_cur_x < cur_x:
                        cur_y += 1
                else:
                    next_cur_y = (cur_y + 1) % self.grid_y
                    if next_cur_y < cur_y:
                        cur_x += 1
            else:
                self.layout.addWidget(plot)
            plot.setup(self)
                
        self.timer = Qt.QTimer()
        self.app.connect(self.timer, Qt.SIGNAL("timeout()"), self.wake)
        self.timer.start(self.sleep_time)

        self.win.show()
        self.app.exec_()

    def wake(self):
        for pipe, plot, auto_redraw in zip(self.pipes, self.plots, self.plot_redraw):
            try: # This is probably too much of a hack
                if pipe.poll():
                    data = pipe.recv()
                    if data == "redraw":
                        plot.redraw()
                    else:
                        plot.update(data)
                        if auto_redraw:
                            plot.redraw()
            except:
                return
        self.timer.start(self.sleep_time)
        
    def add_guiqwt_manager(self, plot):
        if not self.manager:
            self.manager = guiqwt.plot.PlotManager(self.win)
            self.manager.add_plot(plot)
            self.manager.register_all_curve_tools()
    
    def screenshot(self):
        print "snap"
        self.ssn += 1
        pm = Qt.QPixmap()
        pm.grabWindow(self.win.winId())
        i = pm.toImage()
        fn = self.screenshotdir + "ss_" + str(self.ssn) + ".png"
        i.save(fn)
        print "saved", fn

# Example usage
# Layout
# Toolbars
# GuiQwt
# Autoscale checkbox
# screenshot?
# Allow manual replotting rules!
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
    
def multli_line_updater(figure, data):
    n_plot, (x, y) = data
    a = figure.get_axes()[n_plot]
    a.plot(x, y)    

def test():
    win = ScriptPlotWin()
    p1, p2 = win.add_multiLinePlot(2, title="Sine & Cosine", auto_redraw=False)
    win.go()
        
    sleep(2)
    for i in range(100):
        sleep(0.1)
        xrng = np.arange(0, 0.1 * i, 0.1)
        p1.send((xrng, np.sin(xrng)))
        p2.send((xrng, np.cos(xrng)))
        p1.redraw() # Or p2.redraw, same effect, perhaps this can be more semantic?
    print "done"


if __name__ == "__main__":
    test()