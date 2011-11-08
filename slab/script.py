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
import guiqwt.plot, guiqwt.curve, guiqwt.builder, guiqwt.tools
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

class ScriptLinePlot(Qt.QWidget):
    def __init__(self, n_items, cumulative, title="", style=None, **kwargs):
        Qt.QWidget.__init__(self)
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self.curve_d = guiqwt.plot.CurveDialog(toolbar=True, options=kwargs)
        layout.addWidget(self.curve_d)
        self.plot = self.curve_d.get_plot()
        self.plot.set_title(title)
        self.plot_items = []
        self.cumulative = cumulative
        self.update_fns, self.redraw_fns = ([], [])
        if cumulative:
            self.datasets = []
            for i in range(n_items):
                self.datasets.append((np.array([]), np.array([])))
        if style:
            if type(style) is not list:
                style = [style]
            elif len(style) < n_items:
                style = style * int(np.ceil(float(n_items) / len(style)))
            
        for i in range(n_items):
            if style:
                item = guiqwt.builder.make.mcurve(np.array([]), np.array([]), style[i])
            else:
                item = guiqwt.curve.CurveItem()
            self.plot_items.append(item)
            self.plot.add_item(item)
        
    def update(self, data):        
        n, (x, y) = data
        if self.cumulative:
            np.append(self.datasets[n][0], x)
            np.append(self.datasets[n][1], y)
            x, y = self.datasets[n]
            
        self.plot_items[n].set_data(x, y)
        for f in self.update_fns:
            f()
    
    def redraw(self):
        self.plot.replot()
        for f in self.redraw_fns:
            f()
    
    def add_autoscale_check(self):
        self.as_checkbox = Qt.QCheckBox("autoscale axes")
        self.as_checkbox.setCheckState(Qt.Qt.Checked)
        self.layout().addWidget(self.as_checkbox)
        self.connect(self.as_checkbox, Qt.SIGNAL("clicked(bool)"), self.autoscale)
    
    def autoscale(self, b):
        if b:
            self.plot.do_autoscale()
            self.redraw_fns.append(self.plot.do_autoscale) # I wish I saw a better way
        else:
            try: self.redraw_fns.remove(self.plot.do_autoscale)
            except: pass
            self.plot.disable_autoscale()
            
    def add_tool(self, tool, *args, **kwargs):
        self.curve_d.add_tool(tool, *args, **kwargs)

class ScriptImagePlot(Qt.QWidget):
    def __init__(self, xdim=10, ydim=10, title="", **kwargs):
        Qt.QWidget.__init__(self)
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self.image_d = guiqwt.plot.ImageDialog(toolbar=True, options=kwargs)
        layout.addWidget(self.image_d)
        self.plot = self.image_d.get_plot()
        self.plot.set_title(title)
        self.image = guiqwt.image.ImageItem(np.zeros((xdim, ydim)))
        self.plot.add_item(self.image)
    def update(self, data):
        self.image.set_data(data)
    def redraw(self):
        self.plot.replot()
    def add_tool(self, tool, *args, **kwargs):
        self.curve_d.add_tool(tool, *args, **kwargs)

        

class ScriptMPLPlot(FigureCanvas):
    def __init__(self, figure):
        pass

class ScriptProxy:
    def __init__(self, parent, i=None):
        self.parent = parent
        self.i = i
    def send(self, data, *args, **kwargs):
        if self.i is not None:
            self.parent.send(("update", ((self.i, data),) + args, kwargs))
        else:
            self.parent.send(("update", (data,) + args, kwargs))
#    def redraw(self):
#        self.parent.send("redraw")
    def __getattr__(self, name):
        return lambda *args, **kwargs: self.parent.send((name, args, kwargs))

class ScriptPlotWin(object):
    def __init__(self, grid_x=0, grid_y=0):
        self.SPThread = ScriptPlotThread(grid_x, grid_y)
            

    def add_linePlot(self, **kwargs):
        return apply(self.add_multiLinePlot, (1,), kwargs)[0]
    
    def add_multiLinePlot(self, n, auto_redraw=True, cumulative=False, **kwargs):
        """
        n sets the number of lines to be included on the plot,
        returns a list of n "pipes", which are not actual pipes,
        but still "pipe.send(data)" in the same way
        """
        parent, child = Pipe()
        self.SPThread.add_plot(ScriptLinePlot, (n, cumulative), kwargs, child, auto_redraw)
        return [ ScriptProxy(parent, i) for i in range(n) ] 
    
    def add_imagePlot(self, **kwargs):
        return apply(self.add_customPlot, (ScriptImagePlot, ()), kwargs)
    
    def add_customPlot(self, Plot, args, auto_redraw=True, **kwargs):
        """
        Plot must be a subclass of QWidget which also provides an
        update(self, data) method
        """
        parent, child = Pipe() # mulitPipes?
        self.SPThread.add_plot(Plot, args, kwargs, child, auto_redraw)
        return ScriptProxy(parent)

    def go(self):
        self.SPThread.start()


class ScriptPlotThread(Process):
    screenshotdir = "C:\\Users\\phil\\Desktop\\"
    def __init__(self, grid_x, grid_y, sleep_time=10):
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

    def add_plot(self, PlotC, args, kwargs, pipe, auto_redraw):
        self.make_plot.append(PlotC)
        self.make_plot_args.append((args, kwargs))
        self.plot_redraw.append(auto_redraw)
        self.pipes.append(pipe)

    def run(self):
#        self.logfile = open("C:\\Users\\phil\\Documents\\logfile", "w")
        self.app = guidata.qapplication()
        self.win = Qt.QMainWindow()
        cwidget = Qt.QWidget()
        self.win.setCentralWidget(cwidget)
        clayout = Qt.QVBoxLayout()
        cwidget.setLayout(clayout)
        pwidget = Qt.QWidget()
        clayout.addWidget(pwidget)
        
#        self.toolbar = self.win.addToolBar("tools")
        
        if self.x_capped or self.y_capped:
            self.layout = Qt.QGridLayout()
            self.layout.setSpacing(8)
            cur_x, cur_y = (0,0)
        else:
            self.layout = Qt.QVBoxLayout()

        self.layout = Qt.QGridLayout()
        pwidget.setLayout(self.layout)
       
       # Screenshot Button            
        self.ssn = 0 # Number of screenshots taken
        self.ss_button = Qt.QPushButton("Screenshot")
        clayout.addWidget(self.ss_button)
        self.app.connect(self.ss_button, Qt.SIGNAL("clicked()"), self.screenshot)

        for PlotC, (args, kwargs) in zip(self.make_plot, self.make_plot_args):
            plot = apply(PlotC, args, kwargs)
            self.plots.append(plot)
            if self.x_capped or self.y_capped:
                self.layout.addWidget(plot, cur_y, cur_x)
                if self.x_capped:
                    next_cur_x = (cur_x + 1) % self.grid_x
                    if next_cur_x <= cur_x:
                        cur_y += 1
                else:
                    next_cur_y = (cur_y + 1) % self.grid_y
                    if next_cur_y <= cur_y:
                        cur_x += 1
            else:
                self.layout.addWidget(plot)
#            plot.setup(self)
                
        self.timer = Qt.QTimer()
        self.app.connect(self.timer, Qt.SIGNAL("timeout()"), self.wake)
        self.timer.start(self.sleep_time)

        self.win.show()
        self.app.exec_()

    def wake(self):
        for pipe, plot, auto_redraw in zip(self.pipes, self.plots, self.plot_redraw):
            try: # This is probably too much of a hack
                if pipe.poll():
                    cmd, args, kwargs = pipe.recv()
                    try:
                        apply(getattr(plot, cmd), args, kwargs)
                        if cmd == "update" and auto_redraw:
                            plot.redraw()
                    except Exception as e:
                        print cmd, "failed", e
                        # Logfile stuff
#                    if cmd == "redraw":
#                        plot.redraw()
#                    else:
#                        getattr(plot)
#                        try:
#                            plot.update(data)
#                            if auto_redraw:
#                                plot.redraw()
#                        except:
#                            getattr(plot, data)
            except:
                return
        self.timer.start(self.sleep_time)
        
#    def add_guiqwt_manager(self, plot):
#        if not self.manager:
#            self.manager = guiqwt.plot.PlotManager(self.win)
#            self.manager.add_plot(plot)
#            self.manager.register_all_curve_tools()
    
    def screenshot(self):
        while os.path.exists(self.screenshotdir + "ss_" + str(self.ssn) + ".png"):
            self.ssn += 1
        pm = Qt.QPixmap()
        #pm.grabWindow(self.win.winId())
        pm2 = pm.grabWidget(self.win.centralWidget()) # Apparently this is non-updating?
        i = pm2.toImage()
        fn = self.screenshotdir + "ss_" + str(self.ssn) + ".png"
        i.save(fn)

# [ ] Example usage
# [X] Layout
# [X] Toolbars
# [X] GuiQwt
# [ ] Autoscale checkbox
# [X] screenshot?
# [X] Allow manual replotting rules!
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
    win = ScriptPlotWin(grid_y=1)
    p1, p2 = win.add_multiLinePlot(2, title="Sine & Cosine", auto_redraw=False, style=["r-", "b-"],
                                   xlabel="X Label", ylabel="Y Label")
    p1.add_autoscale_check()
    p1.add_tool(guiqwt.tools.AnnotatedSegmentTool)
    p3 = win.add_linePlot(title="Tangent")
    win2 = ScriptPlotWin()
    ip = win2.add_imagePlot(title="Random Image", xdim=100, ydim=100)
    win.go()
    win2.go()
        
    sleep(2)
    for i in range(100):
        sleep(0.1)
        xrng = np.arange(0, 0.1 * i, 0.1)
        p1.send((xrng, np.sin(xrng)))
        p2.send((xrng, np.cos(xrng)))
        p1.redraw() # Or p2.redraw, same effect, perhaps this can be more semantic?
        p3.send((xrng, np.tan(xrng)))
        ip.send(np.random.rand(100, 100))
    sleep(4)
    print "done"


if __name__ == "__main__":
    test()