# -*- coding: utf-8 -*-
"""
script_viewer.py

author: Phil Reinhold

The goal of this module is to enable plotting of various types
of data in 'real-time', as it comes in. This module takes the
approach of separating the GUI creation from the script. This
has two main benefits

1. Errors in the interface/plotting do not interfere with normal
   operation of the script
2. Multiple asynchronous scripts can all plot to the same window
   without additional hassle

This approach does require, however, that the plotting window be launched independently, and before the script has started

This can be achieved by running this script with::

  $ python -m slab.plotting.script_viewer

Or for convenience, a shortcut is located in the launcher
utility (the green arrow)
"""

import zmq
import numpy as np

class ScriptPlotter():
    """
    ScriptPlotter represents a connection to an active 
    ScriptViewerWindow instance. 
    """
    def __init__(self):
        ctx = zmq.Context()
        self.data_pub = ctx.socket(zmq.PUB)
        self.meta_pub = ctx.socket(zmq.PUB)
        self.text_pub = ctx.socket(zmq.PUB)
        self.data_pub.bind("tcp://127.0.0.1:5556")
        self.meta_pub.bind("tcp://127.0.0.1:5557")
        self.text_pub.bind("tcp://127.0.0.1:5558")
        time.sleep(.5)
    def init_plot(self, ident, rank=1, accum=True, **kwargs):
        """
        Initialize a plot. This will remove existing plots with the same identifier,
        as well as allow for non-default configuration options.

        :param ident: Identifier to associate with the new plot
        :param rank: The dimension of the data to be plotted
        :param accum: Boolean flag. If true, data is appended to existing data
                      as it comes in. Otherwise old data is thrown out, and 
                      replaced with new data
        :param kwargs: additional keyword arguments to pass to 
                       guiqwt.builder.make.curve
        """
        self.meta_pub.send_json({'ident':ident, 
                                 'rank':rank,
                                 'accum':accum,
                                 'plotkwargs':kwargs})
        time.sleep(.25)
    
    def plot(self, data, ident):
        """
        Send data to a plot. If ``ident`` is not associated with any plot
        it is created. Accumulation is assumed, i.e., if the data is an 
        (x,y) pair, a line plot is created, If the data is a rank-1 array,
        an image plot is created.
        """
        self.send_array(np.array(data), ident)
        
    def send_array(self, A, ident, flags=0, copy=False, track=False):
        md = dict(
            ident = ident,
            dtype = str(A.dtype),
            shape = A.shape,
        )
        self.data_pub.send_json(md, flags|zmq.SNDMORE)
        return self.data_pub.send(A, flags, copy=copy, track=track)

    def close(self):
        """
        Close connections to the window
        """
        self.data_pub.close()
        self.meta_pub.close()
        self.text_pub.close()
        
    def msg(self, text):
        """
        Display a text message in the plotting window
        """
        self.text_pub.send_unicode(text)
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, tb):
        self.close()

import slab.gui as gui
class ScriptViewerThread(gui.DataThread):
    def start_polling(self):
        """launches zmq event loop"""
        self.ctx = zmq.Context()
        self.data_sub = self.ctx.socket(zmq.SUB)
        self.data_sub.connect("tcp://127.0.0.1:5556")
        self.data_sub.setsockopt(zmq.SUBSCRIBE, "")
        self.meta_sub = self.ctx.socket(zmq.SUB)
        self.meta_sub.setsockopt(zmq.SUBSCRIBE, "")
        self.meta_sub.connect("tcp://127.0.0.1:5557")
        self.text_sub = self.ctx.socket(zmq.SUB)
        self.text_sub.setsockopt(zmq.SUBSCRIBE, "")
        self.text_sub.connect("tcp://127.0.0.1:5558")

        self.msg("polling started")
        while not self.aborted():
            #time.sleep(.05)
            if self.data_sub.poll(10):
                self.recv_array()
            if self.meta_sub.poll(10):
                self.recv_new_plot()
            if self.text_sub.poll(10):
                self.msg(self.text_sub.recv_unicode())
        self.msg("polling complete")
    
    def recv_new_plot(self):
        d = self.meta_sub.recv_json()
        try:
            self.gui["plotstacker"].add_plot(**d)
        except Exception as e:
            self.msg(e)
        
    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.data_sub.recv_json(flags=flags)
        msg = self.data_sub.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        try:
            self.gui["plotstacker"].update_plot(md['ident'], A.reshape(md['shape']))
        except Exception as e:
            self.msg(e)
            
import PyQt4.Qt as qt
from guiqwt.plot import CurveWidget, ImageWidget
from guiqwt.builder import make

class PlotItem(qt.QWidget):
    def __init__(self, ident, rank, accum, plotkwargs):
        qt.QWidget.__init__(self)
        qt.QVBoxLayout(self)
        
        toolbar = qt.QToolBar()
        self.layout().addWidget(toolbar)
        
        self.ident = ident
        self.rank = rank
        self.accum = accum
        
        if rank == 1:
            self.plot_widget = CurveWidget(title=ident)
            self.plot_widget.add_toolbar(toolbar)
            self.plot_widget.register_all_curve_tools()
            self.item = make.curve([], [], **plotkwargs)
        elif rank == 2:
            print plotkwargs
            self.plot_widget = ImageWidget(title=ident, lock_aspect_ratio=False)
            self.plot_widget.add_toolbar(toolbar)
            self.plot_widget.register_all_image_tools()
            self.item = make.image(np.array([[0]]), **plotkwargs)
        else:
            raise ValueError
        self.plot_widget.plot.add_item(self.item)
        self.layout().addWidget(self.plot_widget)

        buttons = qt.QWidget()
        qt.QHBoxLayout(buttons)
        self.remove_button = qt.QPushButton('Remove')
        self.zoom_button = qt.QPushButton('Zoom')
        self.autoscale_check = qt.QCheckBox('autoscale')
        self.autoscale_check.setChecked(True)
        buttons.layout().addWidget(self.remove_button)
        buttons.layout().addWidget(self.zoom_button)
        buttons.layout().addWidget(self.autoscale_check)
        self.layout().addWidget(buttons)

class PlotStacker(qt.QWidget):
    def __init__(self, mainwin):
        qt.QWidget.__init__(self)
        self.mainwin = mainwin
        #self.setLayout(qt.QVBoxLayout())
        self.setLayout(qt.QHBoxLayout())
        plwidget = qt.QWidget()
        self.plwidget = plwidget
        self.plotlist = qt.QVBoxLayout(plwidget)
        #plwidget.setSizePolicy(qt.QSizePolicy.Maximum)
        plwidget.setMaximumWidth(2 ** 16)
        self.layout().addWidget(plwidget)
        zoom_widget = qt.QWidget()
        self.zoom = qt.QHBoxLayout(zoom_widget)
        self.layout().addWidget(zoom_widget)
        self.row = 0
        self.col = 0
        self.plots = {}
        self.accum = {}
   
    def add_plot(self, ident="", rank=1, accum=True, plotkwargs={}):
        if ident in self.plots:
            self.remove_plot(ident)
        widget = PlotItem(ident, rank, accum, plotkwargs)
        widget.remove_button.clicked.connect(lambda: self.remove_plot(ident))
        widget.zoom_button.clicked.connect(lambda: self.zoom_plot(ident))
        self.plots[ident] = widget
        self.plotlist.addWidget(widget)
    
    def zoom_plot(self, ident):
        item = self.zoom.itemAt(0)
        if item is not None:
            widget = item.widget()
            self.zoom.removeWidget(widget)
            self.plotlist.addWidget(widget)
            if widget.ident == ident:
                self.plwidget.setMaximumWidth(2 ** 16)
                return
        else:
            self.plwidget.setMaximumWidth(2 ** 9)
        widget = self.plots[ident]
        self.plotlist.removeWidget(widget)
        self.zoom.addWidget(widget)
        
                             
    def remove_plot(self, ident):
        widget = self.plots.pop(ident)
        if widget.parentWidget().layout() is self.plotlist:
            self.plotlist.removeWidget(widget)
        else:
            self.zoom.removeWidget(widget)
        widget.close()
    
    def update_plot(self, ident, data):
        try:
            plot = self.plots[ident]
        except KeyError:
            self.add_plot(ident, rank=1 if (len(data) == 2) else 2)
            plot = self.plots[ident]
        item = plot.item
        if plot.accum is True and plot.rank is 1:
            x, y = item.get_data()
            x = np.concatenate((x, [data[0]]))
            y = np.concatenate((y, [data[1]]))
            item.set_data(x, y)
        elif plot.accum in (0, 1, True) and plot.rank is 2:
            img = item.data
            if plot.accum in (0, True):
                if img.shape == (1,1): # Yet to be properly initialized
                    img = np.array([data])
                else:
                    img = np.vstack((img, data))
            else:
                img = np.column_stack((img, data))
            item.set_data(img)
        else:
            if plot.rank is 1:
                try:
                    x, y = data
                except:
                    y = data
                    x = np.arange(len(y))
                item.set_data(x, y)
            else:
                item.set_data(data)
        plot.plot_widget.plot.replot()
        if plot.autoscale_check.isChecked():
            plot.plot_widget.plot.do_autoscale()

from guiqwt.qtdesigner import loadui
UiClass = loadui("C:\_Lib\python\slab\plotting\script_viewer.ui")

class ScriptViewerWindow(gui.SlabWindow, UiClass):
    def __init__(self):
        gui.SlabWindow.__init__(self, ScriptViewerThread)
        self.setupSlabWindow(autoparam=True)
        #self.auto_register_gui()
        ps = PlotStacker(self)
        splitter = qt.QSplitter()
        self.message_box = qt.QTextBrowser()
        self.message_box.setMaximumWidth(250)
        splitter.addWidget(self.message_box)
        splitter.addWidget(ps)
        self.verticalLayout.addWidget(splitter)
        self.gui["plotstacker"] = ps
        self.register_script("start_polling", self.start_button, self.stop_button)
        self.start_thread()
        self.msg("initialized")
        self.start_script("start_polling")
        
import sys
import time
def view():
    print "Starting window"
    sys.exit(gui.runWin(ScriptViewerWindow))

def serve(n):
    print "Serving test data"
    with ScriptPlotter() as plotter:
        plotter.init_plot("sin", rank=2)
        plotter.init_plot("cos", rank=1, color='r')
        plotter.init_plot("tan", rank=1, accum=False)
        t = 0
        x = np.linspace(0, 2*np.pi)
        print "starting"
        for i in range(n):
            time.sleep(.1)
            t += .1
            plotter.msg("time " + str(t))
            plotter.plot(np.sin(x + t), "sin")
            plotter.plot((t, np.cos(t)), "cos")
            plotter.plot(np.tan(x + t), "tan")
        plotter.close()
        print "done"
    
if __name__ == "__main__":
    view()    
        
            
        
