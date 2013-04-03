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

TODO: Implement this api
The general plotting api is to send data via the ScriptPlotter.plot method

- shape=(2,) :: an (x,y) pair for a parametric line plot
- shape=(1,) :: a line plot where the x-value increments by 1 with each point
- shape=(2, n>2), (n>2, 2) :: a line plot where the data is entirely replaced
  with each update, rather than accumulating
- shape=(n>2,) :: Adds a single trace to a image plot where the x-axis has
  fixed dimension n
- shape=(n>2, m>2) :: Creates an image plot which is entirely replaced with
  each update, rather than accumulating
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
    def init_plot(self, ident, rank=1, accum=True, xpts=[], ypts=[], **kwargs):
        """
        Initialize a plot. This will remove existing plots with the same identifier,
        as well as allow for non-default configuration options.
        For instance, to add axes to an image plot

        :param ident: Identifier to associate with the new plot
        :param rank: The dimension of the data to be plotted
        :param accum: Boolean flag. If true, data is appended to existing data
                      as it comes in. Otherwise old data is thrown out, and 
                      replaced with new data
        :param kwargs: additional keyword arguments to pass to 
                       guiqwt.builder.make.{curve,image}
        """
        print ident, accum
        self.meta_pub.send_json({'cmd':'init_plot',
                                 'ident':ident, 
                                 'rank':rank,
                                 'accum':accum,
                                 'xpts':list(xpts),
                                 'ypts':list(ypts),
                                 'plotkwargs':kwargs})
        time.sleep(.25)
        
    def zoom_plot(self, ident):
        """
        Specify the main plot, giving it more screen real estate.
        By default, the first plot added is the main plot.
        """
        self.meta_pub.send_json({'cmd':'zoom_plot', 'ident':ident})

    def clear_plots(self):
        self.clear_plot('_all_plots_')
    
    def clear_plot(self, ident):
        self.meta_pub.send_json({'cmd':'clear_plot', 'ident':ident})
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
        
    def msg(self, *args):
        """
        Display a text message in the plotting window
        """
        text = " ".join(map(str, args))
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
            try:
                if self.data_sub.poll(10):
                    self.recv_array()
                if self.meta_sub.poll(10):
                    self.recv_meta()
                if self.text_sub.poll(10):
                    self.msg(self.text_sub.recv_unicode())
            except Exception as e:
                self.msg(e)
        self.msg("polling complete")
    
    def recv_meta(self):
        d = self.meta_sub.recv_json()
        try:
            cmd = d.pop('cmd')
            if cmd == 'init_plot':
                self.gui["plotstacker"].add_plot(**d)
            elif cmd == 'zoom_plot':
                self.gui["plotstacker"].zoom_plot(d['ident'])
            elif cmd == 'clear_plot':
                self.gui["plotstacker"].clear_plot(d['ident'])
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
    def __init__(self, ident, rank, accum, plotkwargs, xpts=[], ypts=[]):
        qt.QWidget.__init__(self)
        qt.QVBoxLayout(self)
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(5,0,5,0)
        
        self.toolbar = toolbar = qt.QToolBar()
        self.layout().addWidget(toolbar)
        
        self.ident = ident
        self.rank = rank
        self.accum = accum
        self.xpts = xpts
        self.ypts = ypts
        self.update_count = 0
        self.collapsed = False
        
        if rank == 1:
            self.plot_widget = CurveWidget(title=ident)
            self.plot_widget.add_toolbar(toolbar)
            self.plot_widget.register_all_curve_tools()
            self.item = make.curve([], [], **plotkwargs)
        elif rank == 2:
            self.plot_widget = ImageWidget(title=ident, lock_aspect_ratio=False)
            self.plot_widget.add_toolbar(toolbar)
            self.plot_widget.register_all_image_tools()
            if 'interpolation' not in plotkwargs:
                plotkwargs['interpolation'] = 'nearest'
            self.item = make.image(np.array([[0]]), **plotkwargs)
        else:
            raise ValueError
        self.plot_widget.plot.add_item(self.item)
        self.layout().addWidget(self.plot_widget)

        self.buttons_widget = buttons = qt.QWidget()
        qt.QHBoxLayout(buttons)
        self.hide_button = qt.QPushButton('Hide')
        self.hide_button.clicked.connect(self.collapse)        
        self.show_button = qt.QPushButton('Show ' + ident)
        self.show_button.clicked.connect(self.expand)
        self.show_button.hide()
        self.remove_button = qt.QPushButton('Remove')
        self.zoom_button = qt.QPushButton('Zoom')
        self.autoscale_check = qt.QCheckBox('autoscale')
        self.autoscale_check.setChecked(True)
        buttons.layout().addWidget(self.hide_button)
        buttons.layout().addWidget(self.remove_button)
        buttons.layout().addWidget(self.zoom_button)
        buttons.layout().addWidget(self.autoscale_check)
        self.layout().addWidget(buttons)
        self.layout().addWidget(self.show_button)
    
    def collapse(self):
        self.collapsed = True
        self.toolbar.hide()
        self.plot_widget.hide()
        self.buttons_widget.hide()
        self.show_button.show()
        
    def expand(self):
        self.collaped = False
        self.toolbar.show()
        self.plot_widget.show()
        self.buttons_widget.show()
        self.show_button.hide()        

class PlotStacker(qt.QSplitter):
    def __init__(self, mainwin):
        qt.QSplitter.__init__(self)
        self.setStyleSheet("QSplitter::handle {border: 1px solid #CCCCCC;}")
        self.mainwin = mainwin
        plwidget = qt.QWidget()
        self.plwidget = plwidget
        self.plotlist = qt.QVBoxLayout(plwidget)
        self.plotlist.setContentsMargins(5,0,5,0)
        self.plotlist.setSpacing(0)
        self.addWidget(plwidget)
        zoom_widget = qt.QWidget()
        self.zoom = qt.QHBoxLayout(zoom_widget)
        self.addWidget(zoom_widget)
        self.row = 0
        self.col = 0
        self.plots = {}
        self.accum = {}
   
    def add_plot(self, ident="", rank=1, accum=True, xpts=[], ypts=[], plotkwargs={}):
        if ident in self.plots:
            self.remove_plot(ident)
        if len(ypts) > 0 and accum in ('x', True):
            plotkwargs['ydata'] = ypts[0], ypts[-1]
        if len(xpts) > 0 and accum == 'y':
            plotkwargs['xdata'] = xpts[0], xpts[-1]
        widget = PlotItem(ident, rank, accum, plotkwargs, xpts, ypts)
        uncollapsed = lambda ident: not(self.plots[ident].collapsed)
        if len(filter(uncollapsed, self.plots.iterkeys())) > 3:
            widget.collapse()
        widget.remove_button.clicked.connect(lambda: self.remove_plot(ident))
        widget.zoom_button.clicked.connect(lambda: self.zoom_plot(ident))
        self.plots[ident] = widget
        self.plotlist.addWidget(widget)
        zoom_item = self.zoom.itemAt(0)
        if zoom_item is None:
            self.zoom_plot(ident)
            widget.expand()
        
    def zoom_plot(self, ident):
        item = self.zoom.itemAt(0)
        if item is not None:
            widget = item.widget()
            self.zoom.removeWidget(widget)
            self.plotlist.addWidget(widget)
            if widget.ident == ident:
                return
        widget = self.plots[ident]
        self.plotlist.removeWidget(widget)
        self.zoom.addWidget(widget)
        
    def clear_plot(self, ident):
        if ident == '_all_plots_':
            for ident in self.plots:
                self.clear_plot(ident)
        plot = self.plots[ident]
        if plot.rank is 1:
            plot.item.set_data([],[])
        elif plot.rank is 2:
            plot.item.set_data([[]])
    
    def remove_plot(self, ident):
        if ident == "_all_plots_":
            for i in self.plots.keys():
                self.remove_plot(i)
            return
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
        elif plot.accum in ('x', 'y', True) and plot.rank is 2:
            img = item.data
            print plot.accum
            if plot.accum in ('x', True):
                if img.shape == (1,1): # Yet to be properly initialized
                    img = np.array([data]).T
                else:
                    img = np.vstack((img.T, data)).T
                if len(plot.xpts) > 0:
                    if plot.update_count > 0:
                        item.set_xdata(plot.xpts[0], plot.xpts[plot.update_count])
                    plot.update_count += 1 
            else:
                if img.shape == (1,1): # Yet to be properly initialized
                    img = np.array([data])
                img = np.vstack((img, data))
                if len(plot.ypts) > 0:
                    if plot.update_count > 0:
                        item.set_ydata(plot.ypts[0], plot.ypts[plot.update_count])
                    plot.update_count += 1 

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
        ps = PlotStacker(self)
        splitter = qt.QSplitter()
        splitter.setStyleSheet("QSplitter::handle {border: 1px solid #CCCCCC;}")
        self.setStyleSheet("QSplitter::handle {border: 1px solid #CCCCCC;}")
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
        self.remove_plots_button.clicked.connect(lambda: ps.remove_plot("_all_plots_"))
        self.screenshot_button.clicked.connect(self.screenshot)
        
    def screenshot(self):
        d = qt.QFileDialog()
        filename = str(d.getSaveFileName(None, "Save Screenshot", r"S:\_Data", "Image file (*.png)"))
        #filename = str(qt.QFileDialog.getSaveFileName(dir=r"S:\_Data"))
        #if len(d.split('.')) == 1:
        #    d += '.png'
        pm = qt.QPixmap()
        pm2 = pm.grabWidget(self.centralWidget()) # Apparently this is non-updating?
        print filename
        pm2.toImage().save(filename)
        
import sys
import time
def view():
    print "Starting window"
    sys.exit(gui.runWin(ScriptViewerWindow))

def serve(n):
    print "Serving test data"
    with ScriptPlotter() as plotter:
        #plotter.clear_plots()
        #plotter.init_plot("sin", rank=2, xpts=np.linspace(0, 1, n), ypts=np.linspace(0, 2*np.pi), accum='x')
        plotter.init_plot("sin", rank=2, ypts=np.linspace(0, 1, n), xpts=np.linspace(0, 2*np.pi), accum='y')
        plotter.init_plot("cos", rank=1, color='r')
        #plotter.clear_plot("cos")
        plotter.init_plot("tan", accum=False)
        plotter.init_plot("dummy1", accum=False)
        plotter.init_plot("dummy2", accum=False)
        plotter.init_plot("dummy3", accum=False)
        plotter.init_plot("dummy4", accum=False)

        t = 0
        x = np.linspace(0, 2*np.pi)
        print "starting"
        #plotter.zoom_plot("cos")
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
        
            
        
