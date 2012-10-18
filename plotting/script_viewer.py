# -*- coding: utf-8 -*-
"""
script_viewer.py

Created on Thu Oct 18 10:00:10 2012

@author: slab
"""

import zmq
import numpy as np

#sv_addr = "tcp://127.0.0.1:9000"
#sv_meta_addr = "tcp://127.0.0.1:9001"

class ScriptPlotter(zmq.Context):
    def __init__(self):
        zmq.Context.__init__(self)
        self.pub = self.socket(zmq.PUB)
        self.meta_pub = self.socket(zmq.PUB)
        self.pub.bind("tcp://127.0.0.1:5556")
        self.meta_pub.bind("tcp://127.0.0.1:5557")
        time.sleep(2)
    def init_plot(self, ident, rank=1, accum=True):
        self.meta_pub.send_json({'ident':ident, 'rank':rank, 'accum':accum})
    def send_array(self, A, ident, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            ident = ident,
            dtype = str(A.dtype),
            shape = A.shape,
        )
        self.pub.send_json(md, flags|zmq.SNDMORE)
        return self.pub.send(A, flags, copy=copy, track=track)
    def close(self):
        self.pub.close()
        self.meta_pub.close()


import slab.gui as gui
class ScriptViewerThread(gui.DataThread):
    def start_polling(self):
        """launches zmq event loop"""
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5556")
        self.sub.setsockopt(zmq.SUBSCRIBE, "")
        self.meta_sub = self.ctx.socket(zmq.SUB)
        self.meta_sub.setsockopt(zmq.SUBSCRIBE, "")
        self.meta_sub.connect("tcp://127.0.0.1:5557")
        self.msg("polling started")
        while not self.aborted():
            time.sleep(.5)
            if self.sub.poll(10):
                self.msg("data hit")
                self.recv_array()
            if self.meta_sub.poll(10):
                self.msg("new plot hit")
                self.recv_new_plot()
                
        self.msg("polling complete")
    def recv_new_plot(self):
        d = self.meta_sub.recv_json()
        self.gui["plotstacker"].add_plot(**d)
        
    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.sub.recv_json(flags=flags)
        msg = self.sub.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        self.gui["plotstacker"].update_plot(md['ident'], A.reshape(md['shape']))
            
import PyQt4.Qt as qt
from guiqwt.plot import CurveWidget, ImageWidget
from guiqwt.builder import make
class PlotStacker(qt.QWidget):
    def __init__(self):
        qt.QWidget.__init__(self)
        self.setLayout(qt.QGridLayout())
        self.row = 0
        self.col = 0
        self.plots = {}
        self.accum = {}
    def add_plot(self, ident="", rank=1, accum=True):
        if rank < 2:
            widget = CurveWidget(title=ident)
            item = make.curve([],[])
        else:
            widget = ImageWidget(title=ident)
            item = make.image()
        widget.plot.add_item(item)
        self.plots[ident] = widget
        self.layout().addWidget(widget, self.row, self.col)
        # Two columns
        self.row += self.col
        self.col = 1 - self.col
        self.plots[ident] = \
          {"widget":widget.plot, "accum":accum, "item":item, "rank":rank}
    def update_plot(self, ident, data):
        plot = self.plots[ident]
        item = plot["item"]
        if plot["accum"]:
            if plot["rank"] is 1:
                x, y = item.get_data()
                x = np.concatenate((x, [data[0]]))
                y = np.concatenate((y, [data[1]]))
                item.set_data(x, y)
            else:
                raise NotImplementedError
        else:
            if plot["rank"] is 1:
                x, y = data
                item.set_data(x, y)
            else:
                raise NotImplementedError
        plot["widget"].replot()

from guiqwt.qtdesigner import loadui
UiClass = loadui("C:\_Lib\python\slab\plotting\script_viewer.ui")

class ScriptViewerWindow(gui.SlabWindow, UiClass):
    def __init__(self):
        gui.SlabWindow.__init__(self, ScriptViewerThread)
        self.setupSlabWindow(autoparam=True)
        #self.auto_register_gui()
        ps = PlotStacker()
        self.verticalLayout.addWidget(ps)
        self.gui["plotstacker"] = ps
        self.register_script("start_polling", self.start_button, self.stop_button)
        self.start_thread()
        self.msg("initialized")
        
import sys
import time
def view():
    print "Starting window"
    sys.exit(gui.runWin(ScriptViewerWindow))

def serve(n):
    print "Serving test data"
    p = ScriptPlotter()
    p.init_plot("sin")
    t = 0
    print "here"
    for i in range(n):
        t += .1
        p.send_array(np.array([t, np.sin(t)]), "sin")
        time.sleep(.1)    
    p.close()
    
if __name__ == "__main__":
    view()    
        
            
        