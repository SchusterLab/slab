from slab.instruments import Instrument
from qick import QickSoc
import Pyro4
import socket

class QickInstrument(Instrument, QickSoc):
    
    def __init__(self, name, address='', enabled=True, timeout=1, query_sleep=0, bitfile=None, **kwargs):
        Instrument.__init__(self, name=name, address=address, enabled=enabled, timeout=timeout, query_sleep=query_sleep, **kwargs)
        self.bitfile=bitfile
        self.reset()

        # list of objects that need to be registered for autoproxying
        self.autoproxy = [self.streamer, self.tproc]

    def reset(self, force_init_clks=False,ignore_version=True, **kwargs):
        QickSoc.__init__(self,bitfile=self.bitfile)