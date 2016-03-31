__author__ = 'ge'

from slab.instruments import SocketInstrument

class Omega16i(SocketInstrument):
    default_port = 5000

    def __init__(self, name="Omega16i", address=None, enabled = True ):
        SocketInstrument.__init__(self, name, address, enabled=enabled, timeout=10, recv_length=2**20)
        self.query_sleep = 0.05

    def get_id(self):
        return "Omega 16i"

    def get_pressure(self, ch = '1'):
        return self.query("*X0"+ch.upper())

