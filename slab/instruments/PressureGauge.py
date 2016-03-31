

from slab.instruments import SerialInstrument

class PressureGauge(SerialInstrument):

    def __init__(self, name="PressureGauge", address='COM5',enabled=True,timeout=1):
        SerialInstrument.__init__(self, name, address, enabled, timeout)

    def pressure_read(self):
        return float(self.query('R'))