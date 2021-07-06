from slab.instruments import SerialInstrument
import time

class ArduinoADC(SerialInstrument):

    def __init__(self, name="ArduinoADC", address='COM4',enabled=True,timeout=.01):
        SerialInstrument.__init__(self, name, address, enabled, timeout,query_sleep=0.2)
        self.read()
        self.read()
        time.sleep(0.5)
        self.query('S')

    def get_voltages(self):
        s=self.query('S')
        s = self.query('S')
        return [float(n) for n in s.split(",") ]
