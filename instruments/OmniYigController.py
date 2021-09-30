__author__ = 'dave'

from slab.instruments import SerialInstrument
from math import floor
import time

def freq_to_bit(freq):
    if freq < 3.5 or freq > 15:
        raise ValueError('YIG FREQ SET OUT OF RANGE')
        return 0
    else:
        return int((4096/11.5)*(freq - 3.5))

class OmniYigController(SerialInstrument):
    """This is a driver for the Arduino Uno based OmniYig controller
    """

    def __init__(self, name="", address='COM6', enabled=True, timeout=0):
        """Note when initialized the yig will set to 3.5GHz passband"""
        SerialInstrument.__init__(self, name=name, address=address, enabled=enabled, timeout=timeout, query_sleep=0.1)
        self.term_char = '\n'

    def get_id(self):
        """Gets the id string of the instrument"""
        self.ser.flushInput()
        self.id = self.query('I')
        self.id = self.query('I')
        return self.id

    def set_yig(self, freq):
        """Sets the yig passband from 3.5GHz to 15GHz in ~3Mhz intervals, 12bit"""
        binval = freq_to_bit(freq)
        self.write('S%d' % binval)

    def get_yig(self):
        """Returns current attenuation setting of the digital attenuator in dB"""
        self.ser.flushInput()
        val = float(self.query('G'))
        return val
        # return -val * 0.5

