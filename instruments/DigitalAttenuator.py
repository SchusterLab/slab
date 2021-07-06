__author__ = 'dave'

from slab.instruments import SerialInstrument
from math import floor
import time


class DigitalAttenuator(SerialInstrument):


    """This is a driver for the Arduino Uno based digital attenuator
       It is based on a HITITE HMC424 digital attenuator.  It also uses a custom board to invert the
       arduino logic.
    """

    
    def __init__(self, name="", address='COM3', enabled=True, timeout=0):
        """Note when initialized the attenuator board will reset to -31.5dB"""
        SerialInstrument.__init__(self, name, address, enabled, timeout, query_sleep=0.1)
        self.term_char = '\n'
    
    
    def get_id(self):
        """Gets the id string of the instrument"""
        self.ser.flushInput()
        self.id = self.query('I')
        self.id = self.query('I')
        return self.id
    
    
    def set_attenuator(self, atten):
        """Sets the attenuation from -31.5 to 0dB
           can be set in 0.5 dB increments, if not multiple of 0.5 it rounds up
           ignores the sign of the attenuation"""
        val = floor(abs(atten) / 0.5)
        self.write('S%d' % val)
    
    
    def get_attenuator(self):
        """Returns current attenuation setting of the digital attenuator in dB"""
        self.ser.flushInput()
        val = float(self.query('G'))
        return -val * 0.5
