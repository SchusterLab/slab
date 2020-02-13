__author__ = 'Brendan S'

from slab.instruments import SerialInstrument
from math import floor
import time


def numtobin(num):
    """This converts the -10V to 10V scale to the 18bit binary scale for the AD5780 DAC"""
    return int((num) * (2 ** 18) / (20) + (2 ** 18 / 2))

class AD5780(SerialInstrument):
    """This is a driver for the bespoke 8-channel AD5780 flux bias box using the serial instrument class for the
    instrumentmanager - as of 20/02/13 this needs to be initialized twice to function well
    """

    def __init__(self, name="", address='COM6', enabled=True, timeout=10):
        """Note when initialized the attenuator board will reset to -31.5dB"""
        SerialInstrument.__init__(self, name, address, enabled, timeout, query_sleep=0.1)
        self.term_char = '\r\n'

    def initialize(self):
        """initialize the flux box to all channels 0V outputs"""
        self.ser.flushInput()
        combinedstring = 'INIT'
        self.write(combinedstring)

    def read(self,channel):
        """Read the voltage on a channel.  Returns last written voltage on arduino"""
        self.ser.flushInput()
        combinedstring = 'READ' + ' ' + str(channel)
        voltage = self.query(combinedstring)
        return voltage

    def setvoltage(self,channel,voltage):
        """sets a channel to a specific voltage """
        self.ser.flushInput()
        binvolt = numtobin(voltage)
        combinedstring = 'SET' + ' ' + str(channel) + ' ' + str(binvolt)
        self.write(combinedstring)

    def ramp(self,dacnum,voltage,step,speed):
        """Calls set on a loop in arduino.  step discretizes voltage, speed spends time at each step"""
        self.ser.flushInput()
        bvoltage = numtobin(voltage)
        bstep = numtobin(step)
        bspeed = numtobin(speed)
        combinedstring = 'RAMP' + ' ' + str(dacnum) + ' ' + str(bvoltage) + ' ' + str(bstep) + ' ' + str(bspeed)
        self.write(combinedstring)

    """Same as above - step, speed in bits instead of volts"""
    def ramp2(self, dacnum, voltage, step, speed):
        self.ser.flushInput()
        bvoltage = numtobin(voltage)
        combinedstring = 'RAMP' + ' ' + str(dacnum) + ' ' + str(bvoltage) + ' ' + str(step) + ' ' + str(speed)
        self.write(combinedstring)

if __name__ == '__main__':

    a=AD5780(name='dacbox',address = 'COM3',enabled = True,timeout=1)
    time.sleep(1)
    print(a.initialize())
    time.sleep(1)
    print(a.initialize())
    time.sleep(1)
    print(a.setvoltage(1,1))
    time.sleep(1)
    print(a.setvoltage(1,2))
