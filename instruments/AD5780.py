# -*- coding: utf-8 -*-
"""
AD5780 Voltage Source
=====================================
:Author: Larry Chen
"""

from slab.instruments import SocketInstrument, Instrument
import time


class AD5780(SocketInstrument):
    default_port = 23

    def __init__(self, name='AD5780', address='', enabled=True, timeout=10, recv_length=1024):
        SocketInstrument.__init__(self, name, address, enabled, timeout, recv_length)
        self.query_sleep=0.01

    def initialize(self, channel=None):
        if channel is None:
            return self.query('INIT')
        else:
            return self.query('INIT %d' %(int(channel)))

    def set_voltage(self, channel, voltage):
        bitcode = int((voltage + 10.0)*13107.2)
        print 'set target', bitcode
        if bitcode < 0 or bitcode > 262143:
            print('ERROR: voltage out of range')
            return bitcode
        return self.query('SET %d %d' % (channel, bitcode))

    def get_voltage(self, channel):
        return self.query('READ %d' % (channel))

    #Reading from DAC not working, only write instead
    # def ramp(self, channel, voltage, speed):
    #     """Ramp to voltage with speed in (V/S)"""
    #     bitcode = int((voltage + 10.0)*13107.2)
    #     print 'target', bitcode
    #     currbit = int(self.get_voltage(channel).strip())
    #     time.sleep(self.query_sleep)
    #     print 'current', currbit
    #     if bitcode == currbit:
    #         return str(bitcode)
    #     if bitcode < 0 or bitcode > 262143:
    #         print('ERROR: voltage out of range')
    #         return str(bitcode)
    #     step_size = 10 # in bits, about 0.7mV out of +-10V
    #     step_time = int(step_size * 0.0762939453 / speed)
    #     if step_time == 0:
    #         step_time = 1
    #     endbit = self.query('RAMP %d %d %d %d' % (channel, bitcode, step_size, step_time))
    #     time.sleep(self.query_sleep)
    #     if not endbit == bitcode:
    #         endbit = self.set_voltage(channel, voltage)
    #     if not int(endbit.strip())==bitcode:
    #         print 'DAC get_voltage error!!!'
    #     return endbit

    #Remove checks, feedback from DAC on ramp due to readout not working
    def ramp(self, channel, voltage, speed):
        """Ramp to voltage with speed in (V/S)"""
        bitcode = int((voltage + 10.0)*13107.2)
        print 'target', bitcode
        time.sleep(self.query_sleep)
        if bitcode < 0 or bitcode > 262143:
            print('ERROR: voltage out of range')
            return str(bitcode)
        step_size = 10 # in bits, about 0.7mV out of +-10V
        step_time = int(step_size * 0.0762939453 / speed)
        if step_time == 0:
            step_time = 1
        endbit = self.query('RAMP %d %d %d %d' % (channel, bitcode, step_size, step_time))
        print 'end of ramp -', endbit
        time.sleep(self.query_sleep)
        return endbit


    def get_id(self):
        """Get Instrument ID String"""
        return self.query('ID')

    def sweep(self, channel):
        time.sleep(self.query_sleep)
        self.query('SET %d 0' % channel)
        time.sleep(self.query_sleep)
        self.query('RAMP %d %d %d %d' % (channel, 262143, 100, 500))
        time.sleep(self.query_sleep)
        self.query('RAMP %d %d %d %d' % (channel, 0, 100, 500))
        return self.query('READ %d' % (channel))



if __name__ == "__main__":
    """Test script"""
    dac = AD5780(address='192.168.14.158')
    print(dac.get_id())

    dac.initialize()
    time.sleep(1)
    for i in range(1,9):
        dac.ramp(i,0.0,0.1)
