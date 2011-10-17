# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:32:54 2011

@author: Dai
"""
from slab.instruments import SerialInstrument, SocketInstrument
import time

class SpectrumAnalyzer(SerialInstrument, SocketInstrument):
    #general design parameters
    #LO offset is 10.7MHz
    #lo_offset = 10.7e6
    #LO power is 7dBm
    lo_power = 10
    #bandwidth is 180kHz
    bandwidth=180e3    
    
    #calibration information
    calibration_data = None
    #min_power = None
    step = 10
    
    def __init__(self, name='spec_analyzer', protocol='socket', step=10, 
                 address='', port=23, enabled=True, timeout=.1, recv_length=1024, 
                 query_sleep=0.005, lo_power=10, lo_offset=10.55e6, baudrate=115200):
        self.lo_offset = lo_offset
        self.lo_power = lo_power
        
        if protocol == 'serial':
            SerialInstrument.__init__(self, name, port, enabled, 
                                      timeout, recv_length, baudrate=baudrate, querysleep=query_sleep)
            self.term_char = ''
            time.sleep(2)
            print self.read()
        elif protocol == 'socket':
            if ':' in address:
                SocketInstrument.__init__(self, name, address, enabled, 
                                      timeout, recv_length)
            else:
                SocketInstrument.__init__(self, name, address+':'+str(port), enabled, 
                                      timeout, recv_length)
            self.recv_length = recv_length
            self.term_char = ''
            self.query_sleep = query_sleep
        else:
            print 'The protocol requested is not valid.' 
        
    def read(self):
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'socket':
            return SocketInstrument.read(self)
            
    def write(self, s):
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'socket':
            SocketInstrument.write(self, s)
    
    def __del__(self):
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'socket':
            self.write('END')
            SocketInstrument.__del__(self)
                
    def get_power(self):
        return float(self.query('READ'))
        
    def get_avg_power(self):
        self.write('READ_AVG')
        #leaves extra time for average power reading
        time.sleep(0.15)
        return float(self.read())
        

if __name__ == '__main__':
    from instruments import E8257D
    
    sa = SpectrumAnalyzer(address='128.135.35.167:23')
    rf = E8257D(address='rfgen1.circuitqed.com')
    lo = E8257D(address='rfgen2.circuitqed.com')
    rf.set_output(False)
    lo.set_output(False)
    rf.set_frequency(6e9)
    lo.set_frequency(6e9+sa.lo_offset)
    lo.set_power(10)
    rf.set_power(-10)
    lo.set_output()
    rf.set_output()
    
    print sa.get_power()
    print sa.get_avg_power()