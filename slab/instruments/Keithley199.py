# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 09:55:05 2011

@author: Phil
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re
import time

class Keithley199(VisaInstrument):
    
    def __init__(self,name="keithley199",address='GPIB0::26::INSTR',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        
        VisaInstrument.__init__(self,name,address,enabled, term_chars='\r')
        self.query_sleep=0.05
        self.recv_length=65536
        self.term_char='\r' 
    
    def get_id(self):
        return self.query('*IDN?')

    def get_volt(self):
        """Returns power supply voltage"""
        return float(self.query('S1')[5:])
        
    def set_range_auto(self):
        self.write('R0')

if __name__ == '__main__':
    print "HERE"
    #magnet=IPSMagnet(address='COM1')
    V=Keithley199(address='GPIB0::26::INSTR')
    print V.get_id()
    V.set_range_auto()
    print V.get_volt()