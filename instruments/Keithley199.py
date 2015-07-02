# -*- coding: utf-8 -*-
"""
Keithley199 Voltage Source (Keithley199.py)
===========================================
:Author: David Schuster
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re
import time
import numpy as np

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
        return float(self.query('S1')[4:])

    def set_range_auto(self):
        self.write('R0X')

    def set_mode(self, mode):
        """
        :param mode: may be one of the following: "VDC", "VAC", "Ohms", "IDC", "IAC"
        :return:
        """
        conversion_table = {'VDC' : 'F0', 'VAC' : 'F1', 'Ohms' : 'F2', 'IDC' : 'F3', 'IAC' : 'F4'}
        self.write(conversion_table[mode]+'X')

    def set_volt_range(self, range):
        """
        :param range: may be one of the following: 0.3V, 3V, 30V or 300V.
        :return:
        """
        allowed = [0.3, 3, 30, 300]

        if range in allowed:
            conversion_table = {'0.3' : 'R1', '3' : 'R2', '30' : 'R3', '300' : 'R4'}
            self.write(conversion_table[str(range)]+'X')

    def integrate_voltage(self, dt):
        """
        :param time: time in seconds to integrate the signal
        :return: mean and standard deviation of the voltage
        """
        t0 = time.time()
        v = []
        while time.time() < t0+dt:
            V = self.get_volt()
            time.sleep(0.017)
            v.append(V)

        return np.mean(np.array(v)), np.std(np.array(v))

if __name__ == '__main__':
    print "HERE"
    #magnet=IPSMagnet(address='COM1')
    V=Keithley199(address='GPIB0::26::INSTR')
    print V.get_id()
    V.set_range_auto()
    print V.get_volt()
