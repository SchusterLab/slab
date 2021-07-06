# -*- coding: utf-8 -*-
"""
Keithley199 Voltage Source (Keithley199.py)
===========================================
:Author: Gerwin Koolstra
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re
import time
import numpy as np

class HP34401A(VisaInstrument):
    def __init__(self,name="HP34401",address='GPIB0::30::INSTR',enabled=True,timeout=1.):
        #if ':' not in address: address+=':22518'

        VisaInstrument.__init__(self,name,address,enabled)
        self.query_sleep=1
        self.recv_length=65536
        self.term_char=''
        #Default mode for multimeter:
        #self.set_4wireresistance()

    def get_id(self):
        self.write("*IDN?")
        time.sleep(30)
        return self.read()
        #return self.query('*IDN?')

    def get_value(self):
        return float(self.query("READ?"))

    def set_DCvoltage(self):
        self.query_sleep = 3
        self.write("CONF:VOLT:DC")
        self.query("READ?")


    def set_ACvoltage(self):
        self.query_sleep = 3
        self.write("CONF:VOLT:AC")
        self.query("READ?")


    def set_DCcurrent(self):
        self.write("CONF:CURR:DC")
        self.write("READ?")
        self.query_sleep = 3

    def set_ACcurrent(self):
        self.write("CONF:CURR:AC")
        self.write("READ?")
        self.query_sleep = 3
        #self.write("MEAS:CURR:AC?")

    def set_2wireresistance(self):
        self.write("CONF:RES")
        self.query("MEAS:RES?")
        self.query_sleep = 1

    def set_4wireresistance(self):
        self.write("CONF:FRES")
        self.query("MEAS:FRES?")
        self.query_sleep = 1

    def get_errors(self):
        return self.query("SYST:ERR?")

    def get_commands(self):
        return "Commands: get_id, get_value, set_DCvoltage, set_DCcurrent, set_2wireDCresistance, set_4wireDCresistance"

    #def get_derivative(self):

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
    print("HERE")
    # #magnet=IPSMagnet(address='COM1')
    # V=Keithley199(address='GPIB0::26::INSTR')
    # print V.get_id()
    # V.set_range_auto()
    # print V.get_volt()
    dmm=HP34401A()