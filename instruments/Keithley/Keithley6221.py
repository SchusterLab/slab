# -*- coding: utf-8 -*-
"""
Keithley6221 Current Source
===========================================
:Author: Kan-Heng Lee
"""
from slab.instruments import VisaInstrument
import time
import numpy as np
import datetime
import os

class Keithley6221(VisaInstrument):

    def __init__(self,name="Keithley6221",address='GPIB0::12::INSTR',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'
        # VisaInstrument.__init__(self,name,address,enabled, term_chars='\r')
        VisaInstrument.__init__(self, name, address, enabled)
        self.query_sleep=0.001
        self.recv_length=65536
        self.term_char='\r'

    def get_id(self):
        return self.queryb('*IDN?')

    def init(self, curr_range, volt_comp=10):
        """
        1. Abort any previous operation and reset memory
        2. Set current range (amp).
        3. Set voltage_compliance. Defalut is the tool maximum = 12 V.
        """
        self.write('ABOR')
        time.sleep(1)
        self.write('*rst')
        time.sleep(1)
        self.write(':sour:curr:rang ' + str(curr_range))
        self.write(':sour:curr:compliance ' + str(volt_comp))
    
    def abort(self, curr_range, volt_comp=10):
        """
        Abort any operation.
        """
        self.write('ABOR')

    def curr(self, switch):
        """Turn on/off currents"""
        if switch == 'on':
            self.write('OUTP ON')
        elif switch == 'off':
            self.write('OUTP OFF')
        else:
            print('''!!!Warning!!! Input is 'on' or 'off' as str.''')

    def set_curr(self, value):
        """Set current output in Amp."""
        self.write(':sour:curr:lev ' + str(value))

