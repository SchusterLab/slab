# -*- coding: utf-8 -*-
"""
Lakeshore 370 AC Resistance Bridge LakeshoreResistanceBridge
====================================
:Author: David Schuster
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re
import time

class LakeshoreResistanceBridge(SerialInstrument,VisaInstrument):

    def __init__(self,name="Lakeshore",address='COM4',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        else:
            VisaInstrument.__init__(self,name,address,enabled, term_chars='\r')
        self.query_sleep=0.05
        self.recv_length=65536
        self.term_char='\r'

    def read(self):
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'VISA':
            return VisaInstrument.read(self)

    def write(self, s):
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'VISA':
            VisaInstrument.write(self, s)

    def __del__(self):
        return
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'VISA':
            VisaInstrument.__del__(self)

    def get_id(self):
        return self.query('V')