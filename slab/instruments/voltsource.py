# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:45:44 2012

@author: Nitrogen
"""

from slab.instruments import SerialInstrument,VisaInstrument
import re
import time

class SRS900(SerialInstrument,VisaInstrument):
    
    def __init__(self,name="",address='COM5',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        else:
            VisaInstrument.__init__(self,name,address,enabled)
        self.query_sleep=0.05
        self.recv_length=65536
        #self.term_char='\r'

    def read(self):
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'GPIB':
            return VisaInstrument.read(self)
            
    def write(self, s):
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'socket':
            VisaInstrument.write(self, s)
    
    def __del__(self):
        return
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'visa':
            VisaInstrument.__del__(self)

    def get_id(self):
        return self.query("*IDN?")
        
    def set_volt(self,voltage,channel=1):
        self.write('SNDT %d,\"VOLT %f\"' % (channel,voltage))
        
    
        
        
        
    def on_volt(self):
        self.write('OPON')
        
    def off_volt(self):
        self.write('OPOF')
        
        
    if __name__=="__main__":
        srs=SRS900(address="COM5")
        print srs.get_id()
        srs.set_volt(.5,2)
    

    