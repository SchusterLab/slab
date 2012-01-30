# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 09:55:05 2011

@author: Phil
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re

class IPSMagnet(SerialInstrument,VisaInstrument):
    
    def __init__(self,name="magnet",address='COM6',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        SerialInstrument.__init__(self,name,address,enabled,timeout)
        self.query_sleep=0.1
        self.recv_length=65536
        self.term_char='\r'
    
    def get_version(self):
        return self.query('V')
        
    
    def set_local(self):
        self.write('C1')

if __name__ == '__main__':
    magnet=IPSMagnet ()
    #magnet.set_local()
    print magnet.get_version()
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    