# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 09:55:05 2011

@author: Phil
"""
from slab.instruments import SocketInstrument
import re

class Triton(SocketInstrument):
    
    def __init__(self,name="Triton",address='fridge.circuitqed.com',enabled=True,timeout=10):
        if ':' not in address: address+=':22518'        
        SocketInstrument.__init__(self,name,address,enabled,timeout)
        self.query_sleep=0.1
        self.recv_length=65536
    
    def get_status(self):
        return self.query('status')

    def get_help(self):
        return self.query('help')
    
    def get_pressures(self):
        """get_pressure returns the pressures detected by the fridge as a dictionary"""
        s=self.query('pressures')
        matches=re.findall(r'.*?(?P<name>.*?): (?P<pressure>.*?)bar.*',s)
        pressures={}
        for match in matches:
            pressures[match[0]]=float(match[1])
        return pressures        
        
    def get_pressure(self,channel='Condense'):
        """Get pressure returns the pressure of the desired channel.  Valid options are:
            Condense, Tank, and Forepump"""
        return self.get_pressures()[channel]
    
    def get_temperatures(self):
        temp_string= self.query('temperatures')
        matches=re.findall(r'.*?name: (?P<name>.*?);.*?temperature: (?P<temperature>.*?);.*',temp_string)
        temps={}
        for match in matches:
            temps[match[0]]=float(match[1])
        return temps
        
    def get_temperature(self,channel='MC RuO2'):
        return self.get_temperatures()[channel]
        
    def get_settings(self):
         settings=SocketInstrument.get_settings(self)
         settings.update(self.get_temperatures())
         settings.update(self.get_pressures())
         return settings
         
         
        
if __name__ == '__main__':
    fridge=Triton (address='fridge.circuitqed.com')
    #print fridge.get_status()
    d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    print fridge.get_settings()
    