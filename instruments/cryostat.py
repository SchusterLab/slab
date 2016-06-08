# -*- coding: utf-8 -*-
"""
Oxford Triton Cryostat Controller (cryostat.py)
===============================================
:Author: David Schuster
"""
from slab.instruments import SocketInstrument
import re

class Triton(SocketInstrument):
    default_port=22518
    def __init__(self,name="Triton",address='slab-fridge1.uchicago.edu',enabled=True,timeout=10):
        #if ':' not in address: address+=':22518'        
        SocketInstrument.__init__(self,name,address,enabled,timeout)
        #self.query_sleep=0.1
        self.query_sleep=0.0
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
        
    def get_mc_temperature(self):
        Temperature=self.get_temperature('MC RuO2')
        if not Temperature >0:
            Temperature=self.get_temperature('MC cernox')
        return Temperature

        
    def get_settings(self):
         settings=SocketInstrument.get_settings(self)
         settings.update(self.get_temperatures())
         settings.update(self.get_pressures())
         return settings
         
         
        
if __name__ == '__main__':
    fridge=Triton (address='192.168.14.129')
    #print fridge.get_status()
    d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    print fridge.get_settings()
    
