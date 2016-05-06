# -*- coding: utf-8 -*-
"""
KEPCO Power Supply (KEPCOPowerSupply.py)
========================================
:Author: David Schuster
"""
from slab.instruments import SerialInstrument
import time
from numpy import linspace

class KEPCOPowerSupply(SerialInstrument):
    
    def __init__(self,name="Kepco",address='COM4',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        self.query_sleep=0.05
        self.recv_length=65536
        self.Remote()
        
    def get_id(self):
        return self.query('*IDN?')
  
    def Remote(self):
        self.write('SYST:REM 1')
        
    def Local(self):
        self.write('SYST:REM 0')
     
    def set_output(self,output=True):
        if output:
            self.write('OUTP 1')
        else:
            self.write('OUTP 0')
    
    def set_voltage(self,v):
        if self.protocol == 'serial':
            self.write('VOLT %f'%v)
            
    def get_voltage(self):
        return float(self.query('VOLT?').strip("\x13\r\n\x11"))
    
    def set_current(self,c):
        self.write('CURR %f'%c)
        
    def ramp_to_current(self,c,sweeprate=None):
        if sweeprate is None:
            sweeprate=self.sweeprate
        
        start=self.get_current()
        stop=c
        start_t=time.time()
        self.set_current(start)
        time.sleep(self.query_sleep)
        step_t=time.time()-start_t
        total_t=abs(stop-start)/sweeprate
        steps=total_t/step_t
        
        for ii in linspace(start,stop,steps):
            self.set_current(ii)
            time.sleep(self.query_sleep)
        
    def get_current(self):
        count = 0

        while (count<20):
            try:
                current_value =  self.query('CURR?').strip("\x13\r\n\x11")
                if type(current_value)==float:
                    return float(current_value)
                elif type(current_value)==str:
                    "Converting from String to Float."
                    current_value = float(current_value)
                    return current_value
                else:
                    print  "The Machine didn't return a float"
                    print current_value
                    print type(current_value)
            except:
                    print "Getting the current again..."
            count+=1
            time.sleep(1)
        raise Exception("The Machine wouldn't return a floating number. Try smacking it.")
            
        return float(current_value)
    
    def set_current_mode(self):
        self.write('FUNC:MODE CURR')
    
    def set_voltage_mode(self):
        self.write('FUNC:MODE VOLT')


if __name__ == '__main__':
    p=KEPCOPowerSupply(address="COM6")
    
    #magnet.set_local()
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    
