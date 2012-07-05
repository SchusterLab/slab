# -*- coding: utf-8 -*-
"""
Bing

"""
from slab.instruments import SerialInstrument
import time

class BKPowerSupply(SerialInstrument):
    def __init__(self,name="",address='COM11',enabled=True,timeout=0):
        SerialInstrument.__init__(self,name,address,enabled,timeout,querysleep=0.2)
        
    def get_id(self):
        #self.query('SYST:VERS?')
        self.query('*IDN?')
        
    def set_voltage(self,channel,voltage):
        ch=['FIR','SECO','THI'][channel-1]
        SerialInstrument.write(self,'INST'+' '+ch+'\n')
        SerialInstrument.write(self,'VOLT %fV\n' %voltage)
            
    def set_current(self,channel,current):
        ch=['FIR','SECO','THI'][channel-1]
        SerialInstrument.write(self,'INST'+' '+ch+'\n')
        SerialInstrument.write(self,'CURR %fA\n' %current)
        return self.query('CURR?\n')
        
    def set_voltages(self,ch1,ch2,ch3):
        SerialInstrument.write(self, 'APP:VOLT %f,%f,%f\n' %(ch1,ch2,ch3))
        
    def get_voltages(self):
        self.query('APP:VOLT?\n')
        return self.query('APP:VOLT?\n')        
        
    def set_currents(self,ch1,ch2,ch3):
        SerialInstrument.write(self, 'APP:CURR %f,%f,%f\n' %(ch1,ch2,ch3))
        
    def get_currents(self):
        self.query('APP:CURR?\n')
        return self.query('APP:CURR?\n')        

    def set_output(self,state):
        stat=['0','1'][state]
        SerialInstrument.write(self,'OUTP '+stat+'\n')
        
    def Remote(self):
        SerialInstrument.write(self,'SYST:REM\n')
        
    def Local(self):
        SerialInstrument.write(self,'SYST:LOC\n')
        
if __name__== '__main__':
    
    p=BKPowerSupply(address='COM12')
    print p.get_voltages()
    
    
    