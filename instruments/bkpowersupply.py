# -*- coding: utf-8 -*-
"""
BK PowerSupply (bkpowersupply.py)
=================================
:Author: Bing

Typically, this supply is used to power the amplifier (TODO: Which?)
"""
from slab.instruments import SerialInstrument
import time

class BKPowerSupply(SerialInstrument):
    'Interface to the BK Precision 9130 Power Supply'
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
        ans=self.query('APP:VOLT?\n')
        voltages=[float (s.strip()) for s in ans.split(',')]
        return voltages   
        
    def set_currents(self,ch1,ch2,ch3):
        SerialInstrument.write(self, 'APP:CURR %f,%f,%f\n' %(ch1,ch2,ch3))
        
    def get_currents(self):
        ans=self.query('APP:CURR?\n')
        currents=[float (s.strip()) for s in ans.split(',')]
        return currents     

    def get_voltage(self,channel=None):
        if channel is None: return self.get_voltages()
        else:
            return self.get_voltages()[channel-1]

    def get_current(self,channel=None):
        if channel is None: return self.get_currents()
        else:
            return self.get_voltages()[channel+1]

    def set_output(self, state, channel='all'):
        """
        Sets the output of channel to "state" (True/False). channel may be any
        of the following: ['all', 1, 2, 3]. channel may be a list or a single integer.
        Example usage:
        * set_output([True, False], channel=[1,2])
        * set_output([True]*3, channel=[1,2,3])
        * set_output(True, channel='all')
        * set_output(False, channel=1)
        """
        if channel == 'all':
            for k in range(3):
                if type(state) == list:
                    stat=['0','1'][state[k]]
                else:
                    stat=['0','1'][state]

                SerialInstrument.write(self,'INST CH%d'%(k+1))
                SerialInstrument.write(self,'OUTP '+stat+'\n')
        elif type(channel) == list:
            for k, chan in enumerate(channel):
                stat=['0','1'][state[k]]

                if chan in [1,2,3]:
                    SerialInstrument.write(self,'INST CH%d'%chan)
                    SerialInstrument.write(self,'OUTP '+stat+'\n')
                else:
                    print "Channel should be 1, 2 or 3"
        elif channel in ['all', 1, 2, 3]:
            stat=['0','1'][state]
            SerialInstrument.write(self,'INST CH%d'%channel)
            SerialInstrument.write(self,'OUTP '+stat+'\n')

    def get_output(self):
        """
        Returns output state of the channels.
        """
        outputs = list()
        for k in range(3):
            SerialInstrument.write(self,'INST CH%d'%(k+1))
            x = SerialInstrument.query(self,'CHAN:OUTP?')
            outputs.append(int(x[0]))

        return outputs

    def Remote(self):
        SerialInstrument.write(self,'SYST:REM\n')
        
    def Local(self):
        SerialInstrument.write(self,'SYST:LOC\n')
        
if __name__== '__main__':
    
    p=BKPowerSupply(address='COM12')
    print p.get_voltages()
    
    
    
