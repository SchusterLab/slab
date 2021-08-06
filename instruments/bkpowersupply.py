# -*- coding: utf-8 -*-
"""
BK PowerSupply (bkpowersupply.py)
=================================
:Author: Bing

Typically, this supply is used to power the amplifier (TODO: Which?)
"""
from slab.instruments import SerialInstrument, VisaInstrument
import time

class BK9130A(SerialInstrument):
    'Interface to the BK Precision 9130A (old style) Power Supply'
    def __init__(self,name="",address='COM11',enabled=True,timeout=0.25):
        SerialInstrument.__init__(self,name,address,enabled,timeout,query_sleep=0.2)
        
    def get_id(self):
        self.write('*CLS')
        print(self.query('SYST:VERS?'))
        return self.query('*IDN?')
        
    def set_voltage(self,channel,voltage):
        ch=['FIR','SECO','THI'][channel-1]
        self.write('INST'+' '+ch+'\n')
        self.write('VOLT %fV\n' %voltage)
            
    def set_current(self,channel,current):
        ch=['FIR','SECO','THI'][channel-1]
        self.write('INST'+' '+ch+'\n')
        self.write('CURR %fA\n' %current)
        return self.query('CURR?\n')
        
    def set_voltages(self,ch1,ch2,ch3):
        self.write( 'APP:VOLT %f,%f,%f\n' %(ch1,ch2,ch3))
        
    def get_voltages(self):
        ans=self.query('APP:VOLT?\n')
        voltages=[float (s.strip()) for s in ans.split(',')]
        return voltages   
        
    def set_currents(self,ch1,ch2,ch3):
        self.write( 'APP:CURR %f,%f,%f\n' %(ch1,ch2,ch3))
        
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

                self.write('INST CH%d'%(k+1))
                self.write('OUTP '+stat+'\n')
        elif type(channel) == list:
            for k, chan in enumerate(channel):
                stat=['0','1'][state[k]]

                if chan in [1,2,3]:
                    self.write('INST CH%d'%chan+'\n')
                    self.write('OUTP '+stat+'\n')
                else:
                    print("Channel should be 1, 2 or 3")
        elif channel in ['all', 1, 2, 3]:
            stat=['0','1'][state]
            self.write('INST CH%d'%channel+'\n')
            self.write('OUTP '+stat+'\n')

    def get_output(self):
        """
        Returns output state of the channels.
        """
        outputs = list()
        for k in range(3):
            self.write('INST CH%d\n'%(k+1))
            x = self.query('CHAN:OUTP?\n')
            outputs.append(int(x[0]))

        return outputs

    def Remote(self):
        self.write('SYST:REM\n')
        
    def Local(self):
        self.write('SYST:LOC\n')


class BK9130B(VisaInstrument):
    '''Interface to the BK Precision 9130B (new style) Power Supply

        Revised by Andrew Oriani, 4/2021
    '''

    def __init__(self, name="", address='', enabled=True, timeout=1.0):
        VisaInstrument.__init__(self, name, address, enabled, timeout)

    def get_id(self):
        self.write('*CLS')
        print(self.query('SYST:VERS?'))
        return self.query('*IDN?').split('\n')[0]

    def set_voltage(self, channel, voltage):
        self.write('INST CH%d'%(channel))
        self.write('VOLT %fV' % voltage)
        return float(self.query('VOLT?').split('\n')[0])

    def set_current(self, channel, current):
        self.write('INST CH%d'%(channel))
        self.write('CURR %fA' % current)
        return float(self.query('CURR?').split('\n')[0])

    def set_voltages(self, ch1, ch2, ch3):
        ch_v=[ch1, ch2, ch3]
        out=[]
        for ch, V in enumerate(ch_v):
            out.append(self.set_voltage(ch+1, V))
        return out

    def get_set_voltages(self):
        return list(map(float, self.query('APP:VOLT?').split('\n')[0].split(',')))

    def set_currents(self, ch1, ch2, ch3):
        ch_v=[ch1, ch2, ch3]
        out=[]
        for ch, V in enumerate(ch_v):
            out.append(self.set_current(ch+1, V))
        return out

    def get_set_currents(self):
        return list(map(float, self.query('APP:CURR?').split('\n')[0].split(',')))

    def get_voltage(self, channel=None):
        if channel!=None:
            self.write('INST CH%d'%(channel))
            return float(self.query('MEAS:VOLT?').split('\n')[0])
        else:
            return list(map(float, self.query('MEAS:VOLT:ALL?').split('\n')[0].split(',')))

    def get_set_current(self, channel):
        self.write('INST CH%d'%(channel))
        return float(self.query('CURR?').split('\n')[0])

    def get_set_voltage(self, channel):
        self.write('INST CH%d'%(channel))
        return float(self.query('VOLT?').split('\n')[0])

    def get_current(self, channel=None):
        if channel!=None:
            self.write('INST CH%d'%(channel))
            return float(self.query('MEAS:CURR?').split('\n')[0])
        else:
            return list(map(float, self.query('MEAS:CURR:ALL?').split('\n')[0].split(',')))

    def get_currents(self):
        return list(map(float, self.query('MEAS:CURR:ALL?').split('\n')[0].split(',')))

    def get_voltages(self):
        return list(map(float, self.query('MEAS:VOLT:ALL?').split('\n')[0].split(',')))

    def set_output(self, channel, state):
        if type(state)!=bool:
            raise TypeError('State nust be a boolean')

        self.write('INST CH%d'%channel)
        self.write('CHAN:OUTP %d'%state)

    def get_output(self, ch):
        """
        Returns output state of the channels.
        """
        self.write('INST CH%d' %ch)
        return int(self.query('CHAN:OUTP?').split('\n')[0])

    def get_outputs(self):
        outputs=[]
        for ch in (1,2,3):
            outputs.append(self.get_output(ch))
        return outputs

    def Remote(self):
        self.write('SYST:REM')

    def Local(self):
        self.write('SYST:LOC')



if __name__== '__main__':
    
    p=BK9130A(address='COM5')
    print(p.get_id())
    #print p.get_voltages()
    
    
    
