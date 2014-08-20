# -*- coding: utf-8 -*-
"""
Agilent E8257D (rfgenerators.py)
================================
:Author: David Schuster
"""

from slab.instruments import SocketInstrument
import time

class E8257D(SocketInstrument):
    """
    The interface to the Agilent E8257D RF Generator, implemented on top of 
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    default_port=5025
    def __init__(self,name='E8257D',address='',enabled=True,timeout=10, recv_length=1024):
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()
    
    def set_output(self,state=True):
        """Set Output State On/Off"""
        if state: self.write(':OUTPUT:STATE ON')
        else:     self.write(':OUTPUT:STATE OFF')
        
    def get_output(self):
        """Query Output State"""
        return int(self.query(':OUTPUT?')) == 1
    
    def set_mod(self,state=True):
        """Set Modulation State On/Off"""
        if state: self.write(':OUTPUT:MOD:STATE ON')
        else:     self.write(':OUTPUT:MOD:STATE OFF')
        
    def get_mod(self):
        """Query Modulation State"""
        return bool(self.query(':OUTPUT:MOD?'))
            
    def set_frequency(self, frequency):
        """Set CW Frequency in Hz"""
        self.write(':FREQUENCY %f' % frequency)
    
    def get_frequency(self):
        """Query CW Frequency"""
        return float(self.query(':FREQUENCY?'))

    def set_phase(self,phase):
        """Set signal Phase in radians"""
        self.write(':PHASE %f' % phase)
    
    def get_phase(self):
        """Query signal phase in radians"""
        return float(self.query(':PHASE?'))
        
    def set_power(self,power):
        """Set CW power in dBm"""
        self.write(':POWER %f' % power)
        
    def get_power(self):
        return float(self.query(':POWER?'))
        
    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['frequency']=self.get_frequency()
        settings['power']=self.get_power()
        settings['phase']=self.get_phase()
        settings['mod']=self.get_mod()
        settings['output']=self.get_output()
        settings['id']=self.get_id()
        return settings
        
    def get_settled(self):
        """Get source settled state"""
        return bool(self.query(':OUTPut:SETTled?'))
        
    def wait_to_settle(self, timeout=1):
        """Block until source settled"""
        start=time.time()
        while not self.get_settled() and time.time()-start<timeout: pass
    
    def set_internal_pulse(self,pulse_time=10e-6):
        self.write(':SOUR:PULM:SOUR:INT FRUN')
        self.write(':SOUR:PULM:INT:PERIOD %f S' %(pulse_time))
        self.write(':SOUR:PULM:INT:PWIDTH %f S' %(pulse_time))
        self.write(":SOUR:PULM:STAT ON")
        self.set_mod()
    
    def set_ext_pulse(self,mod=True):
        self.write(':SOUR:PULM:SOUR EXT')
        if mod:
            self.write(":SOUR:PULM:STAT ON")
        else:
            self.write(":SOUR:PULM:STAT OFF")
        self.set_mod()
        
class BNC845(SocketInstrument):
    """
    The interface to the BNC845 RF Generator, implemented on top of 
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    default_port=18
    def __init__(self,name='BNC845',address='', enabled=True,timeout=10, recv_length=1024):
        #SocketInstrument.__init__(self,name,address,5025,enabled,timeout,recv_length)        
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
        
        #default set to external reference
        self.set_reference_source("EXT")
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()
    
    def set_output(self,state=True):
        """Set Output State On/Off"""
        if state: self.write(':OUTPUT:STATE ON')
        else:     self.write(':OUTPUT:STATE OFF')
        
    def get_output(self):
        """Query Output State"""
        return int(self.query(':OUTPUT?')) == 1
                
    def set_frequency(self, frequency):
        """Set CW Frequency in Hz"""
        self.write(':FREQUENCY %f' % frequency)
    
    def get_frequency(self):
        """Query CW Frequency"""
        return float(self.query(':FREQUENCY?'))

    def set_phase(self,phase):
        """Set signal Phase in radians"""
        self.write(':PHASE %f' % phase)
    
    def get_phase(self):
        """Query signal phase in radians"""
        return float(self.query(':PHASE?'))
    
    #NOTE: The BNC is a fixed output power...this does nothing!!    
    def set_power(self,power):
        """Set CW power in dBm"""
        self.write(':POWER %f' % power)
        print "BNC845 is fixed output power - 13dBm"
        
        
    def get_power(self):
        return float(self.query(':POWER?'))
        
    def set_reference_source(self,source='INT',ref_freq=10e6):
        """Sets reference oscillator source: 'INT' or 'EXT'"""
        if source!='INT' and source!='EXT':
            raise Exception('BNC845: Invalid reference oscillator source %s, must be either INT or EXT' % source)
        self.write(':ROSCillator:SOURce %s' % source)
        #Note that the BNC845 cannot autodetect the reference oscillator frequency
        if source=='EXT':
            self.write(':ROSCillator:EXTernal:FREQuency %f' % ref_freq)
        
    def get_reference_source(self):
        """Gets reference oscillator source: 'INT' or 'EXT'"""
        return self.query(':ROSCillator:SOURce?').strip()
        
    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['frequency']=self.get_frequency()
        settings['power']=self.get_power()
        settings['phase']=self.get_phase()
        #settings['mod']=self.get_mod()
        settings['output']=self.get_output()
        settings['id']=self.get_id()
        return settings
        
    def get_settled(self):
        """Get source settled state"""
        #return bool(self.query(':OUTPut:SETTled?'))
        #no call for the BNC845 to tell if the output has settled
        #data sheet says the frequency settles in <100us
        return True        
        
    def wait_to_settle(self, timeout=1):
        """Block until source settled"""
        start=time.time()
        while not self.get_settled() and time.time()-start<timeout: pass
    
    def set_pulse_state(self,state=True):
        """Set pulse state True/False"""
        if state: self.write(":SOUR:PULM:STAT ON")
        else:     self.write(":SOUR:PULM:STAT OFF")

    def set_internal_pulse(self,width,period,state=True):
        """Set up an internally generated pulse with: width, period, and state"""
        #self.write(':SOUR:PULM:SOUR:INT FRUN')
        self.write(':SOUR:PULM:INT:PERIOD %f S' %(width))
        self.write(':SOUR:PULM:INT:PWIDTH %f S' %(period))
        self.set_pulse_state(state)
    
    def set_ext_pulse(self,state=True):
        self.write(':SOUR:PULM:SOUR EXT')
        self.set_pulse_state(state)
        

def test_BNC845(rf=None):
    if rf is None:
        rf=BNC845(address='192.168.14.151')
    
    print rf.get_id()
    rf.set_output(False)
    print "Output: ", rf.get_output()
    rf.set_frequency(10e9)
    print "Frequency: %g" % rf.get_frequency()
    print "Reference source: %s" % rf.get_reference_source()
    
    
    
def test_8257D (rf=None):   
    if rf is None:
        rf=E8257D(address='128.135.35.30')
    print rf.query('*IDN?')
    rf.set_output(False)
    rf.set_frequency(10e9)
    print "Frequency: %f" % rf.get_frequency()
    rf.set_power(-10)
    print "Power: %f" % rf.get_power()
    
if __name__=="__main__":
    #test_8257D()
    test_BNC845()
