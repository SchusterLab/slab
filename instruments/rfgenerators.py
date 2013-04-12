# -*- coding: utf-8 -*-
"""
Agilent E8257D (rfgenerators.py)
================================
:Author: David Schuster
"""

from slab.instruments import SocketInstrument

class E8257D(SocketInstrument):
    """
    The interface to the Agilent E8257D RF Generator, implemented on top of 
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    def __init__(self,name='E8257D',address='', enabled=True,timeout=10, recv_length=1024):
        #SocketInstrument.__init__(self,name,address,5025,enabled,timeout,recv_length)        
        if ':' in address:
            SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
        else:
            SocketInstrument.__init__(self,name,address+':'+str(5025),enabled,timeout,recv_length)
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?')
    
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
        
    
if __name__=="__main__":
    rf=E8257D(address='128.135.35.30')
    print rf.query('*IDN?')
    rf.set_output(False)
    rf.set_frequency(10e9)
    print "Frequency: %f" % rf.get_frequency()
    rf.set_power(-10)
    print "Power: %f" % rf.get_power()
