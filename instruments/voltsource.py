# -*- coding: utf-8 -*-
"""
SRS900 Voltage Source (voltsource.py)
=====================================
:Author: David Schuster
"""

from slab.instruments import SocketInstrument,SerialInstrument,VisaInstrument,Instrument
import re
import time

class SRS900(SerialInstrument,VisaInstrument):
    'Interface to the SRS900 voltage source'
    def __init__(self,name="",address='COM5',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        else:
            VisaInstrument.__init__(self,name,address,enabled)
        self.query_sleep=0.05
        self.recv_length=65535
        self.escapekey='XXYYXX'
        #self.term_char='\r'
    

    def read(self,port=None):
        if port is not None:
            self.write("CONN %x,'%s'\n" % (port,self.escapekey))
            self.read()
            self.write(self.escapekey)
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'GPIB':
            return VisaInstrument.read(self)
            
    def write(self, s,port=None):
        if port is not None:
            self.write("SNDT %x,#3%03d%s\n" % (port, len(s),s) )
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'socket':
            VisaInstrument.write(self, s)
            
    def query(self,s,port=None):
        if port is not None:
            self.write("CONN %x, '%s'\n" % (port,self.escapekey))
            self.write(s)
            time.sleep(self.query_sleep)
            ans=self.read()
            self.write(self.escapekey)
        else: 
            self.write(s)
            time.sleep(self.query_sleep)
            ans=self.read()

        return ans
    
    
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
        
    def get_volt(self,channel=1):
        return float(self.query("VOLT?",channel) )
        
    def set_output(self,channel=1,state=True):
        if state:
            self.write('SNDT %d,\"OPON\"' % (channel)) 
        else:
            self.write('SNDT %d,\"OPOF\"' % (channel))   
            
    def get_output(self,channel=1):
        return bool(int(self.query('EXON?',channel)))

class YokogawaGS200(SocketInstrument):

    default_port=7655    
    def __init__(self,name='YOKO',address='', enabled=True,timeout=10, recv_length=1024):
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()
    
    def set_output(self,state=True):
        """Set output mode default state=True"""
        self.write(':OUTPUT:STATE %d' % (int(state)))

    def get_output(self):
        """Get output mode return result as bool"""
        return bool(self.query(':OUTPUT:STATE?').strip())
    
    def set_mode(self,mode='voltage'):
        """Set yoko mode, valid inputs are mode='VOLTage' or mode='CURRent' """
        self.write(':SOURCE:FUNCTION %s' % mode)
    
    def get_mode(self):
        """Get yoko mode, returns either 'CURR' or 'VOLT'"""
        return self.query(':SOURCE:FUNCTION?').strip()

    def set_level(self,level):
        """Set yoko level"""
        self.write(':SOURCE:LEVEL %f' % level)
    
    def get_level(self):
        """Get level return as float"""
        return float(self.query(':SOURCE:LEVEL?').strip())
        
    def set_current(self,current):
        """Set yoko current"""
        if self.get_mode() == "CURR":
            self.set_level(current)
        else:
            raise Exception("ERROR: Need to set Yoko current in voltage mode")

    def get_current(self):
        """Get yoko current"""
        if self.get_mode() == "CURR":
            return self.get_level()        
        else:
            raise Exception("ERROR: Need to set Yoko voltage in current mode")
                    
    def set_volt(self,voltage):
        """Set yoko voltage"""
        if self.get_mode() == "VOLT":
            self.set_level(voltage)
        else:
            raise Exception("ERROR: Need to set Yoko voltage in current mode")
    
    def get_volt(self):
        """Get yoko voltage"""
        if self.get_mode() == "VOLT":
            return self.get_level()
        else:
            raise Exception("ERROR: Need to set Yoko voltage in current mode")
    def set_measure_state(self,state=True):
        """Set measurement state of instrument"""
        if state:   self.write(':SENSE:STATE ON')
        else:       self.write(':SENSE:STATE OFF')
        
    def get_measure(self):
        """Get measured value"""
        return float(self.query(':MEASURE?').strip())
        
    
#class SRS928(Instrument):
#    
#    def __init__(self,mainframe,name="",address=None,enabled=True,timeout=1):
#        """Initialized with link to mainframe and the address should be the port # on the mainframe"""
#        self.mainframe=mainframe
#        Instrument.__init__(self,name,address,enabled,timeout)
#        self.escape='xZZxy'
#        
#    def write(self,s):
#        self.mainframe.write('')
        
def test_yoko(yoko=None):
    if yoko is None:
        yoko=YokogawaGS200(address='10.120.35.219')
        
    print yoko.get_id()
    yoko.set_mode('VOLT')
    print yoko.get_mode()
    print yoko.get_volt()
    yoko.set_measure_state()
    print yoko.get_measure()
    

        
if __name__=="__main__":
    #srs=SRS900(address="COM17")
    #print srs.get_id()
    #srs.set_volt(.5,2)
    yoko=YokogawaGS200(address='10.120.35.219')
    test_yoko(yoko)
