# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:45:44 2012

@author: Nitrogen
"""

from slab.instruments import SerialInstrument,VisaInstrument,Instrument
import re
import time

class SRS900(SerialInstrument,VisaInstrument):
    
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
        
        

        
if __name__=="__main__":
    srs=SRS900(address="COM17")
    print srs.get_id()
    srs.set_volt(.5,2)