# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:09:25 2011

@author: Phil
"""
from slab.instruments import SocketInstrument
import time

class BNCAWG(SocketInstrument):
    def __init__(self,name='BNCAWG',address='', enabled=True,timeout=10, recv_length=1024):
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
        if state: self.write('OUTPUT ON')
        else:     self.write('OUTPUT OFF')
        
    def get_output(self):
        """Query Output State"""
        return int(self.query('OUTPUT?')) == 1
    
    def set_function(self,ftype="sine"):
        ftypes={"SINE":"SIN","SQUARE":"SQU","RAMP":"RAMP","PULSE":"PULSE","NOISE":"NOISE","DC":"DC","USER":"USER"}
        ftype_str=ftypes[ftype.upper()]
        self.write('FUNCtion %s' % ftype_str)
        
    def get_function(self):
        return self.query('FUNCtion?')
        
    def set_frequency(self,frequency):
        self.write('FREQ %f' % (frequency))        
        
    def get_frequency(self):
        return float(self.query('FREQ?'))
        
    def set_voltage(self,voltage):
        self.write('VOLT %f' % voltage)
        
    def get_voltage(self):
        return float(self.query('VOLT?'))
        
    def set_offset(self,offset):
        self.write("VOLT:OFFSET %f" % offset)
        
    def get_offset(self):
        return float(self.query("VOLT:OFFSET?"))
        
    def set_amplitude(self,voltage):
        self.set_voltage(voltage)
    
    def get_amplitude(self):
        return self.get_voltage()
        
    def set_trigger_source(self,source="INT"):
        trig_types={'INT':'IMM','INTERNAL':'IMM','EXTERNAL':'EXT','EXT':'EXT','BUS':'BUS'}
        trig_type_str=trig_types[source.upper()]
        self.write('TRIG:SOURCE %s' % trig_type_str)
        
    def get_trigger_source(self):
        return self.query('TRIG:SOURCE?')
        
    def trigger(self):
        self.write('TRIGGER')
        
    def set_burst_cycles(self,cycles=1):
        self.write('BURST:NCYCLES %d' % cycles)
          
    def get_burst_cycles(self,):
        return int(self.query('BURST:NCYCLES?'))
        
    def set_burst_period(self,period):
        self.write('BURSt:INTernal:PERiod %f' % period)
    
    def get_burst_period(self):
        return float(self.query('BURSt:INTernal:PERiod?'))
        
    def set_burst_state(self,state=True):
        if state: self.write('BURst:STATe ON')
        else:     self.write('BURst:STATe OFF')
        
    def get_burst_state(self):
        return int(self.query('BURST:STATE?')) == 1
        
    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['id']=self.get_id()
        settings['output']=self.get_output()
        settings['frequency']=self.get_frequency()
        return settings

class FilamentDriver(BNCAWG):
    
    def setup_driver(self,amplitude,offset,frequency,pulse_length):
        self.set_output(False)
        self.set_amplitude(amplitude)
        self.set_offset(offset)
        self.set_frequency(frequency)
        
        self.set_burst_state(True)
        self.set_burst_cycles(round(pulse_length *frequency))
        self.set_trigger_source('bus')
        
        self.set_output(True)
        
    def fire_filament(self, pulses=1, delay=0):
        for ii in range(pulses):
            self.trigger()
            time.sleep(delay)
            

        
    
if __name__=="__main__":
    #bnc=BNCAWG(address='192.168.14.133')
    filament=FilamentDriver(address='192.168.14.133')
    print filament.query('*IDN?')
