# -*- coding: utf-8 -*-
"""
BNC Function Generator (function_generator.py)
==============================================
:Author: David Schuster
"""
from slab.instruments import SocketInstrument
import time

class BNCAWG(SocketInstrument):
    'Interface to the BNC function generator'
    def __init__(self,name='BNCAWG',address='', enabled=True,timeout=0.01, recv_length=1024):
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
    
    def set_termination(self,load=None):
        """Set Output State On/Off"""
        if load:  self.write('OUTPUT:LOAD %s' % load)
    
    def get_termination(self):
        """Set Output State On/Off"""
        return float(self.querry('OUTPUT:LOAD?'))
        
    
    def set_function(self,ftype="sine"):
        ftypes={"SINE":"SIN","SQUARE":"SQU","RAMP":"RAMP","PULSE":"PULSE","NOISE":"NOISE","DC":"DC","USER":"USER"}
        ftype_str=ftypes[ftype.upper()]
        self.write('FUNCtion %s' % ftype_str)
    
    def set_square(dutycycle=50):
        """
        There are limitations to the duty cycle you can set. 
        for frequency lower than 10MHz, it is limited to 20% and 80%."""
        self.write('FUNC:SQU')        
        return self.write('FUNC:SQU:DCYC %s' % str(dutycycle))
        
    def get_function(self):
        return self.query('FUNCtion?')
        
    def set_frequency(self,frequency):
        self.write('FREQ %f' % (frequency))        
        
    def get_frequency(self):
        return float(self.query('FREQ?'))
        
    def set_amplitude(self,voltage):
        self.write('VOLT %f' % voltage)
        
    def get_amplitude(self):
        return float(self.query('VOLT?'))
        
    def set_autorange(self,range):
        """OFF,ON,ONCE"""
        self.write('VOLT:RANGE:AUTO %s' % range.upper())
        
    def get_autorange(self):
        return self.query('VOLT:RANGE:AUTO?').split('\n')
        
    def set_offset(self,offset):
        self.write("VOLT:OFFSET %f" % offset)
        
    def get_offset(self):
        return float(self.query("VOLT:OFFSET?"))
        
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
    
    def set_burst_phase(self,phase):
        """
        phase is between -360 and 360"""
        self.write('BURSt:PHASe %d' % phase)
    
    def get_burst_phase(self):
        return float(self.query('BURSt:PHASe?'))
        
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
        settings['function']=self.get_function()
        settings['amplitude']=self.get_amplitude()
        settings['offset']=self.get_offset()
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
            
class BiasDriver(BNCAWG):
    """
    want a sticky response. Don't want voltage to change no matter what we do. """
    def setup_driver(self,pulse_length, pulse_voltage,rest_voltage,autorange='off'):
        #set the duty cycle to 40/60, 
        #set the starting phase to be         
#        self.set_output(False)
        amp=abs(pulse_voltage-rest_voltage)
        offset=(pulse_voltage+rest_voltage)/2.
        freq=1/(pulse_length*2)   #Need Integer     
        if pulse_voltage<rest_voltage: 
            phase=180
        else: phase=0
        self.set_autorange(autorange)        
        self.set_termination(load='INFinity')
        self.set_function('square')        
        self.set_offset(offset)
        self.set_amplitude(amp)
        self.set_burst_phase(phase)        
        self.set_frequency(freq)
        self.set_burst_state(True)
        self.set_burst_cycles(1)
        self.set_trigger_source('bus')
        self.set_output(True)
        #self.set_autorange(autorange)
        
    def pulse_voltage(self, pulses=1, delay=0):
        for ii in range(pulses):
            self.trigger()
            time.sleep(delay)
#    def set_voltage(self,volt):
#        phase=self.get_burst_phase()
#        amp=self.get_amplitude()        
#        offset=self.get_offset()
#        if volt < offset:
#            self.set_burst_phase(0)
#            self.set_amplitude((offset-volt)*2)
#        else:
#            self.set_burst_phase(180)
#            self.set_amplitude((volt-offset)*2)
    def set_voltage(self,volt):
        phase=self.get_burst_phase()
        amp=self.get_amplitude()        
        offset=self.get_offset()
        if phase==180:
            self.set_offset(volt-amp/2.)
        else:
            self.set_offset(volt+amp/2.)
    def set_volt(self,volt):
        self.set_voltage(volt)
        
if __name__=="__main__":
    #bnc=BNCAWG(address='192.168.14.133')
    filament=FilamentDriver(address='192.168.14.133')
    print filament.query('*IDN?')
