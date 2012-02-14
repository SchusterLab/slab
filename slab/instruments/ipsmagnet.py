# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 09:55:05 2011

@author: Phil
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re

class IPSMagnet(SerialInstrument,VisaInstrument):
    
    def __init__(self,name="magnet",address='COM4',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        else:
            VisaInstrument.__init__(self,name,address,enabled)
        self.query_sleep=0.05
        self.recv_length=65536
        self.term_char='\r'
        self.set_extended_resolution()
        
    def read(self):
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'GPIB':
            return VisaInstrument.read(self)
            
    def write(self, s):
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'socket':
            VisaInstrument.write(self, s)
    
    def __del__(self):
        return
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'visa':
            VisaInstrument.__del__(self)
    
    def get_id(self):
        return self.query('V')
           
    def set_mode(self,mode='local_unlocked'):
        """Sets mode to (remote/local)_(locked/unlocked)"""
        modestr={'local_locked':'C0','remote_locked':'C1','local_unlocked':'C2','remote_unlocked':'C3'}[mode]
        self.query(modestr)
        
    def get_current(self):
        """Returns magnet current in amps"""
        return float(self.query('R0')[1:])
        
    def get_volt(self):
        """Returns power supply voltage"""
        return float(self.query('R1')[1:])
        
    def get_setpoint(self):
        """Returns current set point in amps"""
        for i in range(5):        
            try:
                return float(self.query('R5')[1:])
            except:
                print "Warning: get_setpoint failed, trying again"
        raise Exception ("Error: get_setpoint failed several times giving up!")
       
       
    def get_sweeprate(self):
        """Returns current sweep rate in amp/min"""
        return float(self.query('R6')[1:])
        
    def get_persistent_current(self):
        """returns the persistent magnet current in amps"""
        return float(self.query('R16')[1:])
        
    def hold(self):
        """Hold current state"""
#        self.remote()
        self.query('A0')
        
    def ramp(self):
        """Ramp to Set point"""
        self.remote()
        self.query('A1')
        
    def zero(self):
        """Ramp to zero"""
        self.remote()
        self.query('A2')
        
    def clamp(self):
        """Clamped"""
        self.remote()
        self.query('A3')
        
    def set_heater(self, state=True, safe=True):
        """If state is True/False turns heater On/Off, 
        if safe=True will only open heater if PSU current = magnet current"""
        self.remote()
        if state:
            if safe:
                self.query('H1')
            else:
                self.query('H2')
        else:
            self.query("H0")

    def set_target_current(self, current):
        """Sets target current to current (in amps)"""
        tol=.001
        count=0
        self.remote()
        while (count<5):
            try:
                print 'I%07.4f' % current
                self.query('I%07.4f' % current)
                setpt=self.get_setpoint()
                if abs(current-setpt)<tol:
                    return
                else:
                    print self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (current,setpt)
            except:
                print "Warning: could not set set_point trying again..."
            count+=1
        raise Exception("Can't set target current correctly!\nSet to: %f\tRead back: %f" % (current,setpt))
        
        
    def set_target_field (self,field):
        """Sets target magnetic field to field (in Tesla)"""
        self.query('I%08.5f' % field)
        
    def set_current_sweeprate(self,sweeprate):
        self.query('S%07.4f' % sweeprate)
        
    def set_field_sweeprate(self,sweeprate):
        self.query('T%08.5f' % sweeprate)
        
    def get_status(self):
        status=self.query('X')
        return status
        
    def get_mode(self):
        done=False
        while not done:
            try:
                status=self.query('X')
                n=int(status.split('M')[1][1])
                mode={0:"Rest",1:"Sweeping",2:"Sweep Limiting",3:"Sweep and Sweep Limit"}[n]
                done = True
            except:
                print "Warning: get_mode failed!"
            
        return mode
        
        
    def remote(self):
        self.set_mode('remote_unlocked')
        
    def set_extended_resolution(self):
        self.write('Q4')
        
    def ramp_to_current(self,current,sweeprate=None):
        self.remote()
        self.hold()
        self.set_target_current(current)
        if sweeprate is not None: self.set_current_sweeprate(sweeprate)
        self.ramp()
        
    def ramp_to_field(self,field, sweeprate=None):
        self.remote()
        self.hold()
        self.set_target_field(field)
        if sweeprate is not None: self.set_field_sweeprate(sweeprate)
        self.ramp()
        
    def wait_for_ramp(self,timeout=None):
        waiting=True        
        while (waiting):
            if self.get_mode()=='Rest': return

    def get_settings(self):
        return {'status':self.get_status()}
        #,'mode':self.get_settings(),
        #        'current':self.get_current(),'persistent field':self.get_persistent_current(),
        #        'volt':self.get_volt(),'setpoint':self.get_setpoint(),'sweeprate':self.get_sweeprate()}
                


if __name__ == '__main__':
    magnet=IPSMagnet (address="COM8")
    #magnet.set_local()
    print magnet.get_id()
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    