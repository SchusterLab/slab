# -*- coding: utf-8 -*-
"""
IPS Magnet Controller (ipsmagnet.py)
====================================
:Author: David Schuster
"""
from slab.instruments import SerialInstrument,VisaInstrument
import re
import time

class IPSMagnet(SerialInstrument,VisaInstrument):
    
    def __init__(self,name="magnet",address='COM4',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        else:
            VisaInstrument.__init__(self,name,address,enabled, term_chars='\r')
        self.query_sleep=0.05
        self.recv_length=65536
        self.term_char='\r'
        self.set_mode()
        self.set_extended_resolution()
        
    def read(self):
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'VISA':
            return VisaInstrument.read(self)
            
    def write(self, s):
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'VISA':
            VisaInstrument.write(self, s)
    
    def __del__(self):
        return
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'VISA':
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

    def get_field(self):
        """Returns magnet field in Tesla"""
        return float(self.query('R7')[1:])
        
    def get_volt(self):
        """Returns power supply voltage"""
        return float(self.query('R1')[1:])
        
    def get_current_setpoint(self):
        """Returns current set point in amps"""
        for i in range(5):        
            try:
                return float(self.query('R5')[1:])
            except:
                print("Warning: get_setpoint failed, trying again")
        raise Exception ("Error: get_setpoint failed several times giving up!")
       
    def get_field_setpoint(self):
        """Returns current set point in amps"""
        for i in range(5):        
            try:
                return float(self.query('R8')[1:])
            except:
                print("Warning: get_setpoint failed, trying again")
        raise Exception ("Error: get_setpoint failed several times giving up!")

       
    def get_sweeprate(self):
        """Returns current sweep rate in amp/min"""
        return float(self.query('R6')[1:])
        
    def get_persistent_current(self):
        """returns the persistent magnet current in amps"""
        return float(self.query('R16')[1:])

    def get_persistent_field(self):
        """returns the persistent magnet current in amps"""
        return float(self.query('R18')[1:])

    def hold(self):
        """Hold current state"""
#        self.remote()
        self.query('A0')
        
    def ramp(self):
        """Ramp to Set point"""
#        self.remote()
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
        tol=.005
        count=0
        self.remote()
        while (count<20):
            try:
                #print 'I%07.4f' % current
                self.query('I%06.3f' % current)
                time.sleep(0.05)
                setpt=self.get_current_setpoint()
                if abs(current-setpt)<tol:
                    return
                else:
                    print(self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (current,setpt))
            except:
                print("Warning: could not set set_point trying again...")
            count+=1
            self.reset_connection()
            time.sleep(1)
        raise Exception("Can't set target current correctly!\nSet to: %f\tRead back: %f" % (current,setpt))
        
    def set_target_field (self,field):
        """Sets target magnetic field to field (in Tesla)"""
        tol=.001
        count=0
        self.remote()
        while (count<20):
                        
            print("This is attempt %s" %(float(count)))
            if count==15:
                self.hold()
                print("Holding the Magnet")
            elif count==18:
                self.query('J%08.5f' % field)
                time.sleep(0.1)
                setpt=self.get_field_setpoint()
                new_tol = 0.0021
                if abs(field-setpt)<new_tol:
                    return
                else:
                    print(self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (field,setpt))
                
                
            try:
                self.query('J%08.5f' % field)
                time.sleep(0.1)
                setpt=self.get_field_setpoint()
                if abs(field-setpt)<tol:
                    return
                else:
                    print(self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (field,setpt))
            except:
                print("Warning: could not set set_point trying again...")
            count+=1
            self.reset_connection()
            time.sleep(1)
            
            if count==19:
                print("Can't set target field, but it's less than 2mT.")
                time.sleep(2)
                self.query('J%08.5f' % field)
                time.sleep(0.5)
                setpt=self.get_field_setpoint()
                if abs(field-setpt)<0.0021:
                    return
                else:
                    print(self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (field,setpt))
                
                
        raise Exception("Can't set target field correctly!\nSet to: %f\tRead back: %f" % (field,setpt))
        
    def set_current_sweeprate(self,sweeprate):
        """Sets current sweep rate in Amps/minute"""
        self.query('S%07.4f' % sweeprate)
        
    def set_field_sweeprate(self,sweeprate):
        """Sets current sweep rate in Tesla/minute"""
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
                print("Warning: get_mode failed!")
            
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
        'Field in Teslas, sweeprate in Amps/minute'
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
    print("HERE")
    #magnet=IPSMagnet(address='COM1')
    magnet=IPSMagnet(address='GPIB0::25::INSTR')
    magnet.set_mode('remote_unlocked')
    #magnet.set_local()
    print(magnet.get_id())
    #magnet.set_heater()
    #magnet.set_current_sweeprate(0.3) 
    #magnet.hold()
    
    print("done")
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    
