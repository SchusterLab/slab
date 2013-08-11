# -*- coding: utf-8 -*-
"""
Magnet Power Supplies
Created on Sun Aug 11 13:11:39 2013

@author: David Schuster
"""

from slab.instruments import SerialInstrument,VisaInstrument
import re, time
from numpy import linspace

class IPSMagnet(SerialInstrument,VisaInstrument):
    """Oxford IPS 120 Magnet power supply    """
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
                setpt=self.get_setpoint()
                if abs(current-setpt)<tol:
                    return
                else:
                    print self.name+": set_point out of tolerance range\nSet to: %f\tRead back: %f" % (current,setpt)
            except:
                print "Warning: could not set set_point trying again..."
            count+=1
            self.reset_connection()
            time.sleep(1)
        raise Exception("Can't set target current correctly!\nSet to: %f\tRead back: %f" % (current,setpt))
        
    def set_target_field (self,field):
        """Sets target magnetic field to field (in Tesla)"""
        self.query('J%08.5f' % field)
        
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
            
    def set_soft_limit(self,current_limit=None,field_limit=None):
        if field_limit is not None: self.field_limit=field_limit
        if current_limit is not None: self.current_limit=current_limit

    def get_settings(self):
        return {'status':self.get_status()}
        #,'mode':self.get_settings(),
        #        'current':self.get_current(),'persistent field':self.get_persistent_current(),
        #        'volt':self.get_volt(),'setpoint':self.get_setpoint(),'sweeprate':self.get_sweeprate()}
                



class KEPCOPowerSupply(SerialInstrument):
    
    def __init__(self,name="Kepco",address='COM4',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'        
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout)
        self.query_sleep=0.05
        self.recv_length=65536
        self.Remote()
        
    def get_id(self):
        return self.query('*IDN?')
  
    def Remote(self):
        self.write('SYST:REM 1')
        
    def Local(self):
        self.write('SYST:REM 0')
     
    def set_output(self,output=True):
        if output:
            self.write('OUTP 1')
        else:
            self.write('OUTP 0')
    
    def set_voltage(self,v):
        if self.protocol == 'serial':
            self.write('VOLT %f'%v)
            
    def get_voltage(self):
        return float(self.query('VOLT?').strip("\x13\r\n\x11"))
    
    def set_current(self,c):
        self.write('CURR %f'%c)
        
    def ramp_to_current(self,c,sweeprate=None):
        if sweeprate is None:
            sweeprate=self.sweeprate
        
        start=self.get_current()
        stop=c
        start_t=time.time()
        self.set_current(start)
        time.sleep(self.query_sleep)
        step_t=time.time()-start_t
        total_t=abs(stop-start)/sweeprate
        steps=total_t/step_t
        
        for ii in linspace(start,stop,steps):
            self.set_current(ii)
            time.sleep(self.query_sleep)
        
    def get_current(self):
        return float(self.query('CURR?').strip("\x13\r\n\x11"))
    
    def set_current_mode(self):
        self.write('FUNC:MODE CURR')
    
    def set_voltage_mode(self):
        self.write('FUNC:MODE VOLT')
        
    def set_soft_limit(self,current_limit=None,field_limit=None):
        if field_limit is not None: self.field_limit=field_limit
        if current_limit is not None: self.current_limit=current_limit


class VectorSupply(Instrument):
    """VectorMagnet Composite Instrument Class, consisting of IPS Magnet, and 3 Kepcos"""
    def __init__(self,name,address='',enabled=True):
        """address string of form M:COMM X:COMX Y:COMY Z:COMZ"""
        if main_address is None: self.main=None
        else: self.main=IPSMagnet(name=name+"_main",address=main_address)
        if x_address is None: self.x=None
        else: self.x=KEPCOPowerSupply(name=name+"_x",address=x_address)
        if y_address is None: self.y=None
        else: self.y=KEPCOPowerSupply(name=name+"_y",address=y_address)
        if z_address is None: self.z=None
        else: self.z=KEPCOPowerSupply(name=name+"_z",address=z_address)

    def set_current_sweep_rates(self,mainrate=None,xrate=None,yrate=None,zrate=None):
        for m,r in zip([self.main,self.x,self.y,self.z],[mainrate,x_rate,yrate,zrate]):
            if m is not None: m.set_current_sweep_rate(r)

    def set_field_sweep_rates(self,mainrate=None,xrate=None,yrate=None,zrate=None):
        for m,r in zip([self.main,self.x,self.y,self.z],[mainrate,x_rate,yrate,zrate]):
            if m is not None: m.set_field_sweep_rate(r)

    def set_soft_current_limits(self,main_limit=None,x_limit=None,y_limit=None,z_limit=None):
        for m,l in zip([self.main,self.x,self.y,self.z],[main_limit,x_limit,y_limit,z_limit]):
            if m is not None: m.set_soft_limit(current_limit=l)

    def set_soft_field_limits(self,main_limit=None,x_limit=None,y_limit=None,z_limit=None):
        for m,l in zip([self.main,self.x,self.y,self.z],[main_limit,x_limit,y_limit,z_limit]):
            if m is not None: m.set_soft_limit(field_limit=l)

    def ramp_to_current(self,x=None,y=None,z=None):
        pass

    def ramp_to_field(self,x=None,y=None,z=None):
        pass
        
        
    def load_calibration(self,fname=None):
        pass        
    def save_calibration(self,fname=None):
        pass        


if __name__ == '__main__':
    p=KEPCOPowerSupply(address="COM6")
    
    #magnet.set_local()
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    
if __name__ == '__main__':
    print "HERE"
    #magnet=IPSMagnet(address='COM1')
    magnet=IPSMagnet(address='GPIB0::25::INSTR')
    magnet.set_mode('remote_unlocked')
    #magnet.set_local()
    print magnet.get_id()
    #magnet.set_heater()
    #magnet.set_current_sweeprate(0.3) 
    #magnet.hold()
    
    print "done"
    #print fridge.get_status()
    #d=fridge.get_temperatures()
    #print fridge.get_temperatures()
    #print fridge.get_settings()
    
