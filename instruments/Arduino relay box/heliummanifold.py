# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 23:39:28 2012

@author: Julia
"""
from slab.instruments import RelayBox
import time

DEBUG_HELIUM_MANIFOLD = True
DEBUG_HELIUM_MANIFOLD_VERBOSE = False

class HeliumManifold(RelayBox):

    inlet_port=1
    pump_port=4        
    outlet_port=8
    pressure_port=1

    atm_level=340.6
    vacuum_offset=0.
    vacuum_threshold=0.01

    def __init__(self,name="Helium Manifold",address="COM6",enabled=True,timeout=0):
        RelayBox.__init__(self,name,address,enabled,timeout)
        self.puffs=0
        self.query_sleep=.1
        
    def get_pressure(self,avgs=1):
        pressure=0
        for avg in range(avgs):
            pressure=pressure*float(avg)/(avg+1)+self.vacuum_offset+float(self.get_analog_input(self.pressure_port))/self.atm_level/(avg+1.)
        self.pressure=pressure
        return self.pressure
        
    def get_manifold_status(self):
        relay_states=self.get_relay()
        self.inlet_state=relay_states[self.inlet_port-1]
        self.pump_state=relay_states[self.pump_port-1]
        self.outlet_state=relay_states[self.outlet_port-1]
        self.get_pressure()
        status_str="Inlet: %s / Pump: %s / Outlet: %s / Pressure: %f bar" % (str(self.inlet_state),str(self.pump_state),str(self.outlet_state),self.pressure)
        if DEBUG_HELIUM_MANIFOLD_VERBOSE: print status_str
        return status_str
    
    def set_inlet(self,state=False):
        if DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set Inlet: %s"% str(state)
        if not state:
            self.set_relay(self.inlet_port,state)
        else:
            self.get_manifold_status()
            if self.pump_state or self.outlet_state:
                raise Exception("Unsafe operation attempted: Tried to open inlet while other ports open")
            else:
                self.set_relay(self.inlet_port,state)
        self.get_manifold_status()
                
    def set_pump(self,state=False):
        if DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set Pump: %s"% str(state)
        if not state:
            self.set_relay(self.pump_port,state)
        else:
            self.get_manifold_status()
            if self.inlet_state:
                raise Exception("Unsafe operation attempted: Tried to open pump port while inlet port is open")
            else:
                self.set_relay(self.pump_port,state)
        self.get_manifold_status()                
        
    def set_outlet(self,state=False):
        if DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set Outlet: %s"% str(state)
        if not state:
            self.set_relay(self.outlet_port,state)
        else:
            self.get_manifold_status()
            if self.inlet_state:
                raise Exception("Unsafe operation attempted: Tried to outlet port while inlet port is open")
            else:
                self.set_relay(self.outlet_port,state)
        self.get_manifold_status()

    def wait_for_pressure_rise(self,threshold,timeout=None):
        done=False
        start_time=time.time()
        while not done:
            if DEBUG_HELIUM_MANIFOLD_VERBOSE: print self.get_pressure()
            if self.get_pressure()>threshold: done=True
            if timeout is not None:            
                if time.time()-start_time > timeout: 
                    done=True
                    print "HeManifold Pressure timeout final P=%f bar" %self.get_pressure()
        self.get_manifold_status()

    def wait_for_pressure_fall(self,threshold,timeout=None):
        done=False
        start_time=time.time()
        while not done:
            if DEBUG_HELIUM_MANIFOLD_VERBOSE: print self.get_pressure()
            if self.get_pressure()<threshold: done=True
            if timeout is not None:            
                if time.time()-start_time > timeout: 
                    done=True
                    print "HeManifold Pressure timeout final P=%f bar" %self.get_pressure()
        self.get_manifold_status()
           
    def wait_for_vacuum(self,min_time=0,timeout=None):
        start_time=time.time()
        evacuated=False
        while not evacuated:
            self.wait_for_pressure_fall(self.vacuum_threshold,timeout)
            if (time.time()-start_time>min_time): evacuated=True

          
    def pump_manifold(self,min_time=0,timeout=None):
        if DEBUG_HELIUM_MANIFOLD: print "Pump manifold."
        self.set_inlet(False)
        self.set_outlet(False)
        self.set_pump(True)
        self.wait_for_vacuum(min_time=min_time,timeout=timeout)
        self.get_manifold_status()
        
    def pump_outlet(self,timeout=None):
        if DEBUG_HELIUM_MANIFOLD: print "Pump outlet."
        self.set_inlet(False)
        self.set_outlet(False)
        self.set_pump(True)
        self.wait_for_vacuum(timeout=timeout)
        self.set_outlet(True)
#        self.wait_for_vacuum(min_time=min_time,timeout=timeout)
#        self.get_manifold_status()
        
    def fill_manifold(self,fill_level=0.99,timeout=None):
        if DEBUG_HELIUM_MANIFOLD: print "Fill manifold to %f bar." % (fill_level)
        self.set_outlet(False)
        self.set_pump(False)
        self.set_inlet(True)
        self.wait_for_pressure_rise(fill_level,timeout=timeout)
        self.set_inlet(False)
        self.get_manifold_status()
        
    def puff(self,pressure,n=1,min_time=0,timeout=None):
        for i in range(n):
            if DEBUG_HELIUM_MANIFOLD: print "Puff #%d" % (self.puffs+1)
            self.pump_manifold(min_time=min_time,timeout=timeout)
            self.fill_manifold(pressure,timeout)
            self.set_outlet(True)
            self.puffs+=1
            if DEBUG_HELIUM_MANIFOLD: print "Letting puff out."
            self.wait_for_vacuum(min_time=min_time,timeout=timeout)
            self.set_outlet(False)

    def clean_manifold(self, n=1,min_time=0,timeout=None):
        if DEBUG_HELIUM_MANIFOLD: print "Clean manifold %d times." % n
        self.pump_manifold(min_time=min_time,timeout=timeout)
        for i in range (n):
            self.fill_manifold(timeout=timeout)
            self.pump_manifold(min_time=min_time,timeout=timeout)
            
    def seal_manifold(self):
        if DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Seal manifold."
        self.set_inlet(False)
        self.set_outlet(False)
        self.set_pump(False)
        self.get_manifold_status()
            
if __name__=="__main__":
    heman=HeliumManifold(address="COM10")
