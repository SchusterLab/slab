# -*- coding: utf-8 -*-
"""
Helium Manifold Controller (relaybox.heliummanifold.py)
=======================================================
:Author: Ge
"""
from slab.instruments import RelayBox
import time



class HeliumManifold(RelayBox):

    DEBUG_HELIUM_MANIFOLD = True
    DEBUG_HELIUM_MANIFOLD_VERBOSE = True
    
    gas_port=1
    pump_port=4        
    cryostat_port=8
    pressure_port=7 #port 1 if connected with COM, port 7 if as http server

    atm_level=340.6
    vacuum_offset=0.
    vacuum_threshold=0.03

    def __init__(self,name="Helium Manifold",address="COM6",enabled=True,timeout=0,puffs=0):
        RelayBox.__init__(self,name,address,enabled,timeout)
        print "Manifold", name, "at address", address, " is initiated!"         
        self.puffs=puffs
        self.query_sleep=.1
        
    def get_pressure(self,avgs=1):
        pressure=0
        for avg in range(avgs):
            pressure=pressure*float(avg)/(avg+1)+self.vacuum_offset+float(self.get_analog_input(self.pressure_port))/self.atm_level/(avg+1.)
        self.pressure=pressure
        return self.pressure
        
    def get_manifold_status(self):
        relay_states=self.get_relay()
        self.gas_state=relay_states[self.gas_port-1]
        self.pump_state=relay_states[self.pump_port-1]
        self.cryostat_state=relay_states[self.cryostat_port-1]
        self.get_pressure()
        status_str="gas: %s / Pump: %s / cryostat: %s / Pressure: %f bar" % (str(self.gas_state),str(self.pump_state),str(self.cryostat_state),self.pressure)
        if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print status_str
        return status_str
    
    def set_gas(self,state=False,override=False):
        if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set gas: %s"% str(state)
        if not state or override:
            self.set_relay(self.gas_port,state)
        else:
            self.get_manifold_status()
            if self.pump_state or self.cryostat_state:
                raise Exception("Unsafe operation attempted: Tried to open gas while other ports open")
            else:
                self.set_relay(self.gas_port,state)
        self.get_manifold_status()
                

    def set_pump(self,state=False,override=False):
        if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set Pump: %s"% str(state)
        if not state or override:
            self.set_relay(self.pump_port,state)
        else:
            self.get_manifold_status()
            if self.gas_state:
                raise Exception("Unsafe operation attempted: Tried to open pump port while gas port is open")
            else:
                self.set_relay(self.pump_port,state)
        self.get_manifold_status()                
        
    def set_cryostat(self,state=False,override=False):
        if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Set cryostat: %s"% str(state)
        if not state or override:
            self.set_relay(self.cryostat_port,state)
        else:
            self.get_manifold_status()
            if self.gas_state:
                raise Exception("Unsafe operation attempted: Tried to cryostat port while gas port is open")
            else:
                self.set_relay(self.cryostat_port,state)
        self.get_manifold_status()

    def wait_for_pressure_rise(self,threshold,timeout=None):
        done=False
        start_time=time.time()
        while not done:
            if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print self.get_pressure()
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
            if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print self.get_pressure()
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
        if self.DEBUG_HELIUM_MANIFOLD: print "Pump manifold."
        self.set_gas(False)
        self.set_cryostat(False)
        self.set_pump(True)
        self.wait_for_vacuum(min_time=min_time,timeout=timeout)
        self.get_manifold_status()
        
    def pump_cryostat(self, min_time=0, timeout=None):
        if self.DEBUG_HELIUM_MANIFOLD: print "Pump cryostat."
        self.set_gas(False)
        self.set_cryostat(False)
        self.set_pump(True)
        self.wait_for_vacuum(min_time=min_time, timeout=timeout)
        self.set_cryostat(True)
#        self.wait_for_vacuum(min_time=min_time,timeout=timeout)
#        self.get_manifold_status()
        
    def fill_manifold(self,fill_level=0.20,timeout=None):
        if self.DEBUG_HELIUM_MANIFOLD: print "Fill manifold to %f bar." % (fill_level)
        self.set_cryostat(False)
        self.set_pump(False)
        self.set_gas(True)
        self.set_pump(True,override=True)
        self.wait_for_pressure_fall(threshold=fill_level,timeout=timeout)
        self.set_pump(False)
        self.wait_for_pressure_rise(fill_level,timeout=timeout)
        self.set_gas(False)
        self.get_manifold_status()
        
    def puff(self,pressure,n=1,min_time=0,timeout=None):
        for i in range(n):
            if self.DEBUG_HELIUM_MANIFOLD: print "Puff #%d" % (self.puffs+1)
            self.pump_manifold(min_time=min_time,timeout=timeout)
            self.fill_manifold(pressure,timeout)
            self.set_cryostat(True)
            self.puffs+=1
            if self.DEBUG_HELIUM_MANIFOLD: print "Letting puff out."
            self.wait_for_vacuum(min_time=min_time,timeout=timeout)
            self.set_cryostat(False)
            
    def get_puffs(self):
        return self.puffs
        
    def set_puffs(self,puffs=0):
        self.puffs=puffs

    def clean_manifold(self, n=1,min_time=0,timeout=None):
        if self.DEBUG_HELIUM_MANIFOLD: print "Clean manifold %d times." % n
        self.pump_manifold(min_time=min_time,timeout=timeout)
        for i in range (n):
            self.fill_manifold(timeout=timeout)
            self.pump_manifold(min_time=min_time,timeout=timeout)
            
    def seal_manifold(self):
        if self.DEBUG_HELIUM_MANIFOLD_VERBOSE: print "Seal manifold."
        self.set_gas(False)
        self.set_cryostat(False)
        self.set_pump(False)
        self.get_manifold_status()
            
if __name__=="__main__":
    heman=HeliumManifold(address="COM10")
