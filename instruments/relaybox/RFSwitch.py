# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:49:01 2012

Relay Numbers(Ports) are always 1-8 unless activating multiple relays at a time.
Ports 1-6 can be activated/deactivated
If using USB cable, set address to appropriate COM.
If using web access, set address to appropriate URL.
@author: ThomasLaptop
"""

from slab.instruments.relaybox import RelayBox
import time

class RFSwitch(RelayBox):    

    def __init__(self,name="",address="",enabled=True, RF_Status=[False,False,False,False,False,False]):
        print "Switch", name, "at address", address, " is initiated!" 
        RelayBox.__init__(self,name,address,enabled)
        self.RF_Status=RF_Status
        
    def set_switch (self, switch=0,state=True):
        """activates or deactivates RF Switch according to state, 
        if switch=0 does for all switches, if switch=1..6 does individual port"""
        if state: self.activate(switch)
        else: self.deactivate(switch)        
        
    def get_switch (self,switch=0):
        """Get switch (cached) state of _switch_ if _switch_ = 0 return array of switch states """
        if switch == 0: return self.RF_Status
        else:           return self.RF_Status[switch-1]
                
    def activate(self, switch=0):
        """Activate _switch_ if _switch_=0 activate all switches"""
        if switch>0:
            self.set_relay(switch, True)
            self.pulse_relay(7)
            self.set_relay(switch, False)
            self.RF_Status[switch-1]= True
        elif switch == 0:
            for i in range(1,7):
                self.activate(i)
      
    def deactivate(self, switch=0):
        """Activate _switch_ if _switch_=0 activate all switches"""
        if switch>0:
            self.set_relay(switch, True)
            self.pulse_relay(8)
            self.set_relay(switch, False)
            self.RF_Status[switch-1]= False
        elif switch == 0:
            for i in range(1,7):
                self.deactivate(i)
                
if __name__=="__main__":
    RF_1= 'http://192.168.14.20'
    RF_2='http://192.168.14.21'
    
    rfs_1=RFSwitch(name="Switch1", address=RF_1)
    rfs_2=RFSwitch(name="Switch2", address=RF_2)

    for i in range(6):
       rfs_1.activate(i+1)
       print rfs_1.RF_Status
    
    for i in range(6):
       rfs_1.deactivate(i+1)
       print rfs_1.RF_Status
    
    
    
    
    
