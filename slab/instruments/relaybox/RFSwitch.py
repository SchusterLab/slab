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
    
    def __init__(self,name="",address="",enabled=True):
       RelayBox.__init__(self,name,address,enabled)
        
    def activate(self, port=0):    
        self.set_relay(port, True)
        self.pulse_relay(7)
        self.set_relay(port, False)
      
    
    def deactivate(self, port=0):
        self.set_relay(port, True)
        self.pulse_relay(8)
        self.set_relay(port, False)


if __name__=="__main__":
    RF_1= 'http://192.168.14.20/relaybox/json?'
    RF_2='http://192.168.14.21/relaybox/json?'
    
    rfs_1=RFSwitch(address=RF_1)
    rfs_2=RFSwitch(address=RF_2)
    rfs_1.activate(4)
    rfs_2.activate(4)
    
    
    
    
    
    rfs_1.get_relay(0)
    rfs_2.get_relay(0)