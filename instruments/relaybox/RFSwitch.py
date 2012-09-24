# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:49:01 2012

Relay Numbers(Ports) are always 1-8 unless activating multiple relays at a time.
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
    
    rfs=RFSwitch(address=RF_1)
    rfs.set_relay(3, True)