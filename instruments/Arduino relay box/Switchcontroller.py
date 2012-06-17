# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:49:01 2012

Relay Numbers(Ports) are always 1-8 unless activating multiple relays at a time.
@author: ThomasLaptop
"""

from slab.instruments import RelayBox
import time

class Switch(Relay Box):
    
    def __init__(self,name="Switch",address="COM6",enabled=True,timeout=0):
        RelayBox.__init__(self,name,address,enabled,timeout)
        
        self.set_relay()
        
    def get_status(self,port=0)
        ans=self.query('@%s RS %d' % (self.boxaddress,port))       
        relay_status=[x=='1' for x in bin(256+int(ans[4:-2]))[-8:]]
        relay_status.reverse()
        if port !=0: return relay_status[port-1]
        else: return relay_status
    
    def connect(self, port=0, sign="high")
        self.set_relay()
        if sign is "high":
            self.set_relay(port, True)
            self.pulse_relay(self,port=7,pulse_width=1)
        elif sign is "low":
            self.set_relay(port, True)
            self.pulse_relay(self,port=8,pulse_width=1)
        else:
            print "Type high or low!"
            
    
        
    def reset(self)
    
        self.set_relay()