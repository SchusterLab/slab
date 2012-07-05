# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:49:01 2012

Relay Numbers(Ports) are always 1-8 unless activating multiple relays at a time.
@author: ThomasLaptop
"""

from slab.instruments import RelayBox
import time

class RFSwitch(RelayBox):    
    
    def __init__(self,name="",address="COM8",enabled=True,timeout=0):
        RelayBox.__init__(self,name,address,enabled,timeout)
        
        self.set_relay()
        
    def get_status(self,port=0):
        ans=self.query('@%s RS %d' % (self.boxaddress,port))       
        relay_status=[x=='1' for x in bin(256+int(ans[4:-2]))[-8:]]
        relay_status.reverse()
        if port !=0: return relay_status[port-1]
        else: return relay_status
    
    def connect(self, port=0, sign= '+'):
        self.set_relay()
        if sign is "+":
            self.set_relay(port, True)
            self.pulse_relay(port=7,pulse_width=1)
            self.set_relay(port, False)           
        elif sign is "-":
            self.set_relay(port, True)
            self.pulse_relay(port=8,pulse_width=1)
            self.set_relay(port, False)

        else:
            print "Type + or -!"
            
    
        
    def reset(self):
    
        self.set_relay()
        
        
if __name__=="__main__":
    rfs=RFSwitch()
    print rfs.get_status()
    rfs.connect(port=1)