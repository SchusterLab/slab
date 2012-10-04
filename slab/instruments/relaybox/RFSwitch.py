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

    def __init__(self,name="",address="",enabled=True, RF_Status="000000"):
        #print "Switch", name, "at address", address, " is initiated!" 
        RelayBox.__init__(self,name,address,enabled)
        self.RF_Status=RF_Status
        
    def activate(self, port=0,pulse_width=1):    
        if port >0:
            self.set_relay(port, True)
            self.pulse_relay(7)
            self.set_relay(port, False)
            if port == 1:
                self.RF_Status= '1' + self.RF_Status[1:]
            else:
                self.RF_Status= self.RF_Status[0:port-1] + '1' + self.RF_Status[port:]
        elif port == 0:
            for ii in range(1,7):
                self.activate(ii,pulse_width)
      
    
    def deactivate(self, port=0,pulse_width=1):
        if port>0:
            self.set_relay(port, True)
            self.pulse_relay(8)
            self.set_relay(port, False)
            if port == 1:
                self.RF_Status= '0' + self.RF_Status[1:]
            else:
                self.RF_Status= self.RF_Status[0:port-1] + '0' + self.RF_Status[port:]
        elif port == 0:
            for ii in range(1,7):
                self.deactivate(ii,pulse_width)
                
if __name__=="__main__":
    RF_1= 'http://192.168.14.20'
    RF_2= 'http://192.168.14.21'
    
    rfs_1=RFSwitch(name="Switch1", address=RF_1)
    rfs_2=RFSwitch(name="Switch2", address=RF_2)

#    for i in range(6):
#       rfs_1.activate(i+1)
#       print rfs_1.RF_Status
#    
#    for i in range(6):
#       rfs_1.deactivate(i+1)
#       print rfs_1.RF_Status
    
    
    
    
    
