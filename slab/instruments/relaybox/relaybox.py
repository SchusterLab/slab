# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:27:19 2011

This program utilizes the pyserial library in a 32 bit environment, and can not
be ran on a 64bit python evironment. 

The serial.read() function requires a specific size parameter. During reading 
from the relay box, a string of length varying from SSS to !#$ is generated, 
therefore sometimes the program has to wait for the timeout. This is currently 
source of delay in the program. 

@author: Ge Yang


"""
from slab.instruments import SerialInstrument, Instrument 
import time
import urllib2


class IPInstrument(Instrument):
    def __init__(self,name,address='',enabled=True):
        Instrument.__init__(self,name,address,enabled)
        self.protocol='IP'
        self.enabled=enabled

class RelayBox(SerialInstrument, IPInstrument):
    def __init__(self,name="",address='COM6',enabled=True,timeout=0):
        if address[:3].upper()=='COM':
            SerialInstrument.__init__(self,name,address,enabled,timeout,querysleep=0.1)
            self.term_char='\r'
            self.boxaddress = '00'      
        elif address == 'RF1' or address == 'RF2':
            IPInstrument.__init__(self,name,address,enabled)
    def set_relay(self, port=0, state=False):
        
        if self.protocol == "serial":
            if state: self.query('@%s ON %d' % (self.boxaddress,port))
            else:     self.query('@%s OF %d' % (self.boxaddress,port))
        if self.protocol == "IP":
            port = str(port)
            if self.address == "RF1":
                if state: 
                    f = urllib2.urlopen("http://192.168.14.20/relaybox/json?ON" + port)
                    print f.read()
                else: 
                    f = urllib2.urlopen("http://192.168.14.20/relaybox/json?OF" + port)
                    print f.read()
            if self.address == "RF2":
                if state:  
                    f = urllib2.urlopen("http://192.168.14.21/relaybox/json?ON" + port)
                    print f.read()
                else:  
                    f = urllib2.urlopen("http://192.168.14.21/relaybox/json?OF" + port)
                    print f.read()
        
    
    def get_relay(self,port=0):
        if self.protocol=="serial":
            ans=self.query('@%s RS %d' % (self.boxaddress,port))       
            relay_status=[x=='1' for x in bin(256+int(ans[4:-2]))[-8:]]
            relay_status.reverse()
            if port !=0: return relay_status[port-1]
            else: return relay_status
        if self.protocol=="IP":
            if self.address=="RF1":
                 f = urllib2.urlopen("http://192.168.14.20/relaybox/json?RS" + port)
                 print f.read()
            if self.address=="RF2":
                f = urllib2.urlopen("http://192.168.14.21/relaybox/json?RS" + port)
                print f.read()
            
        
    def get_analog_input (self,port=0):
        ans=self.query('@%s AI %d' % (self.boxaddress,port))
        analog_inputs=[int(x) for x in ans.split()[1:]]
        if port!=0: return analog_inputs[port-1]
        else: return analog_inputs
        
    def pulse_relay(self,port=0,pulse_width=1):
        if self.protocol=="serial":  
            self.query('@%s TR %d %03d' % (self.boxaddress,port,pulse_width))
        if self.protocol=="IP":
            if self.address=="RF1":
                 f = urllib2.urlopen("http://192.168.14.20/relaybox/json?TR" + port)
                 print f.read()
            if self.address=="RF2":
                f = urllib2.urlopen("http://192.168.14.21/relaybox/json?RS" + port)
                print f.read()
        
    def keep_alive(self,time_to_live=0):
        self.query('@%s KA %d' % (self.boxaddress,time_to_live))
        
    def write_relays(self,relaystate=0):
        self.query('@%s WR %d' % (self.boxaddress,relaystate))
   
if __name__== '__main__':
    re=RelayBox(address='RF2')
    #re.write_relays(0b11011111)
    re.set_relay(1, False)
# #   re.close()
#    re.relay(1)
#    re.relay(1,'ON')
#    print 'now wait for 1 second'
#    time.sleep(1)
#    re.relay(1,'OF')
#    LoopSize=100
##    for i in range(LoopSize):
#    re.relay(0,'ON')
##        time.sleep(0.1)
#    re.relay(0,'OF')
    
#    re.pulseOn(1,5)
#    re.pulseOn(2,5)
#    re.pulseOn(3,10)
#    re.pulseOn(4,20)
#    re.pulseOn(5,40)
#    re.pulseOn(6,80)
#    re.relay(1,'ON')
#    time.sleep(1)
#    re.relay(1,"OF")
    
#    
    
    