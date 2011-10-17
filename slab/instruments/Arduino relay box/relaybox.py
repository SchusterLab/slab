# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:27:19 2011

This program utilizes the pyserial library in a 32 bit environment, and can not
be ran on a 64bit python evironment. 
@author: Ge Yang
"""

import serial
import time
ser=serial.Serial()
OnOff={'0':'OF', '1':'ON'}

class relaybox():
    def __init__(self, com=2, timeout=1):
        self.c=com
        self.t=timeout
        self.Open()
                     
    def hello(self):
        return "Hello World!"
        
    def Open(self):
        ser.timeout=self.t
        ser.port=self.c
        if ser.isOpen()==True:
            print 'serial port COM{0} is already open'.format(self.c+1)
        else:
            print 'serial port COM{0} is not open, I am trying to open it...'.format(self.c+1)
            ser.open()
            if ser.isOpen()==True:
                print 'serial port COM{c} now is open with {t}s timeout.'.format(c=self.c+1,t=self.t)
            else:
                print 'serial port COM{0} is can\'t be opened, there might be something wrong!'.format(self.c+1)
                return
    def Close(self):
        ser.port=self.c        
        ser.close()
        if ser.isOpen()==False:
            print 'serial port COM{0} is now closed'.format(self.c+1)
        else:
            print 'serial port COM{0} is still open, please try again~'.format(self.c+1)
            
    def Relay(self, port=False, state=''):
        if ser.isOpen()==False:
            self.Open()
        if port==False:
            print "0 operates to all ports"
                       
        if state=='':
            self.Read(port)
            print '*****************************include ON or OF as the state if want to change it.'
        elif state=='ON' or state=='OF':
            print 'Setting Port{0} to {1}'.format(port, state), "------",            
            cmd = '@00 {0} {1}\r'.format(state, port)
            print cmd
            ser.write(cmd)
            #time.sleep(0.1)
            self.Read(port)
    def Read(self, port=1):
        if port == 0:     
            for i in [1, 2, 3, 4, 5, 6, 7, 8]:
              ser.write('@00 RS {0}\r'.format(i))
              read=ser.read(14).split()
              print 'Port {0}\'s state is '.format(i) + OnOff[read[-1]]
        else:
            print 'asking state of Port {0}'.format(port),
            ser.write('@00 RS {0}\r'.format(port))
            #time.sleep(0.1)
            read=ser.read(14).split()
            print read
            print 'Port {0}\'s state is '.format(port) + OnOff[read[-1]]
            #IT TAKES TIME for the writing command to finish writing and reading from the         


    
    print 'done loading!'
    
if __name__== '__main__':
    re=relaybox()
 #   re.Close()
    re.Relay(1)
    re.Relay(1,'ON')
    print 'now wait for 1 second'
    time.sleep(1)
    re.Relay(1,'OF')
    LoopSize=100
#    for i in range(LoopSize):
    re.Relay(0,'ON')
#        time.sleep(0.1)
    re.Relay(0,'OF')
#    
    
    