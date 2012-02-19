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

import serial
import time


class relaybox():
    OnOff={'0':'OF', '1':'ON'}
    ser=serial.Serial()
            
    def __init__(self, com=2, timeout=5):
        self.c=com
        self.t=timeout
        self.baudrate='9600'
        self.address = '00'
        self.stopbit = 1
        self.databits=8
        self.parity = None
        self.open()
        self.setBaudRate()
        self.open()
                     
    def hello(self):
        return "Hello World!"
        
    def open(self):
        self.ser.timeout=self.t
        self.ser.port=self.c
        if self.ser.isOpen()==True:
            print 'serial port COM{0} is already open'.format(self.c+1)
        else:
            print 'serial port COM{0} is not open, I am trying to open it...'.format(self.c+1)
            self.ser.open()
            if self.ser.isOpen()==True:
                print 'serial port COM{c} now is open with {t}s timeout.'.format(c=self.c+1,t=self.t)
            else:
                print 'serial port COM{0} is can\'t be opened, there might be something wrong!'.format(self.c+1)
                return
    def close(self):
        self.ser.port=self.c        
        self.ser.close()
        if self.ser.isOpen()==False:
            print 'serial port COM{0} is now closed'.format(self.c+1)
        else:
            print 'serial port COM{0} is still open, please try again~'.format(self.c+1)
    def setBaudRate(self):
        if self.ser.isOpen(): 
            write_string='@'+self.address+' BS '+self.baudrate+'\r'
            self.ser.write(write_string)
            print(write_string)
        else: print 'port is not open...'
    def relay(self, port=False, state=''):
        if self.ser.isOpen()==False:

            self.open()
        if port==False:
            print "0 operates to all ports"
                       
        if state=='':
            self.read(port)
            print '*****************************include ON or OF as the state if want to change it.'
        elif state=='ON' or state=='OF':
            print 'Setting Port{0} to {1}'.format(port, state), "------",            
            cmd = '@'+self.address+' {0} {1}\r'.format(state, port)
            print cmd
            self.ser.write(cmd)
            #time.sleep(0.1)
            self.read(port)
    def read(self, port=1):
        if port == 0:     
            write_str='@'+self.address+' RS 0\r'
            self.ser.flushInput()            
            self.ser.write(write_str)
            read=bin(int(self.ser.read(7).split()[-1])+256)[-8:]
            print 'relay status are',read
#            for i in range(8):
#              write_str='@'+self.address+' RS {0}\r'
#              self.ser.write('@'+self.address+' RS {0}\r'.format(i))
#              read=self.ser.read(7).split()
#              print 'Port {0}\'s state is '.format(i) + self.OnOff[read[-1]]
        else:
            print 'asking state of Port {0}'.format(port),
            self.ser.flushInput()            
            self.ser.write('@'+self.address+' RS {0}\r'.format(port))
            #time.sleep(0.1)
            read=self.ser.read(7)
            readR=read.split()
            print(read)
            print 'splited read is',readR
            print 'Port {0}\'s state is '.format(port) + self.OnOff[readR[-1]]

            #IT TAKES TIME for the writing command to finish writing and reading from the         
    def pulseOn(self,port,t):
        """
            Two parameters are required. 
        port is a number from 1-8 (0 is not included, aka no way to operate all relays)
        time is a number from 1 to 255, in unit of 100us. 
            for example, 100 corresponds to 10s. 001 cor. to 0.1s.
            Note: Relay X is locked from further pulses until the current operation is finished.
        """
        bk=0
        if port >8 or port <0 : print'port number no good. Need 1-8' ; bk=1
        if t >255 or t<0 : print 'Need Pulse On time between 0-255'; bk=1
        if bk==1 : return
        else:
            #Need to cast t into a three digit # from 001 to 255 to be used in the writing string
            t=str(t+1000)[-3:]
            write_str='@'+str(self.address)+' TR '+str(port)+' '+t+'\r'
            print(write_str)
            self.ser.write(write_str)
    print 'done loading!'
    
if __name__== '__main__':
    re=relaybox()
    re.close()
    re.relay(1)
    re.relay(1,'ON')
    print 'now wait for 1 second'
    time.sleep(1)
    re.relay(1,'OF')
    LoopSize=100
#    for i in range(LoopSize):
    re.relay(0,'ON')
#        time.sleep(0.1)
    re.relay(0,'OF')
    
    re.pulseOn(1,5)
    re.pulseOn(2,5)
    re.pulseOn(3,10)
    re.pulseOn(4,20)
    re.pulseOn(5,40)
    re.pulseOn(6,80)
    

    
    
    