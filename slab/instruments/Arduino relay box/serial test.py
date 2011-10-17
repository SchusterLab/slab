# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 00:38:26 2011

@author: Ge Yang
"""

import serial
import time
ser = serial.Serial(port=2,timeout=1)
ser.close()
ser.open()
print ser.portstr

write=ser.write('@00 RS 1\r')
rdLn=ser.inWaiting()
print rdLn
read=ser.read(10)
print read

print 'I am waiting now!!'
time.sleep(1)


write=ser.write("@00 OF 1\r")
time.sleep(0.1)
print "Did it Work?"
write=ser.write('@00 RS 1\r')
rdLn=ser.inWaiting()
print rdLn
read=ser.read(10)
print read

#ser.close()