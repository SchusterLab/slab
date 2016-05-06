"""


Author: Falcon Dai
Date: 08/17/2011
"""
#using pySerial library
import serial

import time


#scan for devices connected through serial ports
#returns a list of tuples of the port number and the name of the device
def scan():
    
    port = []
    for i in range(256):
        try:
            s = serial.Serial(i)
            port.append( (i, s.name) )
            s.close()
        except serial.SerialException:
            pass
    return port


if __name__ == '__main__':
    print scan()
    numPort = int(raw_input('Enter the port number: '))
    try:
        ser = serial.Serial(numPort)
    except serial.SerialException:
        print 'Serial object cannot be created on port '+str(numPort)+'.\n'
        exit()
        
    time.sleep(0.1)
    print ser.readline()
    
    #create data files to record data
    fout = open('data/data.txt', 'w')
    fout.write('LO freq=6GHz, Amp=1dBm\n')
    
    for i in range(100):
        ser.write('READ')
        time.sleep(0.1)
        fout.write(ser.readline())
    
    ser.close()
    fout.close()
    print 'END'