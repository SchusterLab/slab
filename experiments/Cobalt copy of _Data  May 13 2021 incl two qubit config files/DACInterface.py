#DAC Serial interface
#Brendan Saxberg

import serial
import socket
import time
import logging
import sys
import time
import io

def numtobin(num):
    temp = int ((num)*(2**18)/(20) + (2**18/2))
    if temp < 0 or temp >= (2**18):
        print("INVALID VOLTAGE RANGE")
    else:
        return int ((num)*(2**18)/(20) + (2**18/2))

class AD5780_serial():
    def __init__(self):
        ard = serial.Serial('COM7', 9600, timeout=1000)
        self.ard = ard

    #Sets the voltage of dac dacnum to voltage voltage
    def set(self,dacnum, voltage):
        binvolt = numtobin(voltage)#Convert from voltage -10 to 10 to 2^18 bit num
        combinedstring = 'SET' + ' ' + str(dacnum) + ' ' + str(binvolt) + ' \r\n'
        # print('Combined String: ' + str(combinedstring))
        self.ard.write(combinedstring.encode())

    #Initializes all of the dacs
    def init(self):
        self.ard.write(b'INIT \r\n')
        # combinedstring = 'INIT \r\n'
        # print('Combined String: ' + combinedstring)
        # self.ard.write(combinedstring.encode())

    #in volts
    def ramp(self,dacnum,voltage,step,speed):
        bvoltage = numtobin(voltage)
        bstep = numtobin(step)
        bspeed = numtobin(speed)
        combinedstring = 'RAMP' + ' ' + str(dacnum) + ' ' + str(bvoltage) + ' ' + str(bstep) + ' ' + str(bspeed) + ' \r\n'
        self.ard.write(combinedstring.encode())

    #step, speed in bits
    def ramp2(self, dacnum, voltage, step, speed):
        bvoltage = numtobin(voltage)
        combinedstring = 'RAMP' + ' ' + str(dacnum) + ' ' + str(bvoltage) + ' ' + str(step) + ' ' + str(
            speed) + ' \r\n'
        self.ard.write(combinedstring.encode())

    #string buffered ramp2
    def ramp3(self, dacnum, voltage, step, speed):

        if voltage < 10.0 and voltage >= -10.0:

            dummy = self.ard.read_all()
            bvoltage = numtobin(voltage)
            combinedstring = 'RAMP' + ' ' + str(dacnum) + ' ' + str(bvoltage) + ' ' + str(step) + ' ' + str(
                speed) + ' \r\n'
            self.ard.write(combinedstring.encode())
            time.sleep(0.05)
            self.ard.readline()
            time.sleep(0.05)
            self.ard.readline()
            time.sleep(0.05)
            self.ard.readline()
            print("DAC Ch",dacnum,"ramped to", voltage, "V")

        else:
            print("DAC Ch",dacnum," overlimit - setpoint", voltage, "V")

if __name__ == '__main__':

    a=AD5780_serial()
    print(a.ard.is_open)
    a.init()
    time.sleep(1)
    # print(a.ard.read(1000))
    # a.set(5,1)
    # print(a.ard.read(1000))

    # ~ 0.05 V per second
    # dummy = a.ard.read_all()
    # a.ramp2(dacnum=5, voltage=0.5, step=5, speed=5)
    # dummy = a.ard.readline()
    # dummy = a.ard.readline()
    # ramp_done = a.ard.readline()

    #previous command for ramp2
    # for i in range(1,9):
    #     dummy = a.ard.read_all()
    #     a.ramp2(dacnum=i, voltage=0.2, step=5, speed=5)
    #     dummy = a.ard.readline()
    #     dummy = a.ard.readline()
    #     print(a.ard.readline())


    #now try ramp3 which waits for ramp to finish properly
    #for i in range(1,9):
    #i=7
    #a.init
    a.set(dacnum=1,voltage=1)
    #a.ramp3(dacnum=2, voltage= .2, step=4, speed=5)
    time.sleep(1)


