#DAC Serial interface
#Brendan Saxberg

import serial
import socket
import time
import logging
import sys
import time
import io
import json

def numtobin(num):
    temp = int ((num)*(2**18)/(10) + (2**18/2))
    if temp < 0 or temp >= (2**18):
        #print("INVALID VOLTAGE RANGE")
        raise Exception("INVALID VOLTAGE RANGE")
    else:
        return int ((num)*(2**18)/(10) + (2**18/2))

def json_log(filename, V_vec):
    t_obj = time.localtime()
    t = time.asctime(t_obj)
    with open(filename, 'r+') as f:
        data = json.load(f)
        data[t] = V_vec
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

class AD5780_serial():
    def __init__(self):
        ard = serial.Serial('COM3', 9600, timeout=1000)
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

        if voltage < 4.98 and voltage >= -4.98:

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

    def parallelramp(self, voltagearray, stepsize, steptime, filename="dac_parallelramp_log.json"):

        dummy = self.ard.read_all()
        bvoltagearray = [0,0,0,0,0,0,0,0]
        for i in range(8):
            bvoltagearray[i] = numtobin(voltagearray[i])
        combinedstring = 'PARALLELRAMP' + ' ' + str(bvoltagearray[0]) + ' ' + str(bvoltagearray[1]) + ' ' + str(bvoltagearray[2]) + ' ' + str(bvoltagearray[3]) + ' ' + str(bvoltagearray[4]) + ' ' + str(bvoltagearray[5]) + ' ' + str(bvoltagearray[6]) + ' ' + str(bvoltagearray[7]) +  ' ' + str(stepsize) + ' ' + str(
            steptime) + ' \r\n'
        self.ard.write(combinedstring.encode())
        time.sleep(0.05)
        self.ard.readline()
        time.sleep(0.05)
        self.ard.readline()
        time.sleep(0.05)
        self.ard.readline()
        print("DAC ","ramped to", str(voltagearray), "V")
        # import os
        # print(os.getcwd())
        try:
            json_log(filename, voltagearray)
        except:
            print("Could not write DAC array to json file")

    # def read(self,dacnum):
    #     combinedstring = 'READ %s'%(dacnum)
    #     self.ard.write(combinedstring.encode())
    #     time.sleep(0.5)
    #     return
if __name__ == '__main__':

    a=AD5780_serial()
    print(a.ard.is_open)
    a.init()
    time.sleep(1)

    #for i in range(1,9):
    #i=7
    #a.init
    a.ramp3(dacnum=1,voltage=2,step = 1,speed = 1)
    #a.ramp3(dacnum=2, voltage= .2, step=4, speed=5)
    time.sleep(1)


