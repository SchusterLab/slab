# Authored by Meg, Oct. 28, 2019
# Just some switching code to check that the switches are behaving properly

import h5py
import numpy as np
from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
import time
from slab.instruments import *
from slab.instruments import InstrumentManager
path = os.getcwd()
expt_path = os.getcwd() + '\data'
import serial
import socket
import logging
import sys
import time


class switchcontroller():
    def __init__(self):
        # switches gives relation switch no i <-> COM no j
        self.switches = ["", "", "", "", "", "", ""]
        self.serial = []  # Holds all the switches
        #    self.corresp = {0:0, 1:5, 2:6, 3:1, 4:2, 5:3} # Holds the correspondence between which switch goes to which channel on the 6-way
        #    self.corresp = {0:0, 6:1, 1:2, 2:3, 3:4, 4:5,5:1} #Clai dictionary 12/15/16
        self.corresp = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}  # Brendan dictionary 12/16/16
        # self.corresp = {0:0,1:6,2:1,3:2,4:4,5:4,6:5}#dictionary for retaking data 1/11/17

    # makes connection between channel and common out on switch specified
    def on(self, switchno, channelno):
        self.serial[switchno].write(('on %sQ' % channelno).encode())

    # breaks connection between channel and common out on switch specified
    def off(self, switchno, channelno):
        self.serial[switchno].write(('off %sQ' % channelno).encode())

    # includes 6-way connection on top
    def onfull(self, switchno, channelno):
        self.serial[0].write(('on %sQ' % self.corresp[switchno]).encode())
        time.sleep(.01)
        self.on(switchno, channelno)
        time.sleep(.01)

    # includes 6-way connection on top
    def offfull(self, switchno, channelno):
        self.serial[0].write(('off %sQ' % self.corresp[switchno]).encode())
        time.sleep(.01)
        self.off(switchno, channelno)
        time.sleep(.01)

    # turn everything off
    def reset(self):
        for i in range(0, 7):
            if (self.switches[i] == ""):
                continue
            for j in range(1, 11):
                time.sleep(.12)
                self.off(i, j)

    def sense(self, switchno, channelno):
        self.serial[switchno].write(('sense %sQ' % channelno).encode())
        out = self.serial[switchno].readline().decode()
        return out

    # Housekeeping, after running setup the list serialarray contains serial communications between computer and switches 0-5, with 6-way being 0.z
    def setup(self):
        # Populate switches list with corresponding COM ports
        for j in range(5, 7):

            ser = serial.Serial('COM%s' % j, 9600, timeout=1)
            time.sleep(1)
            ser.write('whoQ'.encode())
            response = ser.readline()
            print("COM" + str(j) + " has switch " + response.decode())
            # print("")
            print(response)
            print(chr(response[0]))
            ser.close()
            try:
                #              print(chr(response[0]))
                self.switches[int(chr(response[0]))] = 'COM%d' % j
            #                     print(self.switches)
            except(IndexError, ValueError):
                print('ERROR')

        # Open serial communication with switches, populate serialarray with serial communications
        self.serial = [None, None, None, None, None, None, None]
        for i in range(1, 3):
            try:
                print(self.switches[i])
                self.serial[i] = serial.Serial(self.switches[i], 9600)
            except serial.SerialException:
                print("Switch {} not loaded!".format(i))


if __name__ == '__main__':
    a=switchcontroller()
    a.setup()
    # a.reset()
    # i = 6
    # a.on(1, i)
    # a.off(1,i)



    # a.off(1,i)
    # time.sleep(1)
    # print(a.sense(1, i))
    #
    # time.sleep(1)
    # print("moving to 2")
    #
    # print(a.sense(2,i))
    # a.on(2,i)
    # time.sleep(2)
    # print(a.sense(2,i))
    # a.off(2,i)
    # time.sleep(2)
    # print(a.sense(2, i))




    # print("1,10", a.sense(1,10))
    #
    # for i in range (1,11):
    #     print("sw1 channel", i, a.sense(1,i))
    #     time.sleep(1)
    #     a.on(1, i)
    #     time.sleep(1)
    #     print("sw1 channel", i, a.sense(1,i))
    #     time.sleep(1)
    #     a.off(1, i)
    #     time.sleep(1)
    #     print("sw1 channel", i, a.sense(1,i))
    #     time.sleep(1)
        # print("sw2 channel", i, a.sense(2,i))
        # time.sleep(1)
        # a.on(2, i)
        # time.sleep(1)
        # print("sw2 channel", i, a.sense(2, i))
        # time.sleep(1)
        # a.off(2, i)
        # time.sleep(1)
        # print("sw2 channel", i, a.sense(2, i))
        # time.sleep(1)
