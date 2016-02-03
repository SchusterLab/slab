import getpass
import sys
import telnetlib
import time
from time import strftime
from slab import *
import os
import winsound
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
from numpy import *

class PPMS:

    def __init__(self):
        # Connect to the PPMS
        HOST = "ppms.qtl.uchicago.edu"
        PORT = 5000
        self.tn = telnetlib.Telnet(HOST, PORT)
        # Remember to open up the server on the PPMS computer.
        # This will print "Connected to QDinstrument socket server" if connection is successful. Do not remove.
        print self.tn.read_some()


    def get_temp(self, verbose=False):

        self.tn.write("TEMP?\r\n")
        response = self.tn.read_some().strip().split(", ")
        if response[0]=="0": # call successful
            text = "Temp. = " + response[1] + " K, Status = " + response[2]
            if verbose:
                print text
            return [float(response[1]), int(response[2])]
        else:
            return -1

    def set_temp(self, setpoint, rate=20, mode=0):

        #modes:
        #0: Fast settle
        #1: No overshoot
        command = "TEMP " + str(setpoint) + ", " + str(rate) + ", " + str(mode) + "\r\n"
        print "Setting PPMS temperature:"
        print command.strip()
        self.tn.write(command)
        response = self.tn.read_some().strip()

        if response=="1": # call successful
            time.sleep(1)
            while True:
                try:
                    temp, temp_status = self.get_temp()
                    if temp_status == 1: # stable
                        print "Temperature settled."
                        break
                    else:
                        time.sleep(10)
                except Exception:
                    pass
            return 0
        else:
            print "Call failed"
            return -1

    def get_field(self):

        #field status:
        # 1: persistent
        # 4: holding in driven mode

        self.tn.write("FIELD?\r\n")
        response = self.tn.read_some().strip().split(", ")
        if response[0]=="0": # call successful
            text = "Field = " + response[1] + " Oe, Status = " + response[2]
            print text
            return [float(response[1]), int(response[2])]
        else:
            return -1

    def set_field(self, setpoint, rate=250, approach_mode=0, mode=1):

        #approach:
        #0: Linear
        #1: No overshoot
        #2: Oscillate

        #mode:
        #0: Persistent
        #1: Driven

        command = "FIELD " + str(setpoint) + ", " + str(rate) + ", " + str(approach_mode) + ", " + str(mode) + "\r\n"
        print "Setting PPMS field:"
        print command.strip()
        self.tn.write(command)
        response = self.tn.read_some().strip()

        if response=="1": # call successful
            time.sleep(1)
            while True:
                try:
                    field, field_status = self.get_field()
                    if field_status == 1 or field_status == 4: # stable
                        print "Field settled."
                        break
                    else:
                        time.sleep(2)
                except Exception:
                    pass
            return 0
        else:
            print "Call failed"
            return -1


if __name__ == "__main__":
    # connect to PPMS
    HOST = "ppms.qtl.uchicago.edu"
    PORT = 5000
    tn = telnetlib.Telnet(HOST, PORT)
    # This will print "Connected to QDinstrument socket server" if connection is successful. Do not remove.
    print tn.read_some()

    # Test
    get_temp()
    time.sleep(0.1)
    #print tn.read_some()
    #tn.write("CLOSE\r\n")
    #print tn.read_all()
