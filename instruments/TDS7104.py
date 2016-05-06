__author__ = 'slab'

# -*- coding: utf-8 -*-

# NOT WORKING #
"""
:Tektronix TDS 7104 Digital Oscilloscope (TDS7104.py)
=====================================================
:Author: Aman LaChapelle
"""
from slab.instruments import VisaInstrument
import numpy as np
from time import strftime
import os
import matplotlib.pyplot as plt
import visa

class TekTDS7104(VisaInstrument):

    def __init__(self, name = "TekTDS7104", address = "TCPIP0::MSPBMTEK"):
        VisaInstrument.__init__(self, name=name, address = address)
        try:
            rm = visa.ResourceManager()

            self.scope = rm.open_resource(address)

            reply = self.scope.query("*IDN?")

            rpl = reply.split(",")

            if rpl[0] == "TEKTRONIX" and rpl[1] == "TDS7104":
                #print "Loaded %s %s" % (rpl[0], rpl[1])
                pass
            else:
                print "Error, loaded wrong instrument?"
                print reply

        except Exception:
            print "Instrument not connected"
            print "Make sure utilites->LAN server status->VXI-11 Server is started"
            print "Otherwise check NI-MAX"


    def _preamble(self, channel = 3):
        # Select source here
        self.scope.write("DATa:SOUrce CH%d" % channel)
        # Select data format ASCII
        self.scope.write("DATa:ENCdg ASCII")

        #need something for fast acquire


        # Get the preamble from the scope
        pre = self.scope.query_ascii_values("WFMPRe?", converter='s')
        #print pre ######################

        # Get essential values that are already split out.  -2 in num_pts because we just cut the first and last points
        y_scale = pre[2]
        x_scale = pre[3]
        self.num_pts = int(pre[5].split(";")[1].split(" ")[1]) - 2

       # print self.num_pts  ###############################

        # Need to split up the last value in the preamble to parse out the zero, units, multipliers, etc.
        temp = pre[-1].split(";")

        vals = {}

        for item in temp[3:-1]:
            lhs, rhs = item.split(" ")
            try:
                vals[lhs] = float(rhs)
            except ValueError:
                vals[lhs] = str(rhs)

        self.preamble = vals

        # Now need to get the ending off the y_scale and x_scale
        i = 0
        for char in list(y_scale)[1:]:
            if char == " " or char == ".":
                i += 1
            else:
                try:
                    int(char)
                    i+=1
                except Exception:
                    break

        j = 0
        for char in list(x_scale)[1:]:
            if char == " " or char == ".":
                j += 1
            else:
                try:
                    int(char)
                    j+=1
                except Exception:
                    break

        y_scale_n = float(y_scale[1:i+1])
        x_scale_n = float(x_scale[1:j+1])
        y_scale_u = y_scale[i+1:]
        x_scale_u = x_scale[j+1:]

        # Now store the values in a dictionary so we can get to them later
        # Use units to figure out which is y and which is x
        scale = {}
        scale[y_scale_u] = y_scale_n
        scale[x_scale_u] = x_scale_n

        self.scale = scale

    def _get_raw_data(self):
        # Data values come as strings, need to convert.  First and last are not useful so we discard.
        self.raw = np.array(self.scope.query_ascii_values("CURVe?", converter='s')[1:-1], dtype = np.float64)

    def _fmt_data(self, channel = 3):
        self._preamble(channel = channel)
        self._get_raw_data()

        #print len(self.raw)   #######################

        xdata = []
        # xdata[n] = XZERO + XINcr (n - PT_OFF)
        for i in range(0, self.num_pts):
            xdata.append(self.preamble["XZERO"] + self.preamble["XINCR"] * (i - self.preamble["PT_OFF"]))

        # Remove reference to i just in case
        i = None

        ydata = []
        # ydata[n] = YZERO + YMULT (y[n] - YOFF)
        for i in range(0, self.num_pts):
            ydata.append(self.preamble["YZERO"] + self.preamble["YMULT"] * (self.raw[i] - self.preamble["YOFF"]))

        self.ydata = ydata
        self.xdata = xdata

    def plot_data(self, channel = 3, title = "TDS7104 Trace"):
        self._fmt_data(channel = channel)

        plt.plot(self.xdata, self.ydata)
        plt.xlabel(self.preamble["XUNIT"])
        plt.ylabel(self.preamble["YUNIT"])
        plt.title(title)
        plt.savefig("C:\\Users\\slab\\desktop\\20160203_test\\fast_trace.png")
        plt.show()

    def save_trace(self, name, dirname, channel = 3, output = "C:\\Users\\slab\\Dropbox\\data"):

        # Make or change to the directory that we want to save data to
        try:
            os.chdir(r'%s\\%s_%s' % (output, strftime('%Y%m%d'), dirname))
        except Exception:
            os.mkdir(r'%s\\%s_%s' % (output, strftime('%Y%m%d'), dirname))

        expt = Experiment(r'%s\\%s_%s' % (output, strftime('%Y%m%d'), dirname), prefix = name)

        self._fmt_data(channel = channel)

        with expt.datafile() as f:
            f.append_line("Time %s" % (self.preamble["XUNIT"]), self.xdata)
            f.append_line("Amplitude %s" % (self.preamble["YUNIT"]), self.ydata)
            f.append_pt("%s" % self.scale.keys()[0], self.scale[self.scale.keys()[0]])
            f.append_pt("%s" % self.scale.keys()[1], self.scale[self.scale.keys()[1]])
            f.close()


if __name__ == '__main__':
    tek = TekTDS7104(address='TCPIP0::MSPBMTEK')
