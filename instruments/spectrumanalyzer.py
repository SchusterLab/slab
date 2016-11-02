# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 21:26:31 2011

@author: Dave
"""
from slab.instruments import SocketInstrument
import time
import numpy as np
import glob
import os.path


class E4440(SocketInstrument):
    MAXSWEEPPTS = 1601
    default_port = 5025

    def __init__(self, name="E4440", address=None, enabled=True):
        SocketInstrument.__init__(self, name, address, enabled=enabled, timeout=1.0, recv_length=2 ** 20)
        self.query_sleep = 0.05

    def get_id(self):
        return self.query('*IDN?')

    def get_query_sleep(self):
        return self.query_sleep

    def calibrate(self):
        """
        Performs a full alignment
        :return:
        """
        print "Performing full alignment..."
        self.write(":CAL:ALL")

    #### Frequency setup
    def set_start_frequency(self, freq):
        self.write(":SENS:FREQ:START %f" % (freq))

    def get_start_frequency(self):
        return float(self.query(":SENS:FREQ:START?"))

    def set_stop_frequency(self, freq):
        self.write(":SENS:FREQ:STOP %f" % (freq))

    def get_stop_frequency(self):
        return float(self.query(":SENS:FREQ:STOP?"))

    def set_center_frequency(self, freq):
        self.write(":SENS:FREQ:CENTer %f" % (freq))

    def get_center_frequency(self):
        return float(self.query(":SENS:FREQ:CENTer?"))

    def set_span(self, span):
        return self.write(":SENS:FREQ:SPAN %f" % (span))

    def get_span(self):
        return float(self.query(":SENS:FREQ:SPAN?"))

    def set_sweep_points(self, numpts=8192):
        self.write(":SENSe:SWEep:POINts %f" % (numpts))

    def get_sweep_points(self):
        return float(self.query(":SENSe:SWEep:POINts?"))

    #### Averaging
    def set_averages(self, averages):
        self.write(":SENS:AVERage:COUNt %d" % (averages))

    def get_averages(self):
        return int(self.query(":SENS:AVERage:COUNt?"))

    def set_average_state(self, state=True):
        if state:
            s = "ON"
        else:
            s = "OFF"
        self.write(":SENS:AVERage:STATe %s" % (s))

    def get_average_state(self):
        return bool(self.query(":SENS:AVERage:STATe?")[0])

    def clear_averages(self):
        self.write(":SENS:average:clear")

    def set_resbw(self, bw, auto=None):
        if auto is not None:
            if auto:
                self.write(":SENS:BANDwidth:RESolution:AUTO ON")
            else:
                self.write(":SENS:BANDwidth:RESolution:AUTO OFF")
        else:
            self.write(":SENS:BANDwidth:RESolution %f" % (bw))

    def get_resbw(self):

        return float(self.query(":SENS:BANDwidth:RESolution?"))

    def set_vidbw(self, bw, auto=None):
        if auto is not None:
            if auto:
                self.write(":SENS:BANDwidthVIDEO:AUTO ON")
            else:
                self.write(":SENS:BANDwidth:VIDEO:AUTO OFF")
        else:
            self.write(":SENS:BANDwidth:VIDEO %f" % (bw))

    def get_vidbw(self):

        return float(self.query(":SENS:BANDwidth:VIDEO?"))

    def wait_for_completion(self):
        """
        from the e4440 documentation:
        *OPC? query stops new commands from being processed until the current processing is complete.
        """
        return self.query("*OPC?")

    def restart_measurement(self):
        self.write(':INITiate:RESTart')

    def set_continuous_sweep_state(self, state=True):
        if state:
            stateString = 'ON'
        else:
            stateString = 'OFF'
        self.write(':INITiate:CONTinuous ' + stateString)

    def get_continuous_sweep_state(self, state=True):
        if state:
            stateString = 'ON'
        else:
            stateString = 'OFF'
        return bool(self.query(':INITiate:CONTinuous?' + stateString))

    def trigger_single(self):
        self.write(':INIT:IMM')

    def set_trigger_source(self, source="IMMEDIATE"):  #IMMEDIATE, LINE, EXTERNAL
        self.write('TRIG:SOURCE ' + source)

    def get_trigger_source(self):  #INTERNAL, MANUAL, EXTERNAL,BUS
        return self.query(':TRIGGER:SOURCE?')

        #### File Operations
        # def save_file(self,fname):
        #     self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')
        #
        # def set_format(self,trace_format='MLOG',channel=1):
        #     """set_format: valid options are
        #     {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        #     """
        #     self.write(":CALC:FORMAT "+trace_format)
        # def get_format(self,channel=1):
        #     """set_format: valid options are
        #     {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        #     """
        #     return self.query(":CALC:FORMAT?")

    def read_data(self, channel=1):
        """Read current NWA Data, return fpts,mags,phases"""
        #        self.write(":CALC1:PAR1:SEL")
        #        self.write(":INIT1:CONT OFF")
        #        self.write(":ABOR")
        self.write(":FORM:DATA ASC")
        #self.write(":CALC1:DATA:FDAT?")
        self.write(":TRAC:DATA? TRACE1")
        data_str = ''

        done = False
        ii = 0
        while not done:
            time.sleep(self.query_sleep)
            ii += 1
            try:
                s = self.read()
            except:
                print "read %d failed!" % ii
            data_str += s
            done = data_str[-1] == '\n'
        #print data_str
        data = np.fromstring(data_str, dtype=float, sep=',')
        # data=data.reshape((-1,2))
        data = data.transpose()
        #self.data=data
        fpts = np.linspace(self.get_start_frequency(), self.get_stop_frequency(), len(data))
        return np.vstack((fpts, data))

    #### Meta
    def take_one(self):
        #print "Acquiring single trace"
        #time.sleep(na.query_sleep*2)
        #na.set_format()
        #time.sleep(na.query_sleep)
        self.clear_averages()
        self.trigger_single()
        time.sleep(self.get_query_sleep())
        self.wait_for_completion()
        ans = self.read_data()
        return ans

    def get_settings(self):
        settings = {"start": self.get_start_frequency(), "stop": self.get_stop_frequency(),
                    "power": self.get_power(), "ifbw": self.get_ifbw(),
                    "sweep_points": self.get_sweep_points(),
                    "averaging": self.get_average_state(), "averages": self.get_averages()
        }
        return settings

    def configure(self, start=None, stop=None, center=None, span=None, resbw=None, vidbw=None, sweep_pts=None,
                  avgs=None, defaults=False, remote=False):
        if defaults:       self.set_default_state()
        if remote:                          self.set_remote_state()
        if start is not None:            self.set_start_frequency(start)
        if stop is not None:            self.set_stop_frequency(stop)
        if center is not None:          self.set_center_frequency(center)
        if span is not None:            self.set_span(span)
        if resbw is not None:         self.set_resbw(resbw)
        if vidbw is not None:            self.set_vidbw(vidbw)
        if sweep_pts is not None:   self.set_sweep_points(sweep_pts)
        if avgs is not None:            self.set_averages(avgs)

    def set_remote_state(self):
        self.set_trigger_source('EXT')
        self.set_timeout(10)

    def set_default_state(self):
        self.set_sweep_points()
        self.set_trigger_source()
        self.write(":INIT:CONT ON")

    def play_sound(self, freq=None, music=False):
        import winsound
        import time
        soundfile = "tone_files/1kHz_44100Hz_16bit_30sec.wav"
        winsound.PlaySound(soundfile, winsound.SND_FILENAME|winsound.SND_ASYNC)
        time.sleep(1.5)
        # play the system exit sound if set
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

if __name__ == '__main__':
    #    condense_nwa_files(r'C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data','C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\test')
    sa = E4440("E4440", address="192.168.14.152")
    print sa.get_id()
    print"Taking data"
    data = sa.take_one()
    sa.set_default_state()
    print "Finished"
    #print data
    from matplotlib.pyplot import *
    #data=sa.read_data()

    plot(data[0], data[1])
    show()


    #print "Setting window"

    #from guiqwt.pyplot import *
    #nwa_test2(na)
#    nwa_test2(na)
#nwa_test2(na)


#    nwa_test3(na)
