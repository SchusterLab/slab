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

    MAXSWEEPPTS=1601    
    default_port=5025
    def __init__(self,name="E4440",address=None,enabled=True):
        SocketInstrument.__init__(self,name,address,enabled=enabled,timeout=10,recv_length=2**20)
        self.query_sleep=0.05

    def get_id(self):
        return self.query('*IDN?')
        
    def get_query_sleep(self):
        return self.query_sleep

#### Frequency setup
    def set_start_frequency(self,freq):
        self.write(":SENS:FREQ:START %f" % (freq))

    def get_start_frequency(self):
        return float(self.query(":SENS:FREQ:START?"))

    def set_stop_frequency(self,freq):
        self.write(":SENS:FREQ:STOP %f" % (freq))

    def get_stop_frequency(self):
        return float(self.query(":SENS%d:FREQ:STOP?" ))

    def set_center_frequency(self,freq):
        self.write(":SENS:FREQ:CENTer %f" % (freq))

    def get_center_frequency(self):
        return float(self.query(":SENS:FREQ:CENTer?"))

    def set_span(self,span):
        return self.write(":SENS:FREQ:SPAN %f" % (span))

    def get_span(self):
        return float(self.query(":SENS:FREQ:SPAN?" ))

    def set_sweep_points(self,numpts=8192):
        self.write(":SENSe:SWEep:POINts %f" % (numpts))

    def get_sweep_points(self):
        return float(self.query(":SENSe:SWEep:POINts?" ))

#### Averaging
    def set_averages(self,averages):
        self.write(":SEN:AVERage:COUNt %d" % (averages))

    def get_averages(self):
        return int(self.query(":SENS:average:count?" ))

    def set_average_state(self,state=True):
        if state: s="ON"
        else: s="OFF"
        self.write(":SENS:AVERage:state %s" % (s))

    def get_average_state(self):
        return bool(self.query(":SENS:average:state?"))

    def clear_averages(self):
        self.write(":SENS:average:clear")

    def set_resbw(self,bw,auto=None):
        if auto is not None:
            if auto:
                self.write(":SENS:BANDwidth:RESolution:AUTO ON")
            else:
                self.write(":SENS:BANDwidth:RESolution:AUTO OFF")
        else:
            self.write(":SENS:BANDwidth:RESolution %f" %(bw))

    def get_resbw(self):
        
        return float(self.query(":SENS:BANDwidth:RESolution?"))

    def set_vidbw(self,bw,auto=None):
        if auto is not None:
            if auto:
                self.write(":SENS:BANDwidthVIDEO:AUTO ON")
            else:
                self.write(":SENS:BANDwidth:VIDEO:AUTO OFF")
        else:
            self.write(":SENS:BANDwidth:VIDEO %f" %(bw))

    def get_vidbw(self):
        
        return float(self.query(":SENS:BANDwidth:VIDEO?"))

    def averaging_complete(self):
        #if self.query("*OPC?") == '+1\n': return True
        self.write("*OPC?")
        self.read()
#        else: return False


    def trigger_single (self):
        self.write(':INIT:IMM')

    def set_trigger_average_mode(self,state=True):
        if state: self.write(':TRIG:AVER ON')
        else: self.write(':TRIG:AVER OFF')

    def get_trigger_average_mode(self):
        return bool(self.query(':TRIG:AVER?'))

    def set_trigger_source (self,source="INTERNAL"):  #INTERNAL, MANUAL, EXTERNAL,BUS
        self.write(':TRIG:SEQ:SOURCE ' + source)

    def get_trigger_source (self):  #INTERNAL, MANUAL, EXTERNAL,BUS
        return self.query(':TRIG:SEQ:SOURCE?')


#### File Operations

#    def save_file(self,fname):
#        self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')
#
#    def set_format(self,trace_format='MLOG',channel=1):
#        """set_format: valid options are
#        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
#        """
#        self.write(":CALC:FORMAT "+trace_format)
#    def get_format(self,channel=1):
#        """set_format: valid options are
#        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
#        """
#        return self.query(":CALC:FORMAT?")

    def read_data(self,channel=1):
        """Read current NWA Data, return fpts,mags,phases"""
#        self.write(":CALC1:PAR1:SEL")
#        self.write(":INIT1:CONT OFF")
#        self.write(":ABOR")
        self.write(":FORM:DATA ASC")
        #self.write(":CALC1:DATA:FDAT?")
        self.write("TRACe:DATA?")
        data_str=''
        
        done=False
        ii=0
        while not done:
            time.sleep(self.query_sleep)
            ii+=1
            try:
                s=self.read()
            except:
                print "read %d failed!" % ii
            data_str+=s
            done = data_str[-1]=='\n'
        #print data_str
        data=np.fromstring(data_str,dtype=float,sep=',')
        data=data.reshape((-1,2))
        data=data.transpose()
        #self.data=data
        fpts=np.linspace(self.get_start_frequency(),self.get_stop_frequency(),len(data[0]))
        return np.vstack((fpts,data))

#### Meta

 
    def take_one(self,fname=None):
        """Tell Network Analyzer to take a single averaged trace and grab data, 
        either saving it to fname or returning it.  This function does not set up
        the format or anything else it just starts, blocks, and grabs the next trace."""
        #print "Acquiring single trace"
        #time.sleep(na.query_sleep*2)
        #na.set_format()
        #time.sleep(na.query_sleep)
        self.clear_averages()
        self.trigger_single()
        time.sleep(self.get_query_sleep())
        self.averaging_complete()
        #na.set_format('slog')
#        if fname is not None:
#            self.save_file(fname)
        ans=self.read_data()
        #time.sleep(na.query_sleep)
        #na.set_format()
        return ans

    def get_settings(self):
        settings={"start": self.get_start_frequency(),"stop":self.get_stop_frequency(),
         "power":self.get_power(),"ifbw":self.get_ifbw(),
         "sweep_points":self.get_sweep_points(),
         "averaging":self.get_average_state(),"averages":self.get_averages()
         }
        return settings
        
    def configure(self,start=None,stop=None,center=None,span=None,resbw=None,vidbw=None,sweep_pts=None,avgs=None,defaults=False,remote=False):
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
         self.set_trigger_source('BUS')
         self.set_trigger_average_mode(True)
         self.set_timeout(10000)  
         self.set_format('slog')
     
    def set_default_state(self):
        self.set_sweep_points()
        self.set_format()
        self.set_trigger_source()
        self.set_trigger_average_mode(False)
        self.write(":INIT:CONT ON")






if __name__ =='__main__':
#    condense_nwa_files(r'C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data','C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\test')
    sa=E4440("E4440",address="192.168.14.152")
    print sa.get_id()
    #print "Setting window"

    #from guiqwt.pyplot import *
    #nwa_test2(na)
#    nwa_test2(na)
    #nwa_test2(na)


#    nwa_test3(na)
