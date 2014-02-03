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
    def set_start_frequency(self,freq,channel=1):
        self.write(":SENS%d:FREQ:START %f" % (channel,freq))

    def get_start_frequency(self,channel=1):
        return float(self.query(":SENS%d:FREQ:START?" % channel))

    def set_stop_frequency(self,freq,channel=1):
        self.write(":SENS%d:FREQ:STOP %f" % (channel,freq))

    def get_stop_frequency(self,channel=1):
        return float(self.query(":SENS%d:FREQ:STOP?" % channel))

    def set_center_frequency(self,freq,channel=1):
        self.write(":SENS%d:FREQ:CENTer %f" % (channel,freq))

    def get_center_frequency(self,channel=1):
        return float(self.query(":SENS%d:FREQ:CENTer?" % channel))

    def set_span(self,span,channel=1):
        return self.write(":SENS%d:FREQ:SPAN %f" % (channel,span))

    def get_span(self,channel=1):
        return float(self.query(":SENS%d:FREQ:SPAN?" % channel))

    def set_sweep_points(self,numpts=1600,channel=1):
        self.write(":SENSe%d:SWEep:POINts %f" % (channel,numpts))

    def get_sweep_points(self,channel=1):
        return float(self.query(":SENSe%d:SWEep:POINts?" % (channel)))

#### Averaging
    def set_averages(self,averages,channel=1):
        self.write(":SENS%d:AVERage:COUNt %d" % (channel,averages))

    def get_averages(self,channel=1):
        return int(self.query(":SENS%d:average:count?" % channel))

    def set_average_state(self,state=True,channel=1):
        if state: s="ON"
        else: s="OFF"
        self.write(":SENS%d:AVERage:state %s" % (channel,s))

    def get_average_state(self,channel=1):
        return bool(self.query(":SENS%d:average:state?" % channel))

    def clear_averages(self,channel=1):
        self.write(":SENS%d:average:clear" % channel)

    def set_ifbw(self,bw,channel=1):
        self.write(":SENS%d:BANDwidth:RESolution %f" %(channel,bw))

    def get_ifbw(self,channel=1):
        
        return float(self.query(":SENS%d:BANDwidth:RESolution?" %(channel)))

    def averaging_complete(self):
        #if self.query("*OPC?") == '+1\n': return True
        self.write("*OPC?")
        self.read()
#        else: return False

    def trigger_single (self):
        self.write(':TRIG:SING')

    def set_trigger_average_mode(self,state=True):
        if state: self.write(':TRIG:AVER ON')
        else: self.write(':TRIG:AVER OFF')

    def get_trigger_average_mode(self):
        return bool(self.query(':TRIG:AVER?'))

    def set_trigger_source (self,source="INTERNAL"):  #INTERNAL, MANUAL, EXTERNAL,BUS
        self.write(':TRIG:SEQ:SOURCE ' + source)

    def get_trigger_source (self):  #INTERNAL, MANUAL, EXTERNAL,BUS
        return self.query(':TRIG:SEQ:SOURCE?')


#### Source

    def set_power(self,power,channel=1):
        self.write(":SOURCE%d:POWER %f" % (channel,power))

    def get_power(self,channel=1):
        return float(self.query(":SOURCE%d:POWER?" % channel))

    def set_output(self,state=True):
        if state: self.write(":OUTPUT ON")
        else: self.write(":OUTPUT OFF")

    def get_output(self):
        return bool(self.query(":OUTPUT?"))
        
    def set_measure(self,mode='S21') :
        self.write(":CALC1:PAR1:DEF %s"  % (mode))

#### File Operations

    def save_file(self,fname):
        self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')

    def set_format(self,trace_format='MLOG',channel=1):
        """set_format: valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        self.write(":CALC:FORMAT "+trace_format)
    def get_format(self,channel=1):
        """set_format: valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        return self.query(":CALC:FORMAT?")

    def read_data(self,channel=1):
        """Read current NWA Data, return fpts,mags,phases"""
#        self.write(":CALC1:PAR1:SEL")
#        self.write(":INIT1:CONT OFF")
#        self.write(":ABOR")
        self.write(":FORM:DATA ASC")
        self.write(":CALC1:DATA:FDAT?")
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

    def take_one_averaged_trace(self,fname=None):
        """Setup Network Analyzer to take a single averaged trace and grab data, either saving it to fname or returning it"""
        #print "Acquiring single trace"
        self.set_trigger_source('BUS')
        time.sleep(self.query_sleep*2)
        old_timeout=self.get_timeout()
#        old_format=self.get_format()
        self.set_timeout(10000)
        self.set_format()
        time.sleep(self.query_sleep)
        old_avg_mode=self.get_trigger_average_mode()
        self.set_trigger_average_mode(True)
        self.clear_averages()
        self.trigger_single()
        time.sleep(self.query_sleep)
        self.averaging_complete()    #Blocks!
        self.set_format('slog')
        if fname is not None:
            self.save_file(fname)
        ans=self.read_data()
        time.sleep(self.query_sleep)
#       self.set_format(old_format)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')
        self.set_format()
        return ans

    def segmented_sweep(self,start,stop,step):
        """Take a segmented sweep to achieve higher resolution"""
        span=stop-start
        total_sweep_pts=span/step
        if total_sweep_pts<=1601:
            print "Segmented sweep unnecessary"
        segments=np.ceil(total_sweep_pts/1600.)
        segspan=span/segments
        starts=start+segspan*np.arange(0,segments)
        stops=starts+segspan

        print span
        print segments
        print segspan

        #Set Save old settings and set up for automated data taking
        time.sleep(self.query_sleep)
        old_format=self.get_format()
        old_timeout=self.get_timeout()
        old_avg_mode=self.get_trigger_average_mode()

        self.set_timeout(10000)
        self.set_trigger_average_mode(True)
        self.set_trigger_source('BUS')
        self.set_format('slog')

        self.set_span(segspan)
        segs=[]
        for start,stop in zip(starts,stops):
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)

            self.clear_averages()
            self.trigger_single()
            time.sleep(self.query_sleep)
            self.averaging_complete()    #Blocks!

            seg_data=self.read_data()

            seg_data=seg_data.transpose()
            last=seg_data[-1]
            seg_data=seg_data[:-1].transpose()
            segs.append(seg_data)
        segs.append(np.array([last]).transpose())
        time.sleep(self.query_sleep)
        self.set_format(old_format)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')

        return np.hstack(segs)
        
    def segmented_sweep2(self,start,stop,step,sweep_pts=None,fname=None,save_segments=True):
        """Take a segmented sweep to achieve higher resolution"""
        span=stop-start
        total_sweep_pts=span/step
        if total_sweep_pts<1600:
            print "Segmented sweep unnecessary"
            self.set_sweep_points(max(sweep_pts,total_sweep_pts))
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)
            return self.take_one_averaged_trace(fname)
        segments=np.ceil(total_sweep_pts/1600.)
        segspan=span/segments
        starts=start+segspan*np.arange(0,segments)
        stops=starts+segspan

#        print span
#        print segments
#        print segspan

        #Set Save old settings and set up for automated data taking
        time.sleep(self.query_sleep)
        old_format=self.get_format()
        old_timeout=self.get_timeout()
        old_avg_mode=self.get_trigger_average_mode()

        self.set_timeout(10000)
        self.set_trigger_average_mode(True)
        self.set_trigger_source('BUS')
        self.set_format('mlog')

        self.set_span(segspan)
        segs=[]
        for start,stop in zip(starts,stops):
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)

            self.clear_averages()
            self.trigger_single()
            time.sleep(self.query_sleep)
            self.averaging_complete()    #Blocks!
            self.set_format('slog')
            seg_data=self.read_data()
            self.set_format('mlog')
            seg_data=seg_data.transpose()
            last=seg_data[-1]
            seg_data=seg_data[:-1].transpose()
            segs.append(seg_data)
            if (fname is not None) and save_segments:
                np.savetxt(fname,np.transpose(segs),delimiter=',')
        segs.append(np.array([last]).transpose())
        time.sleep(self.query_sleep)
        self.set_format(old_format)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')
        ans=np.hstack(segs)
        if fname is not None:
            np.savetxt(fname,np.transpose(ans),delimiter=',')
        return ans
        
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
        if fname is not None:
            self.save_file(fname)
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
        
    def configure(self,start=None,stop=None,center=None,span=None,power=None,ifbw=None,sweep_pts=None,avgs=None,defaults=False,remote=False):
        if defaults:       self.set_default_state()
        if remote:                          self.set_remote_state()
        if start is not None:            self.set_start_frequency(start)
        if stop is not None:            self.set_stop_frequency(stop)
        if center is not None:          self.set_center_frequency(center)
        if span is not None:            self.set_span(span)
        if power is not None:         self.set_power(power)
        if ifbw is not None:            self.set_ifbw(ifbw)
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
        self.write(":INIT1:CONT ON")






if __name__ =='__main__':
#    condense_nwa_files(r'C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data','C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\test')
    sa=E4440("E4440",address="192.168.14.151")
    print sa.get_id()
    #print "Setting window"

    #from guiqwt.pyplot import *
    #nwa_test2(na)
#    nwa_test2(na)
    #nwa_test2(na)


#    nwa_test3(na)
