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

class E5071(SocketInstrument):

    MAXSWEEPPTS=1601    
    
    def __init__(self,name="E5071",address='nwa.circuitqed.com:5025',enabled=True):
        if ':' not in address: address+=':5025'
        SocketInstrument.__init__(self,name,address,enabled=enabled,timeout=10,recv_length=2**20)
        self.query_sleep=0.05

    def get_id(self):
        return self.query('*IDN?')

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
        na.clear_averages()
        na.trigger_single()
        time.sleep(na.get_query_sleep())
        na.averaging_complete()
        #na.set_format('slog')
        if fname is not None:
            na.save_file(fname)
        ans=na.read_data()
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

    def set_default_state(self):
        self.set_sweep_points()
        self.set_format()
        self.set_trigger_source()
        self.set_trigger_average_mode(False)
        self.write(":INIT1:CONT ON")



def condense_nwa_files(datapath,prefix):
    prefixes,data=load_nwa_dir(datapath)
    np.save(prefix,np.array(data))


def load_nwa_file(filename):
    """return three arrays: frequency magnitude and phase"""
    return np.transpose(np.loadtxt(filename,skiprows=3,delimiter=','))

def load_nwa_dir(datapath):
    fnames=glob.glob(os.path.join(datapath,"*.CSV"))
    fnames.sort()
    prefixes = [os.path.split(fname)[-1] for fname in fnames]
    data = [load_nwa_file(fname) for fname in fnames]
    return prefixes, data


def nwa_watch_temperature_sweep(na,fridge,datapath,fileprefix,windows,powers,ifbws,avgs,timeout=10000,delay=0):
    """nwa_watch_temperature_sweep monitors the temperature (via fridge) and tells the network analyzer (na) to watch certain windows
    at the specified powers, ifbws, and avgs specified
    windows= [(center1,span1),(center2,span2), ...]
    powers= [power1,power2,...]
    ifbws=[ifbw1,ifbw2,...]
    avgs=[avgs1,avgs2,...]"""
    f=open(datapath+fileprefix+".cfg",'w')
    f.write('datapath: %s\nfileprefix: %s\n\Window #\tCenter\tSpan\tPower\tIFBW\tavgs' % (datapath,fileprefix))
    for ii,w in enumerate(windows):
        f.write('%d\t%f\t%f\t%f\t%f\t%d' % (windows[ii][0],windows[ii][1],powers[ii],ifbws[ii],avgs[ii]))
    f.close()
    na.set_trigger_average_mode(True)
    na.set_sweep_points()
    na.set_average_state(True)
    na.set_timeout(timeout)
    na.set_format('slog')
    na.set_trigger_source('BUS')
    count=0
    while (True):
        for ii,w in enumerate(windows):
            Temperature=fridge.get_temperature('MC RuO2')
            if not Temperature >0:
                Temperature=fridge.get_temperature('MC cernox')
            print "Trace: %d\t\tWindow: %d\tTemperature: %3.3f" % (count,ii,Temperature)
            na.set_center_frequency(w[0])
            na.set_span(w[1])
            na.set_power(powers[ii])
            na.set_ifbw(ifbws[ii])
            na.set_averages(avgs)
            na.trigger_single()
            na.averaging_complete()    #Blocks!
            na.save_file("%s%04d-%3d-%s-%3.3f.csv" % (datapath,count,ii,fileprefix,Temperature))
            time.sleep(delay)


def nwa_test1(na):
    """Read NWA data and plot results"""
    fpts,mags,phases=na.read_data()

    figure(1)
    subplot(2,1,1)
    xlabel("Frequency (GHz)")
    ylabel("Transmission, S21 (dB)")
    plot(fpts/1e9,mags)
    subplot(2,1,2)
    xlabel("Frequency (GHz)")
    ylabel("Transmitted Phase (deg)")
    plot(fpts/1e9,phases)
    show()

def nwa_test2(na):
    """Test segmented Sweep"""
    na.set_default_state()
    na.set_power(-20)
    na.set_ifbw(1e3)
    na.set_averages(1)

    freqs,mags,phases=na.segmented_sweep(9e9,15e9,50e3)

    figure(1)
    subplot(2,1,1)
    xlabel("Frequency (GHz)")
    ylabel("Transmission, S21 (dB)")
    plot(freqs/1e9,mags)
    subplot(2,1,2)
    xlabel("Frequency (GHz)")
    ylabel("Transmitted Phase (deg)")
    plot(freqs/1e9,phases)
    show()

def nwa_test3(na):
    na.set_trigger_average_mode(False)
    na.set_power(-20)
    na.set_ifbw(1e3)
    na.set_sweep_points()
    na.set_averages(10)
    na.set_average_state(True)

    print na.get_settings()
    na.clear_averages()
    na.take_one_averaged_trace("test.csv")
    
def convert_nwa_files_to_hdf(nwa_file_dir, h5file, sweep_min, sweep_max, sweep_label="B Field", ext=".CSV"):
    import glob, h5py, csv
    hfile = h5py.File(h5file)
    files = glob.glob(nwa_file_dir + "*" + ext)
    n_files = len(files)
    for j, fn in enumerate(files):
        f = open(fn, 'r')
        # Skip Header
        for i in range(3):
            f.readline()
        rows = np.array((csv.reader(f)))
        if j is 0:
            n_points = len(rows)
            for t in ["mag", "phase"]:
                hfile[t] = np.zeros((n_points, n_files))
                hfile[t].attrs["_axes"] = ((rows[0][0], rows[-1][0]), (sweep_min, sweep_max))
                hfile[t].attrs["_axes_labels"] = ("Frequency", sweep_label, "S21")
        hfile["mag"][:,j] = rows[:,1]
        hfile["phase"][:,j] = rows[:,2]
    


if __name__ =='__main__':
#    condense_nwa_files(r'C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data','C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\test')
    na=E5071("E0571",address="192.168.14.130")
    print na.get_id()
    #print "Setting window"

    #from guiqwt.pyplot import *
    #nwa_test2(na)
#    nwa_test2(na)
    #nwa_test2(na)


#    nwa_test3(na)
