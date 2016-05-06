# -*- coding: utf-8 -*-
"""
Agilent AWG81180A (awg.py)
==========================
:Author: David Schuster
"""

from numpy import *
from slab.instruments import *
import time

#for debug purposes
file_output_trace = True
debug_file = "C:\\Users\\Phil\\Desktop\\agilentdebug.txt"

class AWG81180A(SocketInstrument):
    default_port=5025
    def __init__(self,name,address='',enabled=True,timeout=10, recv_length=1024):
        
        
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
        self.set_couple()
        self.set_trigger(level=0.7)
        self.set_ext_ref()
        self.set_clockrate(4.2e9)
        for ch in (1,2):
            self.select_channel(ch)
            self.set_marker(1)
            self.set_marker(2)
              
    def get_id(self):
        return self.query("*IDN?\n")

    def select_channel(self,channel=1):    #Set Channel
        self.write(":INSTrument:SELect " + str(channel)+"\n")
        
    def get_mode(self):
        return self.query(":FUNCtion:MODE?")
        
    def set_mode(self,mode="USER"):        #Set Mode
        self.write(":FUNCtion:MODE " + mode+"\n")
        
    def get_clockrate(self):
        return float(self.query(":FREQuency:RASTer?"))

    def set_clockrate(self,clockrate=1e9):
        self.write( (":FREQuency:RASTer %E\n" % (clockrate)))
        
    def set_ext_ref(self,ext=True,frequency=10e6):
        """If ext=True sets clock reference to external with specified frequency"""
        if ext: 
            self.write('ROSC:SOURCE EXT')
            self.write('ROSC:FREQUENCY %d' % frequency)
        else: self.write('ROSC:SOURCE INT')
        
    def get_amplitude(self):
        return float(self.query(":VOLTage:LEVel:AMPLitude?"))
    
    def set_amplitude(self,amp):
        self.write((":VOLTage:LEVel:AMPLitude %f\n" % (amp)))
    
    def get_offset(self):
        return float(self.query(":VOLTage:LEVel:OFFSet?"))

    def set_offset(self,offset=0):
        self.write((":VOLTage:LEVel:OFFSet %f\n" % (offset)))

    def get_output(self):
        return bool(self.query(":OUTPut:STATe?"))

    def set_output(self,output=True):
        if output:
            self.write(":OUTPut:STATe ON\n")
        else:
            self.write(":OUTPut:STATe OFF\n")      

    def get_sync_output(self):
        return bool(self.query(":OUTPut:SYNC:STATe?"))

    def set_sync_output(self,syncoutput=True):
        if syncoutput:
            self.write(":OUTPut:SYNC:STATe ON\n")
        else:
            self.write(":OUTPut:SYNC:STATe OFF\n")

    def set_trigger(self,src='ext',level=1.0):
        """Valid options for src are CONT, EXT, and BUS"""
        if src.upper() is 'CONT':
            self.write(':INIT:CONT:STATE 1')
        else:
            self.write(':INIT:CONT:STATE 0')
            self.write(':TRIG:MODE NORM')
            self.write(':TRIG:SOURCE %s' % (src.upper()))
            if level != -1:
                self.write(':TRIG:LEV %g' % level)
            
    def select_trace(self,trace=1):
        self.write((":TRACe:SELect %d\n") % trace)
    
    def get_trace(self):
        return int(self.query(":TRACe:SELect?"))
        
    def delete_trace(self,trace):
        self.write(":TRACe:DELete %d\n" % trace)

    def delete_all_traces(self):
        self.write(":TRACe:DELete:ALL")
        
    def delete_sequence(self,sequence):
        self.write(":SEQuence:DELete %d\n" % trace)

    def delete_all_sequences(self):
        self.write(":SEQuence:DELete:ALL")  
        
    def delete_all(self):
        self.delete_all_traces()
        self.delete_all_sequences()
        
    def set_marker(self,marker=1,state=True,position=0,width=0,high=1.25,low=0):
        """Sets marker properties for current channel"""
        s=('MARKER%d:STATE %d;MARKER%d:POSITION %d;MARKER%d:WIDTH %d; MARKER%d:VOLTAGE:HIGH %f;MARKER%d:VOLTAGE:LOW %f' %
                   (marker,state,marker,position,marker,width,marker,high,marker,low))
        print s.split(';')
        for cmd in s.split(';'):
            self.write(cmd)
        
    def set_couple(self,state=True):
        self.write(':INSTRUMENT:COUPLE:STATE %d' % (state))
    
    def define_trace(self,trace_num,length):
        self.write(":TRACe:DEFine %d, %d" % (trace_num, length))   #define segment length

    def convert_float_to_int_data(self,floatdata, float_range=None):
        #print "Converting waveform to long"
#        plot(idata)
#        show()
#        print "idata max/min: %d /%d" %(max (idata),min(idata))
        if float_range is None: # Scale to full output range
            maxd = max(floatdata)
            mind = min(floatdata)
        else:
            mind, maxd = float_range
        if maxd==mind:
            idata=zeros(len(floatdata),dtype=long)
        else:
            idata=array(4095*((floatdata-mind)/(maxd-mind)),dtype=long)
        
        return idata

    #takes data from -1.0 to 1.0
    def add_floatsegment(self,data,marker1=None,marker2=None,segnum=1, float_range=None):
        self.add_intsegment(self.convert_float_to_int_data(data, float_range),marker1,marker2,segnum)

    def add_intsegment (self,data,marker1=None, marker2=None,segnum=1):
        
        
        
        if len(data) % 32 != 0:
            raise RuntimeError("AWG81180A segment length not divisible by 32")
        #print "Converting waveform to bytecode"
        self.define_trace(segnum,len(data))
        self.select_trace(segnum)
        if marker1 is None: marker1=zeros(len(data),dtype=int)
        else:
            if len(marker1) != len(data)/4:
                raise RuntimeError("AWG81180A Markers must have 1/4 resolution of data")
            mr1=reshape(marker1,(len(marker1)/8,8))
            marker1=concatenate(hstack((mr1,mr1,mr1,mr1)))
        if marker2 is None: marker2=zeros(len(data),dtype=int)
        else:
            if len(marker2) != len(data)/4:
                raise RuntimeError("AWG81180A Markers must have 1/4 resolution of data")
            mr2=reshape(marker2,(len(marker2)/8,8))
            marker2=concatenate(hstack((mr2,mr2,mr2,mr2)))
        data+=2**12*marker1+2**13*marker2
        b=bytearray(data.__len__()*2)
        #for ii in range(len(data)/32):
        #    tmpdata = data[ii:(ii+32)]
        #    for jj in range(32):
        #        #interlace?
        #        data[jj+ii] = tmpdata[(jj%4)*8+floor(jj/4)]
        for ii in xrange(len(data)):
            b[ii*2] = (data[ii] & 0xFF)
            b[ii*2+1] = (data[ii] & 0xFF00)>>8
        
        #output the data to a file for debug purposes
        if file_output_trace:
            f = open(debug_file,'a')
            for i in range(len(data)):
                f.write(str(b[2*i]+((b[2*i+1] & 0x3F)<<8))+',')
            f.write('\n')
            f.close()
            
        print "Uploading trace %d to AWG" % segnum
        #print len(b)
        self.binblockwrite(":TRACe:DATA",b)

    #for testing purposes
    def add_intsegment2 (self,data,marker1=None, marker2=None,segnum=1):
            
        if len(data) % 32 != 0:
            raise RuntimeError("AWG81180A segment length not divisible by 32")
        
        
        if marker1 is None: marker1=zeros(len(data),dtype=int)
        else:
            
            mr1=reshape(marker1,(len(marker1)/8,8))
            marker1=concatenate(hstack((mr1,mr1,mr1,mr1)))
        if marker2 is None: marker2=zeros(len(data),dtype=int)
        else:
            
            mr2=reshape(marker2,(len(marker2)/8,8))
            marker2=concatenate(hstack((mr2,mr2,mr2,mr2)))
        
        data+=2**12*marker1+2**13*marker2
        b=bytearray(data.__len__()*2)
        #for ii in range(len(data)/32):
        #    tmpdata = data[ii:(ii+32)]
        #    for jj in range(32):
        #        #interlace?
        #        data[jj+ii] = tmpdata[(jj%4)*8+floor(jj/4)]
        for ii in xrange(len(data)):
            b[ii*2] = (data[ii] & 0xFF)
            b[ii*2+1] = (data[ii] & 0xFF00)>>8
        #print "Uploading trace %d to AWG" % segnum
        return b

    def binblockwrite(self,commandstring, blockarray):
        #self.write("*OPC?")
        #response = self.read()
        blockarraylength=str(blockarray.__len__())
        blockarraylengthposition=blockarraylength.__len__()
        cmd = commandstring+" #"+str(blockarraylengthposition)+blockarraylength
        #print "bbw cmd string: ", cmd
        #self.instrument.term_chars = ""
        self.write(str(bytearray(cmd)+blockarray))
        #self.instrument.term_chars = None
        #response= self.query("*OPC?")
        #print "BinBlockWrite Response: ", response
        #return response

    def select_sequence(self,seq_num=1):
        self.write(":SEQuence:SELect %d" % seq_num)
        
    def get_sequence(self):
        return int(self.query(":SEQuence:SELect?"))
    
    def define_sequence_step(self,step,seg_num,loops=1,jump_flag=False):
        if jump_flag: jump=1
        else: jump=0
        self.write(":SEQuence:DEFine %d, %d, %d, %d" % (step,seg_num,loops,jump))
        
    def define_sequence_advance(self,adv='STEP'):
        self.write(':SEQuence:ADVance %s' % (adv.upper()))

    def get_sequence_step(self):
        return self.query(":SEQuence:DEFine?")
        
    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['amplitude']=self.get_amplitude()
        settings['offset']=self.get_offset()
        settings['output']=self.get_output()
        settings['sync_output']=self.get_sync_output()
        settings['clockrate']=self.get_clockrate()
        settings['mode']=self.get_mode()
        return settings

    def reset(self, sequences_only=False):
        self.select_channel(1)
        self.delete_all_sequences()
        if not sequences_only:
            self.delete_all_traces()
        self.select_channel(2)
        self.delete_all_sequences()
        if not sequences_only:
            self.delete_all_traces()
            
        if file_output_trace:
            #clear the debug file
            f = open(debug_file,'w')
            f.close()
        
    def stop(self):
        self.presetup_for_sequences()
        
    def presetup_for_sequences(self):
        #define channel properties
        self.select_channel(1)
        self.set_output(False)
        self.select_channel(2)
        self.set_output(False)
        
        self.select_channel(1)
        self.set_mode("USER")    
        self.define_sequence_advance(adv='STEP')
        self.write(':SEQuence:JUMP BUS')
        
        self.select_channel(2)
        self.set_mode("USER")    
        self.define_sequence_advance(adv='STEP')
        self.write(':SEQuence:JUMP BUS')
        #put the AWG into an idle state to load pulse sequences
        self.set_trigger(src='BUS',level=-1) 
        
    def set_amps_offsets(self,channel_amps=[2.0,2.0],channel_offsets=[0.0,0.0],marker_amps=[1.0,1.0,1.0,1.0]):
        
        #define channel properties
        self.select_channel(1)
        self.set_amplitude(channel_amps[0])
        #awg.set_offset(1.0)
        self.set_offset(channel_offsets[0])
   
        self.select_channel(2)
        self.set_amplitude(channel_amps[1])
        self.set_offset(channel_offsets[1])
        
        #set marker's (to do)
        
    def prep_experiment(self):
        
        self.set_to_sequence()
       
        
    def set_to_sequence(self):
        #start awg 
        self.select_channel(1)
        self.set_mode("SEQ") 
        self.select_channel(2)
        self.set_mode("SEQ") 
        
    def run(self):
        
        self.select_channel(1)
        self.set_output(True)
        self.select_channel(2)
        self.set_output(True)
        self.set_trigger(src='EXT',level=-1)
        
    def set_clock_all(self,clockrate):
        self.set_clockrate(clockrate)
        
    def set_to_trace(self,ch1_trace=1,ch2_trace=2):
        self.select_channel(1)
        self.select_trace(ch1_trace) 
        self.select_channel(2)
        self.select_trace(ch2_trace) 
        
     

#class Tek5014 (SocketInstrument):
#    """Tektronix 5014 Arbitrary Waveform Class"""
#    def __init__(self,name,address='',port=4321,enabled=True,timeout=10, recv_length=1024):
#        
#        pass
    
      

# try:
#     from guiqwt.pyplot import *
# except:
#     print "Could not load guiqwt"
        
if __name__=="__main__":
#    a=mod(arange(32*2048),2048)
#    a=arange(1000*2048.)    
#    b=zeros(a.__len__())
#    teeth=100
#    for f in arange(0.001,.1,0.001):
#        b+=cos(2*pi*f*a)
#    plot(a,b)
#    show()

    print "Initializing Instrument"
    #awg=AWG81180A (name='awg',address="GPIB0::04::INSTR")
    awg=AWG81180A (name='awg',address="192.168.14.134:5025")
    print awg.get_id()
    print "Changing arb waveform"
    awg.set_output(False)
    awg.select_channel(1)
    awg.set_mode("USER")    
    awg.set_amplitude(2.0)
    awg.set_offset(1.0)
    awg.delete_all()
    awg.select_sequence(1)
    arr=[]
    print "Calculating waveform"
    for i in range(20):
        b=zeros(100000)
        for j in range(i*5000):
            b[j]=4095
        arr.append(b)
    print "Convert to integer waveform"
    idata=[awg.convert_float_to_int_data(seg) for seg in arr]    
    print "Uploading waveform"
    for ii,seg in enumerate(idata):
        awg.add_intsegment(seg,ii+1)
        awg.define_sequence_step(ii+1,ii+1)
    #figure(1)
    #imshow(array(arr))
        
    awg.set_output(True) 
    awg.set_mode("SEQUENCE")
     
    print "Finished uploading data."
    
        