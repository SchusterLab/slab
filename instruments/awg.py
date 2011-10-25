# -*- coding: utf-8 -*-
"""
Created on Sun May 08 17:03:09 2011

@author: Phil
"""

from numpy import *
from slab.instruments import *
from guiqwt.pyplot import *
import time

class AWG81180A(VisaInstrument):
              
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

    def select_trace(self,trace=1):
        self.write((":TRACe:SELect %d\n") % trace)
    
    def get_trace(self):
        return int(self.query(":TRACe:SELect?"))
        
    def delete_trace(self,trace):
        self.write(":TRACe:DELete %d\n" % trace)

    def delete_all(self):
        self.write(":TRACe:DELete:ALL\n")
        
    def define_trace(self,trace_num,length):
        self.write(":TRACe:DEFine %d, %d" % (trace_num, length))   #define segment length

    #takes data from -1.0 to 1.0
    def add_floatsegment(self,data,segnum=1):
        #print "Converting waveform to long"
        maxd=max(data)
        mind=min(data)
        if maxd==mind:
            idata=zeros(data.__len__(),dtype=long)
        else:
            idata=array(4095*((data-mind)/(maxd-mind)),dtype=long)
#        plot(idata)
#        show()
#        print "idata max/min: %d /%d" %(max (idata),min(idata))
        self.add_intsegment(idata,segnum)

    def add_intsegment (self,data,segnum=1):
        if data.__len__() % 32 != 0:
            raise RuntimeError("AWG81180A segment length not divisible by 32")
        #print "Converting waveform to bytecode"
        self.define_trace(segnum,data.__len__())
        self.select_trace(segnum)
        b=bytearray(data.__len__()*2)
        for ii in xrange(data.__len__()):
            b[ii*2] = (data[ii] & 0xFF)
            b[ii*2+1] = (data[ii] & 0xFF00)>>8
        print "Uploading trace %d to AWG" % segnum
        self.binblockwrite(":TRACe:DATA",b)

    def binblockwrite(self,commandstring, blockarray):
        self.write("*OPC?")
        response = self.read()
        blockarraylength=str(blockarray.__len__())
        blockarraylengthposition=blockarraylength.__len__()
        cmd = commandstring+" #"+str(blockarraylengthposition)+blockarraylength
        #print "bbw cmd string: ", cmd
        self.instrument.term_chars = ""
        self.write(str(bytearray(cmd)+blockarray))
        self.instrument.term_chars = None
        response= self.query("*OPC?")
        #print "BinBlockWrite Response: ", response
        return response

    def select_sequence(self,seq_num=1):
        self.write(":SEQuence:SELect %d" % seq_num)
        
    def get_sequence(self):
        return int(self.query(":SEQuence:SELect?"))
    
    def define_sequence_step(self,step,seg_num,loops=1,jump_flag=False):
        if jump_flag: jump=1
        else: jump=0
        self.write(":SEQuence:DEFine %d, %d, %d, %d" % (step,seg_num,loops,jump))

    def get_sequence_step(self):
        return self.query(":SEQuence:DEFine?")
        
    def get_settings(self):
        settings=VisaInstrument.get_settings(self)
        settings['amplitude']=self.get_amplitude()
        settings['offset']=self.get_offset()
        settings['output']=self.get_output()
        settings['sync_output']=self.get_sync_output()
        settings['clockrate']=self.get_clockrate()
        settings['mode']=self.get_mode()
        
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
    awg=AWG81180A (name='awg',address="GPIB0::04::INSTR")
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
    for i in range(200):
        b=zeros(12800)
        for j in range(i*50):
            b[j]=4095
        arr.append(b)
        awg.add_floatsegment(b,i+1)
        awg.define_sequence_step(i+1,i+1)
    figure(1)
    imshow(array(arr))
        
    awg.set_output(True) 
    awg.set_mode("SEQUENCE")
    
    print "Finished uploading data."
    
        