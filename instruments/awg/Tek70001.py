# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:59:04 2013

@author: Dave
"""

#DCM: This was directly copied from the TEK5014 class
#*SOME* methods have been updated, but don't assume they have!

from slab.instruments import SocketInstrument 
from numpy import array, floor
    
class Tek70001 (SocketInstrument):
    """Tektronix 70001 Arbitrary Waveform Class"""
    default_port=4000
    def __init__(self,name='Tek70001',address='',enabled=True,timeout=100, recv_length=1024):
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
#        pass     

    def get_id(self):
        return self.query("*IDN?")

    def set_amplitude(self,channel,amp):
        self.write('SOURce%d:VOLTage:AMPLitude %f' % (channel,amp))

    def get_amplitude(self,channel):
        return float(self.query('SOURce%d:VOLTage:AMPLitude?' % (channel)))
        
    def set_analogHigh(self,channel,value):
        self.write('SOURce%d:VOLTage:HIGH %f' % (channel,value))

    def get_analogHigh(self,channel):
        return float(self.query('SOURce%d:VOLTage:HIGH?' % (channel)))
        
    def set_analogLow(self,channel,value):
        self.write('SOURce%d:VOLTage:LOW %f' % (channel,value))

    def get_analogLow(self,channel):
        return float(self.query('SOURce%d:VOLTage:LOW?' % (channel)))
        
    def set_DACResolution(self,channel,value):
        checkset= [8,10,14]
        if value not in checkset: raise Exception('Error: invalid DAC Resolution in Tek5014')
        self.write('SOURce%d:DAC:RESolution %d' %(channel,value))            

    def get_DACResolution(self,channel):
        return int(self.query('SOURce%d:DAC:RESolution %d' %(channel)))
        
    def set_delay(self,channel,value):
        self.write('SOURce%d:DELay %f' % (channel,value))
        
    def get_delay(self,channel):
        return float(self.query('SOURce%d:DELay?' % (channel)))
        
    def set_enabled(self,channel,state):
        if state in ['on','ON',True,1,'1']: 
            self.write('OUTPut%d:STATe ON' % channel)
        elif state in ['off','OFF',False, 0,'0']:
            self.write('OUTPut%d:STATe OFF' % channel)
        else:
            raise Exception('Tek5014: Invalid enabled state')
        
    def get_enabled(self,channel):            
        return bool(self.query('OUTPut%d:STATe?' % channel))
        
    def set_lowpassFilterFrequency(self,channel,value):
        self.write('OUTPut%d:FILTer:LPASs:FREQuency %f' % (channel,value))
        
    def get_lowpassFilterFrequency(self,channel):
        return float(self.query('OUTPut%d:FILTer:LPASs:FREQuency %f' % (channel)))
        
    def set_markerHigh(self,channel,marker_index,value):
        self.write('SOURce%d:MARKER%d:VOLTage:HIGH %f' % (channel,marker_index,value))
        
    def get_markerHigh(self,channel,marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:HIGH?' % (channel,marker_index)))

    def set_markerLow(self,channel,marker_index,value):
        self.write('SOURce%d:MARKER%d:VOLTage:LOW %f' % (channel,marker_index,value))
        
    def get_markerLow(self,channel,marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:LOW?' % (channel,marker_index)))

    def set_markerOffset(self,channel,marker_index,value):
        self.write('SOURce%d:MARKER%d:VOLTage:OFFSet %f' % (channel,marker_index,value))
        
    def get_markerOffset(self,channel,marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:OFFSet?' % (channel,marker_index)))

    def set_markerAmplitude(self,channel,marker_index,value):
        self.write('SOURce%d:MARKER%d:VOLTage:AMPLitude %f' % (channel,marker_index,value))
        
    def get_markerAmplitude(self,channel,marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:AMPLitude?' % (channel,marker_index)))
        
    def set_marker1High(self,channel,value):    self.set_markerHigh(channel,1,value)
    def get_marker1High(self,channel):          return self.get_markerHigh(channel,1)
    def set_marker2High(self,channel,value):    self.set_markerHigh(channel,2,value)
    def get_marker2High(self,channel):          return self.get_markerHigh(channel,2)

    def set_marker1Low(self,channel,value):    self.set_markerHigh(channel,1,value)
    def get_marker1Low(self,channel):          return self.get_markerHigh(channel,1)
    def set_marker2Low(self,channel,value):    self.set_markerHigh(channel,2,value)
    def get_marker2Low(self,channel):          return self.get_markerHigh(channel,2)

    def set_marker1Offset(self,channel,value):    self.set_markerHigh(channel,1,value)
    def get_marker1Offset(self,channel):          return self.get_markerHigh(channel,1)
    def set_marker2Offset(self,channel,value):    self.set_markerHigh(channel,2,value)
    def get_marker2Offset(self,channel):          return self.get_markerHigh(channel,2)

    def set_marker1Amplitude(self,channel,value):    self.set_markerHigh(channel,1,value)
    def get_marker1Amplitude(self,channel):          return self.get_markerHigh(channel,1)
    def set_marker2Amplitude(self,channel,value):    self.set_markerHigh(channel,2,value)
    def get_marker2Amplitude(self,channel):          return self.get_markerHigh(channel,2)
   
    def set_outputWaveformName(self,channel,value):
        self.write('SOURce%d:WAVeform "%s"' % (channel,value))             
    
    def get_outputWaveformName(self,channel):
        return self.query('SOURce%d:WAVeform?' % (channel))             
        
    def set_offset(self,channel,value):
        self.write('SOURce%d:VOLTage:OFFSet %f' % (channel,value) )
        
    def get_offset(self,channel):
        return float(self.query('SOURce%d:VOLTage:OFFSet?' % (channel) ))
        
    def set_skew(self,channel,value):
        self.write('SOURce%d:SKEW %f' %(channel,value))
    
    def get_skew(self,channel):
        return float(self.query('SOURce%d:SKEW?' %(channel)))


    def get_setup_filename(self):
        return self.query('AWGControl:SNAMe?')
        
    def save_config(self,name):
        self.write('AWGControl:SSAVe "%s"' % name)
        
    def open_config(self,name):
        self.write('AWGControl:SREStore "%s"' % name)
    
    def import_waveform(self,waveform_name,filename,file_type):
        self.write('MMEMory:IMPort "%s", "%s", "%s"' % (waveform_name,filename,file_type))
        
    def make_directory(self,name):
        self.write('MMEMory:MDIRectory "%s"' % name )
        
    def reset(self):
        self.write('*RST')
        
    def clear_reg_and_queue(self):
        self.write('*CLS')
        
    def calibrate(self):
        return self.query('*CAL?')
        
    def operation_complete(self):
        return self.query('*OPC?')
        
    def run(self):
        self.write('AWGControl:RUN')
        self.operation_complete()
        
    def stop(self):
        self.write('AWGControl:STOP')
        self.operation_complete()
        
    def sync(self):
        self.write('*OPC')
        
    def create_waveform(self,name,size,waveform_type):
        self.write(':WLISt:WAVeform:NEW "%s", %d, %s' % (name,size,waveform_type))

    def delete_waveform(self,name):
        self.write(':WLISt:WAVeform:DEL "%s"' % name)
       
    def set_current_working_directory(self,value):
        return self.query('MMEMory:CDIRectory "%s"' % value)

    def get_current_working_directory(self):
        return self.query('MMEMory:CDIRectory?')
        
    def set_impedance(self,value):
        """value can be '50' or '1k'"""
        d={'50':'50',50:'50','1k':'1k',1000:'1k','1000':'1k'}
        self.write('TRIGger:IMPedance %s' % d[value])
        
    def get_impedance(self):
        return self.query('TRIGger:IMPedance?')
        
    def set_trigger_interval(self,value):
        self.write('TRIGger:TIMer %f' % value)
        
    def get_trigger_interval(self):
        return float(self.query('TRIGger:Timer?'))
        
    def set_trigger_level(self,value):
        self.write('TRIGger:LEVel %f' % value)
        
    def get_trigger_level(self):
        return float(self.query('TRIGger:LEVel?'))
        
    def set_trigger_slope(self,value):
        """Sets trigger slope, value = 'POS' or 'NEG'"""
        self.write('TRIGger:Polarity %s' % value)
    
    def get_trigger_slope(self):
        return self.query('TRIGger:Polarity?')
    
    def set_trigger_source(self, value):
        """valid values = 'INT', 'EXT' """
        self.write('TRIGger:SOURce %s' % value)
        
    def get_trigger_source(self):
        return self.query('TRIGger:SOURce?')
        
    def set_clock_source(self,value):
        """valid values = 'INT' , 'EXT' """
        self.write('AWGControl:CLOCk:SOURce %s' % value)

    def get_clock_source(self):
        return self.query('AWGControl:CLOCk:SOURce?')
        
    def set_run_mode(self,value):
        """valid values = 'TRIG','SEQ','GAT','CONT' """
        self.write ('AWGControl:RMODe %s' % value )
        
    def get_run_mode(self,value):
        return self.query ('AWGControl:RMODe?')\
        
    def set_reference_source(self,value):
        """valid values = 'INT', 'EXT' """        
        self.write('SOURce1:ROSCillator:SOURce %s' % value )
        
    def get_reference_source(self):
        return self.query('SOURce1:ROSCillator:SOURce?')
        
    def set_repetition_rate(self,value):
        self.write('AWGControl:RRATe %f' % value)
        
    def get_repetition_rate(self):
        return float(self.query('AWGControl:RRATe?'))
        
    def set_sampling_rate(self,value):
        self.write('SOURCe1:FREQuency %f' % value)
        
    def get_sampling_rate(self):
        return float(self.query('SOURCe1:FREQuency?' ))
        
    def set_sequence_length(self,value):
        self.write('SEQuence:LENGth %d' % value)
        
    def get_sequence_length(self):
        return int(self.query('SEQuence:LENGth?'))
        
    def set_loop_count(self,value):
        self.write('SEQuence:ELEMent:LOOP:COUNt %d' % value )
    
    def get_loop_count(self):
        return int(self.query('SEQuence:ELEMent:LOOP:COUNt?'))
        
    def set_waveform_name(self,index,name):
        self.write('SEQuence:ELEMent1:WAVeform%d "%s"' % (index,name))
    
    def get_waveform_name(self,index):
        return self.query('SEQuence:ELEMent1:WAVeform%d?' % (index))
        
    def set_divider_rate(self,value):
        """valid options 1,2,4,8"""
        self.write('AWGControl:CLOCk:DRATe %d' % value)
        
    def get_divider_rate(self):
        return int(self.query('AWGControl:CLOCk:DRATe?' ))
        
    def set_reference_oscillator_type(self,value):
        """valid values = 'FIX', 'VAR' """
        self.write('SOURce1:ROSCillator:TYPE %s' % value)
        
    def get_reference_oscillator_type(self):
        return self.query('SOURce1:ROSCillator:TYPE?')
        
    def set_reference_oscillator_frequency(self,value):
        """valid values '10MHz', '20MHz', '100MHz' """
        self.write('SOURce1:ROSCillator:FREQuency %s' % value)
        
    def get_reference_oscillator_frequency(self):
        return self.query('SOURce1:ROSCillator:FREQuency?')
        
        
    def pack_pattern(self,pattern,marker1,marker2,pattern_high=1.,pattern_low=-1.):
        """AWG 5000 series binary data format
	     m2 m1 d14 d13 d12 d11 d10 d9 d8 d7 d6 d5 d4 d3 d2 d1
	  """
        
        #create binary data pattern from waveform given by pattern  
        lpattern = array((pattern - pattern_low)/(pattern_high-pattern_low)*(2**14-1),dtype=long)
        
#        for i in range(len(pattern)):
#            if pattern[i]<0:
#                print "<0"
#                pattern[i] = 0
#                
#            if pattern[i]>=2**14:
#                print "Too large"
#                pattern[i] = 2**14-1
   
        
        #append marker data
        lpattern+=2**15*marker2+2**14*marker1
        b=bytearray(len(lpattern)*2)
        for ii in xrange(len(lpattern)):
            b[ii*2] = (lpattern[ii] & 0xFF)
            b[ii*2+1] = (lpattern[ii] & 0xFF00) >>8
            
        return b

    def pack_intpattern(self,pattern,marker1,marker2,pattern_high=1.,pattern_low=-1.):
        """AWG 5000 series binary data format
	     m2 m1 d14 d13 d12 d11 d10 d9 d8 d7 d6 d5 d4 d3 d2 d1
	  """
        
        #create binary data pattern from waveform given by pattern  
        #lpattern = array((pattern - pattern_low)/(pattern_high-pattern_low)*(2**14-1),dtype=long)
        
#        for i in range(len(pattern)):
#            if pattern[i]<0:
#                print "<0"
#                pattern[i] = 0
#                
#            if pattern[i]>=2**14:
#                print "Too large"
#                pattern[i] = 2**14-1
   
        
        #append marker data
        #lpattern=pattern+2**15*marker2+2**14*marker1
        lpattern=pattern
        b=bytearray(len(lpattern)*2)
        for ii in xrange(len(lpattern)):
            b[ii*2] = (lpattern[ii] & 0xFF)
            b[ii*2+1] = (lpattern[ii] & 0xFF00) >>8

            
        return b

    def get_error_log(self):
        done=False
        log=''
        while not done:
            s=self.query('SYSTEM:ERR?')
            done=s[0]=='0'
            log+=s
        return log

    def set_waveform_data(self,waveform_name,waveform_data,start_index=0):
        #waveform_data should be a byte array from pack_pattern
        #self.write(':WLISt:WAVeform:DATA "%s", %s' % (waveform_name,str(waveform_data)))
        self.binblockwrite('WLISt:WAVeform:DATA "%s",%d,%d,' % (waveform_name,start_index,len(waveform_data)/2), waveform_data)        
        
    def binblockwrite(self,commandstring, blockarray):
        #response= self.query("*OPC?")
        #print "BinBlockWrite Response: ", response
        blockarraylength=str(len(blockarray))
        blockarraylengthposition=len(blockarraylength)
        cmd = commandstring+"#"+str(blockarraylengthposition)+blockarraylength
        #print "bbw cmd string: ", cmd
        #self.term_chars = ""
        #print len(blockarray)
        print cmd
        print len(str(blockarray))
        self.write(cmd+str(blockarray))
        #self.instrument.term_chars = None
        #response= self.query("*OPC?")
        #print "BinBlockWrite Response: ", response
        #return response
        
    def pre_load(self):
        
        self.stop()
        self.clear_waveforms_seqs()
        
    def clear_waveforms_seqs(self):
        
        self.write('SLISt:SEQuence:DEL ALL')
        self.write('WLISt:WAVeform:DEL ALL')
        
    def load_waveform_file(self,filename):
        
        self.write('MMEM:OPEN "' + filename + '"')
        
    def load_sequence_file(self,filename):
        
        self.socket.send("AWGControl:SREStore '%s' \n" % (filename))
        for i in range(1,5):
            self.set_enabled(i,True)
        
    def prep_experiment(self):
        
        #load sequence        
        self.write('SOUR:CASS:SEQ "seq1",1')
        self.write("SOUR:JUMP:FORC 1")
        self.operation_complete()
            
    
    def set_amps_offsets(self,channel_amps=[1.0],channel_offsets = [0.0], marker_amps=[1.0]):
    
            #Note that there is no offset!
         
            #The amplitude range is 0.25 --> 0.5
            self.set_amplitude(1,channel_amps[0])
            #self.set_offset(i,channel_offsets[i-1])
            
            #maker functions not set for now!!
            
    
    def new_sequence(self,seq_name='seq1',num_steps=1):
        self.write('SLISt:SEQuence:NEW "' + seq_name + '",' + str(num_steps))
        
    def assign_seq_waveform(self,step,waveform,last_step=False,seq_name='seq1',):
        
        self.write('SLISt:SEQuence:STEP'+str(step)+':TASSet1:WAVeform "' + seq_name + '","' + waveform + '"')
        
        #setup wait trigger 
        self.write('SLISt:SEQuence:STEP'+str(step)+':WINPUT "' + seq_name + '", ATR')        
        
        #setup goto
        if last_step:
            self.write('SLISt:SEQuence:STEP'+str(step)+':GOTO "' + seq_name + '", FIRST')
        
if __name__=="__main__":
    awg=Tek70001(address='192.168.14.137')
    print awg.get_id()