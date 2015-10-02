# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 15:01:07 2011

@author: Phil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:14:11 2011

@author: Phil
"""
import ctypes as C
import numpy as np
import sys
from slablayout import *
# try:
#     from guiqwt.pyplot import *
# except:
#     print "Warning unable to import guiqwt.pyplot"
from scipy.fftpack import fft,rfft
from slab.dataanalysis import heterodyne
from numpy import sin,cos,pi
#import matplotlib.pyplot as mplt
#import operator
import time
#import fftw3
import os

U8 = C.c_uint8
U8P = C.POINTER(U8)
U32 = C.c_uint32
U32P = C.POINTER(U32)
U32PP = C.POINTER(U32P)

DEBUGALAZAR=False

#try:
#    CTHelp = C.CDLL(r'C:\Users\Phil\Desktop\AlazarApi\trunk\Debug\ctypes_help.dll')
#except:
#    print "Couldn't load ctypes_help.dll"

# Helper Functions
def int_linterp(x, a1, a2, b1, b2):
    'Moves x from interval [a1, a2] to interval [b1, b2]'
    return int(b1 + ((x - a1) * (b2 - b1)) / float((a2 - a1)))

def ret_to_str(retCode, Az):
    Az.AlazarErrorToText.restype = C.c_char_p
    return Az.AlazarErrorToText(U32(retCode))   

class DMABuffer:
    '''Buffer suitable for DMA transfers.

    AlazarTech digitizers use direct memory access (DMA) to transfer
    data from digitizers to the computer's main memory. This class
    abstracts a memory buffer on the host, and ensures that all the
    requirements for DMA transfers are met.

    DMABuffers export a 'buffer' member, which is a NumPy array view
    of the underlying memory buffer

    Args:

      bytes_per_sample (int): The number of bytes per samples of the
      data. This varies with digitizer models and configurations.

      size_bytes (int): The size of the buffer to allocate, in bytes.

    '''
    def __init__(self, bytes_per_sample, size_bytes):
        self.size_bytes = size_bytes

        cSampleType = U8
        npSampleType = np.uint8
        if bytes_per_sample > 1:
            cSampleType = U16
            npSampleType = np.uint16

        self.addr = None
        if os.name == 'nt':
            MEM_COMMIT = 0x1000
            PAGE_READWRITE = 0x4
            C.windll.kernel32.VirtualAlloc.argtypes = [C.c_void_p, C.c_long, C.c_long, C.c_long]
            C.windll.kernel32.VirtualAlloc.restype = C.c_void_p
            self.addr = C.windll.kernel32.VirtualAlloc(
                0, C.c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
        elif os.name == 'posix':
            C.libc.valloc.argtypes = [C.c_long]
            C.libc.valloc.restype = C.c_void_p
            self.addr = libc.valloc(size_bytes)
            print("Allocated data : " + str(self.addr))
        else:
            raise Exception("Unsupported OS")


        ctypes_array = (cSampleType * (size_bytes // bytes_per_sample)).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes_array, dtype=npSampleType)
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def __exit__(self):
        if os.name == 'nt':
            MEM_RELEASE = 0x8000
            windll.kernel32.VirtualFree.argtypes = [C.c_void_p, C.c_long, C.c_long]
            windll.kernel32.VirtualFree.restype = C.c_int
            windll.kernel32.VirtualFree(C.c_void_p(self.addr), 0, MEM_RELEASE);
        elif os.name == 'posix':
            libc.free(self.addr)
        else:
            raise Exception("Unsupported OS")



class AlazarConstants():
    # Clock Constants
    clock_source = {"internal" : U32(1),
              "reference": U32(7),
              "60 MHz" : U32(4),
              "1 GHz" : U32(5)}
    sample_rate = [(1,U32(1)), (2,U32(2)), (5,U32(4)), (10,U32(8)), (20,U32(10)),
            (50,U32(12)), (100,U32(14)), (200,U32(16)), (500,U32(18)), (1000,U32(20)),
            (2000,U32(24)), (5000,U32(26)), (10000,U32(28)), (20000,U32(30)), (50000,U32(34)),
            (100000,U32(36)), (250000,U32(43)), (500000,U32(48)), (1000000,U32(53))]
#    sample_rate_txt = {"1 kHz":U32(1),"2 kHz":U32(2),"5 kHz":U32(4),"10 kHz":U32(8),"20 kHz":U32(10),
#                      "100 kHz": U32(14),"200 kHz": U32(16),"500 kHz":U32(18),
#                      "1 MHz":U32(20),"2 MHz":U32(24),"5 MHz":U32(26),"10 MHz":U32(28),
#                      "20 MHz":U32(30),"50 MHz":U32(34),
#                      "100 MHz":U32(36),"250 MHz":U32(43),"500 MHz":U32(48),"1 GHz":U32(53)}
    sample_rate_txt = {"1 kHz":1,"2 kHz":2,"5 kHz":5,"10 kHz":10,"20 kHz":20,
                      "100 kHz": 100,"200 kHz": 200,"500 kHz":500,
                      "1 MHz":1000,"2 MHz":2000,"5 MHz":5000,"10 MHz":10000,
                      "20 MHz":20000,"50 MHz":50000,"100 MHz":100000,
                      "250 MHz":250000,"500 MHz":500000,"1 GHz":1000000}
    sample_rate_external = U32(64)
    sample_rate_reference = U32(1000000000)
    clock_edge = {"rising": U32(0),
                  "falling": U32(1)}
    
    # Trigger Constants
    trigger_source = {"CH_A":U32(0), "CH_B":U32(1), "external":U32(2), "disabled":U32(3)}
    trigger_operation = {"or":U32(2), "and":U32(3), "xor":U32(4), "and not":U32(5)}
    trigger_ext_coupling = {"AC":U32(1), "DC":U32(2)}
    trigger_single_op = U32(0)
    trigger_engine_1 = U32(0)
    trigger_engine_2 = U32(1)
    trigger_edge = {"rising":U32(1), "falling":U32(2)}

    # Channel Constants     
    channel = {"CH_A":U8(1), "CH_B":U8(2)}
    input_coupling = {"AC":U32(1), "DC":U32(2)}
    input_range = [(0.04,U32(2)), (0.1,U32(5)), (0.2,U32(6)), (0.4,U32(6)),
                   (1,U32(10)), (2,U32(11)), (4,U32(12))]
    input_range_txt = {"40 mV":0.04,"100 mV":0.1,"200 mV":0.2,"400 mV":0.4,
                       "1 V":1.0,"2 V":2.0,"4 V":4.0}
    input_filter = {False:U32(0), True:U32(1)}
    
    ApiSuccess = 512
    
class AlazarConfig1():
    _form_fields_ =[("samplesPerRecord","Samples",2048),
                    ("recordsPerBuffer","Records per Buffer",10),
                    ("recordsPerAcquisition","Total Records",10),
                    ("bufferCount","Number of Buffers",1),
                    ("buffers_per_merge", "Buffers per Merge", 100),
                    ("clock_source","Clock Source", ["internal",("internal","internal"),("reference","reference"),("60 MHz","60 MHz"),("1 GHz","1 GHz")]),
                    ("clock_edge","Clock Edge", ["rising",("rising","Rising"),("falling","Falling")]),
                    ("sample_rate", "Sample Rate", [1000,(1,"1 kHz"),(2, "2 kHz"),(5,"5 kHz"),(10,"10 kHz"),(20,"20 kHz"),(1000,"1 MHz"),(2000,"2 MHz"),(5000, "5 MHz"),(10000,"10 MHz"),(20000,"20 MHz"),(50000, "50 MHz"),(100000, "100 MHz"),(250000,"250 MHz"),(500000,"500 MHz"),(1000000,"1 GHz")]),
                    ("trigger_source1","Trigger Source 1", ["CH_A", ("CH_A", "Channel 1"),("CH_B", "Channel 2"),("external","External"),("disabled","Disabled")]),
                    ("trigger_edge1", "Trigger Edge 1",["rising",("rising","Rising"),("falling","Falling")]),
                    ("trigger_level1", "Trigger Level 1", 0),
                    ("trigger_source2","Trigger Source 2", ["disabled", ("CH_A", "Channel 1"),("CH_B", "Channel 2"),("external","External"),("disabled","Disabled")]),
                    ("trigger_edge2", "Trigger Edge 2", ["rising", ("rising","Rising"),("falling","Falling")]),
                    ("trigger_level2", "Trigger Level 2", 0),
                    ("trigger_operation", "Trigger Operation", ["or",("or","OR"), ("and","AND"),("xor","XOR"),("and not","AND NOT")]),
                    ("trigger_coupling", "Trigger Coupling", ["DC",("DC","DC"),("AC","AC")]),
                    ("timeout","Timeout",5000),
                    ("ch1_enabled","Ch1 Enabled", True),
                    ("ch1_coupling","Ch1 Coupling", ["DC",("DC","DC"),("AC","AC")]),
                    ("ch1_range","Ch1 Range", [1,(0.04,"40 mV"),(0.1,"100 mV"),(0.2,"200 mV"),(0.4,"400 mV"),(1, "1 V"),(2,"2 V"),(4,"4 V")]),
                    ("ch1_filter","Ch1 Filter",False),
                    ("ch2_enabled","Ch2 Enabled", False),
                    ("ch2_coupling","Ch2 Coupling", ["DC",("DC","DC"),("AC","AC")]),
                    ("ch2_range","Ch2 Range", [1,(0.04,"40 mV"),(0.1,"100 mV"),(0.2,"200 mV"),(0.4,"400 mV"),(1, "1 V"),(2,"2 V"),(4,"4 V")]),
                    ("ch2_filter","Ch2 Filter",False),
                   ]
    
    def __init__(self,config_dict=None):
        if config_dict is not None:
            self.from_dict(config_dict)
            
    def from_dict(self,config_dict):
        for k,v in config_dict.items():
            self.__dict__[k]=v
        
    def get_formdata (self):
        fd=[]
        for field_name,field_label,field_value in self._form_fields_:
            if isinstance(field_value,(list,tuple)):
                field_value[0]=self.__dict__[field_name]
            else:
                field_value=self.__dict__[field_name]
            fd.append((field_name,field_name,field_value))
        return fd

    def get_dict(self):
        d={}
        for field_name in type(self)._form_fields_:
            d[field_name]=self.__getattribute__(field_name)
        return d       

    @property
    def samples_per_buffer(self):
        print 'A'
        return self.samplesPerRecord * self.recordsPerBuffer
    samplesPerBuffer = samples_per_buffer
    
    @property
    def buffers_per_acquisition(self):
        #assert (self.records_per_acquisition % self.records_per_buffer) == 0
        return int(self.recordsPerAcquisition / self.recordsPerBuffer)
    buffersPerAcquisition = buffers_per_acquisition

    @property
    def buffers_per_worker(self):
       # assert (self.records_per_acquisition % self.records_per_buffer) == 0
        return self.buffers_per_acquisition / self.bufferCount
    buffersPerWorker = buffers_per_worker

    bytesPerSample = bytes_per_sample = 1
    @property
    def bytes_per_record(self):
        return self.bytes_per_sample * self.samplesPerRecord
    bytesPerRecord = bytes_per_record

    @property
    def bytes_per_buffer(self):
        return self.bytesPerRecord * self.recordsPerBuffer
    bytesPerBuffer = bytes_per_buffer

class AlazarConfig():
    _fields_ = [ 'samplesPerRecord',
                 'recordsPerBuffer',
                 'recordsPerAcquisition',
                 'bufferCount',
                 'clock_source',
                 'clock_edge',
                 'sample_rate',
                 'trigger_source1',
                 'trigger_edge1',
                 'trigger_level1',
                 'trigger_source2',
                 'trigger_edge2',
                 'trigger_level2',
                 'trigger_operation',
                 'trigger_coupling',
                 'trigger_delay',
                 'timeout',
                 'ch1_enabled',
                 'ch1_coupling',
                 'ch1_range',
                 'ch1_filter',
                 'ch2_enabled',
                 'ch2_coupling',
                 'ch2_range',
                 'ch2_filter'
                 ]
    
    def __init__(self,config_dict=None):
        if config_dict is not None:
            self.from_dict(config_dict)
            
    def from_dict(self,config_dict):
        for k,v in config_dict.items():
            self.__dict__[k]=v
        
    def get_dict(self):
        d={}
        for field_name in self._fields_:
            d[field_name]=getattr(self,field_name)
        return d       
    
    def interpret_constants(self):
        self.sample_rate = AlazarConstants.sample_rate_txt[self.sample_rate_txt]
        self.ch1_range = AlazarConstants.input_range_txt[self.ch1_range_txt]
        self.ch2_range = AlazarConstants.input_range_txt[self.ch2_range_txt]

    
    def from_form(self,widget):
        self.samplesPerRecord = widget.samplesSpinBox.value()
        self.recordsPerBuffer = widget.recordsSpinBox.value()
        self.bufferCount = widget.buffersSpinBox.value()
        self.recordsPerAcquisition = max(self.recordsPerBuffer,widget.recordsPerAcquisitionSpinBox.value())
        self.clock_source = str(widget.clocksourceComboBox.currentText())
        self.clock_edge = str(widget.clockedgeComboBox.currentText())
        self.sample_rate = AlazarConstants.sample_rate_txt[str(widget.samplerateComboBox.currentText())]
        self.trigger_source1 = str(widget.trig1_sourceComboBox.currentText())
        self.trigger_edge1 = str(widget.trig1_edgeComboBox.currentText())
        self.trigger_level1 = widget.trig1_levelSpinBox.value()
        self.trigger_source2 = str(widget.trig2_sourceComboBox.currentText())
        self.trigger_edge2 = str(widget.trig2_edgeComboBox.currentText())
        self.trigger_level2 = widget.trig2_levelSpinBox.value()

        self.trigger_operation = str(widget.trigOpComboBox.currentText())
        self.trigger_coupling = str(widget.trigCouplingComboBox.currentText())
        self.trigger_delay=0.

        self.timeout = widget.timeoutSpinBox.value()

        self.ch1_enabled = widget.ch1_enabledCheckBox.isChecked()
        self.ch1_coupling = str(widget.ch1_couplingComboBox.currentText())
        self.ch1_range = AlazarConstants.input_range_txt[str(widget.ch1_rangeComboBox.currentText())]
        self.ch1_filter = widget.ch1_filteredCheckBox.isChecked()

        self.ch2_enabled = widget.ch2_enabledCheckBox.isChecked()
        self.ch2_coupling = str(widget.ch2_couplingComboBox.currentText())
        self.ch2_range = AlazarConstants.input_range_txt[str(widget.ch2_rangeComboBox.currentText())]
        self.ch2_filter = widget.ch2_filteredCheckBox.isChecked()
        
        
def round_samples(x, base=64):
    return int(base * round(float(x)/base))

class Alazar():
    def __init__(self,config=None, handle=None):
        self.Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
        if handle:
            self.handle = handle
        else:
            self.handle = self.get_handle()
        if not self.handle:
            raise RuntimeError("Board could not be found")

        if config is None:
            self.config = AlazarConfig()
        else:
            self.configure( AlazarConfig(config))
            

            
    def close(self):
       del self.Az
        
            
    def get_handle(self):
        return self.Az.AlazarGetBoardBySystemID(U32(1), U32(1))

    def configure(self,config=None):
        if config is not None: self.config=config
        if self.config.samplesPerRecord<256 or (self.config.samplesPerRecord  % 64)!=0:
            print "Warning! invalid samplesPerRecord!"
            print "Frames will not align properly!"
            print "Try %d or %d" % (max(256,self.config.samplesPerRecord-(self.config.samplesPerRecord  % 64)),
                                                                 max(256,self.config.samplesPerRecord+64-(self.config.samplesPerRecord  % 64)))
        if self.config.recordsPerAcquisition < self.config.recordsPerBuffer:
            raise ValueError("recordsPerAcquisition: %d < recordsPerBuffer: %d" % (self.config.recordsPerAcquisition , self.config.recordsPerBuffer))
        if DEBUGALAZAR: print "Configuring Clock"
        self.configure_clock()
        if DEBUGALAZAR: print "Configuring triggers"
        self.configure_trigger()
        if DEBUGALAZAR: print "Configuring inputs"
        self.configure_inputs()
        if DEBUGALAZAR: print "Configuring buffers"
        self.configure_buffers()
           
    def configure_clock(self, source=None, rate=None, edge=None):
        """
        :param source: 'internal' to use internal clock with rate specified by
                       rate parameter. '60 MHz' for external clock with rate <= 60 MHz.
                       '1 GHz' for external clock with rate <= 1 GHz.
                       'reference' --> see Alazar documentation for AlazarSetCaptureClock
        :param rate: Rate (in KHz) for the internal clock, ignored for external
                     This will be rounded down to closest value specified by AlazarSetCaptureClock
                     documentation
        :param edge: 'rising' or 'falling'
        """
        if source is not None: self.config.clock_source= source        
        if rate is not None: self.config.clock_rate = rate   #check this to make sure it's behaving properly
        if edge is not None: self.config.edge = edge 
        

        #convert clock config
        if self.config.clock_source not in ["internal", "external", "reference"]:
            raise ValueError("source must be one of internal, external, or reference")
        if self.config.clock_edge not in ["rising", "falling"]:
            raise ValueError("edge must be one of rising or falling")

        if self.config.clock_source == "internal":
            source = AlazarConstants.clock_source["internal"]
            decimation=U32(0)
            for (rate, value) in AlazarConstants.sample_rate:
                if rate >= self.config.sample_rate:
                    if rate > self.config.sample_rate:
                        print "Warning: sample_rate not found. Using first smaller value", rate, "Khz"
                        self.config.sample_rate = rate
                    sample_rate = value
                    break
        elif self.config.clock_source == "external":
            sample_rate = AlazarConstants.sample_rate_external
            decimation=U32(0)
            if self.config.sample_rate < 60000:
                source = AlazarConstants.clock_source["60 MHz"]
            elif self.config.sample_rate < 1000000:
                source = AlazarConstants.clock_source["1 GHz"]
            else:
                raise ValueError("Not supported (yet?)")
        elif self.config.clock_source == "reference":
            source = AlazarConstants.clock_source["reference"]
            sample_rate=U32(1000000000)
            decimation=int(1e9/(self.config.sample_rate*1e3))
            if (decimation != 1) and (decimation != 2) and (decimation != 1) and (decimation %10 != 0):
                print "Warning: sample_rate must be 1Gs/s / 1,2,4 or a multiple of 10. Using 1Gs/s."
                decimation=1
            decimation=U32(decimation)
        else:
            raise ValueError("reference signal not implemented yet")
        ret = self.Az.AlazarSetCaptureClock(self.handle, source, sample_rate, AlazarConstants.clock_edge[self.config.clock_edge], decimation)
        if DEBUGALAZAR: print "ClockConfig:", ret_to_str(ret, self.Az)

    def configure_trigger(self,source=None, source2=None, edge=None, edge2=None,
                 level=None, level2=None, operation=None, coupling=None, timeout=None,delay=0):
        """
        Can set up to two trigger operations to be performed

        :param source: Where the first trigger engine should take its input.
                       'CH_A' for channel A, 'CH_B' for channel B, "external" for external source,
                       'disabled' to disable trigger engine
        :param edge: 'rising' or 'falling'
        :param level: integer in interval [-100, 100], i.e. a percent of the input range
                      at which to trigger a capture
        :param coupling: 'AC' or 'DC'
        :param operation: How to combine two enabled triggers to generate capture events
                          'or' to trigger on either engine, 'and' to trigger only when both go high,
                          'xor', 'and not' offered as well.
        :param timeout: How long to wait for a trigger before giving up (milliseconds)
        """
        if source is not None: self.config.trigger_source1 = source
        if source2 is not None: self.config.trigger_source2 = source2
        if edge is not None: self.config.trigger_edge1 = edge
        if edge2 is not None: self.config.trigger_edge2 = edge2
        if level is not None: self.config.trigger_level1 = level
        if level2 is not None: self.config.trigger_level2 = level2
        
        if not (self.config.trigger_level1 >= -100 and self.config.trigger_level1 < 100):
            raise ValueError("Level must be value in [-100,100]")
        if not (self.config.trigger_level2 >= -100 and self.config.trigger_level2 < 100):
            raise ValueError("Level must be value in [-100,100]")
        if operation is not None: self.config.trigger_operation = operation
        if coupling is not None: self.config.trigger_coupling= coupling
        if timeout is not None: self.config.timeout= timeout
        if delay is not None: self.config.trigger_delay = delay
        
        
        if source2 == "disabled":
            op = AlazarConstants.single_op
        else:
            op = AlazarConstants.trigger_operation[self.config.trigger_operation]
            
        print 'Z', op,  AlazarConstants.trigger_engine_1, AlazarConstants.trigger_source[self.config.trigger_source1], AlazarConstants.trigger_edge[self.config.trigger_edge1], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)),  AlazarConstants.trigger_engine_2, AlazarConstants.trigger_source[self.config.trigger_source2], AlazarConstants.trigger_edge[self.config.trigger_edge2], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255))

        ret = self.Az.AlazarSetTriggerOperation(self.handle, op,
                AlazarConstants.trigger_engine_1, AlazarConstants.trigger_source[self.config.trigger_source1], AlazarConstants.trigger_edge[self.config.trigger_edge1], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)),
                AlazarConstants.trigger_engine_2, AlazarConstants.trigger_source[self.config.trigger_source2], AlazarConstants.trigger_edge[self.config.trigger_edge2], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)))
        if DEBUGALAZAR: print "Set Trigger:", ret_to_str(ret, self.Az)
        if self.config.trigger_source1 == "external":
            ret = self.Az.AlazarSetExternalTrigger(self.handle, AlazarConstants.trigger_ext_coupling[self.config.trigger_coupling], U32(0))
            if DEBUGALAZAR: print "Set External Trigger:", ret_to_str(ret, self.Az)
            
        
        self.triggerDelay_samples = int(self.config.trigger_delay * self.config.sample_rate*1e3 + 0.5)
        ret = self.Az.AlazarSetTriggerDelay(self.handle, U32(self.triggerDelay_samples))
        if DEBUGALAZAR: print "Set Trigger Delay:", ret_to_str(ret, self.Az)
        
        
    def configure_inputs(self, enabled1=None, coupling1=None, range1=None, filter1=None, enabled2=None, coupling2=None, range2=None, filter2=None):
        """
        :param channel: 'CH_A' or 'CH_B'. Create two InputConfig classes for both
        :param coupling: 'AC' or 'DC'
        :param input_range: Input range in volts. rounds down to the closest value
                            provided by AlazarInputControl
        :param filter_above_20MHz: if True, enable the 20MHz BW filter
        """
        
        if enabled1 is not None: self.config.ch1_enabled = enabled1
        if coupling1 is not None: self.config.ch1_coupling= coupling1
        if range1 is not None: self.config.ch1_range = range1
        if filter1 is not None: self.config.ch1_filter

        if enabled2 is not None: self.config.ch2_enabled = enabled2
        if coupling2 is not None: self.config.ch2_coupling= coupling2
        if range2 is not None: self.config.ch2_range = range2
        if filter2 is not None: self.config.ch2_filter
        
        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch1_range <= voltage:
                if self.config.ch1_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch1_range = voltage
                ch1_range_value=value
                break
        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch2_range <= voltage:
                if self.config.ch2_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch2_range = voltage
                ch2_range_value=value
                break

        if self.config.ch1_enabled:  
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_A"], AlazarConstants.input_coupling[self.config.ch1_coupling], ch1_range_value, U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_A"], AlazarConstants.input_filter[self.config.ch1_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)
        if self.config.ch2_enabled:  
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_B"], AlazarConstants.input_coupling[self.config.ch2_coupling], ch2_range_value, U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_B"], AlazarConstants.input_filter[self.config.ch2_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)


    def configure_buffers(self,samplesPerRecord=None,recordsPerBuffer=None,recordsPerAcquisition=None,bufferCount=None):
        if samplesPerRecord is not None: self.config.samplesPerRecord=samplesPerRecord
        if recordsPerBuffer is not None: self.config.recordsPerBuffer=recordsPerBuffer
        if recordsPerAcquisition is not None: self.config.recordsPerAcquisition=recordsPerAcquisition
        if bufferCount is not None: self.config.bufferCount = bufferCount
        
        self.config.channelCount=0
        channel=0        #Create channel flag
        if self.config.ch1_enabled: 
            channel= channel | 1
            self.config.channelCount+=1
        if self.config.ch2_enabled: 
            channel= channel | 2
            self.config.channelCount+=1
        
        pretriggers=C.c_long(0) #no pretriggering support for now
        flags = U32 (513) #ADMA flags, should update to be more general
        
        ret = self.Az.AlazarSetRecordSize(self.handle,U32(0),U32(self.config.samplesPerRecord))        
        if DEBUGALAZAR: print "Set Record Size:", ret_to_str(ret,self.Az)
        
        ret = self.Az.AlazarBeforeAsyncRead(self.handle,U32(channel),pretriggers,
                                       U32(self.config.samplesPerRecord), 
                                       U32(self.config.recordsPerBuffer),
                                       U32(self.config.recordsPerAcquisition),
                                       flags)
        if DEBUGALAZAR: print "Before Read:", ret_to_str(ret,self.Az)

        #self.config.bytesPerBuffer=(self.config.samplesPerRecord * self.config.recordsPerBuffer * self.config.channelCount)
        self.bufs=[]
        #self.bufpts = []
        #buftype=U8 * self.config.bytesPerBuffer
        #memorySize_samples, bitsPerSample = self.getChannelInfo()
        #self.config.bytesPerSample = (bitsPerSample.value + 7) // 8
        self.config.bytesPerSample = 1
        self.config.bytesPerRecord = self.config.bytesPerSample * self.config.samplesPerRecord
        self.config.bytesPerBuffer = self.config.bytesPerRecord * self.config.recordsPerBuffer * self.config.channelCount

        for i in range(self.config.bufferCount):
            #self.bufs.append(buftype())        #changing to alazar code
            self.bufs.append(DMABuffer(self.config.bytesPerSample, self.config.bytesPerBuffer))
            #self.bufpts.append(C.cast(self.bufs[i], C.POINTER(U8)))
            #for j in range(self.config.bytesPerBuffer):
            #    self.bufs[i].buffer[j]=U8(0)
            #ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[i],U32(self.config.bytesPerBuffer))
        if DEBUGALAZAR: print "Posted buffers: ", ret_to_str(ret,self.Az)
        
        self.arrs=[np.ctypeslib.as_array(b.buffer) for b in self.bufs]
#        for a in self.arrs:
#            for i in range(a.__len__()):
#                a[i]=0

    def post_buffers(self):
        self.config.channelCount=0
        channel=0        #Create channel flag
        if self.config.ch1_enabled: 
            channel= channel | 1
            self.config.channelCount+=1
        if self.config.ch2_enabled: 
            channel= channel | 2
            self.config.channelCount+=1

        pretriggers=C.c_long(0) #no pretriggering support for now
        flags = U32 (513) #ADMA flags, should update to be more general
            
        ret = self.Az.AlazarBeforeAsyncRead(self.handle,U32(channel),pretriggers,
                                       U32(self.config.samplesPerRecord), 
                                       U32(self.config.recordsPerBuffer),
                                       U32(self.config.recordsPerAcquisition),
                                       flags)
        for i in range (self.config.bufferCount):
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[i].addr,U32(self.bufs[i].size_bytes))
            #print ret, self.config.bytesPerBuffer
            if ret != 512:
                print "Failed to post Buffer", ret_to_str(ret, self.Az)
           
    def acquire_avg_data(self, excise=None):
        self.post_buffers()
        avg_data1=np.zeros(self.config.samplesPerRecord,dtype=float)
        avg_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        ret = self.Az.AlazarStartCapture(self.handle)
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1      
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            #print buf_idx
            #print np.shape(self.arrs[buf_idx])
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            for n in range(self.config.recordsPerBuffer):
                if self.config.ch1_enabled: avg_data1+=self.arrs[buf_idx][(n*self.config.samplesPerRecord):((n+1)*self.config.samplesPerRecord)]
                if self.config.ch2_enabled: avg_data2+=self.arrs[buf_idx][(n+self.config.recordsPerBuffer)*self.config.samplesPerRecord:(n+self.config.recordsPerBuffer+1)*self.config.samplesPerRecord]
            #plot(self.arrs[buf_idx])
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        avg_data1/=float(self.config.recordsPerAcquisition)
        avg_data2/=float(self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        avg_data1-=128.
        avg_data1*=(self.config.ch1_range/128.)
        avg_data2-=128.
        avg_data2*=(self.config.ch2_range/128.)
        if not self.config.ch2_enabled: avg_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        tpts=np.arange(self.config.samplesPerRecord)/float(self.config.sample_rate*1e3)
        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            return tpts[excise[0]:excise[1]],avg_data1[excise[0]:excise[1]],avg_data2[excise[0]:excise[1]]
        else:
            return tpts,avg_data1,avg_data2

    #added by ds on 4/22/2015
    def acquire_data_by_record(self, prep_function=None, start_function=None,excise=None):
        """Acquire average data, but keep the records aligned
           @start_function:  a callback function to start the AWG's or whatever is doing the triggering
           @excise: (start,stop) range to clip the data out
        """
        self.post_buffers()
        num_chs = 0
        if self.config.ch1_enabled: num_chs+=1
        if self.config.ch2_enabled: num_chs+=1
        data=np.zeros(num_chs*self.config.samplesPerRecord*self.config.recordsPerAcquisition,dtype=float)
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        if prep_function is not None: prep_function()
        ret = self.Az.AlazarStartCapture(self.handle)
        if start_function is not None: start_function()
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        currentIndex=0
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount

            buffersCompleted+=1
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            #print buf_idx
            #print np.shape(self.arrs[buf_idx])
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break
            nextIndex=currentIndex+num_chs*self.config.samplesPerRecord*self.config.recordsPerBuffer
            data[currentIndex:nextIndex]=self.arrs[buf_idx]
            currentIndex=nextIndex
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.bufs[buf_idx].size_bytes))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)

        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        data-=128.
        data*=(self.config.ch1_range/128.)
        if num_chs == 2:
            data1,data2=data.reshape((num_chs,self.config.recordsPerAcquisition,self.config.samplesPerRecord))
        else:
            data1=data.reshape((self.config.recordsPerAcquisition,self.config.samplesPerRecord))
            data2=np.zeros((self.config.recordsPerAcquisition,self.config.samplesPerRecord))
        tpts=np.arange(self.config.samplesPerRecord)/float(self.config.sample_rate*1e3)

        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            tpts=tpts[excise[0]:excise[1]]
            data1=data1[:][excise[0]:excise[1]]
            data2=data2[:][excise[0]:excise[1]]
        return tpts,data1,data2


    #added by ds on 4/17/2015
    def acquire_avg_data_by_record(self, prep_function = None, start_function=None,excise=None):
        """Acquire average data, but keep the records aligned
           @start_function:  a callback function to start the AWG's or whatever is doing the triggering
           @excise: (start,stop) range to clip the data out
        """
        self.post_buffers()
        num_chs = 0
        if self.config.ch1_enabled: num_chs+=1
        if self.config.ch2_enabled: num_chs+=1
        avg_data=np.zeros(num_chs*self.config.samplesPerRecord*self.config.recordsPerBuffer,dtype=float)
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        if prep_function is not None: prep_function()
        ret = self.Az.AlazarStartCapture(self.handle)
        if start_function is not None: start_function()
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            #print buf_idx
            #print np.shape(self.arrs[buf_idx])
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break
            avg_data+=self.arrs[buf_idx]
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        avg_data/=float(buffersPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        avg_data-=128.

        if num_chs == 2:
            avg_data1,avg_data2=avg_data.reshape((num_chs,self.config.recordsPerBuffer,self.config.samplesPerRecord))
        else:
            avg_data1=avg_data.reshape((self.config.recordsPerBuffer,self.config.samplesPerRecord))
            avg_data2=np.zeros((self.config.recordsPerBuffer,self.config.samplesPerRecord))
        avg_data1*=(self.config.ch1_range/128.)
        avg_data2*=(self.config.ch2_range/128.)
        tpts=np.arange(self.config.samplesPerRecord)/float(self.config.sample_rate*1e3)

        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            tpts=tpts[excise[0]:excise[1]]
            avg_data1=avg_data1.T[excise[0]:excise[1]].T
            avg_data2=avg_data2.T[excise[0]:excise[1]].T
        return tpts,avg_data1,avg_data2

    #added by ds on 4/17/2015
    def acquire_singleshot_data(self, prep_function=None,start_function=None, excise=None):
        """Acquire a single number (sum of data in excise window) for each record
           @start_function:  a callback function to start the AWG's or whatever is doing the triggering
           @excise: (start,stop) range to clip the data out if None uses whole record
        """
        self.post_buffers()
        num_chs = 0
        if self.config.ch1_enabled: num_chs+=1
        if self.config.ch2_enabled: num_chs+=1
        if excise is None:
            excise=(0,self.config.samplesPerRecord)
        ss_data=np.zeros((2,self.config.recordsPerAcquisition),dtype=np.int32)
        buffersCompleted=0
        recordsCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        if prep_function is not None: prep_function()
        ret = self.Az.AlazarStartCapture(self.handle)
        if start_function is not None: start_function()
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            #print buf_idx
            #print np.shape(self.arrs[buf_idx])
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)

            for n in range(self.config.recordsPerBuffer):
                for ch in range(num_chs):
                    ss_data[ch][recordsCompleted]=np.sum(self.arrs[buf_idx][((n+ch*self.config.recordsPerBuffer)*self.config.samplesPerRecord)+excise[0]:((n+ch*self.config.recordsPerBuffer)*self.config.samplesPerRecord)+excise[1]])
                recordsCompleted+=1
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)

        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        return ss_data[0],ss_data[1]


    #this takes single shot data
    def acquire_singleshot_heterodyne_data(self, IFreq, excise=None):
        self.post_buffers()
        single_data1=np.zeros((2,self.config.recordsPerAcquisition),dtype=float)
        single_data2=np.zeros((2,self.config.recordsPerAcquisition),dtype=float)
        num_pts = self.config.samplesPerRecord
        tpts=np.arange(num_pts)/float(self.config.sample_rate*1e3)        
        cosdata = cos(2*pi*IFreq*tpts)
        sindata = sin(2*pi*IFreq*tpts)
        single_record1 = 0*tpts
        single_record2 = 0*tpts
        recordsCompleted=0
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        ret = self.Az.AlazarStartCapture(self.handle)
        
        if DEBUGALAZAR: 
            print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: 
            print "Buffers per Acquisition: ", buffersPerAcquisition
            
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: 
                print "Waiting for buffer ", buffersCompleted
                
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1           
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            for n in range(self.config.recordsPerBuffer):
                if self.config.ch1_enabled: 
                    single_record1=self.arrs[buf_idx][n*self.config.samplesPerRecord:(n+1)*self.config.samplesPerRecord]
                    single_record1 = (single_record1-128.)*(self.config.ch1_range/128.)                 
                    single_data1[0,recordsCompleted] = (2*sum(cosdata*single_record1)/num_pts)
                    single_data1[1,recordsCompleted] = (2*sum(sindata*single_record1)/num_pts)**2
                if self.config.ch2_enabled: 
                    single_record2=self.arrs[buf_idx][(n+self.config.recordsPerBuffer)*self.config.samplesPerRecord:(n+self.config.recordsPerBuffer+1)*self.config.samplesPerRecord]
                    single_record2 = (single_record2-128)*(self.config.ch2_range/128.)                    
                    single_data2[0,recordsCompleted] = (2*sum(cosdata*single_record2)/num_pts)
                    single_data2[1,recordsCompleted] = (2*sum(sindata*single_record2)/num_pts)
                recordsCompleted+=1
            #plot(self.arrs[buf_idx])
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if not self.config.ch2_enabled: single_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        
        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            return tpts[excise[0]:excise[1]],single_data1[excise[0]:excise[1]],single_data2[excise[0]:excise[1]]
        else:
            return tpts,single_data1,single_data2,single_record1
            
    #this takes single shot data and does not process it!
    #Exports the entire 2D array so don't take to many points!
    def acquire_singleshot_data2(self, excise=None):
        self.post_buffers()
        single_data1=np.zeros((self.config.recordsPerAcquisition,self.config.samplesPerRecord),dtype=float)
        single_data2=np.zeros((self.config.recordsPerAcquisition,self.config.samplesPerRecord),dtype=float)
        tpts=np.arange(self.config.samplesPerRecord)/float(self.config.sample_rate*1e3)        
        recordsCompleted=0
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        ret = self.Az.AlazarStartCapture(self.handle)
        
        if DEBUGALAZAR: 
            print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: 
            print "Buffers per Acquisition: ", buffersPerAcquisition
            
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: 
                print "Waiting for buffer ", buffersCompleted
                
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1           
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            for n in range(self.config.recordsPerBuffer):
                if self.config.ch1_enabled: 
                    single_data1[recordsCompleted]=self.arrs[buf_idx][n*self.config.samplesPerRecord:(n+1)*self.config.samplesPerRecord]
                    
                if self.config.ch2_enabled: 
                    single_data2[recordsCompleted]=self.arrs[buf_idx][(n+self.config.recordsPerBuffer)*self.config.samplesPerRecord:(n+self.config.recordsPerBuffer+1)*self.config.samplesPerRecord]
                    
                recordsCompleted+=1
            #plot(self.arrs[buf_idx])
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        
        single_data1-= 128.
        single_data2-= 128.
        single_data1*=(self.config.ch1_range/128.)
        single_data2*=(self.config.ch2_range/128.)
        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            return tpts[excise[0]:excise[1]],single_data1[excise[0]:excise[1]],single_data2[excise[0]:excise[1]]
        else:
            return tpts,single_data1,single_data2
        
    #This takes the phase from channel 2 and compensates the channel 1 signal        
    def acquire_phase_compensated_quadratures(self, IFreq, excise=None):
        assert self.config.ch2_enabled
        self.post_buffers()
        avg_cos = 0.0
        avg_sin = 0.0
        phase_array=np.zeros(self.config.recordsPerAcquisition, dtype=float)
        avg_data1=np.zeros(self.config.samplesPerRecord,dtype=float)
        avg_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        acq_time = self.config.samplesPerRecord / (self.config.sample_rate * 1e3)
        tpts=np.arange(self.config.samplesPerRecord)/float(self.config.sample_rate*1e3)
        tpts_len = len(tpts)
        cospts=cos(2.*pi*IFreq*tpts)
        sinpts=sin(2.*pi*IFreq*tpts)
        #avg_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        ret = self.Az.AlazarStartCapture(self.handle)
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        rec_no = 0
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount
            buffersCompleted+=1           
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
            if ret != 512:
                print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            for n in range(self.config.recordsPerBuffer):
                # starting modifying here
                o_ch1_pts = self.arrs[buf_idx][n*self.config.samplesPerRecord:(n+1)*self.config.samplesPerRecord]
                o_ch2_pts = self.arrs[buf_idx][(n+self.config.recordsPerBuffer)*self.config.samplesPerRecord:(n+self.config.recordsPerBuffer+1)*self.config.samplesPerRecord]
                #print np.mean(ch1_pts)
                ch1_pts = o_ch1_pts - 128.
                ch1_pts*=(self.config.ch1_range/128.)
                ch2_pts = o_ch2_pts - 128.
                ch2_pts*=(self.config.ch2_range/128.)
            
                cos_quad1=2.*np.sum(cospts*ch1_pts)/tpts_len
                sin_quad1=2.*np.sum(sinpts*ch1_pts)/tpts_len
                cos_quad2=2.*np.sum(cospts*ch2_pts)/tpts_len
                sin_quad2=2.*np.sum(sinpts*ch2_pts)/tpts_len           
                
                #rec_no = n + buffersCompleted*self.config.recordsPerBuffer
                
                phase_array[rec_no] = np.arctan2(cos_quad2, sin_quad2)
                #phase_array[rec_no] = 0.0
                
                #avg_cos += cos_quad1
                #avg_sin += sin_quad1
                avg_cos += cos_quad1 * np.cos(phase_array[rec_no] - phase_array[0])+sin_quad1 * np.sin(phase_array[rec_no] - phase_array[0])
                avg_sin += sin_quad1 * np.cos(phase_array[rec_no] - phase_array[0])-cos_quad1 * np.sin(phase_array[rec_no] - phase_array[0])
                rec_no += 1
                avg_data1+=ch1_pts
                avg_data2+=ch2_pts
            #plot(self.arrs[buf_idx])
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead, PostAsyncBuffer: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        avg_data1/=float(self.config.recordsPerAcquisition)
        avg_data2/=float(self.config.recordsPerAcquisition)        
        avg_data1-=128.
        avg_data1*=(self.config.ch1_range/128.)
        avg_data2-=128.
        avg_data2*=(self.config.ch2_range/128.)
        avg_cos/=float(self.config.recordsPerAcquisition)
        avg_sin/=float(self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
               
        if not self.config.ch2_enabled: avg_data2=np.zeros(self.config.samplesPerRecord,dtype=float)
        
        if DEBUGALAZAR: print "Acquisition finished."
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        if excise is not None:
            return tpts[excise[0]:excise[1]],avg_data1[excise[0]:excise[1]],avg_data2[excise[0]:excise[1]], phase_array
        else:
            return tpts,avg_data1,avg_data2,np.sqrt(avg_cos**2+avg_sin**2), phase_array

    def argselectdomain(self,xdata,domain):
        ind=np.searchsorted(xdata,domain)
        return ind[0],ind[1]
            
    def acquire_cavity_ringdown_data(self,excise=None,frequency_window=None):
        if self.config.ch2_enabled or not self.config.ch1_enabled: 
            raise ValueError("Channel 1 must be enabled and Channel 2 must NOT be enabled in cavity ringdown mode!")
        self.post_buffers()
        #preprocessing
        nyquist=1./2.*float(self.config.sample_rate*1e3)
        s2=np.sqrt(2.)
        if excise is None:
            excise=(0,self.config.samplesPerRecord)

        freqs=np.arange(0,2*nyquist,2*nyquist/(excise[1]-excise[0]))
        if frequency_window is not None:        
            fex=self.argselectdomain(freqs,frequency_window)
        else: fex=(0,-1)
        freqs2=freqs[fex[0]:fex[1]]

        f0_data=np.zeros(self.config.recordsPerAcquisition,dtype=float)
        kappa_data=np.zeros(self.config.recordsPerAcquisition,dtype=float)

        a=time.time()
        buffersCompleted=0
        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
        print "|"+("  ")*10+" |"
        print "|",
        f0,kappa=0,0        
        ret = self.Az.AlazarStartCapture(self.handle)
        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
        while (buffersCompleted < buffersPerAcquisition):
            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
            buf_idx = buffersCompleted % self.config.bufferCount
            if buffersCompleted % (buffersPerAcquisition/10.) ==0: print "-",
            buffersCompleted+=1           
            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx],U32(self.config.timeout))
            if DEBUGALAZAR: print "WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
            if ret != 512:
                print "Abort AsyncRead: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
            for n in xrange(self.config.recordsPerBuffer):
                clipped_psd=np.absolute(fft(self.arrs[buf_idx][n*self.config.samplesPerRecord+excise[0]:n*self.config.samplesPerRecord+excise[1]])[fex[0]:fex[1]])
                maxloc=np.argmax(clipped_psd)
#                    for i in xrange(maxloc,len(fs2)):
#                        if fs2[i]<fs2[maxloc]/s2: break
                x1=freqs2[maxloc-2]
                x2=freqs2[maxloc]
                x3=freqs2[maxloc+2]
                y1=clipped_psd[maxloc-2]
                y2=clipped_psd[maxloc]
                y3=clipped_psd[maxloc+2]
                f0=(x1**2*y1*(y2 - y3) + x3**2*(y1 - y2)*y3 + x2**2*y2*(-y1 + y3))/(2.*(x1*y1*(y2 - y3) + x3*(y1 - y2)*y3 + x2*y2*(-y1 + y3)))
                kappa=np.sqrt(-(((x1 - x2)**4*y1**2*y2**2 - 2*(x1 - x2)**2*y1*y2*((x1 - x3)**2*y1 + (x2 - x3)**2*y2)*y3 + ((x1 - x3)**2*y1 - (x2 - x3)**2*y2)**2*y3**2)/(x1*y1*(y2 - y3) + x3*(y1 - y2)*y3 + x2*y2*(-y1 + y3))**2))
                f0_data[(buffersCompleted-1)*self.config.recordsPerBuffer+n]=f0#freqs2[maxloc]
                kappa_data[(buffersCompleted-1)*self.config.recordsPerBuffer+n]=kappa#2.*(freqs2[i]-freqs2[maxloc])
            #plot(self.arrs[buf_idx])
            #if buffersCompleted < buffersPerAcquisition:            
            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx],U32(self.config.bytesPerBuffer))
            if ret != 512:
                print "Abort AsyncRead: ", ret_to_str(ret,self.Az)            
                ret = self.Az.AlazarAbortAsyncRead(self.handle)
                break       
            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        if DEBUGALAZAR: print "Acquisition finished."
        print "|"
        print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
        print "time taken: %f s" % (time.time()-a)
        ret = self.Az.AlazarAbortAsyncRead(self.handle)
        return f0_data,kappa_data
        
    def capture_buffer_async(self, buf_idx=0):
        ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx].addr,U32(self.config.timeout))
        if ret != 512:
            print "Abort AsyncRead, WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
            ret2 = self.Az.AlazarAbortAsyncRead(self.handle)
            #print ret_to_str(ret, self.Az)
            return ret 
        ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
        if ret != 512:
            print "Abort AsyncRead, PostBuffer: ", ret_to_str(ret,self.Az)            
            ret2 = self.Az.AlazarAbortAsyncRead(self.handle)
            #print ret_to_str(ret, self.Az)
            return ret         
        return ret

    def getChannelInfo(self):
        '''Get the on-board memory in samples per channe and sample size in bits per sample'''
        memorySize_samples = U32(0)
        bitsPerSample = U8(0)
        self.Az.AlazarGetChannelInfo(self.handle, byref(memorySize_samples), byref(bitsPerSample))
        return (memorySize_samples, bitsPerSample)

        
    def repost_buffer(self, buf_idx=0):
        ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx].addr,U32(self.config.bytesPerBuffer))
        if ret != 512:
            print "Abort AsyncRead: ", ret_to_str(ret,self.Az)            
            ret = self.Az.AlazarAbortAsyncRead(self.handle)
        return ret
        
    def separate_channel_data(self, buf_idx=0):
        ch1_data, ch2_data = None, None
        for ch1_record, ch2_record in self.get_records(buf_idx):
            if self.config.ch1_enabled:
                if ch1_data is not None:
                    ch1_data = np.vstack((ch1_data, ch1_record))
                else:
                    ch1_data = ch1_record
            if self.config.ch2_enabled:
                if ch2_data is not None:
                    ch2_data = np.vstack((ch2_data, ch2_record))
                else:
                    ch2_data = ch2_record
        return ch1_data, ch2_data
            
    
    def get_records(self, buf_idx=0):
        recordsPerBuffer = self.config.recordsPerBuffer
        samplesPerRecord = self.config.samplesPerRecord
        buf = self.arrs[buf_idx]
        for n in range(recordsPerBuffer):
            ch1_data = buf[n*samplesPerRecord:(n+1)*samplesPerRecord]
            ch2_data = buf[(n+recordsPerBuffer)*samplesPerRecord:(n+recordsPerBuffer+1)*samplesPerRecord]
            yield ch1_data, ch2_data
            
    def assert_az(self, retCode, buffers_acquired, name):
        self.Az.AlazarErrorToText.restype = C.c_char_p
        if retCode != 512:
            raise AlazarException("::".join(map(str, (name, buffers_acquired,
                                                      retCode, self.Az.AlazarErrorToText(U32(retCode))))))

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            print self.config.samples_per_buffer, 'Z'
            raise AttributeError('Neither Alazar nor AlazarConfig contains item ' + item)

    def acquire_parallel(self, worker_cls, worker_args, result_shape, plot=False):
        """
        :param worker: Function which the subordinate threads execute as their target
        :param proc_fun: Function used by the subordinate threads to process their buffers
        :param result_shape: Shape of the buffer which is the result of the entire acquisition.
        """
        from multiprocessing import Array, Value, Event
        from slab.plotting import ScriptPlotter
        import time

        #acquire_buffer_time = self.samples_per_buffer / (self.samples_per_second)
        #print 'Acquire buffer time %.2e' % acquire_buffer_time
        #print 'Inter-buffer time %.2e' % self.seconds_per_buffer
        #print 'Duty Cycle', acquire_buffer_time / self.seconds_per_buffer

        try:
            # Initialize buffers
            buffers = [Array(U8, self.bytes_per_buffer) for _ in range(self.bufferCount)]
            for b in buffers:
                ret = self.Az.AlazarPostAsyncBuffer(self.handle, b.get_obj(), U32(self.bytes_per_buffer))
                self.assert_az(ret, 0, 'Initial Post Buffer')

            res_buffer = Array(C.c_longdouble, result_shape)

            # Initialize threads
            bufs_merged = Value(U32, 1)
            buf_ready_events = [Event() for _ in range(self.bufferCount)]
            buf_post_events = [Event() for _ in range(self.bufferCount)]
            workers = [worker_cls(*(worker_args + (self.config, b, bre, bpe, res_buffer, bufs_merged)))
                       for b, bre, bpe in zip(buffers, buf_ready_events, buf_post_events)]

            for w in workers:
                w.start()
            time.sleep(1)

            import atexit
            atexit.register(lambda: [w.terminate() for w in workers])


            # Initialize things used during capture
            if plot:
                plotter = ScriptPlotter()
                plotter.init_plot('Data', rank=1, accum=False)
            buffers_acquired, buffers_completed, plot_count = 0, 0, 0
            start_time = time.time()

            # Begin capture
            ret = self.Az.AlazarStartCapture(self.handle)
            self.assert_az(ret, 0, "Start Capture")
            unready_count = 0
            while buffers_completed < self.buffers_per_acquisition:

                # Post all completed buffers
                while buf_post_events[buffers_completed % self.bufferCount].is_set():
                    buf_post_events[buffers_completed % self.bufferCount].clear()
                    buf = buffers[buffers_completed % self.bufferCount]
                    with buf.get_lock():
                        ret = self.Az.AlazarPostAsyncBuffer(self.handle, buf.get_obj(), U32(self.bytes_per_buffer))
                        self.assert_az(ret, buffers_acquired, 'Post Buffer')
                    buffers_completed += 1

                    # Current buffer rotates in a ring
                buf_idx = buffers_acquired % self.bufferCount
                buf = buffers[buf_idx]

                # Pull data to buffer
                with buf.get_lock():
                    ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle, buf.get_obj(), U32(self.timeout))
                    if ret == 573:
                        unready_count += 1
                        continue # BufferNotReady, go back and try to post some buffers.
                    else:
                        self.assert_az(ret, buffers_acquired, 'Wait Buffer Complete')

                buffers_acquired += 1

                # Tell worker thread to begin processing
                buf_ready_events[buf_idx].set()

                # If a second has elapsed, replot the avg_buffer
                seconds_per_plot = 1.
                if (time.time() - start_time) / seconds_per_plot > plot_count:
                    if plot:
                        with res_buffer.get_lock():
                            plotter.msg(buffers_acquired, buffers_completed, bufs_merged.value)
                            plotter.plot(np.frombuffer(res_buffer.get_obj()), 'Data')
                        plot_count += 1
                    else:
                        print buffers_acquired, buffers_completed
                        plot_count += 1
        finally:
            pass
        
#            self.Az.AlazarAbortAsyncRead(self.handle)
#            if buffers_completed:
#                final_time = time.time()
#                print 'Unready Count', unready_count
#                total_time = final_time - start_time
#                print 'Total time', total_time
#                actual_time_per_buffer = total_time / buffers_completed
#                print 'Time per buffer %.2e' % actual_time_per_buffer
#                errf = lambda a, b: abs(a - b) / min(a, b)
#                print 'Perceived overhead %.1f%%' % (errf(actual_time_per_buffer, seconds_per_buffer) * 100)
#            else:
#                print 'No buffers completed'

        res = np.frombuffer(res_buffer.get_obj())
        return res

    def acquire_average_parallel(self, averages, **kwargs):
        return self.acquire_parallel(AverageWorker, (), self.samples_per_buffer)

    def acquire_homodyne_single_shot(self, frequency):
        sin_arr = np.sin(frequency * np.linspace(0, self.seconds_per_record, self.samples_per_record))
        cos_arr = np.cos(frequency * np.linspace(0, self.seconds_per_record, self.samples_per_record))
        records_per_buffer = self.records_per_buffer
        proc_fun = lambda buf_arr: [(np.sum(sin_arr * arr), np.sum(cos_arr * arr)) for arr in np.split(buf_arr, records_per_buffer)]
        self.acquire_parallel(SingleShotWorker, (proc_fun,), (self.records_per_acquisition, 2))

from multiprocessing import Process

class Worker(Process):
    def __init__(self, az_config, buf, buf_ready_event, buf_post_event, res_buffer, buffers_merged):
        Process.__init__(self)
        self.az_config = az_config
        self.buf = buf
        self.buf_ready_event = buf_ready_event
        self.buf_post_event = buf_post_event
        self.res_buffer = res_buffer
        self.buffers_merged = buffers_merged

class AverageWorker(Worker):
    def run(self):
        arr = np.frombuffer(self.buf.get_obj(), U8)
        avg_buffer_arr = np.frombuffer(self.res_buffer.get_obj(), C.c_longdouble)
        sum_buffer = np.zeros(self.az_config.bytes_per_buffer, np.uint32)
        my_buffers_completed = 0
        for _ in range(self.az_config.buffers_per_worker):
            self.buf_ready_event.wait()
            self.buf_ready_event.clear()
            sum_buffer += arr
            self.buf_post_event.set()
            my_buffers_completed += 1

            if my_buffers_completed % self.az_config.buffers_per_merge == 0:
                with self.avg_buffer.get_lock():
                    n = self.buffers_merged.value
                    avg_buffer_arr *= (n - 1.) / n
                    avg_buffer_arr += sum_buffer / (n * self.az_config.buffers_per_merge)
                    self.buffers_merged.value += 1
                sum_buffer.fill(0)

class SingleShotWorker(Worker):
    def __init__(self, proc_fun, *args):
        self.proc_fun = proc_fun
        Worker.__init__(self, *args)

    def run(self):
        arr = np.frombuffer(self.buf.get_obj(), U8)
        results = []
        my_buffers_completed = 0
        iterations = self.az_config.buffers_per_worker
        for _ in range(iterations):
            self.buf_ready_event.wait()
            self.buf_ready_event.clear()
            results.extend(self.proc_fun(arr))
            self.buf_post_event.set()
            my_buffers_completed += 1

        with self.res_buffer.get_lock():
            n = self.buffers_merged.value
            print iterations, len(results), n, len(self.res_buffer)
            self.res_buffer[(n - 1) * iterations:n * iterations] = np.array(results)
            self.buffers_merged.value += 1



#    def acquire_cavity_ringdown_data_fftw(self,excise=None,frequency_window=None):
#        if self.config.ch2_enabled: 
#            raise ValueError("Channel 2 must not be enabled in cavity ringdown mode!")
#        self.post_buffers()
#        #preprocessing
#        nyquist=1./2.*float(self.config.sample_rate*1e3)
#        s2=np.sqrt(2.)
#        if excise is None:
#            excise=(0,self.config.samplesPerRecord)
#
#        freqs=np.arange(0,2*nyquist,2*nyquist/(excise[1]-excise[0]))
#        if frequency_window is not None:        
#            fex=self.argselectdomain(freqs,frequency_window)
#        else: fex=(0,-1)
#        freqs2=freqs[fex[0]:fex[1]]
#
#        f0_data=np.zeros(self.config.recordsPerAcquisition,dtype=float)
#        kappa_data=np.zeros(self.config.recordsPerAcquisition,dtype=float)
#
#        bufcopies=[]
#        fftbufs=[]
#        plans=[]
#        for i in range(self.config.bufferCount):
#            bufcopies.append(np.zeros(self.config.samplesPerRecord,dtype=float))
#            fftbufs.append(np.zeros(self.config.samplesPerRecord,dtype=np.complex128))
#            plans.append(fftw3.Plan(bufcopies[i],fftbufs[i],direction='forward',create_plan=True,nthreads=7))
#        
#        a=time.time()
#        buffersCompleted=0
#        buffersPerAcquisition=self.config.recordsPerAcquisition/self.config.recordsPerBuffer
#        print "go"
#        print "Taking %d data points." % buffersPerAcquisition
#        print "|"+("  ")*10+" |",
#        print "|",
#        
#        ret = self.Az.AlazarStartCapture(self.handle)
#        if DEBUGALAZAR: print "Start Capture: ", ret_to_str(ret,self.Az)
#        if DEBUGALAZAR: print "Buffers per Acquisition: ", buffersPerAcquisition
#        while (buffersCompleted < buffersPerAcquisition):
#            if DEBUGALAZAR: print "Waiting for buffer ", buffersCompleted
#            buf_idx = buffersCompleted % self.config.bufferCount
#            if buffersCompleted % (buffersPerAcquisition/10.) ==0: print "-",
#            buffersCompleted+=1           
#            ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle,self.bufs[buf_idx],U32(self.config.timeout))
#            if DEBUGALAZAR: print "WaitAsyncBuffer: ", ret_to_str(ret,self.Az)            
#            if ret_to_str(ret,self.Az) != "ApiSuccess":
#                print "Abort AsyncRead: ", ret_to_str(ret,self.Az)            
#                ret = self.Az.AlazarAbortAsyncRead(self.handle)
#                break       
#            for n in range(self.config.recordsPerBuffer):
#                if self.config.ch1_enabled: 
#                    np.copyto(bufcopies[buf_idx],self.arrs[buf_idx][n*self.config.samplesPerRecord+excise[0]:n*self.config.samplesPerRecord+excise[1]],casting='unsafe')
#                    plans[buf_idx].execute()
#                    np.absolute(fftbufs,out=fftbufs)
#                    #fs=abs(fft(self.arrs[buf_idx][n*self.config.samplesPerRecord+excise[0]:n*self.config.samplesPerRecord+excise[1]]))
#                    #fs2=fs[fex[0]:fex[1]]
#                    maxloc=np.argmax(fftbufs)
#                    for i in xrange(maxloc,len(fs2)):
#                        if fftbufs[i]<fftbufs[maxloc]/s2: break
#                    f0_data[(buffersCompleted-1)*self.config.recordsPerBuffer+n]=freqs2[maxloc]
#                    kappa_data[(buffersCompleted-1)*self.config.recordsPerBuffer+n]=2.*(freqs2[i]-freqs2[maxloc])
#            #plot(self.arrs[buf_idx])
#            #if buffersCompleted < buffersPerAcquisition:            
#            ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[buf_idx],U32(self.config.bytesPerBuffer))
#            if ret_to_str(ret,self.Az) != "ApiSuccess":
#                print "Abort AsyncRead: ", ret_to_str(ret,self.Az)            
#                ret = self.Az.AlazarAbortAsyncRead(self.handle)
#                break       
#            if DEBUGALAZAR: print "PostAsyncBuffer: ", ret_to_str(ret,self.Az) 
#        if DEBUGALAZAR: print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
#        if DEBUGALAZAR: print "Acquisition finished."
#        print "|"
#        print "buffersCompleted: %d, self.config.recordsPerAcquisition: %d" % (buffersCompleted, self.config.recordsPerAcquisition)
#        print "time taken: %f s" % (time.time()-a)
#        ret = self.Az.AlazarAbortAsyncRead(self.handle)
#        return f0_data,kappa_data



if __name__ == "__main__":
    ac=AlazarConfig1(fedit(AlazarConfig1._form_fields_))

    card=Alazar(ac)    
    card.configure()
    res = card.acquire_avg_data()
    print res

    print "Buffer bytes: ", card.config.bytesPerBuffer
    print "Buffer length: ",card.arrs[0].__len__()

    # ion()
    # figure(1)
    # plot(card.arrs[0])
    # figure(2)
    # plot(card.arrs[0][:2048])
    # figure(3)
    # plot(card.arrs[-1][:2048])
    # show()
