# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:14:11 2011

@author: Phil
"""
from guiqwt.pyplot import *
import ctypes as C
import numpy as np
#import matplotlib.pyplot as mplt
import operator

U8 = C.c_uint8
U32 = C.c_uint32
U32P = C.POINTER(U32)
U32PP = C.POINTER(U32P)
Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
MyAz = C.CDLL(r'C:\Users\Phil\Desktop\AlazarApi\trunk\Debug\MyAlazarApi.dll')
CTHelp = C.CDLL(r'C:\Users\Phil\Desktop\AlazarApi\trunk\Debug\ctypes_help.dll')

def cop(op, lhs, rhs, T):
    if T == None:
        T = type(lhs)
    try:
        return T(op(lhs.value, rhs.value))
    except:
        try:
            return T(op(lhs.value, rhs))
        except: 
            T = type(rhs)
            return T(op(lhs, rhs.value))
    
def cadd(lhs, rhs, T=None):
    return cop(operator.add, lhs, rhs, T)
def csub(lhs, rhs, T=None):
    return cop(operator.sub, lhs, rhs, T)
def cmul(lhs, rhs, T=None):
    return cop(operator.mul, lhs, rhs, T)
def cdiv(lhs, rhs, T=None):
    return cop(operator.div, lhs, rhs, T)
def cmod(lhs, rhs, T=None):
    return cop(operator.mod, lhs, rhs, T)

def int_linterp(x, a1, a2, b1, b2):
    'Moves x from interval [a1, a2] to interval [b1, b2]'
    return int(b1 + ((x - a1) * (b2 - b1)) / float((a2 - a1)))

class Config:
    def save_params(self, f):
        for (name, value) in self.params:
            f.write(name + str(value))

class ConfigStruct(C.Structure):
    def save_params(self, f):
        for (field_name, field_type) in type(self)._fields_:
            f.write(field_name + str(self.__getattribute__(field_name)))


class Card(object):
    def __init__(self, configs):
        self.configs = configs
        self.handle = Az.AlazarGetBoardBySystemID(U32(1), U32(1))
        if not self.handle:
            raise RuntimeError("Board could not be found")
    def save_config(self, fname="configuration.txt"):
        f = open(fname, "w")
        for conf in self.configs:
            conf.save_params(self, f)
        f.close()
    def configure(self):
        for conf in self.configs:
            conf.run_on_card(self.handle)

class ClockConfig(Config):
    source = {"internal" : U32(1),
              "reference": U32(7),
              "60 MHz" : U32(4),
              "1 GHz" : U32(5)}
    sample_rate = [(1,U32(1)), (2,U32(2)), (5,U32(4)), (10,U32(8)), (20,U32(10)),
            (50,U32(12)), (100,U32(14)), (200,U32(16)), (500,U32(18)), (1000,U32(20)),
            (2000,U32(24)), (5000,U32(26)), (10000,U32(28)), (20000,U32(30)), (50000,U32(34)),
            (100000,U32(36)), (250000,U32(43)), (500000,U32(48)), (1000000,U32(53))]
    sample_rate_external = U32(64)
    sample_rate_reference = U32(1000000000)
    clock_edge = {"rising": U32(0),
                  "falling": U32(1)}

    def __init__(self, source, rate, edge="rising"):
        """
        note -- rate is in KHz
        """
        self.sample_rate = rate
        if source in ["internal", "external", "reference"]:
            self.clock_source = source
        else:
            raise ValueError("source must be one of internal, external, or reference")
        if edge in ["rising", "falling"]:
            self.clock_edge = edge
        else:
            raise ValueError("edge must be one of rising or falling")
        self.params = [("source", source), 
                       ("sample_rate", rate),
                       ("edge", edge)]

    def run_on_card(self, handle):
        if self.clock_source == "internal":
            source = ClockConfig.source["internal"]
            for (rate, value) in ClockConfig.sample_rate:
                if rate >= self.sample_rate:
                    if rate > self.sample_rate:
                        print "Warning: sample_rate not found. Using first smaller value", rate, "Khz"
                        self.sample_rate = rate
                    sample_rate = value
                    break
        elif self.clock_source == "external":
            sample_rate = sample_rate_external
            if self.sample_rate < 60000:
                source = ClockConfig.source["60 MHz"]
            elif self.sample_rate < 1000000:
                source = ClockConfig.source["1 GHz"]
            else:
                raise ValueError("Not supported (yet?)")
        else:
            raise ValueError("reference signal not implemented yet")
        Az.AlazarSetCaptureClock(handle, source, sample_rate, ClockConfig.clock_edge[self.clock_edge], U32(0))


class TriggerConfig(Config):
    source = {"CH_A":U32(0), "CH_B":U32(1), "external":U32(2), "disabled":U32(3)}
    operation = {"or":U32(2), "and":U32(3), "xor":U32(4), "and not":U32(5)}
    ext_coupling = {"AC":U32(1), "DC":U32(2)}
    single_op = U32(0)
    engine_1 = U32(0)
    engine_2 = U32(1)
    edge = {"rising":U32(1), "falling":U32(2)}

    def __init__(self, source="CH_A", source2=None, edge="rising", edge2="rising",
                 level=0, level2=0, operation="or", coupling="AC"):
        self.source = TriggerConfig.source[source]
        if source == "external":
            self.coupling=TriggerConfig.source[coupling]
        self.edge = TriggerConfig.edge[edge]
        if level >= -100 and level < 100:
            self.level = U32(int_linterp(level -100, 100, 0, 255))
        else:
            raise ValueError("Level must be value in [-100,100]")
        if source2:
            self.source2 = TriggerConfig.source[source2]
            self.edge2 = TriggerConfig.edge[edge2]
            if level2 >= -100 and level2 < 100:
                self.level2 = U32(int_linterp(level2, -100, 100, 0, 255))
            else:
                raise ValueError("Level2 must be value in [-100,100]")
            self.operation = TriggerConfig.operation[operation]
        else:
            self.source2 = TriggerConfig.source["disabled"]
            self.operation = TriggerConfig.single_op
        self.params = [("source1", source), 
                       ("edge1", edge),
                       ("level1", level),
                       ("source2", source2), 
                       ("edge2", edge2),
                       ("level2", level2),
                       ("operation", operation)]

    def run_on_card(self, handle):
        Az.AlazarSetTriggerOperation(handle, self.operation,
                                     TriggerConfig.engine_1, self.source, self.edge, self.level,
                                     TriggerConfig.engine_2, self.source2, self.edge2, self.level2)
        if self.source == "external":
            Az.AlazarSetExternalTrigger(handle, self.coupling, U32(0))

class InputConfig(Config):
    channel = {"CH_A":U32(1), "CH_B":U32(2)}
    coupling = {"AC":U32(1), "DC":U32(2)}
    input_range = [(0.04,U32(2)), (0.1,U32(5)), (0.2,U32(6)), (0.4,U32(6)),
                   (1,U32(10)), (2,U32(11)), (4,U32(12))]
    bw_filter = {False:U32(0), True:U32(1)}

    def __init__(self, channel, coupling, input_range, filter_above_20MHZ=False):
        self.channel = InputConfig.channel[channel]
        self.coupling = InputConfig.coupling[coupling]
        for (voltage, value) in InputConfig.input_range:
            if input_range <= voltage:
                if input_range < voltage:
                    print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.input_range_voltage = voltage
                self.input_range = value
        self.params = [("channel", channel), ("coupling", coupling), ("input range", self.input_range_voltage)]


class AverageConfig(ConfigStruct):
    """
    Characterizes what data will be averaged and how
    """
    _fields_ = [("preTriggerSamples", U32),
                ("postTriggerSamples", U32),
                ("channelCount", U32),
                ("bufferCount", U32),
                ("recordsPerMeasurement", U32),
                ("measurementsPerRound", U32),
                ("nRounds", U32)]
    def output_array(self):
        samples = self.preTriggerSamples + self.postTriggerSamples
        inner_arr_type = U32 * samples
        outer_arr_type = U32P * self.measurementsPerRound
        ret_arr = outer_arr_type()
        for i in range(self.measurementsPerRound):
            ret_arr[i] = inner_arr_type()
            for j in range(samples):
                ret_arr[i][j] = 0
        #arr_type = (U32 * samples) * self.measurementsPerRound
        return ret_arr
    def run_on_card(self, handle):
        Az.AlazarSetRecordSize(handle,
                               U32(self.preTriggerSamples),
                               U32(self.postTriggerSamples))

defaultConfig = AverageConfig(0, 1024, 1, 4, 250, 1, 3)

class BufferInfo(ConfigStruct):
    _fields_ = [("bitsPerSample", U8),
                ("bytesPerSample", U32),
                ("bytesPerRecord", U32),
                ("bytesPerBuffer", U32),
                ("maxSamplesPerChannel", U32),
                ("samplesPerRecordPerChannel", U32),
                ("samplesPerRecord", U32),
                ("samplesPerBuffer", U32),
                ("samplesPerAcquisition", U32),
                ("recordsPerBuffer", U32),
                ("recordsPerAcquisition", U32),
                ("buffersPerAcquisition", U32),
                ("channelCount", U32)]
   
    def config(self, handle):
        Az.AlazarGetChannelInfo(handle,
                                C.byref(self.maxSamplesPerChannel),
                                C.byref(self.bitsPerSample))
        self.bytesPerSample = U32((self.bitsPerSample.value + 7) / 8)
    def fromAverageConfig(self, ac):
        self.channelCount = ac.channelCount
        self.recordsPerBuffer = ac.recordsPerMeasurement
        self.preTriggerSamples = ac.preTriggerSamples
        self.samplesPerRecordPerChannel= ac.preTriggerSamples + ac.postTriggerSamples
        self.samplesPerRecord = self.channelCount * self.samplesPerRecordPerChannel
        self.samplesPerBuffer = self.samplesPerRecord * self.recordsPerBuffer
        self.bytesPerRecord = self.bytesPerSample * self.samplesPerRecord
        self.bytesPerBuffer = self.bytesPerRecord * self.recordsPerBuffer
        self.buffersPerAcquisition = ac.measurementsPerRound * ac.nRounds
        self.recordsPerAcquisition = self.buffersPrAcquisition * self.recordsPerBuffer
#    def run_on_card(self, handle):
#        Az.AlazarBeforeAsyncRead(handle,
#                                 BufferInfo.channels[self.channelCount],
#                                 U32(-self.preTriggerSamples),
#                                 U32(self.samplesPerRecord),
#                                 U32(self.recordsPerBuffer),
#                                 U32(self.recordsPerAcquisition),
#                                 U32())

#def before_async_read(handle, ac, bi):\
#    channels = {1: U32(1), 2:U32(3)}
#    chan = channels[ac.channelCount]
#    Az.AlazarBeforeAsyncRead(handle, chan, ac.preTriggerSamples,
#                             bi.samplesPerRecord, bi.recordsPerBuffer,
#                             bi.recordsPerAcquisition,)
def test():
    print "Starting Test"
    handle = Az.AlazarGetBoardBySystemID(C.c_uint32(1), C.c_uint32(1))
    if(MyAz.configureBoard(handle)):
        print "Configured"   
        out_array = defaultConfig.output_array()
        #fmt_out_arr = C.cast(out_array, C.POINTER(C.POINTER(U32)))
        MyAz.acquireData(handle, 
                         C.byref(defaultConfig), 
                         out_array)
        print "Acquired"
        return out_array

def config():
    handle = get_handle()
    trig = TriggerConfig()
    clock = ClockConfig()
    ave = defaultConfig
    bufinfo = BufferInfo()
    bufinfo.fromAverageConfig(ave)
    for conf in [trig, clock, ave]: conf.run_on_card(handle)
    
    
def get_handle():
    return Az.AlazarGetBoardBySystemID(U32(1), U32(1))
     
out_arr = test()
#tmp = []
samples = defaultConfig.postTriggerSamples + defaultConfig.preTriggerSamples
#for i in range(samples):
#    tmp.append(out_arr[0][i])
#np.frombuffer(out_arr, np.u)
nparr = np.ctypeslib.as_array(out_arr[0], (samples,))
plot(np.linspace(0, 1, samples), nparr)
show()
    


