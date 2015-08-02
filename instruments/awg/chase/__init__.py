__author__ = 'Dave'

import ctypes as C
import numpy as np

from slab.instruments import Instrument

CHASE_VERSION = "32bit"

try:
    if CHASE_VERSION == "32bit":

        chase_dll_path = r'C:\_Lib\python\slab\instruments\awg\chase\dax22000_lib_DLL32.dll'
    elif CHASE_VERSION == "64bit":
        chase_dll_path = r'C:\_Lib\python\slab\instruments\awg\chase\dax22000_lib_DLL64.dll'
    chaseDLL = C.CDLL(chase_dll_path)
except:
    print "Warning could not load Chase AWG dll, check that dll located at '%s'" % chase_dll_path


class SegmentStruct(C.Structure):
    """ Segment Structure
    """
    _fields_ = [('SegmentPtr', C.POINTER(C.c_uint16)),
                ('NumPoints', C.c_uint32),
                ('NumLoops', C.c_uint32),
                ('TrigEn', C.c_uint32)]


class DAx22000Segment:
    def __init__(self, waveform, loops=1, triggered=False):
        self.waveform = waveform
        self.loops = loops
        self.triggered = triggered


class DAx22000(Instrument):
    def __init__(self, name, address='', enabled=True):
        Instrument.__init__(self, name, address, enabled)
        self.addr = int(address)
        self.dev = None

        num_devices = chaseDLL.DAx22000_GetNumCards()

        if self.addr > num_devices:
            raise Exception("Tried to load DAx22000 id=%d but only %d devices found!" % (self.addr, num_devices))


    def initialize(self, ext_clk_ref=False):
        x = chaseDLL.DAx22000_Open(C.c_int32(self.addr))
        if x != 0: return x

        x = chaseDLL.DAx22000_Ext10MHz(C.c_int32(self.addr), C.c_int32(ext_clk_ref))
        if x != 0: return x

        x = chaseDLL.DAx22000_Initialize(C.c_int32(self.addr))
        if x != 0: return x

        return 0

    def create_single_segment(self, chan, loops, begin_val, end_val, waveform, triggered):
        arr = (C.c_ushort * len(waveform))(*(np.array(waveform, dtype=np.uint16)))

        return chaseDLL.DAx22000_CreateSingleSegment(C.c_uint32(self.addr),
                                                     C.c_uint32(chan),
                                                     C.c_uint32(len(waveform)),
                                                     C.c_uint32(loops),
                                                     C.c_uint32(begin_val),
                                                     C.c_uint32(end_val),
                                                     arr,
                                                     C.c_bool(triggered))

    def create_segments(self, chan, segments, loops=0, begin_val=1047, end_val=2047, triggered=True):

        seglisttype = SegmentStruct * len(segments)

        seglist = seglisttype()

        for ii, seg in enumerate(segments):
            arr = (C.c_uint16 * len(seg.waveform))(*(np.array(seg.waveform, dtype=np.uint16)))
            seglist[ii].SegmentPtr = arr
            seglist[ii].NumPoints = len(seg.waveform)
            seglist[ii].NumLoops = seg.loops
            seglist[ii].TrigEn = seg.triggered

        return chaseDLL.DAx22000_CreateSegments(C.c_uint32(self.addr), C.c_uint32(chan), C.c_uint32(len(segments)),
                                                C.c_uint32(begin_val), C.c_uint32(end_val), seglist, C.c_bool(True))


    def soft_trigger(self):
        return chaseDLL.DAx22000_SoftTrigger(C.c_int32(self.addr))


    def place_mrkr2(self, mod16_cnt):
        return chaseDLL.DAx22000_Place_MRK2(C.c_int32(self.addr), C.c_int32(mod16_cnt))


    def set_clk_freq(self, freq=2.5e9):
        return chaseDLL.DAx22000_SetClkRate(C.c_int32(self.addr), C.c_double(freq))


    def set_ext_trig(self, ext_trig=True):
        return chaseDLL.DAx22000_SelExtTrig(C.c_int32(self.addr), C.c_bool(ext_trig))


    def run(self, trigger_now=False):
        return chaseDLL.DAx22000_Run(C.c_int32(self.addr), C.c_bool(trigger_now))


    def stop(self):
        return chaseDLL.DAx22000_Stop(C.c_int32(self.addr))









