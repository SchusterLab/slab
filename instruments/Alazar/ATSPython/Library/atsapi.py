'''Python interface to the AlazarTech SDK.

This module provides a thin wrapper on top of the AlazarTech C
API. All the exported methods directly map to underlying C
functions. Please see the ATS-SDK Guide for detailed specification of
these functions. In addition, this module provides a few classes for
convenience.

Attributes:

  Board: Represents a digitizer. Provides methods for configuration
  and data acquisition

  DMABuffer: Holds a memory buffer suitable for data transfer with
  digitizers.
'''

from ctypes import *
import numpy as np
import os

from sys import version_info
if version_info.major == 2:
    import tkinter as tk
    import tkinter.ttk
    import _thread
elif version_info.major == 3:
    import tkinter as tk
    import tkinter.ttk as ttk
    import _thread as thread

'''Types of clocks that a board can use for acquiring data.
Note: Available sources for a given board form a subset of this
class' members. Please see your board's specification as well as
the ATS-SDK manual for more information.
'''
INTERNAL_CLOCK = 0x1
EXTERNAL_CLOCK = 0x2
FAST_EXTERNAL_CLOCK = 0x2
MEDIUM_EXTERNAL_CLOCK = 0x3
SLOW_EXTERNAL_CLOCK = 0x4
EXTERNAL_CLOCK_AC = 0x5
EXTERNAL_CLOCK_DC = 0x6
EXTERNAL_CLOCK_10MHz_REF = 0x7
INTERNAL_CLOCK_10MHz_REF = 0x8
EXTERNAL_CLOCK_10MHz_PXI = 0xA
INTERNAL_CLOCK_DIV_4 = 0xF
INTERNAL_CLOCK_DIV_5 = 0x10
MASTER_CLOCK = 0x11
INTERNAL_CLOCK_SET_VCO = 0x12

'''Sample rates that the internal clock of a board can generate.

Note: Available sample rates for a given board form a subset of
this class' members. Please see your board's specification as well
as the ATS-SDK manual for more information.

'''
SAMPLE_RATE_1KSPS = 0x1
SAMPLE_RATE_2KSPS = 0x2
SAMPLE_RATE_5KSPS = 0x5
SAMPLE_RATE_10KSPS = 0x8
SAMPLE_RATE_20KSPS = 0xA
SAMPLE_RATE_50KSPS = 0xC
SAMPLE_RATE_100KSPS = 0xE
SAMPLE_RATE_200KSPS = 0x10
SAMPLE_RATE_500KSPS = 0x12
SAMPLE_RATE_1MSPS = 0x14
SAMPLE_RATE_2MSPS = 0x18
SAMPLE_RATE_5MSPS = 0x1A
SAMPLE_RATE_10MSPS = 0x1C
SAMPLE_RATE_20MSPS = 0x1E
SAMPLE_RATE_25MSPS = 0x21
SAMPLE_RATE_50MSPS = 0x22
SAMPLE_RATE_100MSPS = 0x24
SAMPLE_RATE_125MSPS = 0x25
SAMPLE_RATE_160MSPS = 0x26
SAMPLE_RATE_180MSPS = 0x27
SAMPLE_RATE_200MSPS = 0x28
SAMPLE_RATE_250MSPS = 0x2B
SAMPLE_RATE_400MSPS = 0x2D
SAMPLE_RATE_500MSPS = 0x30
SAMPLE_RATE_800MSPS = 0x32
SAMPLE_RATE_1000MSPS = 0x35
SAMPLE_RATE_1200MSPS = 0x37
SAMPLE_RATE_1500MSPS = 0x3A
SAMPLE_RATE_1600MSPS = 0x3B
SAMPLE_RATE_1800MSPS = 0x3D
SAMPLE_RATE_2000MSPS = 0x3F
SAMPLE_RATE_2400MSPS = 0x6A
SAMPLE_RATE_3000MSPS = 0x75
SAMPLE_RATE_3600MSPS = 0x7B
SAMPLE_RATE_4000MSPS = 0x80
SAMPLE_RATE_USER_DEF = 0x40

'''Direction of the edge from the external clock signal that the board
syncrhonises with.'''
CLOCK_EDGE_RISING = 0
CLOCK_EDGE_FALLING = 1

'''Board input channel identifiers

Note: The channels available for a given board form a subset of this
class' members. Please see your board's specification as well as
the ATS-SDK manual for more information.

'''
CHANNEL_A = 1
CHANNEL_B = 2
CHANNEL_C = 4
CHANNEL_D = 8
CHANNEL_E = 16
CHANNEL_F = 32
CHANNEL_G = 64
CHANNEL_H = 128
CHANNEL_I = 256
CHANNEL_J = 512
CHANNEL_K = 1024
CHANNEL_L = 2048
CHANNEL_M = 4096
CHANNEL_N = 8192
CHANNEL_O = 16384
CHANNEL_P = 32768

channels = [
    CHANNEL_A,
    CHANNEL_B,
    CHANNEL_C,
    CHANNEL_D,
    CHANNEL_E,
    CHANNEL_F,
    CHANNEL_G,
    CHANNEL_H,
    CHANNEL_I,
    CHANNEL_J,
    CHANNEL_K,
    CHANNEL_L,
    CHANNEL_M,
    CHANNEL_N,
    CHANNEL_O,
    CHANNEL_P
]

'''AutoDMA acquisitions flags

Note: Not all AlazarTech devices are capable of dual-ported
acquisitions. Please see your board's specification for more
information.
'''
ADMA_TRADITIONAL_MODE = 0
ADMA_NPT = 0x200
ADMA_CONTINUOUS_MODE = 0x100
ADMA_TRIGGERED_STREAMING = 0x400
ADMA_EXTERNAL_STARTCAPTURE = 0x1
ADMA_ENABLE_RECORD_HEADERS = 0x8
ADMA_ALLOC_BUFFERS = 0x20
ADMA_FIFO_ONLY_STREAMING = 0x800
ADMA_INTERLEAVE_SAMPLES = 0x1000
ADMA_GET_PROCESSED_DATA = 0x2000

'''Boards'''
ATS850  = 1
ATS310  = 2
ATS330  = 3
ATS855  = 4
ATS315  = 5
ATS335  = 6
ATS460  = 7
ATS860  = 8
ATS660  = 9
ATS665  = 10
ATS9462 = 11
ATS9434 = 12
ATS9870 = 13
ATS9350 = 14
ATS9325 = 15
ATS9440 = 16
ATS9410 = 17
ATS9351 = 18
ATS9310 = 19
ATS9461 = 20
ATS9850 = 21
ATS9625 = 22
ATG6500 = 23
ATS9626 = 24
ATS9360 = 25
AXI9870 = 26
ATS9370 = 27
ATU7825 = 28
ATS9373 = 29
ATS9416 = 30

boardNames = {
    ATS850 : "ATS850" ,
    ATS310 : "ATS310" ,
    ATS330 : "ATS330" ,
    ATS855 : "ATS855" ,
    ATS315 : "ATS315" ,
    ATS335 : "ATS335" ,
    ATS460 : "ATS460" ,
    ATS860 : "ATS860" ,
    ATS660 : "ATS660" ,
    ATS665 : "ATS665" ,
    ATS9462: "ATS9462",
    ATS9434: "ATS9434",
    ATS9870: "ATS9870",
    ATS9350: "ATS9350",
    ATS9325: "ATS9325",
    ATS9440: "ATS9440",
    ATS9410: "ATS9410",
    ATS9351: "ATS9351",
    ATS9310: "ATS9310",
    ATS9461: "ATS9461",
    ATS9850: "ATS9850",
    ATS9625: "ATS9625",
    ATG6500: "ATG6500",
    ATS9626: "ATS9626",
    ATS9360: "ATS9360",
    AXI9870: "AXI9870",
    ATS9370: "ATS9370",
    ATU7825: "ATU7825",
    ATS9373: "ATS9373",
    ATS9416: "ATS9416"
};


'''Board input ranges (amplitudes) identifiers. PM stands for
plus/minus.

Note: Available input ranges for a given board _and_ a given
configuration form a subset of this class' members. Please see
your board's specification as well as the ATS-SDK manual for more
information.

'''
INPUT_RANGE_PM_40_MV = 0x2
INPUT_RANGE_PM_50_MV = 0x3
INPUT_RANGE_PM_80_MV = 0x4
INPUT_RANGE_PM_100_MV = 0x5
INPUT_RANGE_PM_200_MV = 0x6
INPUT_RANGE_PM_400_MV = 0x7
INPUT_RANGE_PM_500_MV = 0x8
INPUT_RANGE_PM_800_MV = 0x9
INPUT_RANGE_PM_1_V = 0xA
INPUT_RANGE_PM_2_V = 0xB
INPUT_RANGE_PM_4_V = 0xC
INPUT_RANGE_PM_5_V = 0xD
INPUT_RANGE_PM_8_V = 0xE
INPUT_RANGE_PM_10_V = 0xF
INPUT_RANGE_PM_20_V = 0x10
INPUT_RANGE_PM_40_V = 0x11
INPUT_RANGE_PM_16_V = 0x12
INPUT_RANGE_HIFI = 0x20
INPUT_RANGE_PM_1_V_25 = 0x21
INPUT_RANGE_PM_2_V_5  = 0x25
INPUT_RANGE_PM_125_MV = 0x28
INPUT_RANGE_PM_250_MV = 0x30

'''Coupling types identifiers for all boards input'''
AC_COUPLING = 1
DC_COUPLING = 2

'''Trigger engine identifiers.'''
TRIG_ENGINE_J = 0
TRIG_ENGINE_K = 1

'''Trigger engine operation identifiers.'''
TRIG_ENGINE_OP_J = 0
TRIG_ENGINE_OP_K = 1
TRIG_ENGINE_OP_J_OR_K = 2
TRIG_ENGINE_OP_J_AND_K = 3
TRIG_ENGINE_OP_J_XOR_K = 4
TRIG_ENGINE_OP_J_AND_NOT_K = 5
TRIG_ENGINE_OP_NOT_J_AND_K = 6

'''Types of input that the board can trig on.'''
TRIG_CHAN_A = 0
TRIG_CHAN_B = 1
TRIG_CHAN_C = 4
TRIG_CHAN_D = 5
TRIG_EXTERNAL = 2
TRIG_DISABLE = 3

'''Edge of the external trigger signal that the board syncrhonises with.'''
TRIGGER_SLOPE_POSITIVE = 1
TRIGGER_SLOPE_NEGATIVE = 2

'''Impedance identifiers for the board inputs.

Note: Available parameters for a given board form a subset of this
class' members. Please see your board's specification as well as
the ATS-SDK manual for more information.

'''
IMPEDANCE_1M_OHM = 1
IMPEDANCE_50_OHM = 2
IMPEDANCE_75_OHM = 4
IMPEDANCE_300_OHM = 8

'''External trigger range identifiers.'''
ETR_5V = 0
ETR_1V = 1
ETR_TTL = 2
ETR_2V5 = 3

'''LED State'''
LED_OFF = 0
LED_ON = 1

'''LSB Values'''
LSB_DEFAULT = 0
LSB_EXT_TRIG = 1
LSB_AUX_IN_0 = 2
LSB_AUX_IN_1 = 3

'''Operating modes for the auxiliary input/output port.'''
AUX_OUT_TRIGGER = 0
AUX_IN_TRIGGER_ENABLE = 1
AUX_OUT_PACER = 2
AUX_IN_AUXILIARY = 13
AUX_OUT_SERIAL_DATA = 14

'''Parameters for setParameter'''
SETGET_ASYNC_BUFFCOUNT = 0x10000040
SET_DATA_FORMAT        = 0x10000041
ECC_MODE               = 0x10000048

'''Parameters that apply to some modes of the auxiliary input/output
port.'''
TRIGGER_SLOPE_POSITIVE = 1
TRIGGER_SLOPE_NEGATIVE = 2

'''Record average options'''
CRA_MODE_DISABLE = 0
CRA_MODE_ENABLE_FPGA_AVE = 1
CRA_OPTION_UNSIGNED = 0
CRA_OPTION_SIGNED = 1

'''Reset timestamp'''
TIMESTAMP_RESET_FIRSTTIME_ONLY = 0
TIMESTAMP_RESET_ALWAYS = 1

'''Sleep State'''
POWER_OFF = 0
POWER_ON = 1

class WaitBar:
    '''Simple class that provides similar functionnality to MATLAB's waitbar.'''
    class WaitBarFrame(tk.Frame):
        '''A Tk Frame that shows a progress bar and a cancel button.'''
        def __init__(self, master=None):
            self.userCancelled = False
            self.progress = 0
            self.displayedProgress = 0
            tk.Frame.__init__(self, master)
            self.grid()
            self.createWidgets()
            self.after(15, self.updateGUI)

        def updateGUI(self):
            while self.displayedProgress < self.progress:
                self.displayedProgress += 1
                self.progressBar.step()
            self.after(15, self.updateGUI)

        def cancelAcquisition(self):
            print("Cancelling acquisition. Please wait.")
            self.userCancelled = True
            self.quit()

        def createWidgets(self):
            self.progressBar = tkinter.ttk.Progressbar(self, length=200)
            self.progressBar.grid(padx=10, pady=10)
            self.cancelButton = tk.Button(self,
                                          text='Cancel',
                                          command=self.cancelAcquisition)
            self.cancelButton.grid(padx=10, pady=10)

    def __init__(self):
        _thread.start_new_thread(self.createAndRunFrame, ())

    def createAndRunFrame(self):
        self.frame = WaitBar.WaitBarFrame()
        self.frame.master.title('Alazar Acquisition')
        self.frame.mainloop()

    def hasUserCancelled(self):
        return self.frame.userCancelled

    def setProgress(self, progress):
        self.frame.progress = int(progress * 100)

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

        cSampleType = c_uint8
        npSampleType = np.uint8
        if bytes_per_sample > 1:
            cSampleType = c_uint16
            npSampleType = np.uint16

        self.addr = None
        if os.name == 'nt':
            MEM_COMMIT = 0x1000
            PAGE_READWRITE = 0x4
            windll.kernel32.VirtualAlloc.argtypes = [c_void_p, c_long, c_long, c_long]
            windll.kernel32.VirtualAlloc.restype = c_void_p
            self.addr = windll.kernel32.VirtualAlloc(
                0, c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
        elif os.name == 'posix':
            libc.valloc.argtypes = [c_long]
            libc.valloc.restype = c_void_p
            self.addr = libc.valloc(size_bytes)
            print(("Allocated data : " + str(self.addr)))
        else:
            raise Exception("Unsupported OS")


        ctypes_array = (cSampleType * (size_bytes // bytes_per_sample)).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes_array, dtype=npSampleType)
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def __exit__(self):
        if os.name == 'nt':
            MEM_RELEASE = 0x8000
            windll.kernel32.VirtualFree.argtypes = [c_void_p, c_long, c_long]
            windll.kernel32.VirtualFree.restype = c_int
            windll.kernel32.VirtualFree(c_void_p(self.addr), 0, MEM_RELEASE);
        elif os.name == 'posix':
            libc.free(self.addr)
        else:
            raise Exception("Unsupported OS")


# Load libraries
ats = None
libc = None
if os.name == 'nt':
    ats = CDLL("ATSApi.dll")
elif os.name == 'posix':
    ats = CDLL("libATSApi.so")
    libc = CDLL("libc.so.6")
else:
    raise Exception("Unsupported OS")

U32 = c_uint32
U8  = c_byte

ats.AlazarErrorToText.restype = c_char_p
ats.AlazarErrorToText.argtypes = [U32]
def returnCodeCheck(result, func, arguments):
    '''Function used internally to check the return code of the C ATS-SDK
    functions.'''
    if (result != 512):
        raise Exception("Error calling function %s with arguments %s : %s" %
                        (func.__name__,
                         str(arguments),
                         str(ats.AlazarErrorToText(result))))

def numOfSystems():
    ats.AlazarNumOfSystems.restype = U32
    ats.AlazarNumOfSystems.argtypes = []
    return ats.AlazarNumOfSystems()

def boardsInSystemBySystemID(sid):
    ats.AlazarBoardsInSystemBySystemID.restype = U32
    ats.AlazarBoardsInSystemBySystemID.argtypes = [U32]
    return ats.AlazarBoardsInSystemBySystemID(sid)

class Board:
    '''Interface to an AlazarTech digitizer.

    The Board class represents an acquisition device on the local
    system. It can be used to control configuration parameters, to
    start acquisitions and to retrieve the acquired data.

    Args:

      systemId (int): The board system identifier of the target
      board. Defaults to 1, which is suitable when there is only one
      board in the system.

      boardId (int): The target's board identifier in it's
      system. Defaults to 1, which is suitable when there is only one
      board in the system.

    '''
    def __init__(self, systemId=1, boardId=1):
        ats.AlazarGetBoardBySystemID.restype = U32
        ats.AlazarGetBoardBySystemID.argtypes = [U32, U32]
        self.systemId = systemId
        self.boardId = boardId
        self.handle = ats.AlazarGetBoardBySystemID(systemId, boardId)
        if self.handle == 0:
            raise Exception("Board %d.%d not found" % (systemId, boardId))


        ats.AlazarGetBoardKind.restype = U32
        ats.AlazarGetBoardKind.argtypes = [U32]
        self.type = ats.AlazarGetBoardKind(self.handle)

    ats.AlazarAbortAsyncRead.restype = U32
    ats.AlazarAbortAsyncRead.argtypes = [U32]
    ats.AlazarAbortAsyncRead.errcheck = returnCodeCheck
    def abortAsyncRead(self):
        '''Cancels any asynchronous acquisition running on a board.'''
        ats.AlazarAbortAsyncRead(self.handle)

    ats.AlazarAbortCapture.restype = U32
    ats.AlazarAbortCapture.argtypes = [U32]
    ats.AlazarAbortCapture.errcheck = returnCodeCheck
    def abortCapture(self):
        '''Abort an acquisition to on-board memory.'''
        ats.AlazarAbortCapture(self.handle)



    ats.AlazarBeforeAsyncRead.restype = U32
    ats.AlazarBeforeAsyncRead.argtypes = [U32, U32, c_long, U32, U32, U32, U32]
    ats.AlazarBeforeAsyncRead.errcheck = returnCodeCheck
    def beforeAsyncRead(self, channels, transferOffset, samplesPerRecord,
                        recordsPerBuffer, recordsPerAcquisition, flags):
        '''Prepares the board for an asynchronous acquisition.'''
        ats.AlazarBeforeAsyncRead(self.handle, channels, transferOffset, samplesPerRecord,
                                  recordsPerBuffer, recordsPerAcquisition, flags)

    ats.AlazarBusy.restype = U32
    ats.AlazarBusy.argtypes = [U32]
    def busy(self):
        '''Determine if an acquisition to on-board memory is in progress.'''
        return True if (ats.AlazarBusy(self.handle) > 0) else False

    ats.AlazarConfigureAuxIO.restype = U32
    ats.AlazarConfigureAuxIO.argtypes = [U32, U32, U32]
    ats.AlazarConfigureAuxIO.errcheck = returnCodeCheck
    def configureAuxIO(self, mode, parameter):
        '''Configures the auxiliary output.'''
        ats.AlazarConfigureAuxIO(self.handle, mode, parameter)

    ats.AlazarConfigureLSB.restype = U32
    ats.AlazarConfigureLSB.argtypes = [U32, U32, U32]
    ats.AlazarConfigureLSB.errcheck = returnCodeCheck
    def configureLDB(self, valueLSB0, valueLSB1):
        '''Change unused bits to digital outputs.'''
        ats.AlazarConfigureLSB(self.handle, valueLSB0, valueLSB1)

    ats.AlazarConfigureRecordAverage.restype = U32
    ats.AlazarConfigureRecordAverage.argtypes = [U32, U32, U32, U32, U32]
    ats.AlazarConfigureRecordAverage.errcheck = returnCodeCheck
    def configureRecordAverage(self, mode, samplesPerRecord, recordsPerAverage, options):
        '''Co-add ADC samples into accumulator record.'''
        ats.AlazarConfigureRecordAverage(self.handle, mode, samplesPerRecord, recordsPerAverage, options)

    ats.AlazarForceTrigger.restype = U32
    ats.AlazarForceTrigger.argtypes = [U32]
    ats.AlazarForceTrigger.errcheck = returnCodeCheck
    def forceTrigger(self):
        '''Generate a software trigger event.'''
        ats.AlazarForceTrigger(self.handle)

    ats.AlazarForceTriggerEnable.restype = U32
    ats.AlazarForceTriggerEnable.argtypes = [U32]
    ats.AlazarForceTriggerEnable.errcheck = returnCodeCheck
    def forceTriggerEnable(self):
        '''Generate a software trigger enable event.'''
        ats.AlazarForceTriggerEnable(self.handle)

    ats.AlazarGetChannelInfo.restype = U32
    ats.AlazarGetChannelInfo.argtypes = [U32, c_void_p, c_void_p]
    def getChannelInfo(self):
        '''Get the on-board memory in samples per channe and sample size in bits per sample'''
        memorySize_samples = U32(0)
        bitsPerSample = U8(0)
        ats.AlazarGetChannelInfo(self.handle, byref(memorySize_samples), byref(bitsPerSample))
        return (memorySize_samples, bitsPerSample)

    ats.AlazarInputControl.restype = U32
    ats.AlazarInputControl.argtypes = [U32, U8, U32, U32, U32]
    ats.AlazarInputControl.errcheck = returnCodeCheck
    def inputControl(self, channel, coupling, inputRange, impedance):
        '''Configures one input channel on a board.'''
        ats.AlazarInputControl(self.handle, channel, coupling, inputRange, impedance)

    ats.AlazarNumOfSystems.restype = U32
    ats.AlazarNumOfSystems.argtypes = []
    def numOfSystems():
        '''Returns the number of board systems installed.'''
        ats.AlazarNumOfSystems()

    ats.AlazarPostAsyncBuffer.restype = U32
    ats.AlazarPostAsyncBuffer.argtypes = [U32, c_void_p, U32]
    ats.AlazarPostAsyncBuffer.errcheck = returnCodeCheck
    def postAsyncBuffer(self, buffer, bufferLength):
        '''Posts a DMA buffer to a board.'''
        ats.AlazarPostAsyncBuffer(self.handle, buffer, bufferLength)

    ats.AlazarReadEx.restype = U32
    ats.AlazarReadEx.argtypes = [U32, U32, c_void_p, c_int, c_long, c_int64, U32]
    ats.AlazarReadEx.errcheck = returnCodeCheck
    def read(self, channelId, buffer, elementSize, record, transferOffset, transferLength):
        '''Read all or part of a record from on-board memory.'''
        ats.AlazarReadEx(self.handle, channelId, buffer, elementSize, record, transferOffset, transferLength)

    ats.AlazarResetTimeStamp.restype = U32
    ats.AlazarResetTimeStamp.argtypes = [U32, U32]
    ats.AlazarResetTimeStamp.errcheck = returnCodeCheck
    def resetTimeStamp(self, option):
        '''Control record timestamp counter reset.'''
        ats.AlazarResetTimeStamp(self.handle, option)

    ats.AlazarSetBWLimit.restype = U32
    ats.AlazarSetBWLimit.argtypes = [U32, U32, U32]
    ats.AlazarSetBWLimit.errcheck = returnCodeCheck
    def setBWLimit(self, channel, enable):
        '''Activates or deactivates the low-pass filter on a given channel.'''
        ats.AlazarSetBWLimit(self.handle, channel, enable)

    ats.AlazarSetCaptureClock.restype = U32
    ats.AlazarSetCaptureClock.argtypes = [U32, U32, U32, U32, U32]
    ats.AlazarSetCaptureClock.errcheck = returnCodeCheck
    def setCaptureClock(self, source, rate, edge, decimation):
        '''Configures the board's acquisition clock.'''
        ats.AlazarSetCaptureClock(self.handle,
                                  int(source),
                                  int(rate),
                                  int(edge),
                                  decimation)

    ats.AlazarSetExternalClockLevel.restype = U32
    ats.AlazarSetExternalClockLevel.argtypes = [U32, c_float]
    ats.AlazarSetExternalClockLevel.errcheck = returnCodeCheck
    def setExternalClockLevel(self, level_percent):
        '''Set the external clock comparator level'''
        ats.AlazarSetExternalClockLevel(self.handle, level_percent)

    ats.AlazarSetExternalTrigger.restype = U32
    ats.AlazarSetExternalTrigger.argtypes = [U32, U32, U32]
    ats.AlazarSetExternalTrigger.errcheck = returnCodeCheck
    def setExternalTrigger(self, coupling, range):
        '''Configure the external trigger.'''
        ats.AlazarSetExternalTrigger(self.handle, coupling, range)

    ats.AlazarSetLED.restype = U32
    ats.AlazarSetLED.argtypes = [U32, U32]
    ats.AlazarSetLED.errcheck = returnCodeCheck
    def setLED(self, ledState):
        '''Control LED on a board's mounting bracket.'''
        ats.AlazarSetLED(self.handle, ledState)

    ats.AlazarSetParameter.restype = U32
    ats.AlazarSetParameter.argtypes = [U32, U8, U32, c_long]
    ats.AlazarSetParameter.errcheck = returnCodeCheck
    def setParameter(self, channelId, parameterId, value):
        '''Set a device parameter as a signed long value.'''
        ats.AlazarSetParameter(self.handle, channelId, parameterId, value)

    ats.AlazarSetParameterUL.restype = U32
    ats.AlazarSetParameterUL.argtypes = [U32, U8, U32, c_long]
    ats.AlazarSetParameterUL.errcheck = returnCodeCheck
    def setParameterUL(self, channelId, parameterId, value):
        '''Set a device parameter as a signed long value.'''
        ats.AlazarSetParameterUL(self.handle, channelId, parameterId, value)

    ats.AlazarSetRecordCount.restype = U32
    ats.AlazarSetRecordCount.argtypes = [U32, U32]
    ats.AlazarSetRecordCount.errcheck = returnCodeCheck
    def setRecordCount(self, count):
        '''Configure the record count for single ported acquisitions.'''
        ats.AlazarSetRecordCount(self.handle, count)

    ats.AlazarSetRecordSize.restype = U32
    ats.AlazarSetRecordSize.argtypes = [U32, U32, U32]
    ats.AlazarSetRecordSize.errcheck = returnCodeCheck
    def setRecordSize(self, preTriggerSamples, postTriggerSamples):
        '''Configures the acquisition records size.'''
        ats.AlazarSetRecordSize(self.handle, preTriggerSamples, postTriggerSamples)

    ats.AlazarSetTriggerDelay.restype = U32
    ats.AlazarSetTriggerDelay.argtypes = [U32, U32]
    ats.AlazarSetTriggerDelay.errcheck = returnCodeCheck
    def setTriggerDelay(self, delay_samples):
        '''Configures the trigger delay.'''
        ats.AlazarSetTriggerDelay(self.handle, delay_samples)

    ats.AlazarSetTriggerOperation.restype = U32
    ats.AlazarSetTriggerOperation.argtypes = [U32, U32, U32, U32, U32, U32, U32, U32, U32, U32]
    ats.AlazarSetTriggerOperation.errcheck = returnCodeCheck
    def setTriggerOperation(self, operation,
                            engine1, source1, slope1, level1,
                            engine2, source2, slope2, level2):
        '''Set trigger operation.'''
        ats.AlazarSetTriggerOperation(
            self.handle, operation,
            engine1, source1, slope1, level1,
            engine2,
            source2,
            slope2,
            level2)

    ats.AlazarSetTriggerTimeOut.restype = U32
    ats.AlazarSetTriggerTimeOut.argtypes = [U32, U32]
    ats.AlazarSetTriggerTimeOut.errcheck = returnCodeCheck
    def setTriggerTimeOut(self, timeout_clocks):
        '''Configures the trigger timeout.'''
        ats.AlazarSetTriggerTimeOut(self.handle, timeout_clocks)

    ats.AlazarSleepDevice.restype = U32
    ats.AlazarSleepDevice.argtypes = [U32, U32]
    ats.AlazarSleepDevice.errcheck = returnCodeCheck
    def sleepDevice(self, sleepState):
        '''Control poewr to ADC devices'''
        ats.AlazarSleepDevice(self.handle, sleepState)

    ats.AlazarStartCapture.restype = U32
    ats.AlazarStartCapture.argtypes = [U32]
    ats.AlazarStartCapture.errcheck = returnCodeCheck
    def startCapture(self):
        '''Starts the acquisition.'''
        ats.AlazarStartCapture(self.handle)

    ats.AlazarTriggered.restype = U32
    ats.AlazarTriggered.argtypes = [U32]
    def triggered(self):
        '''Determine if a board has triggered during the current acquisition.'''
        return ats.AlazarTriggered(self.handle)

    ats.AlazarWaitAsyncBufferComplete.restype = U32
    ats.AlazarWaitAsyncBufferComplete.argtypes = [U32, c_void_p, U32]
    ats.AlazarWaitAsyncBufferComplete.errcheck = returnCodeCheck
    def waitAsyncBufferComplete(self, buffer, timeout_ms):
        '''Blocks until the board confirms that buffer is filled with data.'''
        ats.AlazarWaitAsyncBufferComplete(self.handle, buffer, timeout_ms)
