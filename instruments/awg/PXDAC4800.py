# -*- coding: utf-8 -*-
"""
PXDAC4800
================================

:Author: Nelson Leung
"""
import ctypes as C
import numpy as np
import time
from struct import *

class PXDAC4800:
    def __init__(self):
        pass


    def load_sequence_file(self,waveform_file_name):
        U8 = C.c_uint8
        U8P = C.POINTER(U8)
        U32 = C.c_uint32
        U32P = C.POINTER(U32)
        U32PP = C.POINTER(U32P)
        I32 = C.c_int32
        CFLT = C.c_float
        CHARP = C.c_char_p

        unsigned_short_length = 2

        try:
            DACdllpath = r'C:\Program Files\Signatec\PXDAC4800\PXDAC4800_64.dll'
            DACDLL = C.CDLL(DACdllpath)
            print "DPXDAC4800.dll loaded"
        except:
            print "Warning could not load PXDAC4800 dll, check that dll located at '%s'" % DACdllpath

        ppHandle = U32PP(U32P(U32(0)))

        print "Connecting to PXDAC4800 device."
        dll = DACDLL
        dll.ConnectToDeviceXD48(ppHandle, U32(1))

        PXDAC4800.dll = dll


        print "PXDAC4800 device connected."

        pHandle = dll.GetHandleXD48(ppHandle)

        PXDAC4800.pHandle = pHandle

        print "Getting Serial Number."

        SerialNumber = U32(0)
        pSerialNumber = U32P(SerialNumber)
        outSerial = dll.GetSerialNumberXD48(pHandle, pSerialNumber)
        SerialNumber = pSerialNumber.contents.value
        print "Serial Number: " + str(SerialNumber)

        print "Setting DAC"

        dll.SetPowerupDefaultsXD48(pHandle)
        dll.SetTriggerModeXD48(pHandle, U32(0))  # Set trigger mode in PLAY PER TRIGGER mode
        dll.SetPlaybackClockSourceXD48(pHandle, U32(0))  # Set clock source - Internal 1.2 GHz

        dll.SetExternalReferenceClockEnableXD48(pHandle, U32(1)) ## External 10MHz clock

        dll.SetDacSampleFormatXD48(pHandle, U32(1))  # Set DAC format to signed
        dll.SetDacSampleSizeXD48(pHandle, U32(2))  # Set DAC sample size to 16 bit LSB pad
        # dll.SetDigitalIoModeXD48(pHandle, U32(1)) # Set Digital IO pulse at the begining of a playback

        ### Set Active Channel Mask
        dll.SetActiveChannelMaskXD48(pHandle,U32(3))

        ### Set output voltage to max
        dll.SetOutputVoltageCh1XD48(pHandle,U32(1023))
        dll.SetOutputVoltageCh2XD48(pHandle,U32(1023))

        print "Active Channel Mask: "+ str(dll.GetActiveChannelMaskXD48(pHandle,U32(1)))

        print "Load waveform file."

        offset = U32(0)
        PXDAC4800.offset = offset

        filePath = CHARP(waveform_file_name+'\0')
        # fileSize= U32(os.path.getsize(filePath))

        print "Loading.."
        dll._LoadFileIntoRamAXD48(pHandle, offset, U32(0), filePath, U32(0), U32(0), offset)
        print "Loaded!"
        pPlaybackBytes = U32P(U32(0))
        dll._GetActiveMemoryRegionXD48(pHandle, None, pPlaybackBytes)
        PlaybackBytes = pPlaybackBytes.contents
        print "Playback Bytes: " + str(PlaybackBytes.value)

        PXDAC4800.PlaybackBytes = PlaybackBytes


    def run_experiment(self):
        U32 = C.c_uint32
        unsigned_short_length = 2
        # print "Begining Ram Playback."
        PXDAC4800.dll.BeginRamPlaybackXD48(PXDAC4800.pHandle, PXDAC4800.offset, PXDAC4800.PlaybackBytes, U32(2*PXDAC4800.waveform_length*unsigned_short_length)) ## Only play sequence's single waveform byte length for each trigger

    def stop(self):
        # print "Stopping Ram Playback."
        PXDAC4800.dll.EndRamPlaybackXD48(PXDAC4800.pHandle)

def write_PXDAC4800_file(waveforms, filename, seq_name, offsets=[0,0], options=None, do_string_io=False):
    """
    Main function for writing a PXDAC4800 AWG format file (.rd16 binary file).
    """

    ## waveform file
    waveform_file_name = filename

    ## sequence
    sequence_length = len(waveforms[0])
    print "Sequence length: " + str(sequence_length)

    ## max value for signed short
    max_value = 32764
    min_value = -32768
    waveform_length = len(waveforms[0][0])
    clock_rate = 1.2 #G/s

    ## max voltage
    unit_volt = 1.47 # 1.0 in waveform, corresponds to 1.47V in output
    max_output_volt = 1.4 # set output voltage to be 1.4V if 1.0 in waveform
    ## this assumes I,Q offsets < 70mV

    ## offsets
    ch1_offset = offsets[0] / unit_volt
    ch2_offset = offsets[1] / unit_volt

    # generate waveforms
    defined_waveform_ch1_flatten = [y for x in waveforms[0] for y in x]
    defined_waveform_ch2_flatten = [y for x in waveforms[1] for y in x]

    defined_waveform_ch1_flatten_offset = [x*max_output_volt/unit_volt+ch1_offset for x in defined_waveform_ch1_flatten]
    defined_waveform_ch2_flatten_offset = [x*max_output_volt/unit_volt+ch2_offset for x in defined_waveform_ch2_flatten]

    defined_waveform_ch1 = [max_value*x for x in defined_waveform_ch1_flatten_offset]
    defined_waveform_ch2 = [max_value*x for x in defined_waveform_ch2_flatten_offset]

    ## output error if ch values greater than min/max
    if (max(defined_waveform_ch1) > max_value) or (min(defined_waveform_ch1) < min_value):
        raise ValueError('Ch1 waveform value overflow.')
    if (max(defined_waveform_ch2) > max_value) or (min(defined_waveform_ch2) < min_value):
        raise ValueError('Ch2 waveform value overflow.')

    defined_waveform = [None]*(len(defined_waveform_ch1)+len(defined_waveform_ch2))
    defined_waveform[::2] = defined_waveform_ch1
    defined_waveform[1::2] = defined_waveform_ch2

    with open(waveform_file_name,'wb')  as f:
        try:
            for value in defined_waveform:
                # print value
                f.write(pack('h',value))
        finally:
            f.close()
            print 'finished generating waveform file'

