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
from array import array as barray


class PXDAC4800:
    def __init__(self):
        pass

    def load_sequence_file(self, waveform_file_name):
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

        dll.SetExternalReferenceClockEnableXD48(pHandle, U32(1))  ## External 10MHz clock

        dll.SetDacSampleFormatXD48(pHandle, U32(1))  # Set DAC format to signed
        dll.SetDacSampleSizeXD48(pHandle, U32(2))  # Set DAC sample size to 16 bit LSB pad
        # dll.SetDigitalIoModeXD48(pHandle, U32(1)) # Set Digital IO pulse at the begining of a playback

        ### Set Active Channel Mask
        dll.SetActiveChannelMaskXD48(pHandle, U32(3))

        ### Set output voltage to max
        dll.SetOutputVoltageCh1XD48(pHandle, U32(1023))
        dll.SetOutputVoltageCh2XD48(pHandle, U32(1023))

        print "Active Channel Mask: " + str(dll.GetActiveChannelMaskXD48(pHandle, U32(1)))

        print "Load waveform file."

        offset = U32(0)
        PXDAC4800.offset = offset

        filePath = CHARP(waveform_file_name + '\0')
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
        PXDAC4800.dll.BeginRamPlaybackXD48(PXDAC4800.pHandle, PXDAC4800.offset, PXDAC4800.PlaybackBytes, U32(
            2 * PXDAC4800.waveform_length * unsigned_short_length))  ## Only play sequence's single waveform byte length for each trigger

    def stop(self):
        # print "Stopping Ram Playback."
        PXDAC4800.dll.EndRamPlaybackXD48(PXDAC4800.pHandle)


def write_PXDAC4800_file(waveforms, filename, seq_name, offsets=None, options=None, do_string_io=False):
    """
    Main function for writing a PXDAC4800 AWG format file (.rd16 binary file).
    """
    ## sequence
    sequence_length = len(waveforms[0])
    #print "Sequence length: " + str(sequence_length)

    ## max value for signed short
    max_value = 2 ** 15 - 1
    min_value = -2 ** 15
    clock_rate = 1.2  # G/s

    ## max voltage
    max_offset = 0.1
    max_output_volt = 1.47  # set output voltage to be 1.4V if 1.0 in waveform
    unit_volt = max_output_volt - max_offset
    scale = unit_volt / max_output_volt * max_value

    if offsets is None:
        offsets = np.zeros(len(waveforms))

    #     Check for overflows
    if (np.amax(np.abs(offsets)) > max_output_volt - unit_volt):
        raise ValueError('Offset value overflow.')
    #     if np.amax(waveforms) > 1.0 or np.amin(waveforms) <-1.0:
    #         raise ValueError('Waveform value overflow.')

    # generate waveforms
    waveforms = np.reshape(waveforms, (len(waveforms), len(waveforms[0][0]) * len(waveforms[0])))
    #print "reshaped"
    for ii, offset in enumerate(offsets):
        print offset
        waveforms[ii] = waveforms[ii] * scale + offset / max_output_volt * max_value
    #print "scaled"

    interleaved_waveforms = np.ravel(np.column_stack(waveforms))

    #print "interleaved"

    with open(filename, 'wb')  as f:
        interleaved_waveforms.astype('int16').tofile(f)

    #print 'finished generating waveform file'