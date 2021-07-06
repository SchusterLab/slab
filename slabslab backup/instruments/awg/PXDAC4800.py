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

# SERIAL_NUMBER = [301450,301483, 300727]
# SERIAL_NUMBER = [300727, 301483, 301450]

class PXDAC4800:
    def __init__(self,brdNum):
        self.brdNum = brdNum

    def load_sequence_file(self, waveform_file_name,awg):

        print("load_sequence_file inputs:")

        offset_bytes_list  = awg['iq_offsets_bytes']
        clock_speed = awg['clock_speed']
        self.channels_num = awg['channels_num']
        self.sample_size = awg['sample_size']
        DLL_path = awg['DLL_path']

        print('waveform_file_name =', waveform_file_name)
        print('offset_bytes_list =', offset_bytes_list)
        print('clock_speed =', clock_speed)

        U8 = C.c_uint8
        U8P = C.POINTER(U8)
        U32 = C.c_uint32
        U32P = C.POINTER(U32)
        U32PP = C.POINTER(U32P)
        I32 = C.c_int32
        CFLT = C.c_float
        CHARP = C.c_char_p

        clock_divider = 1.2/clock_speed

        if not clock_divider.is_integer():
            raise ValueError('clock speed is not integer divider of 1.2 GHz')

        clock_divider = int(clock_divider)
        if clock_divider == 3:
            raise ValueError('clock divider cannot be equal to 3')

        unsigned_short_length = 2

        try:
            DACdllpath = str(DLL_path)
            DACDLL = C.CDLL(DACdllpath)
            print("DPXDAC4800.dll loaded")
        except:
            print("Warning could not load PXDAC4800 dll, check that dll located at '%s'" % DACdllpath)

        ppHandle = U32PP(U32P(U32(0)))

        print("Connecting to PXDAC4800 Board #", self.brdNum)
        dll = DACDLL
        # dll.ConnectToDeviceXD48(ppHandle, U32(SERIAL_NUMBER[self.brdNum-1]))
        dll.ConnectToDeviceXD48(ppHandle, U32(awg['SERIAL_NUMBER']))

        self.dll = dll

        print("PXDAC4800 device connected.")

        pHandle = dll.GetHandleXD48(ppHandle)

        self.pHandle = pHandle

        print("Getting Serial Number.")

        SerialNumber = U32(0)
        pSerialNumber = U32P(SerialNumber)
        outSerial = dll.GetSerialNumberXD48(pHandle, pSerialNumber)
        SerialNumber = pSerialNumber.contents.value
        print("Serial Number: " + str(SerialNumber))

        print("Setting DAC")

        dll.SetPowerupDefaultsXD48(pHandle)
        # dll._SetOperatingModeXD48(pHandle, U32(1))
        dll.SetTriggerModeXD48(pHandle, U32(0))  # Set trigger mode in PLAY PER TRIGGER mode
        dll.SetPlaybackClockSourceXD48(pHandle, U32(0))  # Set clock source - Internal 1.2 GHz

        dll.SetExternalReferenceClockEnableXD48(pHandle, U32(1))  ## External 10MHz clock

        time.sleep(1)



        dll.SetDacSampleFormatXD48(pHandle, U32(1))  # Set DAC format to signed

        if self.sample_size == 16:
            dll.SetDacSampleSizeXD48(pHandle, U32(2))  # Set DAC sample size to 16 bit LSB pad
        elif self.sample_size == 8:
            dll.SetDacSampleSizeXD48(pHandle, U32(0))  # Set DAC sample size to 8 bit
        else:
            raise ValueError('Invalid sample size %s' %sample_size)
        # dll.SetDigitalIoModeXD48(pHandle, U32(1)) # Set Digital IO pulse at the begining of a playback

        ### Set Active Channel Mask
        if self.channels_num == 2:
            dll.SetActiveChannelMaskXD48(pHandle, U32(3)) # dual chn1/2
            #dll.SetActiveChannelMaskXD48(pHandle, U32(12)) # dual chn 3/4
        elif self.channels_num == 4:
            dll.SetActiveChannelMaskXD48(pHandle, U32(15)) # four channels

        ## start DAC auto calibration
        # if fail, should return -603 or -604, any other error message - should abort
        for ii in range(2):
            calibration_result = dll.StartDacAutoCalibrationXD48(pHandle)
            print("Calibration status: " + str(calibration_result))
            if calibration_result == 0:
                break
            elif calibration_result not in [-603, -604]:
                raise ValueError('PXDAC calibration returns unexpected value - Experiment terminated.')

        ### Set output voltage to max
        dll.SetOutputVoltageCh1XD48(pHandle, U32(1023))
        dll.SetOutputVoltageCh2XD48(pHandle, U32(1023))
        dll.SetOutputVoltageCh3XD48(pHandle, U32(1023))
        dll.SetOutputVoltageCh4XD48(pHandle, U32(1023))

        ### Set Dac Default value
        # works only on the channels enabled
        dll.SetCustomDacValueEnableXD48(pHandle, U32(15))  # chn 1234
        for ii in range(1,5,1):# chn 1/2/3/4
            dll.SetCustomDacDefaultValueXD48(pHandle, U32(ii), I32(offset_bytes_list[ii - 1]))
            # print 'offset_bytes_list on chn', ii, ':', dll.GetCustomDacDefaultValueXD48(pHandle, U32(ii))

        ### Set clock division
        dll.SetClockDivider1XD48(pHandle, U32(clock_divider))


        print("Active Channel Mask: " + str(dll.GetActiveChannelMaskXD48(pHandle, U32(1))))

        print("Load waveform file.")

        offset = U32(0)
        self.offset = offset

        filePath = CHARP((waveform_file_name + '\0').encode())
        # fileSize= U32(os.path.getsize(filePath))

        print("Loading..")
        dll._LoadFileIntoRamAXD48(pHandle, offset, U32(0), filePath, U32(0), U32(0), offset)
        print("Loaded!")
        pPlaybackBytes = U32P(U32(0))
        dll._GetActiveMemoryRegionXD48(pHandle, None, pPlaybackBytes)
        PlaybackBytes = pPlaybackBytes.contents
        print("Playback Bytes: " + str(PlaybackBytes.value))

        self.PlaybackBytes = PlaybackBytes

    def run_experiment(self):
        U32 = C.c_uint32
        unsigned_short_length = int(self.sample_size/8)
        # print "Begining Ram Playback."
        self.dll.BeginRamPlaybackXD48(self.pHandle, self.offset, self.PlaybackBytes, U32(
            self.channels_num * self.waveform_length * unsigned_short_length))  ## Only play sequence's single waveform byte length for each trigger

    def stop(self):
        # print "Stopping Ram Playback."
        self.dll.EndRamPlaybackXD48(self.pHandle)


def write_PXDAC4800_file(waveforms, filename, seq_name, offsets=None, sample_size=None, options=None, do_string_io=False):
    """
    Main function for writing a PXDAC4800 AWG format file (.rd16 binary file).
    """
    ## sequence
    sequence_length = len(waveforms[0])
    #print "Sequence length: " + str(sequence_length)

    ## max value for signed short
    max_value = 2 ** (sample_size-1) - 1
    min_value = -2 ** (sample_size-1)
    # clock_rate = 1.2  # G/s

    ## max voltage
    max_offset = 0.2
    max_output_volt = 1.47  # set output voltage to be 1.4V if 1.0 in waveform
    unit_volt = max_output_volt - max_offset
    scale = unit_volt / max_output_volt * max_value

    if offsets is None:
        offsets = np.zeros(len(waveforms))

    #     Check for overflows
    if (np.amax(np.abs(offsets)*max_output_volt/max_value) > max_output_volt - unit_volt):
        raise ValueError('Offset value overflow.')
    #     if np.amax(waveforms) > 1.0 or np.amin(waveforms) <-1.0:
    #         raise ValueError('Waveform value overflow.')

    # generate waveforms
    waveforms = np.reshape(waveforms, (len(waveforms), len(waveforms[0][0]) * len(waveforms[0])))
    #print "reshaped"

    # old
    # for ii, offset in enumerate(offsets):
    #     # offset_bytes = int((offset + awg_output_offsets[ii]) / max_output_volt * max_value)
    #     # offset_bytes_list.append(offset_bytes)
    #     waveforms[ii] = waveforms[ii] * scale + offset
    # #print "scaled"

    #

    # new
    for ii in range(len(waveforms)):
        waveforms[ii] = waveforms[ii] * scale + offsets[ii]
    #

    interleaved_waveforms = np.ravel(np.column_stack(waveforms))

    #print "interleaved"

    with open(filename, 'wb')  as f:
        interleaved_waveforms.astype('int%d'%sample_size).tofile(f)


    #print 'finished generating waveform file'