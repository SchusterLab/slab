"""
For a comprehensive list of all allowed functions
http://zone.ni.com/reference/en-XX/help/370471AE-01/cdaqmxsupp/ni9260bnc/ (for the card itself)
http://zone.ni.com/reference/en-XX/help/370471AE-01/cdaqmxsupp/cdaq-9188/ (for the chassis)
"""

import time
from tabulate import tabulate
from matplotlib import pyplot as plt
import PyDAQmx
import numpy as np
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
import warnings
warnings.simplefilter("ignore", PotentialGlitchDuringWriteWarning)

int32 = ctypes.c_long
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
float64 = ctypes.c_double
cchar = ctypes.c_char

class NI9260():

    def __init__(self, name="cDAQ9189-1C742CBMod1"):
        self.slot_name = name
        self.query_sleep = 0.05
        self.create_task()

    def get_id(self):
        """
        Indicates the serial number of the device.
        This value is zero if the device does not have a serial number.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/func0632/
        """
        data = uInt32()
        DAQmxGetDevSerialNum(self.slot_name, byref(data))
        return data.value

    def create_task(self, task_name=""):
        """
        Create a task.
        :param task_name:
        :return:
        """
        self.handle = TaskHandle(0)
        self.task_name = task_name
        DAQmxCreateTask(self.task_name, byref(self.handle))

    def setup_channel(self, name=None, channel=0, channelLimits=(-4.23, +4.23)):
        """
        Run after
        - create_task()
        """
        if name is None:
            name = "%s/ao%d" % (self.slot_name, channel)
        DAQmxCreateAOVoltageChan(self.handle, name, "", channelLimits[0], channelLimits[1], DAQmx_Val_Volts, None)

    def setup_all_channels(self, channelLimits=(-4.23, +4.23)):
        """
        Run after
        - create_task()
        """
        name = "%s/ao0:1" % self.slot_name
        DAQmxCreateAOVoltageChan(self.handle, name, "", channelLimits[0], channelLimits[1], DAQmx_Val_Volts, None)

    def set_sample_clock(self, runContinuous=True, clockSource="", sampleSize=1000, sampleRate=1000):
        """
        http://zone.ni.com/reference/en-XX/help/370471AE-01/daqmxcfunc/daqmxcfgsampclktiming/

        Run after
        - create_task()
        - setup_channel()
        :param runContinuous: True/False; If true the waveform is repeated continuously
        :param clockSource: "PFI0" for a 10 MHz reference clock hooked up to the PFI0 port or "" for the onboard clock
        :param sampleSize: Number of samples in the waveform
        :param sampleRate: Waveform timebase. If you use an external source for the Sample Clock, set this value to
        the maximum expected rate of that clock.
        :return: None
        """
        sampleMode = DAQmx_Val_ContSamps if runContinuous else DAQmx_Val_FiniteSamps
        if clockSource is not "":
            clockSource = "PFI0"

        DAQmxCfgSampClkTiming(self.handle, clockSource, float64(sampleRate), DAQmx_Val_Rising, sampleMode, uInt64(sampleSize))

    def get_sample_clock_rate(self):
        """
        Specifies the sampling rate in samples per channel per second.
        If you use an external source for the Sample Clock, set this input to the maximum expected rate of that clock.
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/func1344/
        :return:
        """
        data = float64()
        DAQmxGetSampClkRate(self.handle, byref(data))
        return data.value

    def start_task(self, wait=10.0):
        DAQmxStartTask(self.handle)
        if wait:
            return self.wait_until_operation_completion(maxtime=wait)

    def stop_task(self):
        DAQmxStopTask(self.handle)

    def clear_task(self):
        """
        Clears the task. Before clearing, this function aborts the task, if necessary,
        and releases any resources reserved by the task. You cannot use a task once you
        clear the task without recreating or reloading the task.

        If you use the DAQmxCreateTask function or any of the NI-DAQmx Create Channel
        functions within a loop, use this function within the loop after you finish with
        the task to avoid allocating unnecessary memory.
        """
        DAQmxClearTask(self.handle)
        delattr(self, 'handle')
        self.__init__()

    def get_sample_timing_type(self):
        """
        Sample Timing Type
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr1347/
        :return:
        """
        data = int32()
        DAQmxGetSampTimingType(self.handle, byref(data))
        datadict = {'10388' : 'sample_clock', '10390' : 'on_demand',
                    '12548' : 'burst_handshake', '10389' : 'handshake',
                    '12504' : 'change_detection', '14668' : 'pipelined_sample_clock'}
        return datadict[str(data.value)]

    def set_sample_timing_type(self, mode='on_demand'):
        """
        Sample Timing Type
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr1347/

        There are two relevant types:
        1. Select DAQmx_Val_SampClk when a hardware signal (usually a clock) must acquire or produce samples.
        To perform buffered edge counting, for example, select DAQmx_Val_SampClk and use Source to specify the
        source of the Sample clock.
        2. Select DAQmx_Val_OnDemand to acquire data only when an NI-DAQmx Read function executes or to
        generate data only when an NI-DAQmx Write function executes.
        :return:
        """
        samplingMode = DAQmx_Val_OnDemand if mode == 'on_demand' else DAQmx_Val_SampClk
        DAQmxSetSampTimingType(self.handle, samplingMode)

    def set_volt(self, value, first_time=False):
        """
        Run after
        - self.__init__()

        The DC mode works in user buffer regeneration mode. This means the output buffer is repeated at the computer
        and constantly transferred into the FIFO buffer of the NIDAQ. This means the sampleRate must be fast enough
        such that the buffer is emptied quickly to allow for quick updating of a DC value by the user. The sampleSize
        also plays a role, with this too large, the buffer fills up too quickly and the output won't update either.
        """
        sampleSize = 15
        sampleRate = 25600

        if first_time:
            self.setup_channel()
            self.set_write_regeneration_mode(True)
            self.set_sample_clock(runContinuous=True, sampleRate=sampleRate, sampleSize=sampleSize)
            self.set_write_relative_to('first')
            self.set_idle_output_setting("MaintainCurrentValue")
            self.set_bypass_memory_buffer(False)

        self.set_waveform(value * np.ones(sampleSize), numSamplesPerChan=sampleSize, autostart=True)
        # DAQmxWriteAnalogScalarF64(self.handle, autoStart=False, timeout=self.query_sleep, value=float64(value), reserved=None)

    def set_waveform(self, waveformData, numSamplesPerChan, autostart=True):
        """
        Run after
        - create_task()
        - setup_channel()
        - setup_timing()
        to engage output, run start_task() and trigger, if necessary.

        For multiple channels, use waveformData = np.append(x, y) where x and y are data for ch0 and ch1 respectively.
        Nominally, numSampsPerChan (sampleSize) should be set to the length of x or y. If set to multiples, it will play
        multiple copies of x and/or y.
        """
        samplesWritten = int32()
        DAQmxWriteAnalogF64(self.handle, numSampsPerChan=int32(numSamplesPerChan), autoStart=autostart,
                            timeout=self.query_sleep, dataLayout=DAQmx_Val_GroupByChannel,
                            writeArray=waveformData, sampsPerChanWritten=byref(samplesWritten), reserved=None)
        return samplesWritten.value

    def get_resolution(self):
        """
        Indicates the resolution of the digital-to-analog converter of the channel.
        This value is in the units you specify with Resolution Units.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr182c/
        """
        resolution = float64()
        DAQmxGetAOResolution(self.handle, "", byref(resolution))
        return resolution.value

    def set_idle_output_setting(self, value="MaintainCurrentValue"):
        """
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/attr2240/
        :return:
        """
        allowedValues = ["ZeroVolts", "HighImpedance", "MaintainCurrentValue"]
        allowedDict = {"ZeroVolts" : DAQmx_Val_ZeroVolts,
                       "HighImpedance" : DAQmx_Val_HighImpedance,
                       "MaintainCurrentValue" : DAQmx_Val_MaintainExistingValue}
        if value in list(allowedDict.keys()):
            DAQmxSetAOIdleOutputBehavior(self.handle, "", allowedDict[value])
        else:
            print("Idle output value not valid. Must be one of the following:", allowedDict.keys())

    def get_idle_output_setting(self):
        """
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr2240/
        """
        idle_behavior = int32()
        DAQmxGetAOIdleOutputBehavior(self.handle, "", byref(idle_behavior))
        datadict = {"12526": "ZeroVolts",
                    "12527": "HighImpedance",
                    "12528": "MaintainCurrentValue"}
        return datadict[str(idle_behavior.value)]

    def get_filter_delay_unit(self):
        """
        Specifies the units of Filter Delay and Filter Delay Adjustment.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr3076/
        """
        filter_delay_unit = int32()
        DAQmxGetAOFilterDelayUnits(self.handle, "", byref(filter_delay_unit))
        datadict = {"10364": "Seconds", "10286": "Sample Clock Periods"}
        return datadict[str(filter_delay_unit.value)]

    def get_filter_delay(self):
        """
        Specifies the amount of time between when the sample is written by the
        host device and when the sample is output by the DAC.
        This value is in the units you specify with Filter Delay Units.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr3075/
        """
        filter_delay = float64()
        DAQmxGetAOFilterDelay(self.handle, "", byref(filter_delay))
        return filter_delay.value

    def set_filter_delay(self, delay):
        """
        Specifies the amount of time between when the sample is written by the
        host device and when the sample is output by the DAC.
        This value is in the units you specify with Filter Delay Units.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr3075/
        """
        # NOTE: DOESN'T WORK PROPERLY
        DAQmxSetAOFilterDelay(self.handle, "", float64(delay))

    def get_physical_channel_name(self):
        """
        Specifies the name of the physical channel upon which this virtual channel is based.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/func18f5/
        """
        data = cchar()
        DAQmxGetPhysicalChanName(self.handle, "", byref(data), uInt32(2000))
        return data.value

    def get_minimum_sync_pulse_start_delay(self):
        """
        Specifies in seconds the amount of time that elapses after the master
        device issues the synchronization pulse before the task starts.
        Read Synchronization Time for all slave devices, and set this property
        for the master device to the maximum of those values.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr223f/
        """
        data = float64()
        DAQmxGetSyncPulseMinDelayToStart(self.handle, byref(data))
        return data.value

    def get_write_regeneration_mode(self):
        """
        See documentation for set_write_regeneration_mode()
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr1453/
        """
        data = int32()
        DAQmxGetWriteRegenMode(self.handle, byref(data))
        datadict = {"10097": True, "10158": False}
        return datadict[str(data.value)]

    def set_write_regeneration_mode(self, state):
        """
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/func1453/

        Specifies whether to allow NI-DAQmx to generate the same data multiple times.
        If you enable regeneration and write new data to the buffer,
        NI-DAQmx can generate a combination of old and new data, a phenomenon called glitching.

        if state=True: Allow NI-DAQmx to regenerate samples that the device previously generated.
        When you choose this value, the write marker returns to the beginning of the buffer after the
        device generates all samples currently in the buffer.

        if state=False: Do not allow NI-DAQmx to regenerate samples the device previously generated.
        When you choose this value, NI-DAQmx waits for you to write more samples to the buffer or
        until the timeout expires.
        """
        datadict = {"10097": True, "10158": False}
        state_nr = int32(10097) if state else int32(10158)
        DAQmxSetWriteRegenMode(self.handle, state_nr)

    def set_write_relative_to(self, mode='first'):
        """
        Specifies the point in the buffer at which to write data. If you also specify an offset with Offset,
        the write operation begins at that offset relative to this point you select with this property.
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/func190c/
        :return:
        """
        dataDict = {'first': DAQmx_Val_FirstSample, 'current': DAQmx_Val_CurrWritePos}
        if mode in list(dataDict.keys()):
            DAQmxSetWriteRelativeTo(self.handle, dataDict[mode])
        else:
            print("Mode must be one of the following: ", list(dataDict.keys()))

    def get_write_relative_to(self):
        """
        Specifies the point in the buffer at which to write data. If you also specify an offset with Offset,
        the write operation begins at that offset relative to this point you select with this property.
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/func190c/
        :return:
        """
        data = int32()
        DAQmxGetWriteRelativeTo(self.handle, byref(data))
        dataDict = {'10424' : 'first',
                    '10430' : 'current'}
        return dataDict[str(data.value)]

    def get_current_write_position(self):
        """
        Indicates the position in the buffer of the next sample to generate.
        This value is identical for all channels in the task.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/func1458/
        """
        data = uInt64()
        DAQmxGetWriteCurrWritePos(self.handle, byref(data))
        return data.value

    def get_write_offset(self):
        """
        Specifies in samples per channel an offset at which a write operation begins.
        This offset is relative to the location you specify with Relative To.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr190d/
        :return:
        """
        data = int32()
        DAQmxGetWriteOffset(self.handle, byref(data))
        return data.value

    def set_write_offset(self, offset):
        """
        Specifies in samples per channel an offset at which a write operation begins.
        This offset is relative to the location you specify with Relative To.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr190d/
        :param offset:
        :return:
        """
        if offset < self.get_output_buffer_size():
            DAQmxSetWriteOffset(self.handle, int32(offset))
        else:
            print("Offset must be smaller than the buffer size (%d)!" % self.get_output_buffer_size())

    def get_available_buffer_size(self):
        """
        Indicates in samples per channel the amount of available space in the buffer.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr1460/
        """
        data = uInt32()
        DAQmxGetWriteSpaceAvail(self.handle, byref(data))
        return data.value

    def get_output_buffer_size(self):
        """
        Specifies the number of samples the output buffer can hold for each channel in the task.
        Zero indicates to allocate no buffer. Use a buffer size of 0 to perform a hardware-timed operation
        without using a buffer. Setting this property overrides the automatic output buffer allocation
        that NI-DAQmx performs.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/func186d/
        :return:
        """
        data = uInt32()
        DAQmxGetBufOutputBufSize(self.handle, byref(data))
        return data.value

    def get_onboard_buffer_size(self):
        """
        This is also called the FIFO buffer
        Specifies in samples per channel the size of the onboard output buffer of the device.
        :return:
        """
        data = uInt32()
        DAQmxGetBufOutputOnbrdBufSize(self.handle, byref(data))
        return data.value

    def set_onboard_buffer_size(self, size):
        """
        This is also called the FIFO buffer
        Specifies in samples per channel the size of the onboard output buffer of the device.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr230b/
        :param size:
        :return:
        """
        DAQmxSetBufOutputOnbrdBufSize(self.handle, uInt32(size))

    def get_bypass_memory_buffer(self):
        """
        Specifies whether to write samples directly to the onboard memory of the device, bypassing the memory buffer.
        Generally, you cannot update onboard memory directly after you start the task. Onboard memory includes data FIFOs.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr183a/
        :return:
        """
        data = uInt32()
        DAQmxGetAOUseOnlyOnBrdMem(self.handle, "", byref(data))
        return data.value

    def set_bypass_memory_buffer(self, bypass=True, channel=""):
        """
        Specifies whether to write samples directly to the onboard memory of the device, bypassing the memory buffer.
        Generally, you cannot update onboard memory directly after you start the task. Onboard memory includes data FIFOs.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr183a/
        :return:
        """
        DAQmxSetAOUseOnlyOnBrdMem(self.handle, channel, uInt32(bypass))

    def wait_until_operation_completion(self, maxtime=10.0):
        """
        Waits for the measurement or generation to complete.
        Use this function to ensure that the specified operation is complete before you stop the task.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/daqmxcfunc/daqmxwaituntiltaskdone/
        """
        # Blocks
        return DAQmxWaitUntilTaskDone(self.handle, float64(maxtime))

    def get_operation_completion(self):
        """
        Queries the status of the task and indicates if it completed execution.
        Use this function to ensure that the specified operation is complete before you stop the task.
        http://zone.ni.com/reference/en-XX/help/370471AG-01/daqmxcfunc/daqmxistaskdone/
        """
        # Non-blocking
        data = uInt32()
        DAQmxIsTaskDone(self.handle, byref(data))
        return data.value

    def configure_start_trigger(self):
        """
        Optional, run after
        - create_task()
        - setup_channel()
        - setup_timing()
        """
        DAQmxCfgDigEdgeStartTrig(self.handle, triggerSource="PFI0", triggerEdge=DAQmx_Val_RisingSlope)

    def configure_pause_trigger(self, enable=True, terminal='PFI0', polarity=1):
        """
        Optional, run after
        - create_task()
        - setup_channel()
        - setup_timing()
        http://zone.ni.com/reference/en-XX/help/370471AG-01/mxcprop/attr1366/
        """
        configure = DAQmx_Val_DigLvl if enable else DAQmx_Val_None
        DAQmxSetPauseTrigType(self.handle, configure)
        if enable:
            self.set_pause_trigger_polarity(polarity)
            self.set_pause_trigger_source(terminal)

    def set_pause_trigger_polarity(self, polarity):
        """
        Specifies whether the task pauses while the signal is high or low.
        :param polarity: 1 (pauses when high), 0 (pauses when low)
        :return:
        """
        data = DAQmx_Val_High if polarity else DAQmx_Val_Low
        DAQmxSetDigLvlPauseTrigWhen(self.handle, data)

    def get_pause_trigger_polarity(self):
        """
        Gets whether the task pauses while the signal is high or low.
        :return: Bool
        """
        data = int32()
        DAQmxGetDigLvlPauseTrigWhen(self.handle, byref(data))
        datadict = {"10192" : 1, "10214" : 0}
        return datadict[str(data.value)]

    def get_pause_trigger_source(self):
        """
        Specifies the name of a terminal where there is a digital signal to use as the source of the Pause Trigger.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr1379/
        :return:
        """
        data = cchar()
        DAQmxGetDigLvlPauseTrigSrc(self.handle, byref(data), int32(1000))
        return data.value

    def set_pause_trigger_source(self, terminal='PFI0'):
        """
        Specifies the name of a terminal where there is a digital signal to use as the source of the Pause Trigger.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr1379/
        :return:
        """
        DAQmxSetDigLvlPauseTrigSrc(self.handle, terminal)

    def get_accept_multiple_start_triggers(self):
        """
        Specifies whether a finite task resets and waits for another Start Trigger after the task completes.
        When you set this property to TRUE, the device performs a finite acquisition or generation each time the
        Start Trigger occurs until the task stops.
        The device ignores a trigger if it is in the process of acquiring or generating signals.
        """
        data = uInt32()
        DAQmxGetStartTrigRetriggerable(self.handle, byref(data))
        return data.value

    def set_accept_multiple_start_triggers(self, accept=True):
        """
        Specifies whether a finite task resets and waits for another Start Trigger after the task completes.
        When you set this property to TRUE, the device performs a finite acquisition or generation each time the
        Start Trigger occurs until the task stops.
        The device ignores a trigger if it is in the process of acquiring or generating signals.
        """
        data = uInt32(accept)
        DAQmxSetStartTrigRetriggerable(self.handle, data)

    def get_data_transfer_mechanism(self):
        """
        Specifies the data transfer mode for the device.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr0134/
        :return:
        """
        data = int32()
        DAQmxGetAODataXferMech(self.handle, "", byref(data))
        dataDict = {"10054" : "DMA",
                    "10204" : "IRQ",
                    "10264" : "ProgrammedIO",
                    "12590" : "USBbulk"}
        return dataDict[str(data.value)]

    def set_data_transfer_mechanism(self, mechanism):
        """
        Specifies the data transfer mode for the device.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr0134/
        :return:
        """
        dataDict = {"DMA" : DAQmx_Val_DMA,
                    "IRQ" : DAQmx_Val_Interrupts,
                    "ProgrammedIO" : DAQmx_Val_ProgrammedIO,
                    "USBbulk" : DAQmx_Val_USBbulk}
        if mechanism in list(dataDict.keys()):
            DAQmxSetAODataXferMech(self.handle, "", dataDict[mechanism])

    def get_data_transfer_request_condition(self):
        """
        Specifies under what condition to transfer data from the buffer to the onboard memory of the device.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr183c/
        :return:
        """
        data = int32()
        DAQmxGetAODataXferReqCond(self.handle, "", byref(data))
        dataDict = {"10235" : "empty",
                    "10239" : "halfempty",
                    "10242" : "notfull"}
        return dataDict[str(data.value)]

    def set_data_transfer_request_condition(self, condition):
        """
        Specifies under what condition to transfer data from the buffer to the onboard memory of the device.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr183c/
        :condition: must be one of "empty", "halfempty", "notfull"
        :return:
        """
        dataDict = {"empty" : DAQmx_Val_OnBrdMemEmpty,
                    "halfempty" : DAQmx_Val_OnBrdMemHalfFullOrLess,
                    "notfull" : DAQmx_Val_OnBrdMemNotFull}
        DAQmxSetAODataXferReqCond(self.handle, "", dataDict[condition])

    def get_usb_transfer_request_size(self):
        """
        Specifies the maximum size of a USB transfer request in bytes.
        Modify this value to affect performance under different combinations of operating system and device.
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/attr2a8f/
        :return:
        """
        data = uInt32()
        DAQmxGetAOUsbXferReqSize(self.handle, "", byref(data))
        return data.value

    def set_usb_transfer_request_size(self, size):
        """
        Specifies the maximum size of a USB transfer request in bytes.
        Modify this value to affect performance under different combinations of operating system and device.
        http://zone.ni.com/reference/en-XX/help/370471AA-01/mxcprop/attr2a8f/
        :param size: Integer > 0.
        :return:
        """
        DAQmxSetAOUsbXferReqSize(self.handle, "", uInt32(size))

    def print_settings(self):
        names = ["id", "sample_timing_type", "sample_clock_rate", "resolution",
                 "idle_output_setting", "write_regeneration_mode", "write_relative_to",
                 "current_write_position", "write_offset", "available_buffer_size",
                 "output_buffer_size", "onboard_buffer_size", "bypass_memory_buffer",
                 "data_transfer_mechanism", "data_transfer_request_condition",
                 "usb_transfer_request_size", "usb_transfer_request_count"]
        props = list()
        for n in names:
            try:
                props.append(getattr(self, "get_%s" % n)())
            except:
                props.append("Error")

        print(tabulate(zip(names, props), headers=["Parameter", "Value"],
                       tablefmt="fancy_grid", floatfmt="", numalign="center", stralign="left"))

        return zip(names, props)

if __name__ == "__main__":
    sampleRate = 51200
    ni = NI9260()

    if 1:
        # Sharing a sample clock between different modules across the same device
        sampleSize = 5120

        ni.setup_channel(name="cDAQ9189-1C742CBMod1/ao0:1, cDAQ9189-1C742CBMod2/ao0")
        # ni.setup_channel(name="cDAQ9189-1C742CBMod2/ao0")
        # ni.setup_channel(name="cDAQ9189-1C742CBMod1/ao0:1")

        ni.set_sample_clock(runContinuous=True, sampleSize=sampleSize, sampleRate=sampleRate)
        ni.set_idle_output_setting("MaintainCurrentValue")
        ni.set_write_regeneration_mode(True)
        ni.set_write_relative_to('first')
        actualSampleRate = ni.get_sample_clock_rate()

        t = np.linspace(0, 1 / float(actualSampleRate) * sampleSize, sampleSize);
        y1 = 0.5 * np.sin(2 * np.pi * t * actualSampleRate / float(sampleSize))
        y2 = y1
        y3 = y1

        plt.figure()
        plt.plot(t, y2, '.-b', label='cDAQ9189-1C742CBMod1/ao1')
        plt.plot(t, y1, '.-g', label='cDAQ9189-1C742CBMod1/ao0')
        plt.plot(t, y3, '.-r', label='cDAQ9189-1C742CBMod2/ao0')
        plt.legend(loc=0, prop={"size" : 10}, frameon=False)
        plt.show()

        ni.set_waveform(np.append(np.append(y1, y2), y3), numSamplesPerChan=sampleSize, autostart=True)
        # ni.set_waveform(y1, numSamplesPerChan=sampleSize, autostart=True)
        time.sleep(50)
        # ni.start_task()

    if 0:
        sys.path.append(r"S:\_Data\170422 - EonHe M018V6 with L3 etch\experiment")
        from ni_fast_sweep import triangular, pulse_train

        Vmin = 0.350
        Vmax = 0.600
        triggerPoints = 10

        assert sampleRate == 51200

        t, y1 = triangular(Vmin, Vmax, dV=.5E-3/triggerPoints, dt=1/sampleRate, Ncopies=1)
        y2 = pulse_train(t, frequency=sampleRate/(triggerPoints), waveformperiod=t[-1] - t[0])

        plt.figure()
        plt.plot(t, y2, '.-b', label='ch1')
        plt.plot(t, y1, '.-g', label='ch0')
        plt.legend(loc=0, prop={"size" : 10}, frameon=False)
        plt.show()

        m = 100
        y1 = np.tile(y1, m)
        y2 = np.tile(y2, m)

        actualSampleSize = len(y1)

        ni.setup_all_channels()
        ni.set_sample_clock(runContinuous=False, sampleSize=actualSampleSize, sampleRate=sampleRate)
        ni.set_idle_output_setting("MaintainCurrentValue")
        # ni.set_write_regeneration_mode(True)
        ni.set_write_relative_to('first')
        actualSampleRate = ni.get_sample_clock_rate()

        ni.set_waveform(np.append(y1, y2), numSamplesPerChan=actualSampleSize, autostart=False)
        ni.print_settings()
        ni.start_task(wait=60.0)

    if 0:
        ni.setup_channel()
        ni.set_sample_clock(runContinuous=True, sampleSize=sampleSize, sampleRate=sampleRate)
        ni.set_idle_output_setting("MaintainCurrentValue")
        ni.configure_pause_trigger(polarity=0)
        ni.set_write_regeneration_mode(True)
        ni.set_write_relative_to('first')

        actualSampleRate = ni.get_sample_clock_rate()

        t = np.linspace(0, 1 / float(actualSampleRate) * sampleSize, sampleSize);
        y1 = 0.5 * np.sin(2 * np.pi * t * sampleRate/float(sampleSize))
        # y2 = 0.5 * np.sin(2 * np.pi * t * sampleRate/float(sampleSize))

        plt.figure()
        plt.plot(t, y1, 'g')
        # plt.plot(t, y2, 'b')
        plt.show()

        # ni.set_bypass_memory_buffer(bypass=1, channel="cDAQ9189-1C742CBMod1/ao0")

        ni.set_waveform(y1, numSamplesPerChan=sampleSize, autostart=False)
        # ni.set_waveform(np.append(y1, y2), numSamplesPerChan=sampleSize, autostart=False)
        ni.print_settings()
        ni.start_task(wait=True)

        # for k in range(10):
        #     print("Current write position: ", ni.get_current_write_position())
        #     time.sleep(0.5)

    if 0:
        # DC Mode: dynamically update the value of a constant DC offset
        ni.set_volt(0.5, first_time=True)
        ni.print_settings()

        for k in range(10):
            curr_write_pos = ni.get_current_write_position()
            print(0.0 + k * 0.1, ": Current write position: ", curr_write_pos)
            ni.set_volt(0.0 + k * 0.1)
            time.sleep(1.0)