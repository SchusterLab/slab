"""
For a comprehensive list of all allowed functions
http://zone.ni.com/reference/en-XX/help/370471AE-01/cdaqmxsupp/ni9260bnc/ (for the card itself)
http://zone.ni.com/reference/en-XX/help/370471AE-01/cdaqmxsupp/cdaq-9188/ (for the chassis)
"""

import time
from matplotlib import pyplot as plt
import PyDAQmx
import numpy as np
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *

int32 = ctypes.c_long
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
float64 = ctypes.c_double
cchar = ctypes.c_char

class NI9260():

    def __init__(self, name="cDAQ9189-1C742CBMod1"):
        self.slot_name = name
        self.query_sleep = 0.50
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

    def setup_channel(self, channel=0, channelLimits=(-4.23, +4.23)):
        """
        Run after
        - create_task()
        """
        name = "%s/ao%d" % (self.slot_name, channel)
        DAQmxCreateAOVoltageChan(self.handle, name, "", channelLimits[0], channelLimits[1], DAQmx_Val_Volts, None)

    def setup_all_channels(self, channelLimits=(-4.23, +4.23)):
        """
        Run after
        - create_task()
        """
        name = "%s/ao0:1" % self.slot_name
        DAQmxCreateAOVoltageChan(self.handle, name, "", channelLimits[0], channelLimits[1], DAQmx_Val_Volts, None)

    def set_sample_clock(self, runContinuous=True, sampleSize=1000, sampleRate=1000):
        """
        Run after
        - create_task()
        - setup_channel()
        """
        sampleMode = DAQmx_Val_ContSamps if runContinuous else DAQmx_Val_FiniteSamps
        DAQmxCfgSampClkTiming(self.handle, "", float64(sampleRate), DAQmx_Val_Rising, sampleMode, uInt64(sampleSize))

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

    def start_task(self, wait=True):
        DAQmxStartTask(self.handle)
        if wait:
            return self.wait_until_operation_completion()

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

    def get_voltage_output_mode(self):
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

    def set_voltage_output_mode(self, mode='on_demand'):
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

    def set_volt(self, value):
        """
        Run after
        - create_task()
        - setup_channel()
        - setup_timing()

        to engage output, run start_task() and trigger, if necessary.
        """
        DAQmxWriteAnalogScalarF64(self.handle, autoStart=False, timeout=self.query_sleep, value=float64(value), reserved=None)

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

    def set_bypass_memory_buffer(self, bypass=True):
        """
        Specifies whether to write samples directly to the onboard memory of the device, bypassing the memory buffer.
        Generally, you cannot update onboard memory directly after you start the task. Onboard memory includes data FIFOs.
        http://zone.ni.com/reference/en-XX/help/370471AE-01/mxcprop/attr183a/
        :return:
        """
        DAQmxSetAOUseOnlyOnBrdMem(self.handle, "", bypass)

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

if __name__ == "__main__":
    sampleRate = 20000
    sampleSize = 250

    ni = NI9260()

    if 0:
        self.setup_all_channels()
        self.set_sample_clock(runContinuous=True, sampleSize=sampleSize, sampleRate=sampleRate)
        self.set_idle_output_setting("ZeroVolts")
        self.configure_pause_trigger(polarity=1)
        # self.set_onboard_buffer_size(sampleSize)
        # self.set_write_regeneration_mode(True)
        # self.set_write_relative_to('current')
        # self.set_bypass_memory_buffer(True)

        t = np.linspace(0, 1 / float(sampleRate) * sampleSize, sampleSize);
        a = 0.5
        self.set_waveform(np.append(a * np.cos(2 * np.pi * t), a * np.sin(2 * np.pi * t)), autostart=False)
        # self.start_task()

        # self.setup_timing()
        # self.set_waveform(0.5*np.ones(1000))
        # self.start_task()

        print("Write regeneration mode: ", self.get_write_regeneration_mode())
        print("Pause trigger polarity: ", self.get_pause_trigger_polarity())
        print("Output buffer size: ", self.get_output_buffer_size())
        print("Onboard buffer size: ", self.get_onboard_buffer_size())

        plt.figure()
        plt.plot(t, a * np.cos(2 * np.pi * t), 'g')
        plt.plot(t, a * np.sin(2 * np.pi * t), 'b')
        plt.show()
    if 0:
        ni.setup_channel()
        ni.set_sample_clock(runContinuous=True, sampleSize=sampleSize, sampleRate=sampleRate)
        ni.set_idle_output_setting("MaintainCurrentValue")
        ni.configure_pause_trigger(polarity=0)

        t = np.linspace(0, 1 / float(sampleRate) * sampleSize, sampleSize);
        a = 0.5
        ni.set_waveform(a * np.cos(2 * np.pi * t), autostart=False)

        print("Write regeneration mode: ", self.get_write_regeneration_mode())
        print("Pause trigger polarity: ", self.get_pause_trigger_polarity())
        print("Output buffer size: ", self.get_output_buffer_size())
        print("Onboard buffer size: ", self.get_onboard_buffer_size())

        ni.start_task(wait=True)
    if 1:
        ni.setup_channel()
        ni.set_write_regeneration_mode(True)
        ni.set_sample_clock(runContinuous=True, sampleSize=sampleSize, sampleRate=sampleRate)
        ni.set_idle_output_setting("MaintainCurrentValue")
        ni.set_bypass_memory_buffer(False)
        ni.set_write_relative_to('first')
        ni.set_write_offset(0)

        actualSampleRate = ni.get_sample_clock_rate()

        t = np.linspace(0, 1 / float(actualSampleRate) * sampleSize, sampleSize)
        y = 0.5 * np.cos(2 * np.pi * 4 * t * actualSampleRate)

        ni.set_waveform(y, numSamplesPerChan=sampleSize, autostart=True)
        time.sleep(2.0)

        plt.figure()
        plt.plot(t, y, 'g')
        # plt.plot(t, a * np.sin(2 * np.pi * t), 'b')
        plt.show()

        print("Sample clock rate: ", ni.get_sample_clock_rate())
        print("Write regeneration mode: ", ni.get_write_regeneration_mode())
        print("Write offset: ", ni.get_write_offset())
        print("Write relative to: ", ni.get_write_relative_to())
        print("Output buffer size: ", ni.get_output_buffer_size())
        print("Onboard buffer size: ", ni.get_onboard_buffer_size())

        for k in range(10):
            print(k, )
            y = (0.6 + k * 0.1) * np.cos(2 * np.pi * 4 * t * actualSampleRate)
            print(ni.set_waveform(y, numSamplesPerChan=sampleSize, autostart=True))
            time.sleep(2.0)
            # ni.stop_task()

            # ni.wait_until_operation_completion(maxtime=2.0)
            # time.sleep(0.1)
            # ni.set_write_offset(-ni.get_current_write_position())
            # time.sleep(0.1)# + sampleSize / float(sampleRate))
            # ni.stop_task()

        print("Sample clock rate: ", ni.get_sample_clock_rate())
        print("Write regeneration mode: ", ni.get_write_regeneration_mode())
        print("Write offset: ", ni.get_write_offset())
        print("Write relative to: ", ni.get_write_relative_to())
        print("Output buffer size: ", ni.get_output_buffer_size())
        print("Onboard buffer size: ", ni.get_onboard_buffer_size())



