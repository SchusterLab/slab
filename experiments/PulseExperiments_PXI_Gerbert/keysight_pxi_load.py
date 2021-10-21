# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:41:56 2018

@author: slab
"""

# %pylab inline
from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
from slab.experiments.PulseExperiments_PXI_Gerbert.sequences_pxi import PulseSequences
import time
import numpy as np
import os
import json
import threading
import matplotlib.pyplot as plt
from tqdm import tqdm
from slab.datamanagement import SlabFile


AWG_BANDWIDTH = 10**9 #Samples per second
QB_W_FAST_FLUX = [0, 1, 2, 3, 4, 5, 6, 7] #qubits where we will load fast flux waveforms. Note: hardcoded that Q0-Q3 goes to module ff1, Q4-Q7 goes to module ff2

class KeysightSingleQubit:
    '''Class designed to implement a simple single qubit experiment given pulse sequences from the Sequencer class. Does  not use
    HVI technology.
=
    Module (slot) 5 is an AWG, used for fast flux pulses
    Module 6 is an AWG, used for fast flux pulses
    Module 7 is an AWG. channel 1 goes to the I input to the mixer, channel 2 goes to the Q input
    Module 4 is used as a marker for the LO (ie send signal to switch).
    Module 9 is for I, Q, and marker for stabilizer
        ch4 of this trig for the digitizer.

    Module 10 is the digitizer. channel 1 is for readout of I component and channel 2 is for readout from Q
    component.'''
    ## LO             _/-\_/-\_/-\_/-\_/-\   # LO always on
    ## Marker         _______|------|______  # marker windows LO output in time w/ AWG signals non-zero state (+ wiggle room!)
    ## AWG Signal     _________/??\________  # AWG card outputs IQ signal to mixer, mixer input is windows LO+marker!


    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, sequences, name, save_path=r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials=50 * 1000):  # 1000*10000 if you want to watch sweep by eye

        chassis = key.KeysightChassis(1,
                                      {4: key.ModuleType.OUTPUT,
                                       5: key.ModuleType.OUTPUT,
                                       6: key.ModuleType.OUTPUT,
                                       7: key.ModuleType.OUTPUT,
                                       8: key.ModuleType.OUTPUT,
                                       9: key.ModuleType.OUTPUT,
                                       10: key.ModuleType.INPUT})

        self.hardware_cfg = hardware_cfg
        self.AWG_mod_no = hardware_cfg['awg_info']['keysight_pxi']['AWG_mod_no'] ## AWG_mod_no is qubit AWG output
        self.marker_mod_no = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no'] ## marker is a pulse that's on for length that you want LO switch on
        self.stab_mod_no = hardware_cfg['awg_info']['keysight_pxi']['stab_mod_no'] ## IQ and marker for 
        # stabilizer, + triggers Digitizer
        self.ff1_mod_no = hardware_cfg['awg_info']['keysight_pxi']['ff1_mod_no'] #4 channels for Q0-Q3 fast flux
        self.ff2_mod_no = hardware_cfg['awg_info']['keysight_pxi']['ff2_mod_no']# 4 channels for Q4-Q7 fast flux
        self.dig_mod_no = hardware_cfg['awg_info']['keysight_pxi']['dig_mod_no']  # digitizer card

        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_m = hardware_cfg['awg_info']['keysight_pxi']['dt_m']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        self.dt_M3201A = hardware_cfg['awg_info']['keysight_pxi']['dt_M3201A']
        self.adc_range =  hardware_cfg['awg_info']['keysight_pxi']['digtzr_vpp_range']

        if 'A' in quantum_device_cfg["setups"]:
            self.readoutA_window = np.array(quantum_device_cfg['readout']['A']['window'])

            if quantum_device_cfg["weighted_readout"]["weighted_readout_on"]:
                self.readoutA_weight = np.load(quantum_device_cfg["weighted_readout"]["A"]["filenames"])
            else:
                self.readoutA_weight = np.ones((self.readoutA_window[1]-self.readoutA_window[0],1))
        if 'B' in quantum_device_cfg["setups"]:
            self.readoutB_window = np.array(quantum_device_cfg['readout']['B']['window'])

            if quantum_device_cfg["weighted_readout"]["weighted_readout_on"]:
                self.readoutB_weight = np.load(quantum_device_cfg["weighted_readout"]["B"]["filenames"])
            else:
                self.readoutB_weight = np.ones((self.readoutB_window[1]-self.readoutB_window[0],1))


        
        self.qb_lo_conv = hardware_cfg['awg_info']['keysight_pxi']['qb_lo_conv'] #lo_delay convolves LO marker
        self.stab_lo_conv = hardware_cfg['awg_info']['keysight_pxi']['stab_lo_conv']  # lo_delay convolves LO marker
        self.hardware_delays = hardware_cfg['awg_info']['keysight_pxi']['channels_delay_hardware_10ns']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']

        print ("Module used for generating Q1 IQ  pulses = ",self.AWG_mod_no)
        print ("Module used for generating digital markers for LO = ",self.marker_mod_no)
        print ("Module used to trigger dig and for stabilizer  = ", self.stab_mod_no)
        print("Module used for generating fast flux pluses for Q0-Q3 = ", self.ff1_mod_no)
        print("Module used for generating fast flux pluses for Q4-Q7 = ", self.ff2_mod_no)
        self.out_mod_nums = [self.AWG_mod_no, self.marker_mod_no, self.stab_mod_no, self.ff1_mod_no, self.ff2_mod_no]

        self.on_qubits = quantum_device_cfg['setups']
        self.num_avg = experiment_cfg[name]['acquisition_num']
        self.num_expt = sequences["digtzr_trig"].shape[0] #take arbitrary channel, shape will tell you num_expt
        self.trigger_period = self.hardware_cfg['trigger']['period_us']
        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']
        self.totaltime = self.num_avg * self.num_expt * self.trigger_period * 1e-6 / 60.0

        self.chassis = chassis
        self.awg_channels = range(1,5)
        self.dig_channels = range(1,5)


        # Initialize AWG Cards!

        # initialize modules
        self.AWG_module = chassis.getModule(self.AWG_mod_no)
        self.m_module = chassis.getModule(self.marker_mod_no)
        self.stab_module = chassis.getModule(self.stab_mod_no)
        self.ff1_module = chassis.getModule(self.ff1_mod_no)
        self.ff2_module = chassis.getModule(self.ff2_mod_no)
        self.DIG_module = chassis.getModule(self.dig_mod_no)

        self.out_mods = [self.AWG_module, self.m_module, self.stab_module, self.ff1_module, self.ff2_module]

        # Initialize channels on fast flux card 1
        self.ff1_ch_1 = chassis.getChannel(self.ff1_mod_no, 1)
        self.ff1_ch_2 = chassis.getChannel(self.ff1_mod_no, 2)
        self.ff1_ch_3 = chassis.getChannel(self.ff1_mod_no, 3)
        self.ff1_ch_4 = chassis.getChannel(self.ff1_mod_no, 4)

        # Initialize channels on fast flux card 2
        self.ff2_ch_1 = chassis.getChannel(self.ff2_mod_no, 1)
        self.ff2_ch_2 = chassis.getChannel(self.ff2_mod_no, 2)
        self.ff2_ch_3 = chassis.getChannel(self.ff2_mod_no, 3)
        self.ff2_ch_4 = chassis.getChannel(self.ff2_mod_no, 4)

        # Initialize channels on qubit IQ pulse card.  Ch1 = AWG I, CH2 = AWG Q
        self.AWG_ch_1 = chassis.getChannel(self.AWG_mod_no, 1)
        self.AWG_ch_2 = chassis.getChannel(self.AWG_mod_no, 2)
        self.AWG_ch_3 = chassis.getChannel(self.AWG_mod_no, 3)
        self.AWG_ch_4 = chassis.getChannel(self.AWG_mod_no, 4)

        # Initialize channels on Marker card. Digital markers for qubit, readout
        self.m_ch_1 = chassis.getChannel(self.marker_mod_no, 1)
        self.m_ch_2 = chassis.getChannel(self.marker_mod_no, 2)
        self.m_ch_3 = chassis.getChannel(self.marker_mod_no, 3)
        self.m_ch_4 = chassis.getChannel(self.marker_mod_no, 4)

        # Initialize channels card that generates stabilizer stuff, and also triggers the digitizer
        self.stab_ch_1 = chassis.getChannel(self.stab_mod_no, 1)
        self.stab_ch_2 = chassis.getChannel(self.stab_mod_no, 2)
        self.stab_ch_3 = chassis.getChannel(self.stab_mod_no, 3)
        self.digtzr_trig_ch = chassis.getChannel(self.stab_mod_no, 4)

        # Initialize digitizer card
        # self.DIG_chs = [chassis.getChannel(10, ch) for ch in self.dig_channels]
        self.DIG_ch_1 = chassis.getChannel(self.dig_mod_no, 1)
        self.DIG_ch_2 = chassis.getChannel(self.dig_mod_no, 2)
        self.DIG_ch_3 = chassis.getChannel(self.dig_mod_no,3)
        self.DIG_ch_4 = chassis.getChannel(self.dig_mod_no, 4)


        self.data_1,self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)),np.zeros((self.num_expt, self.DIG_sampl_record))
        self.A_I,self.A_Q = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.data_3, self.data_4 = np.zeros((self.num_expt, self.DIG_sampl_record)), np.zeros(
            (self.num_expt, self.DIG_sampl_record))
        self.B_I, self.B_Q = np.zeros(self.num_expt), np.zeros(self.num_expt)



        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally


    def configureChannels(self, hardware_cfg, experiment_cfg, quantum_device_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in Vpp.'''

        amp_AWG = hardware_cfg['awg_info']['keysight_pxi']['amp_awg']
        amp_mark = hardware_cfg['awg_info']['keysight_pxi']['amp_mark']
        amp_stab = hardware_cfg['awg_info']['keysight_pxi']['amp_stab']
        amp_digtzr_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_stab'][3] ##THIS IS ON PURPOSE
        amp_ff1 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff1']
        amp_ff2 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff2']
        if 'A' in quantum_device_cfg["setups"]:
            IQ_dc_offsetA = quantum_device_cfg['pulse_info']['A']['IQ_dc']
        else:
            IQ_dc_offsetA = quantum_device_cfg['pulse_info']['B']['IQ_dc']
        if 'B' in quantum_device_cfg["setups"]:
            IQ_dc_offsetB = quantum_device_cfg['pulse_info']['B']['IQ_dc']
        else:
            IQ_dc_offsetB = quantum_device_cfg['pulse_info']['A']['IQ_dc']


        DIG_ch_delays = hardware_cfg["awg_info"]['keysight_pxi']["DIG_channels_delay_samples"]


        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        ## Output modules: Configure amplitude, dc offset, and if "trigger" channel on a module is used as a receiver or a trigger source
        # ie triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN) means it's receiving a trigger
        print ("Configuring qubit IQ channels")
        self.AWG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_ch_1.configure(amplitude=amp_AWG[0], offset_voltage = IQ_dc_offsetA)
        self.AWG_ch_2.configure(amplitude=amp_AWG[1], offset_voltage = IQ_dc_offsetA)
        self.AWG_ch_3.configure(amplitude=amp_AWG[2], offset_voltage = IQ_dc_offsetB)
        self.AWG_ch_4.configure(amplitude=amp_AWG[3], offset_voltage = IQ_dc_offsetB)

        print("Configuring marker channels")
        self.m_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_ch_1.configure(amplitude=amp_mark[0])
        self.m_ch_2.configure(amplitude=amp_mark[1])
        self.m_ch_3.configure(amplitude=amp_mark[2])
        self.m_ch_4.configure(amplitude=amp_mark[3])

        print("Configuring trigger channels")
        self.stab_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.stab_ch_1.configure(amplitude=amp_stab[0])
        self.stab_ch_2.configure(amplitude=amp_stab[1])
        self.stab_ch_3.configure(amplitude=amp_stab[2])
        self.digtzr_trig_ch.configure(amplitude=amp_digtzr_trig)
        print ("Dig card trigger amplitude = ",amp_digtzr_trig)

        print("Configuring fast flux module1 channels")
        self.ff1_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.ff1_ch_1.configure(amplitude=amp_ff1[0])
        self.ff1_ch_2.configure(amplitude=amp_ff1[1])
        self.ff1_ch_3.configure(amplitude=amp_ff1[2])
        self.ff1_ch_4.configure(amplitude=amp_ff1[3])

        print("Configuring fast flux module2 channels")
        self.ff2_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.ff2_ch_1.configure(amplitude=amp_ff2[0])
        self.ff2_ch_2.configure(amplitude=amp_ff2[1])
        self.ff2_ch_3.configure(amplitude=amp_ff2[2])
        self.ff2_ch_4.configure(amplitude=amp_ff2[3])


        print ("Setting trigger mode for all channels of all output modules to External")
        for n in range(1, 5):
            self.AWG_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.stab_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.ff1_module.AWGtriggerExternalConfig(nAWG=n,
                                                      externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                                      triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.ff2_module.AWGtriggerExternalConfig(nAWG=n,
                                                      externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                                      triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)


        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")
        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale=self.adc_range, delay=DIG_ch_delays[0], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,delay=DIG_ch_delays[1], points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale=self.adc_range, delay=DIG_ch_delays[2], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale=self.adc_range, delay=DIG_ch_delays[3], points_per_cycle=self.DIG_sampl_record,
                                cycles=num_expt * num_avg, buffer_time_out=100000,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)

    def configureDigChannels(self, hardware_cfg, experiment_cfg, quantum_device_cfg, name):
        '''Configures the DIG channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in Vpp.'''

        DIG_ch_delays = hardware_cfg["awg_info"]['keysight_pxi']["DIG_channels_delay_samples"]


        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")
        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale=self.adc_range, delay=DIG_ch_delays[0], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,delay=DIG_ch_delays[1], points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale=self.adc_range, delay=DIG_ch_delays[2], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale=self.adc_range, delay=DIG_ch_delays[3], points_per_cycle=self.DIG_sampl_record,
                                cycles=num_expt * num_avg, buffer_time_out=100000,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)

    def generatemarkers_old(self,wvs, channel, dt_src=1, dt_mark = 1, conv_width = 1,trig_delay = 0):
        """if given a waveform, it will produce square pulses anywhere that waveform isn't zero. Used to generate
        markers for LOs that get passed through AWGs.
        resample: if the marker is coming from a card with a half the sample/second rate (ie 500MS/s vs 1GS/s,
        it will resample the marker array so that the timing comes out right"""
        
        waveform = wvs[channel]
        
        #dt_src is dt of the channel you are making markers from, given by the dt of its awg
        dt_src = self.get_dt_channel(channel)
        
        markers = np.array([np.append(np.heaviside(np.convolve(abs(np.append(w[int(trig_delay):],np.zeros(int(trig_delay)))), np.ones((int(conv_width),)) / conv_width), 0)[int(conv_width / 2):], np.zeros(int(conv_width / 2))) for w in waveform])
        if dt_src!=dt_mark: return np.array([m[::int(dt_mark/dt_src)] for m in markers])
        else:return markers

    def generatemarkers(self, wvs, channel, dt_mark=1, conv_width=1, trig_delay=0):
        """if given a waveform, it will produce square pulses anywhere that waveform isn't zero. Used to generate
        markers for LOs that get passed through AWGs.
        resample: if the marker is coming from a card with a half the sample/second rate (ie 500MS/s vs 1GS/s,
        it will resample the marker array so that the timing comes out right"""

        waveform = wvs[channel]

        # dt_src is dt of the channel you are making markers from, given by the dt of its awg
        dt_src = self.get_dt_channel(channel)

        markers = np.array([np.append(
                                np.heaviside(
                                    np.convolve(
                                        abs(np.append(
                                            w[int(trig_delay):],
                                            np.zeros(int(trig_delay))
                                        )),
                                        np.ones((int(conv_width),)) / conv_width
                                    ), 0)[int(conv_width / 2):], np.zeros(int(conv_width / 2))) for w in waveform])

        if dt_src != dt_mark:
            return np.array([m[::int(dt_mark / dt_src)] for m in markers])
        else:
            return markers

    def generatemastertrigger(self,length,trig_width = 1,trig_delay = 0):
        trig = np.zeros(length)
        for i in np.arange(int(trig_delay/self.dt_dig),int((trig_delay + trig_width)/self.dt_dig)):
            trig[i] = 1
        return trig

    def sequenceslist(self,sequences,waveform_channels):
        """takes the sequences arrays produced in "pulse squences" class and stored in "sequences" dict and renames
        the to  a dictionary called "wv." also makes sure that if a given sequence doesn't include a channel listed in waveform_channels, it gets listed in wv and
        filled with zeros"""
        wv = {}
        for channel in waveform_channels:
            if channel != None: ##TODO fix this mess
                wv[channel] = sequences[channel]
            else:
                wv[channel] = np.zeros_like(sequences[waveform_channels[0]])
        return wv
    
    def get_dt_channel(self, channel_name):
        if self.hardware_cfg['channels_awg'][channel_name] == "keysight_pxi_M3201A":
            dt = self.dt_M3201A
        else:
            dt = self.hardware_cfg['awg_info'][self.hardware_cfg['channels_awg'][channel_name]]['dt']
            
        return dt
        


    def loadAndQueueWaveforms(self, sequences):
        '''Loads the provided waveforms from a pulse sequence to the appropriate modules.

        Note that all waveforms should consist of values from -1 to 1 (inclusive) only. Amplitude is set in the configureChannels() method.
        If you accidentally pass in an array containing larger values, the code raises a KeysightError: Invalid Waveform.

        Pxi Waveform Channels:
            wv["chargeA_I"]: List of numpy arrays representing waveforms (or a 2D array) for the "I" input AWG to the mixer,
                1 for each unique trial/experiment. Should be the same number of waveforms as num_experiments in the __init__ method.
                The "inner" array is a waveform, and the "outer" dimension corresponds to the experiment/trial.
                The QB lo marker is generated using the known placement of IQ pulses.
            wv["chargeA_Q"]: Same for the "Q" channel.
            readout: Readout waveform used to trigger the readout LO
            wv["digtzr_trig"]: Trigger for the digitizer

            The master trigger for all the cards is generated knowing the length of the AWG waveforms using self.generate_trigger
            '''


        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        num_expt = sequences['digtzr_trig'].shape[0]
        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences,pxi_waveform_channels)

        AWG_module = self.chassis.getModule(self.AWG_mod_no)
        m_module = self.chassis.getModule(self.marker_mod_no)
        stab_module = self.chassis.getModule(self.stab_mod_no)
        ff1_module = self.chassis.getModule(self.ff1_mod_no)
        ff2_module = self.chassis.getModule(self.ff2_mod_no)

        print ("shape of waveform I",np.shape(wv["chargeA_I"]))

        #make sure all waveforms are the same length
        # for i in range(1,len(wv.keys())):
        #     ch1 = wv[list(wv.keys())[i-1]]
        #     dt_ch1 = self.get_dt_channel(list(wv.keys())[i-1])
        #     ch2 = wv[list(wv.keys())[i]]
        #     dt_ch2 = self.get_dt_channel(list(wv.keys())[i])
        #     if len(ch1)*dt_ch1 != len(ch2)*dt_ch2:
        #         raise TypeError("Not all waveform lists are the same length")

        self.AWG_module.clearAll()
        self.m_module.clearAll()
        self.stab_module.clearAll()
        self.ff1_module.clearAll()
        self.ff2_module.clearAll()

        #wv["digtzr_trig"] -> trigger for digitizer card coming from stab card
        #readout_markers -> trigger for readout LO coming from marker card

        key.Waveform._waveform_number_counter = 0

        # Marker array for qb LO, generated from wavefroms_I
        qubitA_marker = self.generatemarkers(wv, "chargeA_I", dt_mark=self.dt_m,
                                                conv_width=self.qb_lo_conv)
        qubitB_marker = self.generatemarkers(wv, "chargeB_I", dt_mark=self.dt_m,
                                                conv_width=self.qb_lo_conv)

        # marker array for stabilizer LO
        stabilizer_marker = self.generatemarkers(wv, "stab_I", dt_mark=self.dt_m, conv_width=self.stab_lo_conv)

        for i in tqdm(range(len(wv[list(wv.keys())[0]]))):
            ## making PXI savvy waveform objects out of the arrays to send out to the PXI
            # Have to include append_zero or the triggers get messed up! idk why

            # Generate master trigger (apparently this is something we might need augh)
            trig_arr_awg = self.generatemastertrigger(len(wv["readoutA"][i]), 2 * self.trig_pulse_length)

            PXIwave_chargeA_I = key.Waveform(np.array(wv["chargeA_I"][i]),append_zero=True)
            PXIwave_chargeA_Q = key.Waveform(wv["chargeA_Q"][i], append_zero=True)
            PXIwave_qubitA_marker = key.Waveform(qubitA_marker[i], append_zero=True)
            PXIwave_readoutA = key.Waveform(wv["readoutA"][i], append_zero=True)

            PXIwave_chargeB_I = key.Waveform(np.array(wv["chargeB_I"][i]), append_zero=True)
            PXIwave_chargeB_Q = key.Waveform(wv["chargeB_Q"][i], append_zero=True)
            PXIwave_qubitB_marker = key.Waveform(qubitB_marker[i], append_zero=True)
            PXIwave_readoutB = key.Waveform(wv["readoutB"][i], append_zero=True)

            PXIwave_stab_I = key.Waveform(np.array(wv["stab_I"][i]),append_zero=True)
            PXIwave_stab_Q = key.Waveform(wv["stab_Q"][i], append_zero=True)
            PXIwave_stabilizer_marker = key.Waveform(wv["stab_I"][i], append_zero=True)

            PXIwave_digtzr_trig = key.Waveform(wv["digtzr_trig"][i], append_zero=True)
            PXIwave_master_trigger = key.Waveform(trig_arr_awg, append_zero=True)

            PXIwave_ff_Q0 = key.Waveform(wv["ff_Q0"][i], append_zero=True)
            PXIwave_ff_Q1 = key.Waveform(wv["ff_Q1"][i], append_zero=True)
            PXIwave_ff_Q2 = key.Waveform(wv["ff_Q2"][i], append_zero=True)
            PXIwave_ff_Q3 = key.Waveform(wv["ff_Q3"][i], append_zero=True)
            PXIwave_ff_Q4 = key.Waveform(wv["ff_Q4"][i], append_zero=True)
            PXIwave_ff_Q5 = key.Waveform(wv["ff_Q5"][i], append_zero=True)
            PXIwave_ff_Q6 = key.Waveform(wv["ff_Q6"][i], append_zero=True)
            PXIwave_ff_Q7 = key.Waveform(wv["ff_Q7"][i], append_zero=True)

            # #####FAST FLUX MODULE1#####
            # # Load FF! waveforms to trigger card
            PXIwave_ff_Q0.loadToModule(ff1_module)
            PXIwave_ff_Q1.loadToModule(ff1_module)
            PXIwave_ff_Q2.loadToModule(ff1_module)
            PXIwave_ff_Q3.loadToModule(ff1_module)

            # Queue ff1 waveforms to ff1 channels
            PXIwave_ff_Q0.queue(self.ff1_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q0"], cycles=1, prescaler=0)
            PXIwave_ff_Q1.queue(self.ff1_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q1"], cycles=1, prescaler=0)
            PXIwave_ff_Q2.queue(self.ff1_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q2"], cycles=1, prescaler=0)
            PXIwave_ff_Q3.queue(self.ff1_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q3"], cycles=1, prescaler=0)

            # Configure ff1 module settings
            self.ff1_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                  syncMode=1, length=10, delay=0)


            #####FAST FLUX MODULE2#####
            # Load FF! waveforms to trigger card
            PXIwave_ff_Q4.loadToModule(ff2_module)
            PXIwave_ff_Q5.loadToModule(ff2_module)
            PXIwave_ff_Q6.loadToModule(ff2_module)
            PXIwave_ff_Q7.loadToModule(ff2_module)

            # Queue ff1 waveforms to ff1 channels
            PXIwave_ff_Q4.queue(self.ff2_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q4"],cycles=1, prescaler=0)
            PXIwave_ff_Q5.queue(self.ff2_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q5"], cycles=1, prescaler=0)
            PXIwave_ff_Q6.queue(self.ff2_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q6"], cycles=1, prescaler=0)
            PXIwave_ff_Q7.queue(self.ff2_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays[
                "ff_Q7"], cycles=1, prescaler=0)

            # Configure ff1 module settings
            self.ff2_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #####QB AWG##### #hardcoded: A:ch1,2 B:ch3,4
            #Send I,Q, qubit A drive waveforms to AWG drive card
            PXIwave_chargeA_I.loadToModule(AWG_module)
            PXIwave_chargeA_Q.loadToModule(AWG_module)
            # Queue I,Q qubitA  drive waveforms on AWG card.
            PXIwave_chargeA_I.queue(self.AWG_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.hardware_delays['chargeA_I'], cycles = 1, prescaler = 0)
            PXIwave_chargeA_Q.queue(self.AWG_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.hardware_delays['chargeA_Q'], cycles = 1, prescaler = 0)
            # Send I,Q, qubit B drive waveforms to AWG drive card
            PXIwave_chargeB_I.loadToModule(AWG_module)
            PXIwave_chargeB_Q.loadToModule(AWG_module)
            # Queue I,Q qubit Bdrive waveforms on AWG card.
            PXIwave_chargeB_I.queue(self.AWG_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                   delay=self.hardware_delays['chargeB_I'], cycles=1, prescaler=0)
            PXIwave_chargeB_Q.queue(self.AWG_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.hardware_delays['chargeB_Q'], cycles=1, prescaler=0)


            self.AWG_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #####Marker#####
            #Send marker waveforms to Marker card
            PXIwave_qubitA_marker.loadToModule(m_module)
            PXIwave_readoutA.loadToModule(m_module)
            # Queue marker waveforms to marker card channels
            PXIwave_qubitA_marker.queue(self.m_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays['qubitA_marker'], cycles=1, prescaler=0)
            PXIwave_readoutA.queue(self.m_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                        delay=self.hardware_delays['qubitA_marker'], cycles=1, prescaler=0)
            #PXIwave_readoutA.queue(self.m_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays['readoutA'], cycles=1, prescaler=0)
            # Send marker waveforms to Marker card
            PXIwave_qubitB_marker.loadToModule(m_module)
            PXIwave_readoutB.loadToModule(m_module)

            # Queue marker waveforms to marker card channels
            PXIwave_qubitB_marker.queue(self.m_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                        delay=self.hardware_delays['qubitB_marker'], cycles=1, prescaler=0)
            PXIwave_readoutB.queue(self.m_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                   delay=self.hardware_delays['readoutB'], cycles=1, prescaler=0)

            # Configure marker card settings
            self.m_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #####STABILIZER and DIGTZR TRIG#####
            # Load waveforms to stabilizer card
            PXIwave_master_trigger.loadToModule((stab_module))
            PXIwave_stab_I.loadToModule(stab_module)
            PXIwave_stab_Q.loadToModule(stab_module)
            PXIwave_stabilizer_marker.loadToModule(stab_module)
            PXIwave_digtzr_trig.loadToModule(stab_module)

            #Queue trigger waveforms to trigger channels
            PXIwave_master_trigger.queue(self.stab_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, cycles=1, prescaler=0)
            PXIwave_master_trigger.queue(self.stab_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, cycles=1, prescaler=0)
            #PXIwave_stab_I.queue(self.stab_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays['stab_I'], cycles=1, prescaler=0)
            # PXIwave_stab_Q.queue(self.stab_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
            #                     delay=self.hardware_delays['stab_Q'],cycles=1, prescaler=0)
            # PXIwave_stabilizer_marker.queue(self.stab_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays['stab_marker'], cycles=1,
            #                    prescaler=0)
            PXIwave_digtzr_trig.queue(self.digtzr_trig_ch, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.hardware_delays["digtzr_trig"], cycles=1, prescaler=0)
            # PXIwave_qubitA_marker.queue(self.digtzr_trig_ch, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
            #                           delay=self.hardware_delays["digtzr_trig"], cycles=1, prescaler=0)
            #
            # #Configure trigger module settings
            self.stab_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                               syncMode=1, length=10, delay=0)


    def run(self):
        print("Experiment starting. Expected time = ", self.totaltime, "mins")
        try:
            # Start all the channels on the AWG and digitizer modules.
            print ("Number of experiments = ",self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_2.clear()
            self.DIG_ch_2.start()
            self.DIG_ch_3.clear()
            self.DIG_ch_3.start()
            self.DIG_ch_4.clear()
            self.DIG_ch_4.start()

            self.AWG_module.startAll()
            self.ff1_module.startAll()
            self.stab_module.startAll()
            self.m_module.startAll()
            self.ff2_module.startAll()


        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self,expt_num=0):
        if "A" in self.on_qubits:
            I = []
            Q = []
            for sweep_ct in tqdm(range(self.num_avg)):
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)#data_1.shape = (num_expt, dig_sample_per_record)
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)
                I.append(ch1)
                Q.append(ch2)

            #avg along num averageers
            I_avg = np.mean(np.asarray(I))
            Q_avg = np.mean(np.asarray(Q))

            fig = plt.figure(figsize = (12,4))
            ax = fig.add_subplot(131,title = 'I QbA')
            plt.imshow(I_avg, aspect='auto')
            ax.set_xlabel('Digitizer bins')
            ax.set_ylabel('Experiment number')
            ax2 = fig.add_subplot(132,title = 'Q QbA')
            plt.imshow(Q_avg, aspect='auto')
            ax2.set_xlabel('Digitizer bins')
            ax2.set_ylabel('Experiment number')
            ax3 = fig.add_subplot(133,title = 'Expt num = ' + str(expt_num))
            ax3.plot(np.arange(self.DIG_sampl_record)*self.dt_dig, I_avg[expt_num])
            ax3.plot(np.arange(self.DIG_sampl_record)*self.dt_dig, I_avg[expt_num])
            ax3.axvspan(self.readoutA_window[0], self.readoutA_window[1], alpha=0.2, color='b')
            ax3.set_xlabel('Time (ns)')
            ax3.set_ylabel('Signal')
            fig.tight_layout()
            plt.show()

        if "B" in self.on_qubits:
            I = []
            Q = []
            for sweep_ct in tqdm(range(self.num_avg)):
                ch1 = np.reshape(self.DIG_ch_3.readDataQuiet(),
                                 self.data_3.shape)  # data_1.shape = (num_expt, dig_sample_per_record)
                ch2 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape)
                I.append(ch1)
                Q.append(ch2)

            # avg along num averageers
            I_avg = np.mean(np.asarray(I))
            Q_avg = np.mean(np.asarray(Q))

            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(131, title='I QbB')
            plt.imshow(I_avg, aspect='auto')
            ax.set_xlabel('Digitizer bins')
            ax.set_ylabel('Experiment number')
            ax2 = fig.add_subplot(132, title='Q QbB')
            plt.imshow(Q_avg, aspect='auto')
            ax2.set_xlabel('Digitizer bins')
            ax2.set_ylabel('Experiment number')
            ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, I_avg[expt_num])
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, I_avg[expt_num])
            ax3.axvspan(self.readoutB_window[0], self.readoutB_window[1], alpha=0.2, color='b')
            ax3.set_xlabel('Time (ns)')
            ax3.set_ylabel('Signal')
            fig.tight_layout()
            plt.show()

        print("The digitzer bins were individually averaged for testing synchronization.")

    def traj_data_one(self):
        """
        Reads off digitizer once, in shape (num_expt * dig_sample_per_record), and windows that data
        Saves it in ch1 -> (num_expt, len(window))
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data = []
        if "A" in self.on_qubits:
            qbA_I = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(self.readoutA_window[0]):int(
                self.readoutA_window[1])].T
            qbA_Q = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(self.readoutA_window[0]):int(
                self.readoutA_window[1])].T
            data.append([qbA_I, qbA_Q])
        if "B" in self.on_qubits:
            qbB_I = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(self.readoutB_window[0]):int(
                self.readoutB_window[1])].T
            qbB_Q = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(self.readoutB_window[0]):int(
                self.readoutB_window[1])].T
            data.append([qbB_I, qbB_Q])
        return np.asarray(data)

    def traj_data_many(self):
        """
        Reads off digitizer num_avg_times, each shot in shape (num_expt , dig_sample_per_record) and windowed.
        Saves it in qbA_I -> (num avg, num_expt, len(window))
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data= []
        qbA_I = []
        qbA_Q = []
        qbB_I = []
        qbB_Q = []
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                qbA_I.append(ch1)
                qbA_Q.append(ch2)
            if "B" in self.on_qubits:
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                qbB_I.append(ch3)
                qbB_Q.append(ch4)
        if "A" in self.on_qubits:
            data.append([qbA_I, qbA_Q])
        if "B" in self.on_qubits:
            data.append([qbB_I, qbB_Q])
        return np.asarray(data)

    def traj_data_many_no_window(self,w):
        """
        Reads off digitizer num_avg_times, each shot in shape (num_expt , dig_sample_per_record) and windowed.
        Saves it in qbA_I -> (num avg, num_expt, len(window))
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data= []
        qbA_I = []
        qbA_Q = []
        qbB_I = []
        qbB_Q = []
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                      int(w[0]):int(w[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                      int(w[0]):int(w[1])].T
                qbA_I.append(ch1)
                qbA_Q.append(ch2)
            if "B" in self.on_qubits:
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[
                      int(w[0]):int(w[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[
                      int(w[0]):int(w[1])].T
                qbB_I.append(ch3)
                qbB_Q.append(ch4)
        if "A" in self.on_qubits:
            data.append([qbA_I, qbA_Q])
        if "B" in self.on_qubits:
            data.append([qbB_I, qbB_Q])
        return np.asarray(data)

    def SSdata_one(self):
        """
        Reads off digitizer once, in shape (num_expt * dig_sample_per_record), and windows that data. averages over
        that window to give array of shape (num_expt).
        Saves averaged in qbA_I, final shape -> (num_expt)
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data = []
        if "A" in self.on_qubits:
            qbA_I = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                  int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
            qbA_Q = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                  int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
            data.append([np.mean(qbA_I, 1), np.mean(qbA_Q, 1)])
        if "B" in self.on_qubits:
            qbB_I = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(self.readoutB_window[0]):int(
                self.readoutB_window[1])].T
            qbB_Q = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(self.readoutB_window[0]):int(
                self.readoutB_window[1])].T
            data.append([np.mean(qbB_I,1), np.mean(qbB_Q,1)])
        return np.asarray(data)

    def SSdata_many(self,w =[0,-1]):
        """
        Reads off digitizer num_avg_times, each shot in shape (num_expt , dig_sample_per_record). Windows it and
        averages over that window to give array shape (num_expt). Saves that average shot in ch1.
        Saves all average shots in qBA_I, whose final shape will be  -> (num_avg, num_expt).
        For whatever reason you take the transpose of this, so your data shape is (num_expt, num avg)?? oh well
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data= []
        qbA_I = []
        qbA_Q = []
        qbB_I = []
        qbB_Q = []
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                qbA_I.append(np.mean(ch1,1))
                qbA_Q.append(np.mean(ch2,1))
            if "B" in self.on_qubits:
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                qbB_I.append(np.mean(ch3,1))
                qbB_Q.append(np.mean(ch4,1))
        if "A" in self.on_qubits:
            data.append([np.array(qbA_I).T, np.array(qbA_Q).T])
        if "B" in self.on_qubits:
            data.append([np.array(qbB_I).T, np.array(qbB_Q).T])
        return np.asarray(data)

    def avg_data_threshold(self, vecA=[0,0], phiA=0, vecB=[0,0], phiB=0):
        """
        Reads off digitizer num_avg_times, each shot in shape (num_expt , dig_sample_per_record). Windows it and
        averages over that window to give array shape (num_expt). Saves that average shot in ch1 (I), ch2 (Q).
        Thresholds data, save that in I_rot.

        Saves all average shots in qBA_I, qbA_state, whose final shape will be  -> (num_avg, num_expt).
        then, average over num_avg, for final shape num_expt.

        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q, state]], if both will return [[qbA_I, qbA_Q, state],[qbB_I, qbB_Q, state]]
        """
        data= []
        qbA_I = []
        qbA_Q = []
        qbB_I = []
        qbB_Q = []
        qbA_state = []
        qbB_state = []
        ran = self.hardware_cfg['awg_info']['keysight_pxi']['digtzr_vpp_range']
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])].T
                qbA_I.append(np.mean(ch1,1))
                qbA_Q.append(np.mean(ch2,1))

                if ran > 0:
                    ch1 = ch1 / 2 ** 15 * ran
                    ch2 = ch2 / 2 ** 15 * ran
                I_centered = np.mean(ch1,1) - vecA[0]
                Q_centered = np.mean(ch2, 1) - vecA[1]
                I_rot = I_centered*np.cos(phiA) + Q_centered*np.sin(phiA)
                qbA_state.append(I_rot>0)

            if "B" in self.on_qubits:
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(self.readoutB_window[0]):int(
                    self.readoutB_window[1])].T
                qbB_I.append(np.mean(ch3,1))
                qbB_Q.append(np.mean(ch4,1))

                if ran > 0:
                    ch3 = ch3 / 2 ** 15 * ran
                    ch4 = ch4 / 2 ** 15 * ran
                I_centered = np.mean(ch3,1) - vecB[0]
                Q_centered = np.mean(ch4, 1) - vecB[1]
                I_rot = I_centered*np.cos(phiB) + Q_centered*np.sin(phiB)
                qbB_state.append(I_rot<0)


        if "A" in self.on_qubits:
            qbA_I = np.mean(np.array(qbA_I),0)
            qbA_Q = np.mean(np.array(qbA_Q), 0)
            qbA_state = np.mean(np.array(qbA_state), 0)
            data.append([qbA_I, qbA_Q, qbA_state])
        if "B" in self.on_qubits:
            qbB_I = np.mean(np.array(qbB_I), 0)
            qbB_Q = np.mean(np.array(qbB_Q), 0)
            qbB_state = np.mean(np.array(qbB_state), 0)
            data.append([qbB_I, qbB_Q, qbB_state])
        return np.asarray(data)

    def acquire_avg_data(self,pi_calibration=False):
        """
        Reads off digitizer num_average times, in shape (num_expt * dig_sample_per_record), and windows that data.
        Averages over that window to get an array of shape (num_expt)
        Adds that to qbA_I, and then at end divides qbA_I by num_avg, final shape -> (num_expt).
        Not sure what pi calibration is doing. I think the last two experiments are readout w/out a pi pulse,
        and readout w/pi pulse. Sets it so that "zero" is readout w/out pi pulse and normalizes whole thing to
        amplitude of wave with pi pulse
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        """
        data = []
        qbA_I, qbA_Q, qbB_I, qbB_Q = np.zeros(self.num_expt), np.zeros(self.num_expt), np.zeros(self.num_expt), np.zeros(self.num_expt)
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                qbA_I += np.mean((np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])]*self.readoutA_weight).T, 1)

                qbA_Q += np.mean((np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                      int(self.readoutA_window[0]):int(self.readoutA_window[1])]*self.readoutA_weight).T, 1)
            if "B" in self.on_qubits:
                qbB_I += np.mean((np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(
                    self.readoutB_window[0]):int(self.readoutB_window[1])]*self.readoutB_weight).T, 1)
                qbB_Q += np.mean((np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(
                    self.readoutB_window[0]):int(self.readoutB_window[1])]*self.readoutB_weight).T, 1)
        if "A" in self.on_qubits:
            qbA_I = qbA_I / self.num_avg
            qbA_Q = qbA_Q / self.num_avg
            # if pi_calibration:
            #     qbA_I = (qbA_I[:-2] - qbA_I[-2]) / (qbA_I[-1] - qbA_I[-2])
            #     qbA_Q = (qbA_Q[:-2] - qbA_Q[-2]) / (qbA_Q[-1] - qbA_Q[-2])
            data.append([qbA_I, qbA_Q])

        if "B" in self.on_qubits:
            qbB_I = qbB_I / self.num_avg
            qbB_Q = qbB_Q / self.num_avg
            # if pi_calibration:
            #     qbB_I = (qbB_I[:-2] - qbB_I[-2]) / (qbB_I[-1] - qbB_I[-2])
            #     qbB_Q = (qbB_Q[:-2] - qbB_Q[-2]) / (qbB_Q[-1] - qbB_Q[-2])
            data.append([qbB_I, qbB_Q])

        return np.asarray(data)


    def acquire_avg_std_data(self, pi_calibration=False):
        """
        Reads off digitizer num_average times, in shape (num_expt * dig_sample_per_record), and windows that data.
        Averages over that window to get an array of shape (num_expt)
        Adds that to qbA_I, and then at end divides qbA_I by num_avg, final shape -> (num_expt).
        Not sure what pi calibration is doing. I think the last two experiments are readout w/out a pi pulse,
        and readout w/pi pulse. Sets it so that "zero" is readout w/out pi pulse and normalizes whole thing to
        amplitude of wave with pi pulse
        Calculates the standard deviation for each expt
        Returns:
        data(np.ndarray):if qb "A" OR "B" on, will return [[I, Q]], if both will return [[qbA_I, qbA_Q],[qbB_I, qbB_Q]]
        data_std(np.ndarray):if qb "A" OR "B" on, will return [[I_std, Q_std]], if both will return [[qbA_I_std,
        qbA_Q_std],[qbB_I_std, qbB_Q_std]]
        """
        data = []
        data_std = []
        qbA_I, qbA_Q, qbB_I, qbB_Q = [],[],[],[]
        for ii in tqdm(range(self.num_avg)):
            if "A" in self.on_qubits:
                qbA_I.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[
                                 int(self.readoutA_window[0]):int(self.readoutA_window[1])].T,1))
                qbA_Q.append(np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[
                                 int(self.readoutA_window[0]):int(self.readoutA_window[1])].T,1))
            if "B" in self.on_qubits:
                qbB_I.append(np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(
                    self.readoutB_window[0]):int(self.readoutB_window[1])].T),1)
                qbB_Q.append(np.mean(np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(
                    self.readoutB_window[0]):int(self.readoutB_window[1])].T),1)
        if "A" in self.on_qubits:
            qbA_I_std = np.std(qbA_I, 0)
            qbA_Q_std = np.std(qbA_Q, 0)
            qbA_I = np.mean(qbA_I)
            qbA_Q = np.mean(qbA_Q)
            if pi_calibration:
                qbA_I = (qbA_I[:-2] - qbA_I[-2]) / (qbA_I[-1] - qbA_I[-2])
                qbA_Q = (qbA_Q[:-2] - qbA_Q[-2]) / (qbA_Q[-1] - qbA_Q[-2])
            data.append([qbA_I, qbA_Q])
            data_std.append([qbA_I_std, qbA_Q_std])

        if "B" in self.on_qubits:
            qbB_I_std = np.std(qbB_I, 0)
            qbB_Q_std = np.std(qbB_Q, 0)
            qbB_I = np.mean(qbB_I)
            qbB_Q = np.mean(qbB_Q)
            if pi_calibration:
                qbB_I = (qbB_I[:-2] - qbB_I[-2]) / (qbB_I[-1] - qbB_I[-2])
                qbB_Q = (qbB_Q[:-2] - qbB_Q[-2]) / (qbB_Q[-1] - qbB_Q[-2])
            data.append([qbB_I, qbB_Q])
            data_std.append([qbB_I_std, qbB_Q_std])

        return np.asarray(data)

def run_keysight(experiment_cfg, hardware_cfg, sequences, name):

    setup = KeysightSingleQubit(experiment_cfg, hardware_cfg, sequences, name)

    try:
        setup.configureChannels(hardware_cfg, experiment_cfg, name)
        setup.loadAndQueueWaveforms(sequences)
        setup.run()
        setup.acquireandplot()

    finally:
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.m_module.stopAll()
        setup.m_module.clearAll()
        setup.stab_module.stopAll()
        setup.stab_module.clearAll()
        setup.chassis.close()
