# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:41:56 2018

@author: slab

Module 6 used as AWG/marker:
    Channel 1 is I
    Channel 2 is Q
    Channel 3 is marker for readout pulse
    Channel 4 is indicator to take data. Should be connected to "Trigger" on
        Module 10.

Module 10 is used for reacout.
    Channel 1 is I
    Channel 2 is Q
"""

# %pylab inline
from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.HVIExperiments import HVIExpLib as exp
import time
import numpy as np
import os
import json
import threading
import matplotlib.pyplot as plt
from tqdm import tqdm
from slab.datamanagement import SlabFile


AWG_BANDWIDTH = 10**9 #Samples per second


class KeysightDoubleQubit:

    # Class designed to implement a two-qubit experiment given pulse sequences from the Sequencer class. Does not use HVI technology.
    # Both qubits will be supplied with drive and readout tones if the relevant channels are included in hardware config.
    # Call of this class should be conditional on 'on_qubits' in expt config being two items long
    # We map 1 to A and 2 to B here. A for "Alice", B for "Bob"

    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, sequences, name,
                 save_path=r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials=50 * 1000):  # 1000*10000 if you want to watch sweep by eye

        chassis = key.KeysightChassis(1,
                                      {6: key.ModuleType.OUTPUT,
                                       8: key.ModuleType.OUTPUT,
                                       7: key.ModuleType.OUTPUT,
                                       9: key.ModuleType.OUTPUT,
                                       10: key.ModuleType.INPUT})
        self.hardware_cfg = hardware_cfg
        self.out_mod_no_A = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no'][0]
        self.out_mod_no_B = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no'][1]
        # Right now 7 is A and 6 is B
        self.marker_mod_no_8 = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no'][0]
        self.marker_mod_no_9 = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no'][1]
        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        self.readout_A_window = np.array(quantum_device_cfg['readout']['1']['window'])
        self.readout_B_window = np.array(quantum_device_cfg['readout']['2']['window'])
        self.lo_delay = hardware_cfg['awg_info']['keysight_pxi']['lo_delay']
        self.abs_trig_delay = hardware_cfg['awg_info']['keysight_pxi']['abs_trig_delay']
        # the below global delay will be added to tek2_trigger_delay and propagated
        self.global_trig_delay_fromawg = hardware_cfg['awg_info']['keysight_pxi']['global_trig_delay_fromawg']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']
        self.trig_delay = hardware_cfg['awg_info']['keysight_pxi']['m3201a_trig_delay']
        self.card_delay = hardware_cfg['awg_info']['keysight_pxi']['m3102a_card_delay']
        self.adc_range = hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
        self.iq_line_delay = hardware_cfg['awg_info']['keysight_pxi']['iq_line_delay']
        self.marker_pulse_delay = hardware_cfg['awg_info']['keysight_pxi']['marker_pulse_delay']
        self.readout_marker_pulse_delay = hardware_cfg['awg_info']['keysight_pxi']['readout_marker_pulse_delay']
        self.bob_delay_rel_alice = hardware_cfg['awg_info']['keysight_pxi']['bob_delay_rel_alice']
        self.tentative_card_8_delay = hardware_cfg['awg_info']['keysight_pxi']['tentative_card_8_delay']
        self.tentative_card_9_delay = hardware_cfg['awg_info']['keysight_pxi']['tentative_card_9_delay']

        print("Module used for generating analog pulses for A = ", self.out_mod_no_A)
        print("Module used for generating analog pulses for B = ", self.out_mod_no_B)
        print("Module used for generating digital markers = ", self.marker_mod_no_8)
        print("Module used for generating digital markers and digitizer trigger = ", self.marker_mod_no_9)
        self.out_mod_nums = [self.out_mod_no_A, self.out_mod_no_A, self.marker_mod_no_8, self.marker_mod_no_9]

        self.num_avg = experiment_cfg[name]['acquisition_num']
        self.on_qubits = experiment_cfg[name]['on_qubits']


        for qubit_id in self.on_qubits:
            self.num_expt = sequences['readout%s' % qubit_id].shape[0]
            num_expt = self.num_expt
            # if there's one it'll pick the right readout and if two will take the second only but that's fine
        self.trigger_period = self.hardware_cfg['trigger']['period_us']

        self.totaltime = self.num_avg * self.num_expt * self.trigger_period * 1e-6 / 60.0

        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']

        if 'sideband' in name:
            print("Yikes you are using legacy tek2 code in keysight pxi load")
            self.prep_tek2 = True
            self.tek2_trigger_delay = int(hardware_cfg['awg_info']['tek70001a']['trig_delay'] / 100.0)
        else:
            # print("Yikes you are using legacy tek2 code in keysight pxi load")
            self.prep_tek2 = False
            self.tek2_trigger_delay = 0

        # hack
        # self.tek2_trigger_delay = self.trig_delay / self.dt_dig


        if 'cavity_drive' in name:
            self.prep_cavity_drive = True
        else:
            self.prep_cavity_drive = False

        # # mgp 4/21 added
        # if 'two_qubits' in name:
        #     self.two_qubits = True
        # else:
        #     self.two_qubits = False

        # if 'cavity_sideband' in name:
        #     self.prep_cavity_drive = True
        # else:self.prep_cavity_drive=False

        self.chassis = chassis
        self.awg_channels = range(1, 5)
        self.dig_channels = range(1, 5)

        # Initialize Module 6 = Analog card for A.  Ch1 = AWG I, CH2 = AWG Q

        # self.AWG_chs = [chassis.getChannel(self.out_mod_no, ch) for ch in self.awg_channels]
        self.AWG_A_ch_1 = chassis.getChannel(self.out_mod_no_A, 1)
        self.AWG_A_ch_2 = chassis.getChannel(self.out_mod_no_A, 2)
        self.AWG_A_ch_3 = chassis.getChannel(self.out_mod_no_A, 3)
        self.AWG_A_ch_4 = chassis.getChannel(self.out_mod_no_A, 4)

        # Initialize Module 7 = Analog card for B.  Ch1 = AWG I, CH2 = AWG Q

        # self.AWG_chs = [chassis.getChannel(self.out_mod_no, ch) for ch in self.awg_channels]
        self.AWG_B_ch_1 = chassis.getChannel(self.out_mod_no_B, 1)
        self.AWG_B_ch_2 = chassis.getChannel(self.out_mod_no_B, 2)
        self.AWG_B_ch_3 = chassis.getChannel(self.out_mod_no_B, 3)
        self.AWG_B_ch_4 = chassis.getChannel(self.out_mod_no_B, 4)

        # Initialize Module 8 = Marker card. Digital markers for qubit A/B, readout
        # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]
        self.m_8_ch_1 = chassis.getChannel(self.marker_mod_no_8, 1)
        self.m_8_ch_2 = chassis.getChannel(self.marker_mod_no_8, 2)
        self.m_8_ch_3 = chassis.getChannel(self.marker_mod_no_8, 3)
        self.m_8_ch_4 = chassis.getChannel(self.marker_mod_no_8, 4)

        # Initialize Module 9 = Marker card. Digital markers for cavity A/B, digitizer trigger
        # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]
        self.m_9_ch_1 = chassis.getChannel(self.marker_mod_no_9, 1)
        self.m_9_ch_2 = chassis.getChannel(self.marker_mod_no_9, 2)
        self.m_9_ch_3 = chassis.getChannel(self.marker_mod_no_9, 3)
        self.m_9_ch_4 = chassis.getChannel(self.marker_mod_no_9, 4)

        # Initialize card that generates Triggers
        # self.trig_chs = [chassis.getChannel(self.trig_mod_no, ch) for ch in self.awg_channels]
        # self.trig_ch_1 = chassis.getChannel(self.trig_mod_no, 1)
        # self.trig_ch_2 = chassis.getChannel(self.trig_mod_no, 2)
        # self.trig_ch_3 = chassis.getChannel(self.trig_mod_no, 3)
        # self.trig_ch_4 = chassis.getChannel(self.trig_mod_no, 4)

        # self.out_modules = [chassis.getModule(num) for num in self.out_mod_nums]
        self.AWG_A_module = chassis.getModule(self.out_mod_no_A)
        self.AWG_B_module = chassis.getModule(self.out_mod_no_B)
        self.m_8_module = chassis.getModule(self.marker_mod_no_8)
        self.m_9_module = chassis.getModule(self.marker_mod_no_9)
        # self.trig_module = chassis.getModule(self.trig_mod_no)

        # Initialize digitizer card
        # self.DIG_chs = [chassis.getChannel(10, ch) for ch in self.dig_channels]
        self.DIG_ch_1 = chassis.getChannel(10, 1)
        self.DIG_ch_2 = chassis.getChannel(10, 2)
        self.DIG_ch_3 = chassis.getChannel(10, 3)
        self.DIG_ch_4 = chassis.getChannel(10, 4)
        self.DIG_module = chassis.getModule(10)


        self.data_1, self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)), np.zeros(
                (self.num_expt, self.DIG_sampl_record))
        self.data_3, self.data_4 = np.zeros((self.num_expt, self.DIG_sampl_record)), np.zeros(
                (self.num_expt, self.DIG_sampl_record))


        # self.I, self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IA, self.QA = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IB, self.QB = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally

    def configureChannels(self, hardware_cfg, experiment_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_AWG_A = hardware_cfg['awg_info']['keysight_pxi']['amplitudes_A']
        amp_AWG_B = hardware_cfg['awg_info']['keysight_pxi']['amplitudes_B']
        amp_mark_8 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_8']
        amp_mark_9 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_9']
        # amp_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_trig']
        # dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        # dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring analog channels")

        self.AWG_A_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_A_ch_1.configure(amplitude=amp_AWG_A[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_2.configure(amplitude=amp_AWG_A[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_3.configure(amplitude=amp_AWG_A[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_4.configure(amplitude=amp_AWG_A[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        self.AWG_B_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_B_ch_1.configure(amplitude=amp_AWG_B[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_2.configure(amplitude=amp_AWG_B[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_3.configure(amplitude=amp_AWG_B[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_4.configure(amplitude=amp_AWG_B[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        print("Configuring marker channels")

        self.m_8_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_8_ch_1.configure(amplitude=amp_mark_8[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_2.configure(amplitude=amp_mark_8[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_4.configure(amplitude=amp_mark_8[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_3.configure(amplitude=amp_mark_8[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_4.configure(amplitude=amp_mark[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # print("tried marker 4")

        self.m_9_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_9_ch_1.configure(amplitude=amp_mark_9[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_2.configure(amplitude=amp_mark_9[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_4.configure(amplitude=amp_mark_9[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_3.configure(amplitude=amp_mark_9[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_4.configure(amplitude=amp_mark[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # print("tried marker 4")

        # print("Configuring trigger channels")
        #
        # self.trig_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.trig_ch_1.configure(amplitude=amp_trig[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_2.configure(amplitude=amp_trig[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_3.configure(amplitude=amp_trig[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_4.configure(amplitude=amp_trig[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)


        print ("Setting trigger config for all channels of all modules to External")

        for n in range(1, 5):
            self.AWG_A_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.AWG_B_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_8_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_9_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            # self.trig_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)

        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")

        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)

    def configureDigitizerChannels(self, hardware_cfg, experiment_cfg, name):
        '''Configures the individual DIGITIZER channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_mark_9 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_9']
        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        # print("Configuring marker channel to digitizer trigger")
        #
        # self.m_9_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.m_9_ch_3.configure(amplitude=amp_mark_9[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        #
        # print ("Setting trigger config for all channels of all modules to External")
        #
        # for n in range(1, 5):
        #     self.m_9_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)

        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")

        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_3.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_4.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)



    def generatemarkers(self,waveform,resample=False,conv_width = 1,trig_delay = 0):

        # markers = np.array([np.append(np.heaviside(np.convolve(abs(w), np.ones((int(self.lo_delay),))/self.lo_delay),0)[int(self.lo_delay/2):],np.zeros(int(self.lo_delay/2))) for w in waveform])
        markers = np.array([np.append(np.heaviside(np.convolve(abs(np.append(w[int(trig_delay):],np.zeros(int(trig_delay)))), np.ones((int(conv_width),)) / conv_width), 0)[int(conv_width / 2):], np.zeros(int(conv_width / 2))) for w in waveform])
        if resample: return np.array([m[::int(self.dt_dig/self.dt)] for m in markers])
        else:return markers

    def generatemastertrigger(self,length,trig_width = 1,trig_delay = 0):
        trig = np.zeros(length)
        for i in np.arange(int(trig_delay/self.dt_dig),int((trig_delay + trig_width)/self.dt_dig)):
            trig[i] = 1
        return trig

    # 4/2021 modified hardware config for current expt to have more channels
    def sequenceslist(self, sequences, waveform_channels):
        wv = {}
        for channel in waveform_channels:
            if not channel == None:
                wv[channel] = sequences[channel]
            else:
                wv[channel] = np.zeros_like(sequences[waveform_channels[0]])
        return wv

    def loadAndQueueWaveforms(self, sequences):
        '''Loads the provided waveforms from a pulse sequence to the appropriate modules.

        Note that all waveforms should consist of values from -1 to 1 (inclusive) only. Amplitude is set in the configureChannels() method.
        If you accidentally pass in an array containing larger values, the code raises a KeysightError: Invalid Waveform.

        Params:
            waveforms_I: List of numpy arrays representing waveforms (or a 2D array) for the "I" input AWG to the mixer,
                1 for each unique trial/experiment. Should be the same number of waveforms as num_experiments in the __init__ method.
                The "inner" array is a waveform, and the "outer" dimension corresponds to the experiment/trial.
            waveforms_Q: Same for the "Q" channel.
            readout: Readout waveform used to trigger the readout LO
            markers_readout: Trigger for the digitizer
            The master trigger for all the cards is generated knowing the length of the AWG waveforms using self.generate_trigger
            '''

        # Modified this for a second set of waveforms. 4/21 mgp

        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']

        for qubit_id in self.on_qubits:
            self.num_expt = sequences['readout%s' % qubit_id].shape[0]
            num_expt = self.num_expt
            # if there's one it'll pick the right readout and if two will take the second only but that's fine

        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences, pxi_waveform_channels)
        # newname=s
        # you need to manually make sure wv is the right length for this to work
        # new list for ref  "waveform_channels": ["charge1A_I", "charge1A_Q","charge1B_I", "charge1B_Q", "readoutA", "readoutB",  "readout_trig","tek2_trig","cavity1A_I", "cavity1A_Q","cavity1B_I", "cavity1B_Q"]

        waveforms_A_I = wv["charge1_I"]
        waveforms_A_Q = wv["charge1_Q"]
        waveforms_B_I = wv["charge2_I"]
        waveforms_B_Q = wv["charge2_Q"]
        readout_A = wv["readout1"]
        readout_B = wv["readout2"]
        markers_readout = wv["readout_trig"]
        # question: we only need 1 marker to digitizer channel yes? it gets fed into trigger port
        # will have to be careful re channels when making actual waveform
        # issue if a,b waveforms differ bc length check error below will get thrown
        # if self.prep_tek2: tek2_marker = wv["tek2_trig"]
        if self.prep_cavity_drive:
            cavity_A_I = wv["cavity1_I"]
            cavity_A_Q = wv["cavity1_Q"]
            cavity_B_I = wv["cavity2_I"]
            cavity_B_Q = wv["cavity2_Q"]

        chassis = self.chassis
        # modified 4/21 mgp
        AWG_A_module = chassis.getModule(self.out_mod_no_A)
        AWG_B_module = chassis.getModule(self.out_mod_no_B)
        m_8_module = chassis.getModule(self.marker_mod_no_8)
        m_9_module = chassis.getModule(self.marker_mod_no_9)

        print("shape of waveform I A", np.shape(waveforms_A_I))
        print("shape of waveform I B", np.shape(waveforms_B_I))

        if len(waveforms_A_I) != len(waveforms_A_Q) or len(waveforms_A_I) != len(markers_readout) or len(
                    waveforms_A_I) != len(readout_A):
            raise TypeError("Not all waveform A lists are the same length")
        if len(waveforms_B_I) != len(waveforms_B_Q) or len(waveforms_B_I) != len(markers_readout) or len(
                    waveforms_B_I) != len(readout_B):
            raise TypeError("Not all waveform B lists are the same length")
        # make duplicate markers readout if this gets hinky
        if len(waveforms_A_I) != len(waveforms_B_I) or len(waveforms_A_Q) != len(waveforms_B_Q) or len(
                    readout_A) != len(readout_B):
            raise TypeError("Not all waveform A and B lists are the same length")

        self.AWG_A_module.clearAll()
        self.AWG_B_module.clearAll()
        self.m_8_module.clearAll()
        self.m_9_module.clearAll()
        # print("attempted to clear modules")

        # dsp stands for downsampling, which to use gets selected later in hardcode depending on card speed
        # delays seem to be showing up here in delays on marker cards or in nelson's channels_delay
        # lo_delay makes output marker to los wider or smaller. a convolution factor altering output marker to lo
        # trig_delay comes from m3201 delay in hardware config. delays marker pulses by adding zeros

        # abs_trig_delay sets delay on markers to awg cards. for anything not trig to digitizer, adds zeros to that marker array
        # m3102 delay in hwcfg gets called card delay here, this affects digitizer waveform. divided by 100 idk why. delays channels from hardware!!!
        # delays w queue waveform less accurate than thru arrays maybe
        # card delay delays channel triggering digitizer
        # change abs trig delay and card delay apr 2021?

        # dsp = downsampling
        # markers_readout -> trigger for digitizer card coming from trigger card
        # readout_markers -> trigger for readout LO coming from marker card
        # if marker card is 500MS/s, use dsp version
        # if marker card is 1Gs/s, use non-dsp-ed version
        # if trigger card is 500MS/s, use dsp version
        # if trigger card is 1Gs/s, use non-dsp-ed version

        # MAY NEED TO DIVERSIFY DELAYS if different qubits need diff things
        key.Waveform._waveform_number_counter = 0

        # Need to uncomment a section further down if u want non downsampled markers
        # Plus the commented stuff here
        # qubit_A_marker = self.generatemarkers(waveforms_A_I)
        qubit_A_marker_dsp = self.generatemarkers(waveforms_A_I, resample=True, conv_width=self.lo_delay,
                                                      trig_delay=self.trig_delay)
        # qubit_B_marker = self.generatemarkers(waveforms_B_I)
        qubit_B_marker_dsp = self.generatemarkers(waveforms_B_I, resample=True, conv_width=self.lo_delay,
                                                      trig_delay=self.trig_delay)
        # readout_A_marker = self.generatemarkers(readout_A, resample=False)
        readout_A_marker_dsp = self.generatemarkers(readout_A, resample=True, trig_delay=self.trig_delay)
        # readout_B_marker = self.generatemarkers(readout_B, resample=False)
        readout_B_marker_dsp = self.generatemarkers(readout_B, resample=True, trig_delay=self.trig_delay)
            # marker array for waveform markers_readout, waveform for triggering digitizer card, already resampled
        card_trig_arr = self.generatemarkers(markers_readout, resample=True)
            # if self.prep_tek2: tek2_marker_dsp = self.generatemarkers(tek2_marker, resample=True, trig_delay=0.0)

        if self.prep_cavity_drive:
            cavity_A_marker_dsp = self.generatemarkers(cavity_A_I, resample=True, conv_width=self.lo_delay,
                                                           trig_delay=self.trig_delay)
            cavity_B_marker_dsp = self.generatemarkers(cavity_B_I, resample=True, conv_width=self.lo_delay,
                                                           trig_delay=self.trig_delay)

            # trig_arr_awg = self.generatemastertrigger(len(readout_marker_dsp[0]), 2 * self.trig_pulse_length,self.abs_trig_delay)
            # triggers other cards. gets loaded to trig channels that don't go to digitizer
            # we dont need to trigger other cards this way????
        # print("generated markers")

        for i in tqdm(range(len(waveforms_A_I))):

            wave_A_I = key.Waveform(np.array(waveforms_A_I[i]),
                                        append_zero=True)  # Have to include append_zero or the triggers get messed up!
            wave_B_I = key.Waveform(np.array(waveforms_B_I[i]),
                                        append_zero=True)  # Have to include append_zero or the triggers get messed up!
            wave_A_Q = key.Waveform(waveforms_A_Q[i], append_zero=True)
            wave_B_Q = key.Waveform(waveforms_B_Q[i], append_zero=True)
            # uncomment if u want non downsampled markers
            # m_A_readout = key.Waveform(readout_A_marker[i], append_zero=True)
            # m_B_readout = key.Waveform(readout_B_marker[i], append_zero=True)
            # m_A_qubit = key.Waveform(qubit_A_marker[i], append_zero=True)  ### this qubit marker is wrong - Vatsan
            # m_B_qubit = key.Waveform(qubit_B_marker[i], append_zero=True)  ### this qubit marker is wrong - Vatsan

            m_A_readout_dsp = key.Waveform(readout_A_marker_dsp[i], append_zero=True)
            m_B_readout_dsp = key.Waveform(readout_B_marker_dsp[i], append_zero=True)
            m_A_qubit_dsp = key.Waveform(qubit_A_marker_dsp[i], append_zero=True)
            m_B_qubit_dsp = key.Waveform(qubit_B_marker_dsp[i], append_zero=True)
            # if self.prep_tek2: m_tek2_dsp = key.Waveform(tek2_marker_dsp[i], append_zero=True)

            if self.prep_cavity_drive:
                wave_cavity_A_I = key.Waveform(np.array(cavity_A_I[i]),
                                                   append_zero=True)  # Have to include append_zero or the triggers get messed up!
                wave_cavity_B_I = key.Waveform(np.array(cavity_B_I[i]),
                                                   append_zero=True)  # Have to include append_zero or the triggers get messed up!
                wave_cavity_A_Q = key.Waveform(cavity_A_Q[i], append_zero=True)
                wave_cavity_B_Q = key.Waveform(cavity_B_Q[i], append_zero=True)
                m_cavity_A_dsp = key.Waveform(cavity_A_marker_dsp[i], append_zero=True)
                m_cavity_B_dsp = key.Waveform(cavity_B_marker_dsp[i], append_zero=True)

            # trig = key.Waveform(trig_arr_awg, append_zero=True)
            card_trig = key.Waveform(card_trig_arr[i], append_zero=True)

            # print(" got thru key.waveform")

            # Load objects to the modules
            wave_A_I.loadToModule(AWG_A_module)
            wave_A_Q.loadToModule(AWG_A_module)
            wave_B_I.loadToModule(AWG_B_module)
            wave_B_Q.loadToModule(AWG_B_module)
            if self.prep_cavity_drive:
                wave_cavity_A_I.loadToModule(AWG_A_module)
                wave_cavity_A_Q.loadToModule(AWG_A_module)
                wave_cavity_B_I.loadToModule(AWG_B_module)
                wave_cavity_B_Q.loadToModule(AWG_B_module)

            # print("got thru ab loadtomodule")

            # Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.

            # rn tek2 trigger delay is 0


            wave_A_I.queue(self.AWG_A_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                               delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay ,
                               cycles=1, prescaler=0)
            wave_A_Q.queue(self.AWG_A_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                               delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay,
                               cycles=1, prescaler=0)
            wave_B_I.queue(self.AWG_B_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                               delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay + self.bob_delay_rel_alice,
                               cycles=1, prescaler=0)
            wave_B_Q.queue(self.AWG_B_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                               delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay + self.bob_delay_rel_alice ,
                               cycles=1, prescaler=0)
            if self.prep_cavity_drive:
                wave_cavity_A_I.queue(self.AWG_A_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                          delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)
                wave_cavity_A_Q.queue(self.AWG_A_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                          delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)
                wave_cavity_B_I.queue(self.AWG_B_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                          delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)
                wave_cavity_B_Q.queue(self.AWG_B_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                          delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)

            # print("got through wave queue")
            self.AWG_A_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                       value=1,
                                                       syncMode=1, length=10, delay=0)
            self.AWG_B_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                       value=1,
                                                       syncMode=1, length=10, delay=0)

            # print("got through ab queue")
            # 4/21 note that in our scheme module 8 supplies qubit readout and drive marker pulses
            # cavity drive are what goes to module 9
            m_A_qubit_dsp.loadToModule(m_8_module)
            m_B_qubit_dsp.loadToModule(m_8_module)
            m_A_readout_dsp.loadToModule(m_8_module)
            m_B_readout_dsp.loadToModule(m_8_module)

            # if self.prep_tek2: m_tek2_dsp.loadToModule(m_module)
            if self.prep_cavity_drive:
                m_cavity_A_dsp.loadToModule(m_9_module)
                m_cavity_B_dsp.loadToModule(m_9_module)

            # trig.loadToModule(trig_module)
            card_trig.loadToModule(m_9_module)
            # pretty sure this is trigger pulse from card 9 ch 3 to digitizer trigger in port



            m_A_qubit_dsp.queue(self.m_8_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay ,
                                    cycles=1, prescaler=0)
            m_A_readout_dsp.queue(self.m_8_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                      delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay + self.readout_marker_pulse_delay , cycles=1, prescaler=0)
            m_B_qubit_dsp.queue(self.m_8_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay,
                                    cycles=1, prescaler=0)
            # possible that sync functions auto-move markers to match qubits, not sure adding self.bob_delay_rel_alice is needed
            m_B_readout_dsp.queue(self.m_8_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                      delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay + self.readout_marker_pulse_delay , cycles=1, prescaler=0)
            # if self.prep_tek2: m_tek2_dsp.queue(self.m_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0,
            #                                     cycles=1, prescaler=0)
            # m_ch_number sets which channel of the pxi emits the trigger tone to source

            if self.prep_cavity_drive:
                m_cavity_A_dsp.queue(self.m_9_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                         delay=self.tek2_trigger_delay+ self.global_trig_delay_fromawg +  self.marker_pulse_delay, cycles=1, prescaler=0)
                m_cavity_B_dsp.queue(self.m_9_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                         delay=self.tek2_trigger_delay+ self.global_trig_delay_fromawg +  self.marker_pulse_delay, cycles=1, prescaler=0)

            # trig.queue(self.trig_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            # trig.queue(self.trig_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            # trig.queue(self.trig_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            card_trig.queue(self.m_9_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                delay=int(self.card_delay / 100) + self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.marker_pulse_delay + self.readout_marker_pulse_delay + self.tentative_card_9_delay,
                                cycles=1, prescaler=0)

            # self.trig_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
            #                                       syncMode=1, length=10, delay=0)
            # print("got to end of loadandqueue", i)
            self.m_8_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                     syncMode=1, length=10, delay=0)

            self.m_9_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                     value=1,
                                                     syncMode=1, length=10, delay=0)

            # print("got thru markers queue")

    def run(self):
        print("Experiment starting. Expected time = ", self.totaltime, "mins")
        try:
            # Start all the channels on the AWG and digitizer modules.
            #print ("Number of experiments = ",self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_2.clear()
            self.DIG_ch_2.start()
            self.DIG_ch_3.clear()
            self.DIG_ch_3.start()
            self.DIG_ch_4.clear()
            self.DIG_ch_4.start()


            self.AWG_A_module.startAll()
            self.AWG_B_module.startAll()
            self.m_8_module.startAll()
            self.m_9_module.startAll()
            # self.trig_module.startAll()
            #print("started all modules")

        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self,expt_num):
        # not sure this saves anything? A check fn

        for sweep_ct in tqdm(range(self.num_avg)):

            self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
            self.data_2 += np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)
            # NOTE we may need two different progress bars here
            self.data_3 += np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape)
            self.data_4 += np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape)


        self.data_1 /= self.num_avg
        self.data_2 /= self.num_avg
        self.data_3 /= self.num_avg
        self.data_4 /= self.num_avg

        print("Processed data shape", np.shape(self.data_1))


        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131, title='I')
        plt.imshow(self.data_1, aspect='auto')
        ax.set_xlabel('Digitizer bins')
        ax.set_ylabel('Experiment number')
        ax2 = fig.add_subplot(132, title='Q')
        plt.imshow(self.data_2, aspect='auto')
        ax2.set_xlabel('Digitizer bins')
        ax2.set_ylabel('Experiment number')
        ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
        ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_1[expt_num])
        ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_2[expt_num])
        ax3.axvspan(self.readout_A_window[0], self.readout_A_window[1], alpha=0.2, color='b')
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Signal')
        fig.tight_layout()
        plt.show()

        fig2 = plt.figure(figsize=(12, 4))
        ax4 = fig2.add_subplot(131, title='I')
        plt.imshow(self.data_3, aspect='auto')
        ax4.set_xlabel('Digitizer bins')
        ax4.set_ylabel('Experiment number')
        ax5 = fig2.add_subplot(132, title='Q')
        plt.imshow(self.data_4, aspect='auto')
        ax5.set_xlabel('Digitizer bins')
        ax5.set_ylabel('Experiment number')
        ax6 = fig2.add_subplot(133, title='Expt num = ' + str(expt_num))
        ax6.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_3[expt_num])
        ax6.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_4[expt_num])
        ax6.axvspan(self.readout_B_window[0], self.readout_B_window[1], alpha=0.2, color='b')
        ax6.set_xlabel('Time (ns)')
        ax6.set_ylabel('Signal')
        fig2.tight_layout()
        plt.show()

    def SSdata_many(self, wA=[0, -1], wB=[0, -1]):

        data = []
        IA = []
        QA = []
        IB = []
        QB = []
        for ii in tqdm(range(self.num_avg)):
            IA.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])], 0))
            QA.append(np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])], 0))
            IB.append(np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])], 0))
            QB.append(np.mean(np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])], 0))
        IAt = np.array(IA).T
        QAt = np.array(QA).T
        IBt = np.array(IB).T
        QBt = np.array(QB).T
        data.append([IAt, QAt])
        data.append([IBt, QBt])
        return np.asarray(data)

    def traj_data_many(self, wA=[0, -1], wB=[0, -1] ):
        data = []
        IA = []
        QA = []
        IB = []
        QB = []
        for ii in tqdm(range(self.num_avg)):
            ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])].T
            ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])].T
            ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])].T
            ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])].T
            IA.append(ch1)
            QA.append(ch2)
            IB.append(ch3)
            QB.append(ch4)
        data.append([IA, QA])
        data.append([IB, QB])
        return np.asarray(data)

    def traj_data_many_nowindow(self):
        data = []
        # default to the most permissive window conditions by force
        wA = [0, -1]
        wB = [0, -1]
        IA = []
        QA = []
        IB = []
        QB = []
        for ii in tqdm(range(self.num_avg)):
            ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])].T
            ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])].T
            ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])].T
            ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])].T
            IA.append(ch1)
            QA.append(ch2)
            IB.append(ch3)
            QB.append(ch4)
        data.append([IA, QA])
        data.append([IB, QB])
        return np.asarray(data)

    def acquire_avg_data(self,wA = [0,-1],wB = [0,-1],pi_calibration=False,rotate_iq_A = False,rotate_iq_B = False,phi_A=0,phi_B=0):

        data = []
        # self.I, self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IA, self.QA = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IB, self.QB = np.zeros(self.num_expt), np.zeros(self.num_expt)

        if rotate_iq_A and rotate_iq_B:
            print("Rotating IQ A&B digitally")
                    # If you want to remove tqdm to get at error messages you can do it here, just leave a note for yourself
            for ii in tqdm(range(self.num_avg)):
                iAd = self.DIG_ch_1.readDataQuiet(timeout=20000)
                qAd = self.DIG_ch_2.readDataQuiet(timeout=20000)
                iBd = self.DIG_ch_3.readDataQuiet(timeout=20000)
                qBd = self.DIG_ch_4.readDataQuiet(timeout=20000)
                    # if you have buffer issues try this verbose version
                    # iAd=self.DIG_ch_1.readData(data_points=self.data_1.shape[0]*self.data_1.shape[1])
                    # qAd = self.DIG_ch_2.readData(data_points=self.data_2.shape[0] * self.data_2.shape[1])
                    # iBd=self.DIG_ch_3.readData(data_points=self.data_3.shape[0]*self.data_3.shape[1])
                    # qBd = self.DIG_ch_4.readData(data_points=self.data_4.shape[0] * self.data_4.shape[1])

                IAtemp = np.reshape(iAd, self.data_1.shape).T[int(wA[0]):int(wA[1])]
                QAtemp = np.reshape(qAd, self.data_2.shape).T[int(wA[0]):int(wA[1])]
                IBtemp = np.reshape(iBd, self.data_3.shape).T[int(wB[0]):int(wB[1])]
                QBtemp = np.reshape(qBd, self.data_4.shape).T[int(wB[0]):int(wB[1])]
                # Presumably we might need different rotation angles here for A, B
                IArot = IAtemp * np.cos(phi_A) + QAtemp * np.sin(phi_A)
                QArot = -IAtemp * np.sin(phi_A) + QAtemp * np.cos(phi_A)
                IBrot = IBtemp * np.cos(phi_B) + QBtemp * np.sin(phi_B)
                QBrot = -IBtemp * np.sin(phi_B) + QBtemp * np.cos(phi_B)

                self.IA += np.mean(IArot, 0)
                self.QA += np.mean(QArot, 0)
                self.IB += np.mean(IBrot, 0)
                self.QB += np.mean(QBrot, 0)

        elif rotate_iq_A and not rotate_iq_B:
            print("Rotating IQ A digitally")
            # If you want to remove tqdm to get at error messages you can do it here, just leave a note for yourself
            for ii in tqdm(range(self.num_avg)):
                iAd = self.DIG_ch_1.readDataQuiet(timeout=20000)
                qAd = self.DIG_ch_2.readDataQuiet(timeout=20000)
                iBd = self.DIG_ch_3.readDataQuiet(timeout=20000)
                qBd = self.DIG_ch_4.readDataQuiet(timeout=20000)
                    # if you have buffer issues try this verbose version
                    # iAd=self.DIG_ch_1.readData(data_points=self.data_1.shape[0]*self.data_1.shape[1])
                    # qAd = self.DIG_ch_2.readData(data_points=self.data_2.shape[0] * self.data_2.shape[1])
                    # iBd=self.DIG_ch_3.readData(data_points=self.data_3.shape[0]*self.data_3.shape[1])
                    # qBd = self.DIG_ch_4.readData(data_points=self.data_4.shape[0] * self.data_4.shape[1])

                IAtemp = np.reshape(iAd, self.data_1.shape).T[int(wA[0]):int(wA[1])]
                QAtemp = np.reshape(qAd, self.data_2.shape).T[int(wA[0]):int(wA[1])]
                IBtemp = np.reshape(iBd, self.data_3.shape).T[int(wB[0]):int(wB[1])]
                QBtemp = np.reshape(qBd, self.data_4.shape).T[int(wB[0]):int(wB[1])]
                # Presumably we might need different rotation angles here for A, B
                IArot = IAtemp * np.cos(phi_A) + QAtemp * np.sin(phi_A)
                QArot = -IAtemp * np.sin(phi_A) + QAtemp * np.cos(phi_A)
                # IBrot = IBtemp * np.cos(phi) + QBtemp * np.sin(phi)
                # QBrot = -IBtemp * np.sin(phi) + QBtemp * np.cos(phi)

                self.IA += np.mean(IArot, 0)
                self.QA += np.mean(QArot, 0)
                self.IB += np.mean(IBtemp, 0)
                self.QB += np.mean(QBtemp, 0)

        elif rotate_iq_B and not rotate_iq_A:
            print("Rotating IQ B digitally")
            # If you want to remove tqdm to get at error messages you can do it here, just leave a note for yourself
            for ii in tqdm(range(self.num_avg)):
                iAd = self.DIG_ch_1.readDataQuiet(timeout=20000)
                qAd = self.DIG_ch_2.readDataQuiet(timeout=20000)
                iBd = self.DIG_ch_3.readDataQuiet(timeout=20000)
                qBd = self.DIG_ch_4.readDataQuiet(timeout=20000)
                    # if you have buffer issues try this verbose version
                    # iAd=self.DIG_ch_1.readData(data_points=self.data_1.shape[0]*self.data_1.shape[1])
                    # qAd = self.DIG_ch_2.readData(data_points=self.data_2.shape[0] * self.data_2.shape[1])
                    # iBd=self.DIG_ch_3.readData(data_points=self.data_3.shape[0]*self.data_3.shape[1])
                    # qBd = self.DIG_ch_4.readData(data_points=self.data_4.shape[0] * self.data_4.shape[1])

                IAtemp = np.reshape(iAd, self.data_1.shape).T[int(wA[0]):int(wA[1])]
                QAtemp = np.reshape(qAd, self.data_2.shape).T[int(wA[0]):int(wA[1])]
                IBtemp = np.reshape(iBd, self.data_3.shape).T[int(wB[0]):int(wB[1])]
                QBtemp = np.reshape(qBd, self.data_4.shape).T[int(wB[0]):int(wB[1])]
                # Presumably we might need different rotation angles here for A, B
                # IArot = IAtemp * np.cos(phi) + QAtemp * np.sin(phi)
                # QArot = -IAtemp * np.sin(phi) + QAtemp * np.cos(phi)
                IBrot = IBtemp * np.cos(phi_B) + QBtemp * np.sin(phi_B)
                QBrot = -IBtemp * np.sin(phi_B) + QBtemp * np.cos(phi_B)

                self.IA += np.mean(IAtemp, 0)
                self.QA += np.mean(QAtemp, 0)
                self.IB += np.mean(IBrot, 0)
                self.QB += np.mean(QBrot, 0)
        else:
            # REMOVED TQDM HERE
            # Reinserted by MGP December 11, 2019, removed April 1 2020, reinserted September 17, 2020
            for ii in tqdm(range(self.num_avg)):
                self.IA += np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(timeout=10000), self.data_1.shape).T[
                                      int(wA[0]):int(wA[1])], 0)
                self.QA += np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(timeout=10000), self.data_2.shape).T[
                                      int(wA[0]):int(wA[1])], 0)
                self.IB += np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(timeout=10000), self.data_3.shape).T[
                                       int(wB[0]):int(wB[1])], 0)
                self.QB += np.mean(np.reshape(self.DIG_ch_4.readDataQuiet(timeout=10000), self.data_4.shape).T[
                                       int(wB[0]):int(wB[1])], 0)
                print(ii)
        IA = self.IA / self.num_avg
        QA = self.QA / self.num_avg
        IB = self.IB / self.num_avg
        QB = self.QB / self.num_avg
        if pi_calibration:
            IA = (IA[:-2] - IA[-2]) / (IA[-1] - IA[-2])
            QA = (QA[:-2] - QA[-2]) / (QA[-1] - QA[-2])
            IB = (IB[:-2] - IB[-2]) / (IB[-1] - IB[-2])
            QB = (QB[:-2] - QB[-2]) / (QB[-1] - QB[-2])
        data.append([IA, QA])
        data.append([IB, QB])
        return np.asarray(data)


class KeysightSingleQubit:
    '''Class designed to implement a simple single qubit experiment given pulse sequences from the Sequencer class. Does  not use
    HVI technology.

    Module (slot) 6 and 7 are used for AWGS. Module 8 is used as a marker. Module 9 is triggered externally and its outputs are used
    to trigger the rest of the modules. ch4 of this trig for the digitizer. Digitizer is module 10.

    Module 6 is currently awg B, module 7 awg A

    On Module 6, channel 1 goes to the I input to the mixer, channel 2 goes to the Q input, channel 3 is the readout pulse, and
    channel 4 is the readout marker. On module 10, channel 1 is for readout of I component and channel 2 is for readout from Q component.'''

    # update 4/21: we are modifying this class for two qubit use rather than writing a new one and worrying abt where it gets called
    # given all the modules are hardcoded it doesn't matter
    # if we hit issues will rethink

    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, sequences, name, save_path=r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials=50 * 1000):  # 1000*10000 if you want to watch sweep by eye

        chassis = key.KeysightChassis(1,
                                      {6: key.ModuleType.OUTPUT,
                                       8: key.ModuleType.OUTPUT,
                                       7: key.ModuleType.OUTPUT,
                                       9: key.ModuleType.OUTPUT,
                                       10: key.ModuleType.INPUT})
        # self.chassis = chassis

        self.hardware_cfg = hardware_cfg
        self.out_mod_no_A = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no'][0]
        self.out_mod_no_B = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no'][1]
        # Right now 7 is A and 6 is B
        self.marker_mod_no_8 = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no'][0]
        self.marker_mod_no_9 = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no'][1]
        # self.trig_mod_no = hardware_cfg['awg_info']['keysight_pxi']['trig_mod_no']

        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        self.readout_A_window = np.array(quantum_device_cfg['readout']['1']['window'])
        self.readout_B_window = np.array(quantum_device_cfg['readout']['2']['window'])
        self.lo_delay = hardware_cfg['awg_info']['keysight_pxi']['lo_delay']
        self.abs_trig_delay = hardware_cfg['awg_info']['keysight_pxi']['abs_trig_delay']
        # the below global delay will be added to tek2_trigger_delay and propagated
        self.global_trig_delay_fromawg = hardware_cfg['awg_info']['keysight_pxi']['global_trig_delay_fromawg']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']
        self.trig_delay = hardware_cfg['awg_info']['keysight_pxi']['m3201a_trig_delay']
        self.card_delay = hardware_cfg['awg_info']['keysight_pxi']['m3102a_card_delay']
        self.adc_range =  hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
        self.iq_line_delay = hardware_cfg['awg_info']['keysight_pxi']['iq_line_delay']
        self.marker_pulse_delay = hardware_cfg['awg_info']['keysight_pxi']['marker_pulse_delay']
        self.readout_marker_pulse_delay = hardware_cfg['awg_info']['keysight_pxi']['readout_marker_pulse_delay']
        self.bob_delay_rel_alice = hardware_cfg['awg_info']['keysight_pxi']['bob_delay_rel_alice']
        self.tentative_card_8_delay = hardware_cfg['awg_info']['keysight_pxi']['tentative_card_8_delay']
        self.tentative_card_9_delay = hardware_cfg['awg_info']['keysight_pxi']['tentative_card_9_delay']



        print ("Module used for generating analog pulses for A = ",self.out_mod_no_A)
        print("Module used for generating analog pulses for B = ", self.out_mod_no_B)
        print ("Module used for generating digital markers = ",self.marker_mod_no_8)
        print("Module used for generating digital markers and digitizer trigger = ", self.marker_mod_no_9)
        # print ("Module used for temp generation of module = ", self.trig_mod_no)
        self.out_mod_nums = [self.out_mod_no_A, self.out_mod_no_A, self.marker_mod_no_8, self.marker_mod_no_9]
        # removed self.trig_mod_no

        self.num_avg = experiment_cfg[name]['acquisition_num']
        self.on_qubits = experiment_cfg[name]['on_qubits']
        # print(self.on_qubits[0], "on qubits!!")
        # may need to modify 4/21
        # self.num_expt = sequences['readout'].shape[0]
        for qubit_id in self.on_qubits:
            self.num_expt = sequences['readout%s' % qubit_id].shape[0]
            # if there's one it'll pick the right readout and if two will take the second only but that's fine
        self.trigger_period = self.hardware_cfg['trigger']['period_us']

        self.totaltime = self.num_avg*self.num_expt*self.trigger_period*1e-6/60.0

        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']


        if 'sideband' in name:
            print("Yikes you are using legacy tek2 code in keysight pxi load")
            self.prep_tek2 = True
            self.tek2_trigger_delay = int(hardware_cfg['awg_info']['tek70001a']['trig_delay']/100.0)
        else:
            # print("Yikes you are using legacy tek2 code in keysight pxi load")
            self.prep_tek2=False
            self.tek2_trigger_delay=0



        if 'cavity_drive' in name:self.prep_cavity_drive = True
        else:self.prep_cavity_drive=False

        # mgp 4/21 added
        # if 'two_qubits' in name:self.two_qubits = True
        # else:self.two_qubits=False

        # if 'cavity_sideband' in name:
        #     self.prep_cavity_drive = True
        # else:self.prep_cavity_drive=False

        self.chassis = chassis
        self.awg_channels = range(1,5)
        # self.dig_channels = range(1,3)
        self.dig_channels = range(1, 5)


        # Initialize Module 6 = Analog card for A.  Ch1 = AWG I, CH2 = AWG Q

        # print("abt to getChannel things")
        # self.AWG_chs = [chassis.getChannel(self.out_mod_no, ch) for ch in self.awg_channels]
        self.AWG_A_ch_1 = chassis.getChannel(self.out_mod_no_A, 1)
        self.AWG_A_ch_2 = chassis.getChannel(self.out_mod_no_A, 2)
        self.AWG_A_ch_3 = chassis.getChannel(self.out_mod_no_A, 3)
        self.AWG_A_ch_4 = chassis.getChannel(self.out_mod_no_A, 4)

        # Initialize Module 7 = Analog card for B.  Ch1 = AWG I, CH2 = AWG Q

        # self.AWG_chs = [chassis.getChannel(self.out_mod_no, ch) for ch in self.awg_channels]
        self.AWG_B_ch_1 = chassis.getChannel(self.out_mod_no_B, 1)
        self.AWG_B_ch_2 = chassis.getChannel(self.out_mod_no_B, 2)
        self.AWG_B_ch_3 = chassis.getChannel(self.out_mod_no_B, 3)
        self.AWG_B_ch_4 = chassis.getChannel(self.out_mod_no_B, 4)


        # Initialize Module 8 = Marker card. Digital markers for qubit A/B, readout
        # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]
        self.m_8_ch_1 = chassis.getChannel(self.marker_mod_no_8, 1)
        self.m_8_ch_2 = chassis.getChannel(self.marker_mod_no_8, 2)
        self.m_8_ch_3 = chassis.getChannel(self.marker_mod_no_8, 3)
        self.m_8_ch_4 = chassis.getChannel(self.marker_mod_no_8, 4)

        # Initialize Module 9 = Marker card. Digital markers for cavity A/B, digitizer trigger
        # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]
        self.m_9_ch_1 = chassis.getChannel(self.marker_mod_no_9, 1)
        self.m_9_ch_2 = chassis.getChannel(self.marker_mod_no_9, 2)
        self.m_9_ch_3 = chassis.getChannel(self.marker_mod_no_9, 3)
        self.m_9_ch_4 = chassis.getChannel(self.marker_mod_no_9, 4)

        # Initialize card that generates Triggers
        # self.trig_chs = [chassis.getChannel(self.trig_mod_no, ch) for ch in self.awg_channels]
        # self.trig_ch_1 = chassis.getChannel(self.trig_mod_no, 1)
        # self.trig_ch_2 = chassis.getChannel(self.trig_mod_no, 2)
        # self.trig_ch_3 = chassis.getChannel(self.trig_mod_no, 3)
        # self.trig_ch_4 = chassis.getChannel(self.trig_mod_no, 4)

        # print("got channels excepting dig")

        # self.out_modules = [chassis.getModule(num) for num in self.out_mod_nums]
        self.AWG_A_module = chassis.getModule(self.out_mod_no_A)
        self.AWG_B_module = chassis.getModule(self.out_mod_no_B)
        self.m_8_module = chassis.getModule(self.marker_mod_no_8)
        self.m_9_module = chassis.getModule(self.marker_mod_no_9)
        # self.trig_module = chassis.getModule(self.trig_mod_no)

        # print("got modules excepting dig")

        # Initialize digitizer card
        # self.DIG_chs = [chassis.getChannel(10, ch) for ch in self.dig_channels]
        self.DIG_ch_1 = chassis.getChannel(10, 1)
        self.DIG_ch_2 = chassis.getChannel(10, 2)
        self.DIG_ch_3 = chassis.getChannel(10, 3)
        self.DIG_ch_4 = chassis.getChannel(10, 4)
        self.DIG_module = chassis.getModule(10)
        # print("got channels, module for digitizer")

        # pretty sure we could have used data_1 and data_2 for any 1 qubit experiment and just reassigned digitizer channels but whatever
        if self.on_qubits[0] == "1":
            self.data_1, self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)), np.zeros(
                    (self.num_expt, self.DIG_sampl_record))
        elif self.on_qubits[0] == "2":
            self.data_3, self.data_4 = np.zeros((self.num_expt, self.DIG_sampl_record)), np.zeros(
                    (self.num_expt, self.DIG_sampl_record))
        else:
            print('You have an issue with the on_qubits config option')

        # self.I, self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)
        if self.on_qubits[0] == "1":
            self.IA,self.QA = np.zeros(self.num_expt), np.zeros(self.num_expt)
        elif self.on_qubits[0] == "2":
            self.IB, self.QB = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally
        # print("got to end of init")


    def configureChannels(self, hardware_cfg, experiment_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_AWG_A = hardware_cfg['awg_info']['keysight_pxi']['amplitudes_A']
        amp_AWG_B = hardware_cfg['awg_info']['keysight_pxi']['amplitudes_B']
        amp_mark_8 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_8']
        amp_mark_9 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_9']
        # amp_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_trig']
        # dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        # dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring analog channels")

        self.AWG_A_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_A_ch_1.configure(amplitude=amp_AWG_A[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_2.configure(amplitude=amp_AWG_A[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_3.configure(amplitude=amp_AWG_A[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_A_ch_4.configure(amplitude=amp_AWG_A[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        self.AWG_B_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_B_ch_1.configure(amplitude=amp_AWG_B[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_2.configure(amplitude=amp_AWG_B[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_3.configure(amplitude=amp_AWG_B[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.AWG_B_ch_4.configure(amplitude=amp_AWG_B[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        print("Configuring marker channels")

        self.m_8_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_8_ch_1.configure(amplitude=amp_mark_8[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_2.configure(amplitude=amp_mark_8[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_4.configure(amplitude=amp_mark_8[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_8_ch_3.configure(amplitude=amp_mark_8[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_4.configure(amplitude=amp_mark[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # print("tried marker 4")

        self.m_9_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_9_ch_1.configure(amplitude=amp_mark_9[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_2.configure(amplitude=amp_mark_9[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_4.configure(amplitude=amp_mark_9[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        self.m_9_ch_3.configure(amplitude=amp_mark_9[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_4.configure(amplitude=amp_mark[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # print("tried marker 4")

        # print("Configuring trigger channels")
        #
        # self.trig_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.trig_ch_1.configure(amplitude=amp_trig[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_2.configure(amplitude=amp_trig[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_3.configure(amplitude=amp_trig[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_4.configure(amplitude=amp_trig[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        print ("Setting trigger config for all channels of all modules to External")

        for n in range(1, 5):
            self.AWG_A_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.AWG_B_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_8_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_9_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            # self.trig_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)


        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")

        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)

        # print("configured digiitzer")

    def configureDigitizerChannels(self, hardware_cfg, experiment_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_mark_9 = hardware_cfg['awg_info']['keysight_pxi']['amp_mark_9']
        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' % num_expt)

        # print("Configuring marker channels")
        #
        # self.m_9_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.m_9_ch_3.configure(amplitude=amp_mark_9[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        #
        # print("Setting trigger config for all channels of all modules to External")
        #
        # for n in range(1, 5):
        #     self.m_9_module.AWGtriggerExternalConfig(nAWG=n,
        #                                              externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
        #                                              triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
        # #     # self.trig_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)

        print("Configuring digitizer. ADC range set to", self.adc_range, "Vpp")

        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale=self.adc_range, points_per_cycle=self.DIG_sampl_record,
                                cycles=num_expt * num_avg, buffer_time_out=100000,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)
        # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale=self.adc_range, points_per_cycle=self.DIG_sampl_record,
                                buffer_time_out=100000, cycles=num_expt * num_avg,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)
        # self.DIG_ch_3.configure(full_scale=self.adc_range, points_per_cycle=self.DIG_sampl_record,
        #                         cycles=num_expt * num_avg, buffer_time_out=100000,
        #                         trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
        #                         cycles_per_return=num_expt)
        # # self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        # self.DIG_ch_4.configure(full_scale=self.adc_range, points_per_cycle=self.DIG_sampl_record,
        #                         buffer_time_out=100000, cycles=num_expt * num_avg,
        #                         trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
        #                         cycles_per_return=num_expt)

        # print("configured digiitzer")

    def generatemarkers(self,waveform,resample=False,conv_width = 1,trig_delay = 0):

        # markers = np.array([np.append(np.heaviside(np.convolve(abs(w), np.ones((int(self.lo_delay),))/self.lo_delay),0)[int(self.lo_delay/2):],np.zeros(int(self.lo_delay/2))) for w in waveform])
        markers = np.array([np.append(np.heaviside(np.convolve(abs(np.append(w[int(trig_delay):],np.zeros(int(trig_delay)))), np.ones((int(conv_width),)) / conv_width), 0)[int(conv_width / 2):], np.zeros(int(conv_width / 2))) for w in waveform])
        if resample: return np.array([m[::int(self.dt_dig/self.dt)] for m in markers])
        else:return markers

    def generatemastertrigger(self,length,trig_width = 1,trig_delay = 0):
        trig = np.zeros(length)
        for i in np.arange(int(trig_delay/self.dt_dig),int((trig_delay + trig_width)/self.dt_dig)):
            trig[i] = 1
        return trig

    # 4/2021 modified hardware config for current expt to have more channels
    def sequenceslist(self, sequences, waveform_channels):
        wv = {}
        for channel in waveform_channels:
            if not channel == None:
                wv[channel] = sequences[channel]
            else:
                wv[channel] = np.zeros_like(sequences[waveform_channels[0]])
        return wv

    #old version with wv[0] etc as calls in loadandqueue below
    # def sequenceslist(self,sequences,waveform_channels):
    #     wv = []
    #     for channel in waveform_channels:
    #         if not channel == None:
    #             wv.append(sequences[channel])
    #         else:
    #             wv.append(np.zeros_like(sequences[waveform_channels[0]]))
    #
    #     return wv
    #
    #from gerbert 4/21
    # def sequenceslist(self,sequences,waveform_channels):
    #     wv = {}
    #     for channel in waveform_channels:
    #         if not channel == None: #TODO: I think this should be if not sequences[channel] == None
    #             wv[channel] = sequences[channel]
    #         else:
    #             wv[channel] = np.zeros_like(sequences[waveform_channels[0]])
    #     return wv

    def loadAndQueueWaveforms(self,sequences):
        '''Loads the provided waveforms from a pulse sequence to the appropriate modules.

        Note that all waveforms should consist of values from -1 to 1 (inclusive) only. Amplitude is set in the configureChannels() method.
        If you accidentally pass in an array containing larger values, the code raises a KeysightError: Invalid Waveform.

        Params:
            waveforms_I: List of numpy arrays representing waveforms (or a 2D array) for the "I" input AWG to the mixer,
                1 for each unique trial/experiment. Should be the same number of waveforms as num_experiments in the __init__ method.
                The "inner" array is a waveform, and the "outer" dimension corresponds to the experiment/trial.
            waveforms_Q: Same for the "Q" channel.
            readout: Readout waveform used to trigger the readout LO
            markers_readout: Trigger for the digitizer
            The master trigger for all the cards is generated knowing the length of the AWG waveforms using self.generate_trigger
            '''

        # Modified this for a second set of waveforms. 4/21 mgp


        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        # print("def'd pxi waveform channels")

        for qubit_id in self.on_qubits:
            self.num_expt = sequences['readout%s' % qubit_id].shape[0]
            num_expt = self.num_expt
            # if there's one it'll pick the right readout
        # print("this is about to print num expt")


        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences,pxi_waveform_channels)
        # print("we at least configured wv")
       # newname=s
        # you need to manually make sure wv is the right length for this to work
        # new list for ref  "waveform_channels": ["charge1A_I", "charge1A_Q","charge1B_I", "charge1B_Q", "readoutA", "readoutB",  "readout_trig","tek2_trig","cavity1A_I", "cavity1A_Q","cavity1B_I", "cavity1B_Q"]

        if self.on_qubits[0] == "1":
            waveforms_I = wv["charge1_I"]
            waveforms_Q = wv["charge1_Q"]
            readout = wv["readout1"]
            markers_readout = wv["readout_trig"]
            # if self.prep_tek2: tek2_marker = wv["tek2_trig"]
            if self.prep_cavity_drive:
                cavity_I = wv["cavity1_I"]
                cavity_Q = wv["cavity1_Q"]
            # print("got thru wv assign 1")
        elif self.on_qubits[0] == "2":
            waveforms_I = wv["charge2_I"]
            waveforms_Q = wv["charge2_Q"]
            readout = wv["readout2"]
            markers_readout = wv["readout_trig"]
            # if self.prep_tek2: tek2_marker = wv["tek2_trig"]
            if self.prep_cavity_drive:
                cavity_I = wv["cavity2_I"]
                cavity_Q = wv["cavity2_Q"]
            # print("got thru wv assign 2")
        else:
            print("Error naming waveforms in keysight_pxi_load.")


        # Based on Nelson's original nomenclature

        # old version w numbers for diff sequenceslist fn
        # modified 4/21 mgp
        # # you need to manually make sure wv is the right length for this to work
        # #new list for ref  "waveform_channels": ["charge1A_I", "charge1A_Q","charge1B_I", "charge1B_Q", "readoutA", "readoutB",  "readout_trig","tek2_trig","cavity1A_I", "cavity1A_Q","cavity1B_I", "cavity1B_Q"]
        # if self.two_qubits:
        #     waveforms_A_I = wv[0]
        #     waveforms_A_Q = wv[1]
        #     waveforms_B_I = wv[2]
        #     waveforms_B_Q = wv[3]
        #     readout_A = wv[4]
        #     readout_B = wv[5]
        #     markers_readout = wv[6]
        #     # question: we only need 1 marker to digitizer channel yes? it gets fed into trigger port
        #     # will have to be careful re channels when making actual waveform
        #     # issue if a,b waveforms differ bc length check error below will get thrown
        #     if self.prep_tek2:tek2_marker = wv[7]
        #     if self.prep_cavity_drive:
        #         cavity_A_I = wv[8]
        #         cavity_A_Q = wv[9]
        #         cavity_B_I = wv[10]
        #         cavity_B_Q = wv[11]
        # elif not self.two_qubits:
        #     waveforms_I = wv[0]
        #     waveforms_Q = wv[1]
        #     readout = wv[2]
        #     markers_readout = wv[3]
        #     if self.prep_tek2: tek2_marker = wv[4]
        #     if self.prep_cavity_drive:
        #         cavity_I = wv[5]
        #         cavity_Q = wv[6]
        # else:
        #     print("Error naming waveforms in keysight_pxi_load.")


        # waveforms_I = wv[0]
        # waveforms_Q = wv[1]
        # readout = wv[2]
        # markers_readout = wv[3]
        # if self.prep_tek2:tek2_marker = wv[4]
        # if self.prep_cavity_drive:
        #     cavity_I = wv[5]
        #     cavity_Q = wv[6]

        # AWG_module = self.chassis.getModule(self.out_mod_no)
        # m_module = self.chassis.getModule(self.marker_mod_no)
        # trig_module = self.chassis.getModule(self.trig_mod_no)


        # modified 4/21 mgp
        # For some reason it's having a hard time remembering what chassis is
        # chassis = key.KeysightChassis(1,
        #                               {6: key.ModuleType.OUTPUT,
        #                                8: key.ModuleType.OUTPUT,
        #                                7: key.ModuleType.OUTPUT,
        #                                9: key.ModuleType.OUTPUT,
        #                                10: key.ModuleType.INPUT})
        chassis = self.chassis
        # these commands need a slot number!!!
        AWG_A_module = chassis.getModule(self.out_mod_no_A)
        AWG_B_module = chassis.getModule(self.out_mod_no_B)
        m_8_module = chassis.getModule(self.marker_mod_no_8)
        m_9_module = chassis.getModule(self.marker_mod_no_9)

        print ("shape of waveform I",np.shape(waveforms_I))

        if len(waveforms_I) != len(waveforms_Q) or len(waveforms_I) != len(markers_readout) or len(waveforms_I) != len(
                readout):
            raise TypeError("Not all waveform lists are the same length")

        # self.AWG_module.clearAll()
        # self.m_module.clearAll()
        # self.trig_module.clearAll()
        self.AWG_A_module.clearAll()
        self.AWG_B_module.clearAll()
        self.m_8_module.clearAll()
        self.m_9_module.clearAll()
        #print("attempted to clear modules")

        # dsp stands for downsampling, which to use gets selected later in hardcode depending on card speed
        # delays seem to be showing up here in delays on marker cards or in nelson's channels_delay
        # lo_delay makes output marker to los wider or smaller. a convolution factor altering output marker to lo
        # trig_delay comes from m3201 delay in hardware config. delays marker pulses by adding zeros

        # abs_trig_delay sets delay on markers to awg cards. for anything not trig to digitizer, adds zeros to that marker array
        # m3102 delay in hwcfg gets called card delay here, this affects digitizer waveform. divided by 100 idk why. delays channels from hardware!!!
        # delays w queue waveform less accurate than thru arrays maybe
        # card delay delays channel triggering digitizer
        # change abs trig delay and card delay apr 2021?

        # dsp = downsampling
        # markers_readout -> trigger for digitizer card coming from trigger card
        # readout_markers -> trigger for readout LO coming from marker card
        # if marker card is 500MS/s, use dsp version
        # if marker card is 1Gs/s, use non-dsp-ed version
        # if trigger card is 500MS/s, use dsp version
        # if trigger card is 1Gs/s, use non-dsp-ed version

        # MAY NEED TO DIVERSIFY DELAYS if different qubits need diff things
        key.Waveform._waveform_number_counter = 0

        # print("about to generate markers kpl")
        qubit_marker = self.generatemarkers(waveforms_I)
        qubit_marker_dsp = self.generatemarkers(waveforms_I,resample=True,conv_width=self.lo_delay,trig_delay=self.trig_delay)
        readout_marker = self.generatemarkers(readout, resample=False)
        readout_marker_dsp = self.generatemarkers(readout, resample=True,conv_width=self.lo_delay, trig_delay=self.trig_delay)

        # DOES THIS NEED TRIG DELAY
        # if self.prep_tek2:tek2_marker_dsp =  self.generateadout,resample=True,trig_delay=self.trig_delay)
        #             card_trig_arr = self.generatemarkers(markers_readout, emarkers(tek2_marker,resample=True,trig_delay=0.0)
        card_trig_arr = self.generatemarkers(markers_readout, resample=True)

        if self.prep_cavity_drive:cavity_marker_dsp = self.generatemarkers(cavity_I,resample=True,conv_width=self.lo_delay,trig_delay=self.trig_delay)
            # trig_arr_awg = self.generatemastertrigger(len(readout_marker_dsp[0]),2*self.trig_pulse_length,self.abs_trig_delay)
            # triggers other cards. gets loaded to trig channels that don't go to digitizer
            #commenting out here too bc even here we don't trigger non digitizer cards thru the pxi

        for i in tqdm(range(len(waveforms_I))):
            # removed TQDM here too -- wrote it down but may want to re-insert if you want a progress bar ever again
            # reinserted by MGP December 11, 2019, just external tqdm parenthetical
            # print("we are running tqdm loop in loadandqueuewaveforms")
            # print("the tqdm ranges over")
            # print(range(len(waveforms_I)))

            # recall tek2 trigger delay is 0 in hardware config

            wave_I = key.Waveform(np.array(waveforms_I[i]),
                                      append_zero=True)  # Have to include append_zero or the triggers get messed up!
            wave_Q = key.Waveform(waveforms_Q[i], append_zero=True)
            m_readout = key.Waveform(readout_marker[i], append_zero=True)
            m_qubit = key.Waveform(qubit_marker[i], append_zero=True)  ### this qubit marker is wrong - Vatsan

            m_readout_dsp = key.Waveform(readout_marker_dsp[i], append_zero=True)
            m_qubit_dsp = key.Waveform(qubit_marker_dsp[i], append_zero=True)
            # if self.prep_tek2: m_tek2_dsp = key.Waveform(tek2_marker_dsp[i], append_zero=True)

            if self.prep_cavity_drive:
                wave_cavity_I = key.Waveform(np.array(cavity_I[i]),
                                                 append_zero=True)  # Have to include append_zero or the triggers get messed up!
                wave_cavity_Q = key.Waveform(cavity_Q[i], append_zero=True)
                m_cavity_dsp = key.Waveform(cavity_marker_dsp[i], append_zero=True)

            #  even if running old version we don't have a trigger card now??
            # trig = key.Waveform(trig_arr_awg, append_zero=True)
            card_trig = key.Waveform(card_trig_arr[i], append_zero=True)

            # Load objects to the modules
            # Need to decide whether doing qubit A or B!!
            # Associate 1 with A, 2 with B
            if self.on_qubits[0] == "1":

                wave_I.loadToModule(AWG_A_module)
                wave_Q.loadToModule(AWG_A_module)
                if self.prep_cavity_drive:
                    wave_cavity_I.loadToModule(AWG_A_module)
                    wave_cavity_Q.loadToModule(AWG_A_module)

            # remove this:
            #     gg = 140
            #     ff = 16
            # Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.
                wave_I.queue(self.AWG_A_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                     delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay,
                                     cycles=1, prescaler=0)
                wave_Q.queue(self.AWG_A_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                     delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay,
                                     cycles=1, prescaler=0)
                if self.prep_cavity_drive:
                    wave_cavity_I.queue(self.AWG_A_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                                delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)
                    wave_cavity_Q.queue(self.AWG_A_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                                delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay, cycles=1, prescaler=0)
                self.AWG_A_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                         value=1,
                                                         syncMode=1, length=10, delay=0)

                # Load and queue marker pulses

                m_qubit_dsp.loadToModule(m_8_module)
                m_readout_dsp.loadToModule(m_8_module)
                # if self.prep_tek2: m_tek2_dsp.loadToModule(m_module)

                m_qubit_dsp.queue(self.m_8_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay,
                                  cycles=1, prescaler=0)
                m_readout_dsp.queue(self.m_8_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.readout_marker_pulse_delay+ self.marker_pulse_delay, cycles=1, prescaler=0)

                self.m_8_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                   syncMode=1, length=10, delay=0)

                # trig.loadToModule(trig_module)
                if self.prep_cavity_drive: m_cavity_dsp.loadToModule(m_9_module)
                card_trig.loadToModule(m_9_module)
                if self.prep_cavity_drive: m_cavity_dsp.queue(self.m_9_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                                                  delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg +  self.marker_pulse_delay, cycles=1, prescaler=0)
                card_trig.queue(self.m_9_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                delay=int(self.card_delay / 100) + self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.marker_pulse_delay + self.readout_marker_pulse_delay + self.tentative_card_9_delay, cycles=1, prescaler=0)

                self.m_9_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                         value=1,
                                                         syncMode=1, length=10, delay=0)
                    # self.trig_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                    #                                       syncMode=1, length=10, delay=0)
                    # print("got to end of loadandqueue", i)

            elif self.on_qubits[0] == "2":

                wave_I.loadToModule(AWG_B_module)
                wave_Q.loadToModule(AWG_B_module)
                if self.prep_cavity_drive:
                    wave_cavity_I.loadToModule(AWG_B_module)
                    wave_cavity_Q.loadToModule(AWG_B_module)

                # Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.
                wave_I.queue(self.AWG_B_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                 delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay + self.bob_delay_rel_alice,
                                 cycles=1, prescaler=0)
                wave_Q.queue(self.AWG_B_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                 delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay + self.bob_delay_rel_alice,
                                 cycles=1, prescaler=0)
                if self.prep_cavity_drive:
                    wave_cavity_I.queue(self.AWG_B_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                            delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay , cycles=1, prescaler=0)
                    wave_cavity_Q.queue(self.AWG_B_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                            delay=self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.iq_line_delay , cycles=1, prescaler=0)
                self.AWG_B_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                           value=1,
                                                           syncMode=1, length=10, delay=0)

                # Load and queue marker pulses

                m_qubit_dsp.loadToModule(m_8_module)
                m_readout_dsp.loadToModule(m_8_module)
                 # if self.prep_tek2: m_tek2_dsp.loadToModule(m_module)


                m_qubit_dsp.queue(self.m_8_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                      delay=self.tek2_trigger_delay+ self.global_trig_delay_fromawg +self.marker_pulse_delay ,
                                      cycles=1, prescaler=0)
                m_readout_dsp.queue(self.m_8_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                        delay=self.tek2_trigger_delay+ self.global_trig_delay_fromawg + self.readout_marker_pulse_delay+ self.marker_pulse_delay , cycles=1, prescaler=0)


                self.m_8_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                         value=1,
                                                         syncMode=1, length=10, delay=0)

                # trig.loadToModule(trig_module)
                if self.prep_cavity_drive: m_cavity_dsp.loadToModule(m_9_module)
                card_trig.loadToModule(m_9_module)
                if self.prep_cavity_drive: m_cavity_dsp.queue(self.m_9_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                                                  delay=self.tek2_trigger_delay+ self.global_trig_delay_fromawg +self.marker_pulse_delay, cycles=1, prescaler=0)
                card_trig.queue(self.m_9_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,
                                    delay=int(self.card_delay / 100) + self.tek2_trigger_delay + self.global_trig_delay_fromawg + self.readout_marker_pulse_delay+ self.marker_pulse_delay , cycles=1, prescaler=0)

                self.m_9_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0,
                                                         value=1,
                                                         syncMode=1, length=10, delay=0)
                # self.trig_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                #                                       syncMode=1, length=10, delay=0)
                # print("got to end of loadandqueue", i)

            else:
                print("You have a problem with qubit labeling in expt config.")
                pass

# A bunch of these functions below are not used by our code as of April 2021. We will put the important ones at the top
# and demarcate the ones we didn't bother to modify below. --MGP

    def run(self):
        print("Experiment starting. Expected time = ", self.totaltime, "mins")
        try:
            # Start all the channels on the AWG and digitizer modules.
            #print ("Number of experiments = ",self.num_expt)



            if self.on_qubits[0] == "1":
                self.DIG_ch_1.clear()
                self.DIG_ch_1.start()
                self.DIG_ch_2.clear()
                self.DIG_ch_2.start()
                # print("cleared digitizer channels 1 and 2 and started")
            elif self.on_qubits[0] == "2":
                self.DIG_ch_3.clear()
                self.DIG_ch_3.start()
                self.DIG_ch_4.clear()
                self.DIG_ch_4.start()
            else:
                print("Error with qubit options as called in keysight pxi load")

            self.AWG_A_module.startAll()
            self.AWG_B_module.startAll()
            self.m_8_module.startAll()
            self.m_9_module.startAll()
            # self.trig_module.startAll()
            #print("started all modules")

        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self,expt_num):
        # not sure this saves anything? A check fn

        for sweep_ct in tqdm(range(self.num_avg)):


            if self.on_qubits[0] == "1":
                self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
                self.data_2 += np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)
            elif self.on_qubits[0] == "2":
                self.data_3 += np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape)
                self.data_4 += np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape)
            else:
                print("Error with qubit options as called in keysight pxi load")



        if self.on_qubits[0] == "1":
            self.data_1 /= self.num_avg
            self.data_2 /= self.num_avg
        elif self.on_qubits[0] == "2":
            self.data_3 /= self.num_avg
            self.data_4 /= self.num_avg
        else:
            print("Error with qubit options as called in keysight pxi load")


        if self.on_qubits[0] == "1":
            print("Processed data shape", np.shape(self.data_1))
        elif self.on_qubits[0] == "2":
            print("Processed data shape", np.shape(self.data_3))
        else:
            print("Error with qubit options as called in keysight pxi load")


        if self.on_qubits[0] == "1":
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(131, title='I')
            plt.imshow(self.data_1, aspect='auto')
            ax.set_xlabel('Digitizer bins')
            ax.set_ylabel('Experiment number')
            ax2 = fig.add_subplot(132, title='Q')
            plt.imshow(self.data_2, aspect='auto')
            ax2.set_xlabel('Digitizer bins')
            ax2.set_ylabel('Experiment number')
            ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_1[expt_num])
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_2[expt_num])
            ax3.axvspan(self.readout_A_window[0], self.readout_A_window[1], alpha=0.2, color='b')
            ax3.set_xlabel('Time (ns)')
            ax3.set_ylabel('Signal')
            fig.tight_layout()
            plt.show()
        elif self.on_qubits[0] == "2":
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(131, title='I')
            plt.imshow(self.data_3, aspect='auto')
            ax.set_xlabel('Digitizer bins')
            ax.set_ylabel('Experiment number')
            ax2 = fig.add_subplot(132, title='Q')
            plt.imshow(self.data_4, aspect='auto')
            ax2.set_xlabel('Digitizer bins')
            ax2.set_ylabel('Experiment number')
            ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_3[expt_num])
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_4[expt_num])
            ax3.axvspan(self.readout_B_window[0], self.readout_B_window[1], alpha=0.2, color='b')
            ax3.set_xlabel('Time (ns)')
            ax3.set_ylabel('Signal')
            fig.tight_layout()
            plt.show()
        else:
            print("Error with qubit options as called in keysight pxi load")

        # print("The digitzer bins were individually averaged for testing synchronization.")


    def SSdata_many(self,wA =[0,-1], wB =[0,-1]):
        data = []
        if self.on_qubits[0] == "1":
            I = []
            Q = []
            for ii in tqdm(range(self.num_avg)):
                I.append(
                        np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])], 0))
                Q.append(
                        np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])], 0))
            IA = np.array(I).T
            QA = np.array(Q).T
            data.append([IA, QA])
        elif self.on_qubits[0] == "2":
            I = []
            Q = []
            for ii in tqdm(range(self.num_avg)):
                I.append(
                        np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])], 0))
                Q.append(
                        np.mean(np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])], 0))
            IB =np.array(I).T
            QB = np.array(Q).T
            data.append([IB, QB])
        else:
            print("You have a problem with the on_qubits option")
        return np.asarray(data)


    def traj_data_many(self, wA=[0, -1], wB=[0, -1] ):
        data = []
        if "1" in self.on_qubits:
            IA = []
            QA = []
            for ii in tqdm(range(self.num_avg)):
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])].T
                IA.append(ch1)
                QA.append(ch2)
            data.append([IA, QA])
        elif "2" in self.on_qubits:
            IB = []
            QB = []
            for ii in tqdm(range(self.num_avg)):
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])].T
                IB.append(ch3)
                QB.append(ch4)
            data.append([IB, QB])
        else:
            print("you have a problem with the on_qubits option or keysight_pxi_load class choices")

        return np.asarray(data)

    def traj_data_many_nowindow(self):
        data = []
        # just default to the most permissive window conditions by force
        wA = [0, -1]
        wB = [0, -1]
        if "1" in self.on_qubits:
            IA = []
            QA = []
            for ii in tqdm(range(self.num_avg)):
                ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(wA[0]):int(wA[1])].T
                ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(wA[0]):int(wA[1])].T
                IA.append(ch1)
                QA.append(ch2)
            data.append([IA, QA])
        elif "2" in self.on_qubits:
            IB = []
            QB = []
            for ii in tqdm(range(self.num_avg)):
                ch3 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_3.shape).T[int(wB[0]):int(wB[1])].T
                ch4 = np.reshape(self.DIG_ch_4.readDataQuiet(), self.data_4.shape).T[int(wB[0]):int(wB[1])].T
                IB.append(ch3)
                QB.append(ch4)
            data.append([IB, QB])
        else:
            print("you have a problem with the on_qubits option or keysight_pxi_load class choices")

        return np.asarray(data)


    def acquire_avg_data(self,wA = [0,-1],wB = [0,-1],pi_calibration=False,rotate_iq_A = False,rotate_iq_B = False,phi_A=0,phi_B=0):
        data = []
        # self.I, self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IA, self.QA = np.zeros(self.num_expt), np.zeros(self.num_expt)
        self.IB, self.QB = np.zeros(self.num_expt), np.zeros(self.num_expt)

        if "1" in self.on_qubits:
            if rotate_iq_A:
                print("Rotating IQ A digitally")
                 # If you want to remove tqdm to get at error messages you can do it here, just leave a note for yourself
                for ii in tqdm(range(self.num_avg)):
                    iAd = self.DIG_ch_1.readDataQuiet(timeout=20000)
                    qAd = self.DIG_ch_2.readDataQuiet(timeout=20000)
                    # if you have buffer issues try this verbose version
                    # iAd=self.DIG_ch_1.readData(data_points=self.data_1.shape[0]*self.data_1.shape[1])
                     # qAd = self.DIG_ch_2.readData(data_points=self.data_2.shape[0] * self.data_2.shape[1])

                    IAtemp = np.reshape(iAd, self.data_1.shape).T[int(wA[0]):int(wA[1])]
                    QAtemp = np.reshape(qAd, self.data_2.shape).T[int(wA[0]):int(wA[1])]
                    # We have different rotation angles for qubits A, B
                    IArot = IAtemp * np.cos(phi_A) + QAtemp * np.sin(phi_A)
                    QArot = -IAtemp * np.sin(phi_A) + QAtemp * np.cos(phi_A)

                    self.IA += np.mean(IArot, 0)
                    self.QA += np.mean(QArot, 0)
            else:
                for ii in tqdm(range(self.num_avg)):
                    self.IA += np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(timeout=10000), self.data_1.shape).T[
                                           int(wA[0]):int(wA[1])], 0)
                    self.QA += np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(timeout=10000), self.data_2.shape).T[
                                           int(wA[0]):int(wA[1])], 0)
            IA = self.IA / self.num_avg
            QA = self.QA / self.num_avg
            if pi_calibration:
                IA = (IA[:-2] - IA[-2]) / (IA[-1] - IA[-2])
                QA = (QA[:-2] - QA[-2]) / (QA[-1] - QA[-2])
            data.append([IA, QA])
        elif "2" in self.on_qubits:
            if rotate_iq_B:
                print("Rotating IQ B digitally")
                # If you want to remove tqdm to get at error messages you can do it here, just leave a note for yourself
                for ii in tqdm(range(self.num_avg)):
                    iBd = self.DIG_ch_3.readDataQuiet(timeout=20000)
                    qBd = self.DIG_ch_4.readDataQuiet(timeout=20000)
                    # if you have buffer issues try this verbose version
                    # iBd=self.DIG_ch_3.readData(data_points=self.data_3.shape[0]*self.data_3.shape[1])
                    # qBd = self.DIG_ch_4.readData(data_points=self.data_4.shape[0] * self.data_4.shape[1])

                    IBtemp = np.reshape(iBd, self.data_3.shape).T[int(wB[0]):int(wB[1])]
                    QBtemp = np.reshape(qBd, self.data_4.shape).T[int(wB[0]):int(wB[1])]
                    # Presumably we might need different rotation angles here for A, B
                    # IArot = IAtemp * np.cos(phi) + QAtemp * np.sin(phi)
                    # QArot = -IAtemp * np.sin(phi) + QAtemp * np.cos(phi)
                    IBrot = IBtemp * np.cos(phi_B) + QBtemp * np.sin(phi_B)
                    QBrot = -IBtemp * np.sin(phi_B) + QBtemp * np.cos(phi_B)
                    self.IB += np.mean(IBrot, 0)
                    self.QB += np.mean(QBrot, 0)
            else:
                # REMOVED TQDM HERE
                # Reinserted by MGP December 11, 2019, removed April 1 2020, reinserted September 17, 2020
                for ii in tqdm(range(self.num_avg)):
                    self.IB += np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(timeout=10000), self.data_3.shape).T[
                                           int(wB[0]):int(wB[1])], 0)
                    self.QB += np.mean(np.reshape(self.DIG_ch_4.readDataQuiet(timeout=10000), self.data_4.shape).T[
                                           int(wB[0]):int(wB[1])], 0)
                    print(ii)
            IB = self.IB / self.num_avg
            QB = self.QB / self.num_avg
            if pi_calibration:
                IB = (IB[:-2] - IB[-2]) / (IB[-1] - IB[-2])
                QB = (QB[:-2] - QB[-2]) / (QB[-1] - QB[-2])
            data.append([IB, QB])
        else:
            print("You have a problem with the on_qubits option at acquireavgdata")

        return np.asarray(data)


#=======================================================================================================================
# Below this line are UNMODIFIED and WILL NOT WORK as of 4/2021 -- MGP


    def column_averaged_data(self,w =[0,-1]):


        for sweep_ct in tqdm(range(self.num_avg)):
            #print(sweep_ct)
            self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
            self.data_2 += np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)


        self.data_1 /= self.num_avg
        self.data_2 /= self.num_avg

        print ("Processed data shape",np.shape(self.data_1))
        return np.array(self.data_1), np.array(self.data_2)

    def runacquireandplot(self):

        try:
            print("Experiment starting")
            print("The digitzer bins were individually averaged for testing synchronization.")
            print("Number of experiments = ", self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_2.clear()
            self.DIG_ch_2.start()
            self.AWG_module.startAll()
            self.m_module.startAll()
            self.trig_module.startAll()

            for sweep_ct in tqdm(range(self.num_avg)):
                self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
                self.data_2 += np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)

            self.data_1 /= self.num_avg
            self.data_2 /= self.num_avg

            print("Processed data shape", np.shape(self.data_1))

            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(131, title='I')
            plt.imshow(self.data_1, aspect='auto')
            ax.set_xlabel('Digitizer bins')
            ax.set_ylabel('Experiment number')
            ax2 = fig.add_subplot(132, title='Q')
            plt.imshow(self.data_2, aspect='auto')
            ax2.set_xlabel('Digitizer bins')
            ax2.set_ylabel('Experiment number')
            ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_1[expt_num])
            ax3.plot(np.arange(self.DIG_sampl_record) * self.dt_dig, self.data_2[expt_num])
            ax3.axvspan(self.readout_window[0], self.readout_window[1], alpha=0.2, color='b')
            ax3.set_xlabel('Time (ns)')
            ax3.set_ylabel('Signal')
            fig.tight_layout()
            plt.show()




        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def traj_data_one(self,w = [0,-1]):
        ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[0])].T
        ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[0])].T
        return ch1,ch2



    def SSdata_one(self,w =[0,-1]):
        ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])]
        ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])]
        return np.mean(ch1,0), np.mean(ch2,0)



    def acquire_avg_std_data(self, w=[0, -1], pi_calibration=False):
        self.I,self.Q = [],[]
        for ii in tqdm(range(self.num_avg)):
            self.I.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])], 0))
            self.Q.append(np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])], 0))
        I,Q = mean(self.I),mean(self.Q)
        Ierr,Qerr = std(self.I),std(self.Q)
        if pi_calibration:
            I = (I[:-2] - I[-2]) / (I[-1] - I[-2])
            Q = (Q[:-2] - Q[-2]) / (Q[-1] - Q[-2])
        return I,Q,Ierr,Qerr

def run_keysight(experiment_cfg, hardware_cfg, sequences, name):



    setup = KeysightSingleQubit(experiment_cfg, hardware_cfg, sequences, name)

    try:
        setup.configureChannels(hardware_cfg, experiment_cfg, name)
        print("configured channels")
        setup.loadAndQueueWaveforms(sequences)
        print("loaded and queued waveforms")
        setup.runacquireandplot()
        print("ranacquiredandplotted")

    finally:
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.m_module.stopAll()
        setup.m_module.clearAll()
        setup.trig_module.stopAll()
        setup.trig_module.clearAll()
        setup.chassis.close()


if __name__ == "__main__":
    setup = KeysightSingleQubit()
    try:
        #waveforms_I, waveforms_Q, readout, qubit = generateWaveforms()
        #print (len(waveforms_I))
        waveforms_I = sequences['charge1']
        setup.loadAndQueueWaveforms(self.out_mod_no, waveforms_I, waveforms_I, waveforms_I, waveforms_I)
        setup.run()
        save_path = r"S:\_Data\180828 - Manipulate cavity and 3D readout - Cooldown 2\Jupyter notebooks\keysight_rabi_test"
        plt.plot(setup.data_1[40])
        plt.show()
        #np.save(os.path.join(save_path, "I"), np.array(setup.data_list_I))
        #np.save(os.path.join(save_path, "Q"), np.array(setup.data_list_Q))
    finally:
        setup.AWG_A_module.stopAll()
        setup.AWG_A_module.clearAll()
        setup.AWG_B_module.stopAll()
        setup.AWG_B_module.clearAll()
        setup.m_8_module.stopAll()
        setup.m_8_module.clearAll()
        setup.m_9_module.stopAll()
        setup.m_9_module.clearAll()
        setup.chassis.close()