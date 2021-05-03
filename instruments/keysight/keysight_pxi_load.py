# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:41:56 2018

@author: slab

Module 6 used as AWG:
    Channel 1 is I
    Channel 2 is Q

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
QB_W_FAST_FLUX = [0, 1, 2, 3, 4, 5, 6, 7] #qubits where we will load fast flux waveforms. Note: hardcoded that Q0-Q3 goes to module ff1, Q4-Q7 goes to module ff2

class KeysightSingleQubit:
    '''Class designed to implement a simple single qubit experiment given pulse sequences from the Sequencer class. Does  not use
    HVI technology.
=
    Module (slot) 5 is an AWG, used for fast flux pulses
    Module 6 is an AWG, used for fast flux pulses
    Module 7 is an AWG. channel 1 goes to the I input to the mixer, channel 2 goes to the Q input
    Module 8 is used as a marker for the LO (ie send signal to switch).
    Module 9 is triggered externally and its outputs are used to trigger the rest of the modules.
        ch4 of this trig for the digitizer.

    Module 10 is the digitizer. channel 1 is for readout of I component and channel 2 is for readout from Q component.'''

    ## Outstanding issue - if we upload waveforms that contain all 0's in time for unused space - how are "markers" used at all???
    ## Answer - markers window the region where pulse sent out to bin output of LO!!!

    ## LO             _/-\_/-\_/-\_/-\_/-\   # LO always on
    ## Marker         _______|------|______  # marker windows LO output in time w/ AWG signals non-zero state (+ wiggle room!)
    ## Trigger        |--|_________________  # Trigger turns on AWG/DIG cards
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
        self.dt_M3201A = hardware_cfg['awg_info']['keysight_pxi']['dt_M3201A']
        self.adc_range =  hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']

        self.readout_window = np.array(quantum_device_cfg['readout']['window'])
        self.lo_delay = hardware_cfg['awg_info']['keysight_pxi']['lo_delay'] #lo_delay convolves LO marker so that marker is bigger to account for switching time delay (ie stretches marker)
        self.qb_lo_delay = hardware_cfg['awg_info']['keysight_pxi']['qb_lo_delay'] #delay to trigger to QB card/module
        self.abs_trig_delay = hardware_cfg['awg_info']['keysight_pxi']['abs_trig_delay']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']
        self.trig_delay = hardware_cfg['awg_info']['keysight_pxi']['m3201a_trig_delay'] #global delay that moves qubit + readout LO around ##TODO check this I think it also moves dig
        self.card_delay = hardware_cfg['awg_info']['keysight_pxi']['m3102a_card_delay'] #-> delay that affects digitizer marker
        self.ff1_card_delay = hardware_cfg['awg_info']['keysight_pxi']['ff1_card_delay']
        self.ff2_card_delay = hardware_cfg['awg_info']['keysight_pxi']['ff2_card_delay']


        print ("Module used for generating Q1 IQ  pulses = ",self.AWG_mod_no)
        print ("Module used for generating digital markers for LO = ",self.marker_mod_no)
        print ("Module used to trigger dig and for stabilizer  = ", self.stab_mod_no)
        print("Module used for generating fast flux pluses for Q0-Q3 = ", self.ff1_mod_no)
        print("Module used for generating fast flux pluses for Q4-Q7 = ", self.ff2_mod_no)
        self.out_mod_nums = [self.AWG_mod_no, self.marker_mod_no, self.stab_mod_no, self.ff1_mod_no, self.ff2_mod_no]

        self.num_avg = experiment_cfg[name]['acquisition_num']
        self.num_expt = sequences['readout'].shape[0]
        self.trigger_period = self.hardware_cfg['trigger']['period_us']

        self.totaltime = self.num_avg*self.num_expt*self.trigger_period*1e-6/60.0

        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']

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
        self.DIG_module = chassis.getModule(10)

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
        self.DIG_ch_1 = chassis.getChannel(10, 1)
        self.DIG_ch_2 = chassis.getChannel(10, 2)
        self.DIG_ch_3 = chassis.getChannel(10,3)
        self.DIG_ch_4 = chassis.getChannel(10, 4)
        

        self.data_1,self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)),np.zeros((self.num_expt, self.DIG_sampl_record))

        self.I,self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally


    def configureChannels(self, hardware_cfg, experiment_cfg, quantum_device_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in Vpp.'''

        amp_AWG = hardware_cfg['awg_info']['keysight_pxi']['amplitudes']
        amp_mark = hardware_cfg['awg_info']['keysight_pxi']['amp_mark']
        amp_stab = hardware_cfg['awg_info']['keysight_pxi']['amp_stab']
        amp_digtzr_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_digtzr_trig']
        amp_ff1 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff1']
        amp_ff2 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff2']
        IQ_dc_offset = quantum_device_cfg['pulse_info']['1']['IQ_dc']


        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        ## Output modules: Configure amplitude, dc offset, and if "trigger" channel on a module is used as a receiver or a trigger source
        # ie triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN) means it's receiving a trigger
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

        print ("Configuring qubit IQ channels")
        self.AWG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.AWG_ch_1.configure(amplitude=amp_AWG[0], offset_voltage = IQ_dc_offset)
        self.AWG_ch_2.configure(amplitude=amp_AWG[1], offset_voltage = IQ_dc_offset)
        self.AWG_ch_3.configure(amplitude=amp_AWG[2], offset_voltage = IQ_dc_offset)
        self.AWG_ch_4.configure(amplitude=amp_AWG[3], offset_voltage = IQ_dc_offset)

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
        self.digtzr_trig_ch.configure(amplitude=amp_digtzr_trig[0])
        print ("Dig card trigger amplitude = ",amp_digtzr_trig[0])


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
        self.DIG_ch_1.configure(full_scale=self.adc_range,points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale=self.adc_range,points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale=self.adc_range, points_per_cycle=self.DIG_sampl_record,
                                cycles=num_expt * num_avg, buffer_time_out=100000,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)

    def generatemarkers(self,wvs, channel, dt_src=1, dt_mark = 1, conv_width = 1,trig_delay = 0):
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

    def sequenceslist(self,sequences,waveform_channels):
        """takes the sequences arrays produced in "pulse squences" class and stored in "sequences" dict and renames
        the to  a dictionary called "wv." also makes sure that if a given sequence doesn't include a channel listed in waveform_channels, it gets listed in wv and
        filled with zeros"""
        wv = {}
        for channel in waveform_channels:
            if not sequences[channel] == None:
                wv[channel] = sequences[channel]
            else:
                wv[channel] = np.zeros_like(sequences[waveform_channels[0]])
        return wv
    
    def get_dt_channel(self, channel_name):
        if self.hardware_cfg['channels_awg'][channel_name] == "keysight_pxi_M3201A":
            dt = self.dt_M3201A
        else:
            dt = self.hardware_cfg['awg_info'][['channels_awg'][channel_name]]['dt']
            
        return dt
        


    def loadAndQueueWaveforms(self, sequences):
        '''Loads the provided waveforms from a pulse sequence to the appropriate modules.

        Note that all waveforms should consist of values from -1 to 1 (inclusive) only. Amplitude is set in the configureChannels() method.
        If you accidentally pass in an array containing larger values, the code raises a KeysightError: Invalid Waveform.

        Pxi Waveform Channels:
            wv["charge1_I"]: List of numpy arrays representing waveforms (or a 2D array) for the "I" input AWG to the mixer,
                1 for each unique trial/experiment. Should be the same number of waveforms as num_experiments in the __init__ method.
                The "inner" array is a waveform, and the "outer" dimension corresponds to the experiment/trial.
                The QB lo marker is generated using the known placement of IQ pulses.
            wv["charge1_Q"]: Same for the "Q" channel.
            readout: Readout waveform used to trigger the readout LO
            wv["digtzr_trig"]: Trigger for the digitizer

            The master trigger for all the cards is generated knowing the length of the AWG waveforms using self.generate_trigger
            '''


        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        num_expt = sequences['readout'].shape[0]
        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences,pxi_waveform_channels)

        AWG_module = self.chassis.getModule(self.AWG_mod_no)
        m_module = self.chassis.getModule(self.marker_mod_no)
        stab_module = self.chassis.getModule(self.stab_mod_no)
        ff1_module = self.chassis.getModule(self.ff1_mod_no)
        ff2_module = self.chassis.getModule(self.ff2_mod_no)

        print ("shape of waveform I",np.shape(wv["charge1_I"]))

        #make sure all waveforms are the same length 
        for i in range(1,len(wv.keys())):
            ch1 = wv[wv.keys()[i-1]]
            dt_ch1 = self.get_dt_channel(wv.keys()[i-1])
            ch2 = wv[wv.keys()[i]]
            dt_ch2 = self.get_dt_channel(wv.keys()[i])
            if len(ch1)*dt_ch1 != len(ch2)*dt_ch2:
                raise TypeError("Not all waveform lists are the same length")

        self.AWG_module.clearAll()
        self.m_module.clearAll()
        self.stab_module.clearAll()
        self.ff1_module.clearAll()
        self.ff2_module.clearAll()

        #wv["digtzr_trig"] -> trigger for digitizer card coming from stab card
        #readout_markers -> trigger for readout LO coming from marker card

        key.Waveform._waveform_number_counter = 0

        # Marker array for qb LO, generated from wavefroms_I
        qubit_marker = self.generatemarkers(wv, "charge1_I", dt_mark=self.dt_m,
                                                conv_width=self.lo_delay,trig_delay=self.trig_delay)

        # marker array for readout LO
        readout_marker = self.generatemarkers(wv, "readout", dt_mark = self.dt_m, trig_delay=self.trig_delay)

        # marker array for stabilizer LO
        stabilizer_marker = self.generatemarkers(wv, "stab_I", dt_mark=self.dt_m, trig_delay=self.trig_delay)

        #marker array for waveform wv["digtzr_trig"], waveform for triggering digitizer card, already resampled
        digtzr_trig = self.generatemarkers(wv, "digtzr_trig", dt_mark=self.dt_m)

        for i in tqdm(range(len(wv["charge1_I"]))):
            ## making PXI savvy waveform objects out of the arrays to send out to the PXI
            PXIwave_I = key.Waveform(np.array(wv["charge1_I"][i]),append_zero=True)  # Have to include append_zero or the triggers get messed up!
            PXIwave_Q = key.Waveform(wv["charge1_Q"][i], append_zero=True)
            PXIwave_stab_I = key.Waveform(np.array(wv["stab_I"][i]),
                                     append_zero=True)  # Have to include append_zero or the triggers get messed up!
            PXIwave_stab_Q = key.Waveform(wv["stab_Q"][i], append_zero=True)
            PXIwave_readout_marker = key.Waveform(readout_marker[i], append_zero=True)
            PXIwave_qubit_marker = key.Waveform(qubit_marker[i], append_zero=True)  ## this qubit marker is wrong - Vatsan | Brendan: In what way?
            PXIwave_stabilizer_marker = key.Waveform(stabilizer_marker[i], append_zero=True)
            PXIwave_digtzr_trig = key.Waveform(digtzr_trig[i], append_zero=True)
            PXIwave_ff_Q0 = key.Waveform(wv["ff_Q0"][i], append_zero=True)
            PXIwave_ff_Q1 = key.Waveform(wv["ff_Q1"][i], append_zero=True)
            PXIwave_ff_Q2 = key.Waveform(wv["ff_Q2"][i], append_zero=True)
            PXIwave_ff_Q3 = key.Waveform(wv["ff_Q3"][i], append_zero=True)
            PXIwave_ff_Q4 = key.Waveform(wv["ff_Q4"][i], append_zero=True)
            PXIwave_ff_Q5 = key.Waveform(wv["ff_Q5"][i], append_zero=True)
            PXIwave_ff_Q6 = key.Waveform(wv["ff_Q6"][i], append_zero=True)
            PXIwave_ff_Q7 = key.Waveform(wv["ff_Q7"][i], append_zero=True)

            #####QB AWG#####
            #Send I,Q, qubit drive waveforms to AWG drive card
            PXIwave_I.loadToModule(AWG_module)
            PXIwave_Q.loadToModule(AWG_module)

            # Queue I,Q qubit drive waveforms on AWG card. Want to set trigger mode to SWHVITRIG to trigger from computer.
            PXIwave_I.queue(self.AWG_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay = self.tek2_trigger_delay, cycles = 1, prescaler = 0)
            PXIwave_Q.queue(self.AWG_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay = self.tek2_trigger_delay, cycles = 1, prescaler = 0)


            self.AWG_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #####Marker#####
            #Send marker waveforms to Marker card
            PXIwave_qubit_marker.loadToModule(m_module)
            PXIwave_readout_marker.loadToModule(m_module)

            # if self.prep_tek2 == True:
            #         m_tek2_dsp.loadToModule(m_module)

            # Queue marker waveforms to marker card channels
            PXIwave_qubit_marker.queue(self.m_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)
            PXIwave_readout_marker.queue(self.m_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)

            # Configure marker card settings
            self.m_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #####TRIGGER#####
            # Load trigger waveforms to stabilizer card
            PXIwave_stab_I.loadToModule(stab_module)
            PXIwave_stab_Q.loadToModule(stab_module)
            PXIwave_stabilizer_marker.loadToModule(stab_module)
            PXIwave_digtzr_trig.loadToModule(stab_module)

            #Queue trigger waveforms to trigger channels
            PXIwave_stab_I.queue(self.stab_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            PXIwave_stab_Q.queue(self.stab_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.qb_lo_delay,
                                 cycles=1, prescaler=0)
            PXIwave_stabilizer_marker.queue(self.stab_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1,
                                prescaler=0)
            PXIwave_digtzr_trig.queue(self.digtzr_trig_ch, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=int(self.card_delay/100)+self.tek2_trigger_delay, cycles=1, prescaler=0)

            #Configure trigger module settings
            self.stab_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                               syncMode=1, length=10, delay=0)

            #####FAST FLUX MODULE1#####
            # Load FF! waveforms to trigger card
            PXIwave_ff_Q0.loadToModule(ff1_module)
            PXIwave_ff_Q1.loadToModule(ff1_module)
            PXIwave_ff_Q2.loadToModule(ff1_module)
            PXIwave_ff_Q3.loadToModule(ff1_module)

            # Queue ff1 waveforms to ff1 channels
            PXIwave_ff_Q0.queue(self.ff1_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay + self.ff1_card_delay, cycles=1, prescaler=0)
            PXIwave_ff_Q1.queue(self.ff1_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff1_card_delay, cycles=1, prescaler=0)
            PXIwave_ff_Q2.queue(self.ff1_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff1_card_delay,
                                cycles=1, prescaler=0)
            PXIwave_ff_Q3.queue(self.ff1_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff1_card_delay,
                                cycles=1, prescaler=0)

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
            PXIwave_ff_Q4.queue(self.ff2_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff2_card_delay,
                                cycles=1, prescaler=0)
            PXIwave_ff_Q5.queue(self.ff2_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff2_card_delay,
                                cycles=1, prescaler=0)
            PXIwave_ff_Q6.queue(self.ff2_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff2_card_delay,
                                cycles=1, prescaler=0)
            PXIwave_ff_Q7.queue(self.ff2_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay+ self.ff2_card_delay,
                                cycles=1, prescaler=0)

            # Configure ff1 module settings
            self.ff2_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
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
            self.m_module.startAll()
            self.stab_module.startAll()
            self.ff1_module.startAll()
            self.ff2_module.startAll()


        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self,expt_num):

        for sweep_ct in tqdm(range(self.num_avg)):
            # pass
            self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
            self.data_2 += np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape)

        self.data_1 /= self.num_avg
        self.data_2 /= self.num_avg

        print ("Processed data shape",np.shape(self.data_1))

        fig = plt.figure(figsize = (12,4))
        ax = fig.add_subplot(131,title = 'I')
        plt.imshow(self.data_1, aspect='auto')
        ax.set_xlabel('Digitizer bins')
        ax.set_ylabel('Experiment number')
        ax2 = fig.add_subplot(132,title = 'Q')
        plt.imshow(self.data_2, aspect='auto')
        ax2.set_xlabel('Digitizer bins')
        ax2.set_ylabel('Experiment number')
        ax3 = fig.add_subplot(133,title = 'Expt num = ' + str(expt_num))
        ax3.plot(np.arange(self.DIG_sampl_record)*self.dt_dig, self.data_1[expt_num])
        ax3.plot(np.arange(self.DIG_sampl_record)*self.dt_dig, self.data_2[expt_num])
        ax3.axvspan(self.readout_window[0], self.readout_window[1], alpha=0.2, color='b')
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Signal')
        fig.tight_layout()
        plt.show()

        print("The digitzer bins were individually averaged for testing synchronization.")

    def runacquireandplot(self):

        try:
            print("Experiment starting")
            print("The digitzer bins were individually averaged for testing synchronization.")
            print("Number of experiments = ", self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_2.clear()
            self.DIG_ch_2.start()
            self.DIG_ch_3.clear()
            self.DIG_ch_3.start()
            self.DIG_ch_4.clear()
            self.DIG_ch_4.start()
            self.AWG_module.startAll()
            self.m_module.startAll()
            self.stab_module.startAll()

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

    def traj_data_many(self,w = [0,-1]):
        I = []
        Q = []
        for ii in tqdm(range(self.num_avg)):
            ch1= np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])].T
            ch2= np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])].T
            I.append(ch1)
            Q.append(ch2)
        return np.array(I),np.array(Q)

    def SSdata_one(self,w =[0,-1]):
        ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])]
        ch2 = np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])]
        return np.mean(ch1,0), np.mean(ch2,0)

    def SSdata_many(self,w =[0,-1]):
        I = []
        Q = []
        for ii in tqdm(range(self.num_avg)):
            I.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])], 0))
            Q.append(np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])], 0))
        return np.array(I).T, np.array(Q).T

    def acquire_avg_data(self,w = [0,-1],pi_calibration=False):
        for ii in tqdm(range(self.num_avg)):

            self.I += np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(),self.data_1.shape).T[int(w[0]):int(w[1])],0)
            self.Q += np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(),self.data_2.shape).T[int(w[0]):int(w[1])],0)


        I = self.I/self.num_avg
        Q = self.Q/self.num_avg
        if pi_calibration:
            I = (I[:-2]-I[-2])/(I[-1]-I[-2])
            Q = (Q[:-2]-Q[-2])/(Q[-1]-Q[-2])
        return I,Q


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
        setup.loadAndQueueWaveforms(sequences)
        setup.runacquireandplot()

    finally:
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.m_module.stopAll()
        setup.m_module.clearAll()
        setup.stab_module.stopAll()
        setup.stab_module.clearAll()
        setup.chassis.close()


if __name__ == "__main__":
    setup = KeysightSingleQubit()
    try:
        #wv["charge1_I"], wv["charge1_Q"], readout, qubit = generateWaveforms()
        #print (len(wv["charge1_I"]))
        wv["charge1_I"] = sequences['charge1']
        setup.loadAndQueueWaveforms(self.AWG_mod_no, wv["charge1_I"], wv["charge1_I"], wv["charge1_I"], wv["charge1_I"])
        setup.run()
        save_path = r"S:\_Data\180828 - Manipulate cavity and 3D readout - Cooldown 2\Jupyter notebooks\keysight_rabi_test"
        plt.plot(setup.data_1[40])
        plt.show()
        #np.save(os.path.join(save_path, "I"), np.array(setup.data_list_I))
        #np.save(os.path.join(save_path, "Q"), np.array(setup.data_list_Q))
    finally:
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.m_module.stopAll()
        setup.m_module.clearAll()
        setup.chassis.close()