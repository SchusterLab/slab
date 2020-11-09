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

class KeysightSingleQubit:
    '''Class designed to implement a simple single qubit experiment given pulse sequences from the Sequencer class. Does  not use
    HVI technology.

    Module (slot) 6 and 7 are used for AWGS. Module 8 is used as a marker. Module 9 is triggered externally and its outputs are used
    to trigger the rest of the modules. ch4 of this trig for the digitizer. Digitizer is module 10.

    On Module 6, channel 1 goes to the I input to the mixer, channel 2 goes to the Q input, channel 3 is the readout pulse, and
    channel 4 is the readout marker. On module 10, channel 1 is for readout of I component and channel 2 is for readout from Q component.'''

    ## Outstanding issue - if we upload waveforms that contain all 0's in time for unused space - how are "markers" used at all???
    ## Answer - markers window the region where pulse sent out to bin output of LO!!!

    ## LO             _/-\_/-\_/-\_/-\_/-\   # LO always on
    ## Marker         _______|------|______  # marker windows LO output in time w/ AWG signals non-zero state (+ wiggle room!)
    ## Trigger        |--|_________________  # Trigger turns on AWG/DIG cards
    ## AWG Signal     _________/??\________  # AWG card outputs IQ signal to mixer, mixer input is windows LO+marker!


    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, sequences, name, save_path=r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials=50 * 1000):  # 1000*10000 if you want to watch sweep by eye

        chassis = key.KeysightChassis(1,
                                      {6: key.ModuleType.OUTPUT,
                                       7: key.ModuleType.OUTPUT,
                                       8: key.ModuleType.OUTPUT,
                                       9: key.ModuleType.OUTPUT,
                                       10: key.ModuleType.INPUT})

        self.hardware_cfg = hardware_cfg
        ## out_mod_no is AWG outupt?
        self.out_mod_no = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no']
        ## what does marker do - windows LO
        self.marker_mod_no = hardware_cfg['awg_info']['keysight_pxi']['marker_mod_no']
        ## what does trigger do - triggers AWG / Digitizer
        self.trig_mod_no = hardware_cfg['awg_info']['keysight_pxi']['trig_mod_no']

        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        self.readout_window = np.array(quantum_device_cfg['readout']['window'])
        self.lo_delay = hardware_cfg['awg_info']['keysight_pxi']['lo_delay']
        self.qb_lo_delay = hardware_cfg['awg_info']['keysight_pxi']['qb_lo_delay']
        self.abs_trig_delay = hardware_cfg['awg_info']['keysight_pxi']['abs_trig_delay']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']
        self.trig_delay = hardware_cfg['awg_info']['keysight_pxi']['m3201a_trig_delay']
        self.card_delay = hardware_cfg['awg_info']['keysight_pxi']['m3102a_card_delay']
        self.adc_range =  hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']



        print ("Module used for generating analog pulses = ",self.out_mod_no)
        print ("Module used for generating digital markers = ",self.marker_mod_no)
        print ("Module used for temp generation of module = ", self.trig_mod_no) # wtf
        self.out_mod_nums = [self.out_mod_no, self.marker_mod_no, self.trig_mod_no]

        self.num_avg = experiment_cfg[name]['acquisition_num']
        self.num_expt = sequences['readout'].shape[0]
        self.trigger_period = self.hardware_cfg['trigger']['period_us']

        self.totaltime = self.num_avg*self.num_expt*self.trigger_period*1e-6/60.0

        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']


        if 'sideband' in name:
            self.prep_tek2 = True
            self.tek2_trigger_delay = int(hardware_cfg['awg_info']['tek70001a']['trig_delay']/100.0)
        else:
            self.prep_tek2=False
            self.tek2_trigger_delay=0


        self.chassis = chassis
        self.awg_channels = range(1,5)
        self.dig_channels = range(1,3)


        # Initialize Module 6 = Analog card.  Ch1 = AWG I, CH2 = AWG Q

        # self.AWG_chs = [chassis.getChannel(self.out_mod_no, ch) for ch in self.awg_channels]
        self.AWG_ch_1 = chassis.getChannel(self.out_mod_no, 1)
        self.AWG_ch_2 = chassis.getChannel(self.out_mod_no, 2)
        self.AWG_ch_3 = chassis.getChannel(self.out_mod_no, 3)
        self.AWG_ch_4 = chassis.getChannel(self.out_mod_no, 4)

        # Initialize Module 8 = Marker card. Digital markers for qubit, readout
        # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]

        self.m_ch_1 = chassis.getChannel(self.marker_mod_no, 1)
        self.m_ch_2 = chassis.getChannel(self.marker_mod_no, 2)
        self.m_ch_3 = chassis.getChannel(self.marker_mod_no, 3)
        self.m_ch_4 = chassis.getChannel(self.marker_mod_no, 4)

        # Initialize card that generates Triggers
        # self.trig_chs = [chassis.getChannel(self.trig_mod_no, ch) for ch in self.awg_channels]
        self.trig_ch_1 = chassis.getChannel(self.trig_mod_no, 1)
        self.trig_ch_2 = chassis.getChannel(self.trig_mod_no, 2)
        self.trig_ch_3 = chassis.getChannel(self.trig_mod_no, 3)
        self.trig_ch_4 = chassis.getChannel(self.trig_mod_no, 4)

        # self.out_modules = [chassis.getModule(num) for num in self.out_mod_nums]
        self.AWG_module = chassis.getModule(self.out_mod_no)
        self.m_module = chassis.getModule(self.marker_mod_no)
        self.trig_module = chassis.getModule(self.trig_mod_no)

        # Initialize digitizer card
        # self.DIG_chs = [chassis.getChannel(10, ch) for ch in self.dig_channels]
        self.DIG_ch_1 = chassis.getChannel(10, 1)
        self.DIG_ch_2 = chassis.getChannel(10, 2)
        self.DIG_ch_3 = chassis.getChannel(10,3)
        self.DIG_module = chassis.getModule(10)

        self.data_1,self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)),np.zeros((self.num_expt, self.DIG_sampl_record))

        self.I,self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally


    def configureChannels(self, hardware_cfg, experiment_cfg, quantum_device_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_AWG = hardware_cfg['awg_info']['keysight_pxi']['amplitudes']
        amp_mark = hardware_cfg['awg_info']['keysight_pxi']['amp_mark']
        amp_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_trig']
        IQ_dc_offset = quantum_device_cfg['pulse_info']['1']['IQ_dc']
        # dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        # dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        num_avg = experiment_cfg[name]['acquisition_num']
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring analog channels")

        self.AWG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        #self.AWG_ch_1.configure(amplitude=amp_AWG[0], trigger_source=4000)
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

        self.trig_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)


        self.trig_ch_1.configure(amplitude=amp_trig[0])
        self.trig_ch_2.configure(amplitude=amp_trig[1])
        self.trig_ch_3.configure(amplitude=amp_trig[2])
        self.trig_ch_4.configure(amplitude=amp_trig[3])
        print ("Card trigger amplitude = ",amp_trig[3])

        #TODO: WHAT IS THAT
        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)

        print ("Setting trigger config for all channels of all modules to External")

        for n in range(1, 5):
            self.AWG_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.trig_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)


        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")

        self.DIG_ch_1.configure(full_scale=self.adc_range,points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG_CYCLE, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale=self.adc_range,points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,use_buffering=True, cycles_per_return=num_expt)
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

    def sequenceslist(self,sequences,waveform_channels):
        wv = []
        for channel in waveform_channels:
            if not channel == None:
                wv.append(sequences[channel])
            else:
                wv.append(np.zeros_like(sequences[waveform_channels[0]]))
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


        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        num_expt = sequences['readout'].shape[0]
        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences,pxi_waveform_channels)

        # Based on Nelson's original nomenclature

        waveforms_I = wv[0]
        waveforms_Q = wv[1]
        readout = wv[2]
        markers_readout = wv[3]
        if self.prep_tek2:tek2_marker = wv[4]

        AWG_module = self.chassis.getModule(self.out_mod_no)
        m_module = self.chassis.getModule(self.marker_mod_no)
        trig_module = self.chassis.getModule(self.trig_mod_no)

        print ("shape of waveform I",np.shape(waveforms_I))

        if len(waveforms_I) != len(waveforms_Q) or len(waveforms_I) != len(markers_readout) or len(waveforms_I) != len(
                readout):
            raise TypeError("Not all waveform lists are the same length")

        self.AWG_module.clearAll()
        self.m_module.clearAll()
        self.trig_module.clearAll()

        #dsp = downsampling
        #markers_readout -> trigger for digitizer card coming from trigger card
        #readout_markers -> trigger for readout LO coming from marker card
        #if marker card is 500MS/s, use dsp version
        #if marker card is 1Gs/s, use non-dsp-ed version


        key.Waveform._waveform_number_counter = 0

        # Marker array for waveform wavefroms_I
        qubit_marker = self.generatemarkers(waveforms_I)
        qubit_marker_dsp = self.generatemarkers(waveforms_I,resample=True,conv_width=self.lo_delay,trig_delay=self.trig_delay)

        # marker array for waveform readout
        readout_marker = self.generatemarkers(readout, resample=False)
        readout_marker_dsp = self.generatemarkers(readout,resample=True,trig_delay=self.trig_delay)

        #marker array for waveform markers_readout, waveform for triggering digitizer card, already resampled
        card_trig_arr = self.generatemarkers(markers_readout,resample=True)

        #if self.prep_tek2:tek2_marker_dsp =  self.generatemarkers(tek2_marker,resample=True,trig_delay=0.0)

        # ?? array made of readout marker (marker of readout waveform) with delays : should be used to trigger awg
        trig_arr_awg = self.generatemastertrigger(len(readout_marker[0]), 2 * self.trig_pulse_length,self.abs_trig_delay)
        trig_arr_awg = self.generatemastertrigger(len(readout_marker_dsp[0]), 2 * self.trig_pulse_length,self.abs_trig_delay)

        for i in tqdm(range(len(waveforms_I))):

            ## Guess: making PXI savvy waveform objects out of the arrays to send out to the PXI
            wave_I = key.Waveform(np.array(waveforms_I[i]),append_zero=True)  # Have to include append_zero or the triggers get messed up!
            wave_Q = key.Waveform(waveforms_Q[i], append_zero=True)
            m_readout = key.Waveform(readout_marker[i], append_zero=True)
            m_qubit = key.Waveform(qubit_marker[i], append_zero=True)  ## this qubit marker is wrong - Vatsan | Brendan: In what way?

            m_readout_dsp = key.Waveform(readout_marker_dsp[i], append_zero=True)
            m_qubit_dsp = key.Waveform(qubit_marker_dsp[i], append_zero=True)
            #if self.prep_tek2:m_tek2_dsp = key.Waveform(tek2_marker_dsp[i], append_zero=True)


            trig = key.Waveform(trig_arr_awg, append_zero=True)
            card_trig = key.Waveform(card_trig_arr[i], append_zero=True)

            #Send I,Q, qubit drive waveforms to AWG drive card
            wave_I.loadToModule(AWG_module)
            wave_Q.loadToModule(AWG_module)

            # Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.
            # Queue I,Q qubit drive waveforms on AWG card
            wave_I.queue(self.AWG_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay = self.tek2_trigger_delay, cycles = 1, prescaler = 0)
            wave_Q.queue(self.AWG_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay = self.tek2_trigger_delay, cycles = 1, prescaler = 0)


            self.AWG_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            #Send marker waveforms to Marker card
            m_qubit_dsp.loadToModule(m_module)
            m_readout_dsp.loadToModule(m_module)

            # if self.prep_tek2 == True:
            #         m_tek2_dsp.loadToModule(m_module)

            # Queue marker waveforms to marker card channels
            m_qubit_dsp.queue(self.m_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)
            m_readout_dsp.queue(self.m_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)

            # if self.prep_tek2 == True:
            #     m_tek2_dsp.queue(self.m_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)

            # Configure marker card settings
            self.m_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            # Load trigger waveforms to trigger card
            trig.loadToModule(trig_module)
            card_trig.loadToModule(trig_module)

            #Queue trigger waveforms to trigger channels
            trig.queue(self.trig_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            trig.queue(self.trig_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.qb_lo_delay, cycles=1, prescaler=0)
            trig.queue(self.trig_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            card_trig.queue(self.trig_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=int(self.card_delay/100)+self.tek2_trigger_delay, cycles=1, prescaler=0)

            #Configure trigger module settings
            self.trig_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                               syncMode=1, length=10, delay=0)

    def run(self):
        print("Experiment starting. Expected time = ", self.totaltime, "mins")
        try:
            # Start all the channels on the AWG and digitizer modules.
            print ("Number of experiments = ",self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_3.clear()
            self.DIG_ch_3.start()
            self.AWG_module.startAll()
            self.m_module.startAll()
            self.trig_module.startAll()

        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self,expt_num):

        for sweep_ct in tqdm(range(self.num_avg)):
            # pass
            self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
            self.data_2 += np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape)

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
            self.DIG_ch_3.clear()
            self.DIG_ch_3.start()
            self.AWG_module.startAll()
            self.m_module.startAll()
            self.trig_module.startAll()

            for sweep_ct in tqdm(range(self.num_avg)):
                self.data_1 += np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape)
                self.data_2 += np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape)

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
        ch2 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[0])].T
        return ch1,ch2

    def traj_data_many(self,w = [0,-1]):
        I = []
        Q = []
        for ii in tqdm(range(self.num_avg)):
            ch1= np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])].T
            ch2= np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])].T
            I.append(ch1)
            Q.append(ch2)
        return np.array(I),np.array(Q)

    def SSdata_one(self,w =[0,-1]):
        ch1 = np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])]
        ch2 = np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])]
        return np.mean(ch1,0), np.mean(ch2,0)

    def SSdata_many(self,w =[0,-1]):
        I = []
        Q = []
        for ii in tqdm(range(self.num_avg)):
            I.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(), self.data_1.shape).T[int(w[0]):int(w[1])], 0))
            Q.append(np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])], 0))
        return np.array(I).T, np.array(Q).T

    def acquire_avg_data(self,w = [0,-1],pi_calibration=False):
        for ii in tqdm(range(self.num_avg)):

            self.I += np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(),self.data_1.shape).T[int(w[0]):int(w[1])],0)
            self.Q += np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(),self.data_2.shape).T[int(w[0]):int(w[1])],0)


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
            self.Q.append(np.mean(np.reshape(self.DIG_ch_3.readDataQuiet(), self.data_2.shape).T[int(w[0]):int(w[1])], 0))
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
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.m_module.stopAll()
        setup.m_module.clearAll()
        setup.chassis.close()