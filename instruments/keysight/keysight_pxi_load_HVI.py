# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 2019

@author: Gabrielle

Module 6 used as AWG for drives/pulses:
    Channel 1 is I qubit
    Channel 2 is Q qubit
    Channel 3 is I cavity
    Channel 4 is Q cavity
    Trigger IO is used for BNC 1
    
Module 7 used as AWG for 
    Channel 1 is 
    Channel 2 is 
    Channel 3 is 
    Channel 4 is 
    Trigger IO is used for BNC 2(cavity)

Module 10 is used for reacout.
    Channel 1 is I
    Channel 2 is Q

This is a rewriting of KeysightSingleQubit to try to incorporate hvi code.
"""

# %pylab inline
from slab.instruments.keysight import KeysightLib as keyLib
from slab.instruments.keysight import keysightSD1 as SD1
from slab.experiments.PulseExperiments_PXI.sequences_pxi_HVI import PulseSequences
#from slab.experiments.HVIExperiments import HVIExpLib as exp
import time
import numpy as np
import os
import json
#import threading
import matplotlib.pyplot as plt
from tqdm import tqdm
from slab.datamanagement import SlabFile


AWG_BANDWIDTH = 10**9 #Samples per second

class KeysightSingleQubit_HVI:
    '''Class designed to implement a simple single qubit experiment given pulse sequences from the Sequencer class. Uses
    HVI technology.

    Module (slot) 6 and 7 are used for AWGS, with Trigger IOs being used for BNCs. Digitizer is module 10.

    On Module 6, channel 1 goes to the I input to the mixer, channel 2 goes to the Q input,
    channel 3 goes to the readout I, and channel 4 is for readout Q.
    The trigger is used for the qubit BNC.
    On Module 7, the trigger is used for the readout BNC.
    On module 10, channel 1 is for readout of I component and channel 2 is for readout from Q component.'''

    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, sequences, name, save_path=r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials=50 * 1000):  # 1000*10000 if you want to watch sweep by eye

        #this is a dictionary, so it's ok if the slots aren't in order
        chassis = keyLib.KeysightChassis(1,
                                      {6: keyLib.ModuleType.OUTPUT,
                                       7: keyLib.ModuleType.OUTPUT,
                                       8: keyLib.ModuleType.OUTPUT,
                                       9: keyLib.ModuleType.OUTPUT,
                                       10: keyLib.ModuleType.INPUT})

        self.hardware_cfg = hardware_cfg

        #set number for apporpriate slots - here there is only one AWG card being used, SLOT 6
        self.out_mod_no = hardware_cfg['awg_info']['keysight_pxi']['out_mod_no']

        #setting clock rates
        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']

        #shows up in plotting things apparently
        self.readout_window = np.array(quantum_device_cfg['readout']['window'])
        # self.lo_delay = hardware_cfg['awg_info']['keysight_pxi']['lo_delay']
        # self.abs_trig_delay = hardware_cfg['awg_info']['keysight_pxi']['abs_trig_delay']
        # self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']
        # self.trig_delay = hardware_cfg['awg_info']['keysight_pxi']['m3201a_trig_delay']
        # self.card_delay = hardware_cfg['awg_info']['keysight_pxi']['m3102a_card_delay']
        
        #setting DAQ input range
        self.adc_range =  hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']

        #makes a list of the different modules being used at AWGs
        print ("Module used for generating  pulses = ",self.out_mod_no)
        self.out_mod_nums = [self.out_mod_no]

        #number of times experiment is repeated
        self.num_avg = experiment_cfg[name]['acquisition_num']
        
        #nb of different parameter sweeps for same basic "exp"
        self.num_expt = sequences['readout'].shape[0]

        #self.trigger_period - how often each experiment is run, set by an external AWG, time lag to make sure that
        #resonator and qubit are back in the ground state at the end of each experiment
        self.trigger_period = self.hardware_cfg['trigger']['period_us']
        #total time is an estimate of how long the experiment will take to run, prints out as a flag
        self.totaltime = self.num_avg*self.num_expt*self.trigger_period*1e-6/60.0
        # 
        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']

        #set HVI parameters hardware (readout wait times, etc)
        #ALSO, all times entered in ns must be divided by the "cycle time" of the device, which from
        #keysight's example code is 10ns.
        #//TODO: look into why 10ns
        self.hvi = None
        self.hvi_file_path = hardware_cfg['HVI_info']['HVI_file_path']
        self.AWG_HVI_hardware_params = hardware_cfg['HVI_info']['AWG_HVI_hardware_params'] #dictionary
        self.DIG_HVI_hardware_params = hardware_cfg['HVI_info']['DIG_HVI_hardware_params'] #dictionary
        self.nanosec_per_cycle = hardware_cfg['HVI_info']['nanosec_per_cycle'] #number

        # set HVI parameters that are experiment specfic (nloops, tau step, etc)
        #self.AWG_HVI_experiement_params = experiment_cfg["hvi"]
        #self.DIG_HVI_experiement_params = experiment_cfg["hvi"]


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

        # # Initialize Module 8 = Marker card. Digital markers for qubit, readout
        # # self.m_chs = [chassis.getChannel(self.marker_mod_no, ch) for ch in self.awg_channels]
        # self.m_ch_1 = chassis.getChannel(self.marker_mod_no, 1)
        # self.m_ch_2 = chassis.getChannel(self.marker_mod_no, 2)
        # self.m_ch_3 = chassis.getChannel(self.marker_mod_no, 3)
        # self.m_ch_4 = chassis.getChannel(self.marker_mod_no, 4)
        # 
        # # Initialize card that generates Triggers
        # # self.trig_chs = [chassis.getChannel(self.trig_mod_no, ch) for ch in self.awg_channels]
        # self.trig_ch_1 = chassis.getChannel(self.trig_mod_no, 1)
        # self.trig_ch_2 = chassis.getChannel(self.trig_mod_no, 2)
        # self.trig_ch_3 = chassis.getChannel(self.trig_mod_no, 3)
        # self.trig_ch_4 = chassis.getChannel(self.trig_mod_no, 4)

        self.out_modules = [chassis.getModule(num) for num in self.out_mod_nums]
        self.AWG_module = chassis.getModule(self.out_mod_no)
        self.AWG_hvi_name = 'SLOT ' + str(self.out_mod_no)
        
        # self.m_module = chassis.getModule(self.marker_mod_no)
        # self.trig_module = chassis.getModule(self.trig_mod_no)

        # Initialize digitizer card
        # self.DIG_chs = [chassis.getChannel(10, ch) for ch in self.dig_channels]
        self.DIG_ch_1 = chassis.getChannel(10, 1)
        self.DIG_ch_2 = chassis.getChannel(10, 2)
        self.DIG_module = chassis.getModule(10)
        self.DIG_hvi_name = 'SLOT ' + str(10)

        #I think this is just for I and Q? idk cuz there are I and Q entries bolw this
        self.data_1,self.data_2 = np.zeros((self.num_expt, self.DIG_sampl_record)),np.zeros((self.num_expt, self.DIG_sampl_record))

        #self.I,self.Q = np.zeros(self.num_expt), np.zeros(self.num_expt)

        self.sleep_time = sleep_time_between_trials / (10 ** 9)  # stored in seconds internally


    def open_hvi(self, file_path):
        """
        Creates an instance of HVI and opens the requested HVI file. Sets self.hvi variable
        :param file_path: the path to the compiled HVI sequence (.HVI extention)
        """
        self.hvi = SD1.SD_HVI()

        # Check to see that the file actually exists
        if os.path.isfile(file_path):
            print(file_path)
            error = self.hvi.open(file_path)
            if self.hvi==None:
                print("DEBUG: just tried to open HVI in open_hvi, but returned none")
            if error < 0:
                if error != -8031 and error != -8038:
                    self.hvi.close()
                    self.hvi = None
                    raise Exception('Error opening HVI file: {}'.format(keyLib.Tools.decodeError(error)))
        else:
            self.hvi.close()
            self.hvi = None
            raise Exception('Failed to open HVI file: {}.  File does not exist.'.format(file_path))

    def compile_and_load_hvi(self):
        """
        Compiles an HVI project after all of the parameters have been applied.
        :return:
        """
        if self.hvi is None:
            raise Exception('No HVI project open.  Open a valid HVI project using the open_hvi function.')

        errors = self.hvi.compile()

        if errors > 0:
            print('Compile errors: {}'.format(errors))
            for error in range(errors):
                print('Error ' + str(error) + ' : ' + self.hvi.compilationErrorMessage(error))

        error = self.hvi.load()
        if error < 0:
            print('load: {}'.format(keyLib.Tools.decodeError(error)))

    def initiate_HVI(self):
        """
        Initiates all the parameters and hardware linkings in the HVI instance
        :param hvi_file_path: the path to the compiled HVI sequence (.HVI extention)
        """

        self.open_hvi(self.hvi_file_path)

        # Assign the hardware modules to the HVI project
        error = self.hvi.assignHardwareWithUserNameAndModuleID(self.AWG_hvi_name, self.AWG_module)
        if error < 0 and error != -8069:
            raise Exception('Error occurred assigning hardware. {}'.format(keyLib.Tools.decodeError(error)))
        error = self.hvi.assignHardwareWithUserNameAndModuleID(self.DIG_hvi_name, self.DIG_module)
        if error < 0 and error != -8069:
            raise Exception('Error occurred assigning hardware. {}'.format(keyLib.Tools.decodeError(error)))

        # Assign all of the parameters to the HVI project.
        for key, value in self.AWG_HVI_hardware_params.items():
            error = self.hvi.writeIntegerConstantWithUserName(self.AWG_hvi_name, key, value)
            if error < 0:
                raise Exception('Error occurred writing integer constant to AWG. {}'.format(keyLib.Tools.decodeError(error)))

        for key, value in self.DIG_HVI_hardware_params.items():
            error = self.hvi.writeIntegerConstantWithUserName(self.DIG_hvi_name, key, value)
            if error < 0:
                raise Exception('Error occurred writing integer constant to DIG. {}'.format(keyLib.Tools.decodeError(error)))

        # Compile and load the HVI sequence
        self.compile_and_load_hvi()

    def configureChannels(self, hardware_cfg, experiment_cfg, name):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''

        amp_AWG = hardware_cfg['awg_info']['keysight_pxi']['amplitudes']
        #amp_mark = hardware_cfg['awg_info']['keysight_pxi']['amp_mark']
        #amp_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_trig']

        num_avg = self.num_avg
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring analog channels")

        #sets the trigger channel on the awg to be OUT (we want this since we are using it for the BNC)
        self.AWG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_OUT)

        #kind of output already configured in setting the module type (this is how Josie's Keysight Lib is set up)
        #NOT setting trigger sources here (either external or internal) because will all be triggered through HVI
        #and that is set in Queue config
        self.AWG_ch_1.configure(amplitude=amp_AWG[0])
        self.AWG_ch_2.configure(amplitude=amp_AWG[1])
        self.AWG_ch_3.configure(amplitude=amp_AWG[2])
        #self.AWG_ch_4.configure(amplitude=amp_AWG[3])

        # print("Configuring marker channels")
        #
        # self.m_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.m_ch_1.configure(amplitude=amp_mark[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_2.configure(amplitude=amp_mark[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_3.configure(amplitude=amp_mark[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.m_ch_4.configure(amplitude=amp_mark[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        #
        # print("Configuring trigger channels")
        #
        # self.trig_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        # self.trig_ch_1.configure(amplitude=amp_trig[0], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_2.configure(amplitude=amp_trig[1], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_3.configure(amplitude=amp_trig[2], trigger_source=SD1.SD_TriggerModes.EXTTRIG)
        # self.trig_ch_4.configure(amplitude=amp_trig[3], trigger_source=SD1.SD_TriggerModes.EXTTRIG)

        #self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)

        #everything will be triggered from the HVI so it should be ok to comment this out - triggering gets set in waveform (I think)
        # print ("Setting trigger config for all channels of all modules to External")
        # for n in range(1, 5):
        #     self.AWG_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
        #     self.m_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
        #     self.trig_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)


        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")
        self.DIG_ch_1.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.SWHVITRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.SWHVITRIG, use_buffering=True, cycles_per_return=num_expt)

    # def generatemarkers(self,waveform,resample=False,conv_width = 1,trig_delay = 0):
    # 
    #     # markers = np.array([np.append(np.heaviside(np.convolve(abs(w), np.ones((int(self.lo_delay),))/self.lo_delay),0)[int(self.lo_delay/2):],np.zeros(int(self.lo_delay/2))) for w in waveform])
    #     markers = np.array([np.append(np.heaviside(np.convolve(abs(np.append(w[int(trig_delay):],np.zeros(int(trig_delay)))), np.ones((int(conv_width),)) / conv_width), 0)[int(conv_width / 2):], np.zeros(int(conv_width / 2))) for w in waveform])
    #     if resample: return np.array([m[::int(self.dt_dig/self.dt)] for m in markers])
    #     else:return markers
    # 
    # def generatemastertrigger(self,length,trig_width = 1,trig_delay = 0):
    #     trig = np.zeros(length)
    #     for i in np.arange(int(trig_delay/self.dt_dig),int((trig_delay + trig_width)/self.dt_dig)):
    #         trig[i] = 1
    #     return trig

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
            sequences - dictionary with channels as keywords and list of expt arrays as entries.
            Here, I'm assuming that we have:
            charge_I
            charge_Q
            readout

            Before HVI triggering:
            The master trigger for all the cards is generated knowing the length of the AWG waveforms using self.generate_trigger
            '''


        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        num_expt = self.num_expt #reminder: this is number of arrays for each channel
        print('num_expt = {}'.format(num_expt))
        wv = self.sequenceslist(sequences, pxi_waveform_channels) #turns sequences from a dictionary into an array

        # Based on Nelson's original nomenclature
        #each of these should be an array of arrays
        waveforms_I = wv[0]
        waveforms_Q = wv[1]
        readout = wv[2]

        #assuming not using tek2 at all
        #if self.prep_tek2:tek2_marker = wv[4]

        AWG_module = self.chassis.getModule(self.out_mod_no)
        #m_module = self.chassis.getModule(self.marker_mod_no)
        #trig_module = self.chassis.getModule(self.trig_mod_no)

        print ("shape of waveform I",np.shape(waveforms_I))

        if len(waveforms_I) != len(waveforms_Q) or len(waveforms_I) != len(readout):
            raise TypeError("Not all waveform lists are the same length")

        self.AWG_module.clearAll()
        #self.m_module.clearAll()
        #self.trig_module.clearAll()

        keyLib.Waveform._waveform_number_counter = 0 #IDK what this is doing

        # qubit_marker = self.generatemarkers(waveforms_I)
        # qubit_marker_dsp = self.generatemarkers(waveforms_I,resample=True,conv_width=self.lo_delay,trig_delay=self.trig_delay)
        # readout_marker = self.generatemarkers(readout, resample=False)
        # readout_marker_dsp = self.generatemarkers(readout,resample=True,trig_delay=self.trig_delay)
        # card_trig_arr = self.generatemarkers(markers_readout,resample=True)
        # if self.prep_tek2:
        #     tek2_marker_dsp =  self.generatemarkers(tek2_marker,resample=True,trig_delay=0.0)
        # trig_arr_awg = self.generatemastertrigger(len(readout_marker_dsp[0]),2*self.trig_pulse_length,self.abs_trig_delay)


        #queue each of the waveforms
        for i in tqdm(range(len(waveforms_I))):

            wave_I = keyLib.Waveform(np.array(waveforms_I[i]),append_zero=True)  # Have to include append_zero or the triggers get messed up!
            wave_Q = keyLib.Waveform(waveforms_Q[i], append_zero=True)
            readout = keyLib.Waveform(readout[i], append_zero=True)

            #m_readout_dsp = keyLib.Waveform(readout_marker_dsp[i], append_zero=True)
            #m_qubit_dsp = keyLib.Waveform(qubit_marker_dsp[i], append_zero=True)
            #if self.prep_tek2:m_tek2_dsp = keyLib.Waveform(tek2_marker_dsp[i], append_zero=True)

            #trig = keyLib.Waveform(trig_arr_awg, append_zero=True)
            #card_trig = keyLib.Waveform(card_trig_arr[i], append_zero=True)

            # Load objects to the modules
            wave_I.loadToModule(AWG_module)
            wave_Q.loadToModule(AWG_module)
            readout.loadToModule(AWG_module)

            # Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.
            #Delay used to be self.tek2_trigger_delay, I'm setting it to zero since hopefully going through HVI everything should be great
            wave_I.queue(self.AWG_ch_1, trigger_mode=SD1.SD_TriggerModes.SWHVITRIG, delay = 0, cycles = 1, prescaler = 0)
            wave_Q.queue(self.AWG_ch_2, trigger_mode=SD1.SD_TriggerModes.SWHVITRIG, delay = 0, cycles = 1, prescaler = 0)
            readout.queue(self.AWG_ch_3, trigger_mode=SD1.SD_TriggerModes.SWHVITRIG, delay=0, cycles=1, prescaler=0)

            # //TODO: wthk does this thing do and why does it matter
            self.AWG_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
                                                 syncMode=1, length=10, delay=0)

            # m_qubit_dsp.loadToModule(m_module)
            # m_readout_dsp.loadToModule(m_module)
            # if self.prep_tek2:m_tek2_dsp.loadToModule(m_module)
            # m_qubit_dsp.queue(self.m_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)
            # m_readout_dsp.queue(self.m_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=self.tek2_trigger_delay, cycles=1, prescaler=0)
            # if self.prep_tek2:m_tek2_dsp.queue(self.m_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            #
            #
            # self.m_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
            #                                      syncMode=1, length=10, delay=0)
            #
            # trig.loadToModule(trig_module)
            # card_trig.loadToModule(trig_module)
            #
            # trig.queue(self.trig_ch_1, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            # trig.queue(self.trig_ch_2, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            # trig.queue(self.trig_ch_3, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=0, cycles=1, prescaler=0)
            # card_trig.queue(self.trig_ch_4, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, delay=int(self.card_delay/100)+self.tek2_trigger_delay, cycles=1, prescaler=0)
            #
            # self.trig_module.AWGqueueMarkerConfig(nAWG=1, markerMode=1, trgPXImask=0b11111111, trgIOmask=0, value=1,
            #                                    syncMode=1, length=10, delay=0)

    def run(self):
        print("Experiment starting. Expected time = ", self.totaltime, "mins")
        try:
            # Start all the channels on the AWG and digitizer modules.
            print("Number of experiments = ", self.num_expt)

            self.DIG_ch_1.clear()
            self.DIG_ch_1.start()
            self.DIG_ch_2.clear()
            self.DIG_ch_2.start()
            self.AWG_module.startAll()
            self.initiate_HVI()
            error = self.hvi.start()
            if error < 0:
                raise Exception('Error occurred starting HVI sequence. {}'.format(keyLib.Tools.decodeError(error)))
            #self.m_module.startAll()
            #self.trig_module.startAll()

        except BaseException as e:  # Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            pass
            print(e)

        finally:  # Clean up threads to prevent zombies. If this fails, you have to restart program.
            pass

    def acquireandplot(self, expt_num):

        for sweep_ct in tqdm(range(self.num_avg)):
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
            self.AWG_module.startAll()
            #self.m_module.startAll()
            #self.trig_module.startAll()

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
        print("DEBUG: about to start looping over num_avg")
        for ii in tqdm(range(self.num_avg)):
            print("reading data")
            I.append(np.mean(np.reshape(self.DIG_ch_1.readDataQuiet(timeout=100), self.data_1.shape).T[int(w[0]):int(w[1])], 0)) #timeout in ms
            Q.append(np.mean(np.reshape(self.DIG_ch_2.readDataQuiet(timeout=100), self.data_2.shape).T[int(w[0]):int(w[1])], 0)) #timeout in ms
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