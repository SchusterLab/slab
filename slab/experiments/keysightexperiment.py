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

from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
from slab.experiments.HVIExperiments import HVIExpLib as exp
import time
import numpy as np
import os
import threading

AWG_BANDWIDTH = 10**9 #Samples per second
NUM_TRIALS = 200 #This is the number of different experiments, or how many values the parameter that is being varied (in this case sigma) will take
SIGMA_STEP=1 #This is the parameter step
READOUT_PULSE=2000 #length in ns
NUM_SWEEPS = 2000 #This is the total number of runs that will be averaged

class KeysightSingleQubit:
    '''Class designed to implement a simple single qubit experiment given a pulse sequence. Note that this will not work for error correction or 
    anything else needing the FPGA (extend HVIEnabledQubitExperiment instead) or anything requiring more than 1 AWG and 1 digitizer module
    (it is impossible to trigger them in synch from Python using the manufacturer-provided code). In either case you will have to use the HVI
    technology.
    
    This currently only works for one hardware configuration. In particular, Module (slot) 6 is used as the AWG/marker, and Module 10 is used as
    the digitizer. On Module 6, channel 1 goes to the I input to the mixer, channel 2 goes to the Q input, channel 3 is the readout pulse, and
    channel 4 is the qubit pulse. Module 6, channel 3 (readout pulse) should also be connected to the trigger input of Module 10 (digitizer). On
    module 10, channel 1 is for readout of I component and channel 2 is for readout from Q component.'''
    
    
    
    def __init__(self, num_sweeps = NUM_SWEEPS, num_experiments = NUM_TRIALS, save_path = r"C:\Users\slab\Documents\Data",
                 sleep_time_between_trials = 1000):
        '''Initializes the experiment.
        Params:
            num_sweeps: the number of total runs that will be averaged.
            num_experiments: The number of values that the parameter being varied will take (i.e. the number of unique experiments being done). 
                Equivalently, this is the number of arrays you'll expect to get back at the end of a single sweep.
            save_path: The path where the data is to be saved.
            sleep_time_between_trials: Sleep time in ns between triggers for each trial. Keep in mind that this should be at least the length
                of the longest trial, plus any qubit relaxation time. Also keep in mind that the latency can very somewhat between trials because
                of the speed at which instructions are delivered to the Keysight hardware, so it is possible for two trials to occur a
                hundred ns or so closer together than expected if the first is delayed and the second is not.
        '''
        
        chassis = key.KeysightChassis(1,
                                  {6: key.ModuleType.OUTPUT,
                                   7: key.ModuleType.OUTPUT,
                                   8: key.ModuleType.OUTPUT,
                                   9: key.ModuleType.OUTPUT,
                                   10: key.ModuleType.INPUT})
        #initialized with our hardware setup. Could also read from .keycfg file using KeysightLib.KeysightChassis.fromFile()
        
        self.chassis = chassis 
        self.AWG_I_channel = chassis.getChannel(6, 1) #module 6, channel 1. AWG for I input to mixer
        self.AWG_Q_channel = chassis.getChannel(6, 2) #AWG for Q input into mixer
        self.readout_pulse_marker = chassis.getChannel(6, 3) #AWG used as marker for readout pulse
        self.qubit_marker = chassis.getChannel(6, 4) #AWG used as marker for qubit pulse
        self.AWG_module = chassis.getModule(6)
        
        self.digitizer_I_channel = chassis.getChannel(10, 1)
        self.digitizer_Q_channel = chassis.getChannel(10, 2)
        self.DIG_module = chassis.getModule(10)
        
        self.configureChannels()
        self.num_sweeps = num_sweeps
        self.num_data_points = num_experiments

        self.data_list_I = [] #Stores the data from the I channel as a list of numpy arrays
        self.data_list_Q = [] #Stores the data from the Q channel analogously
        
        #Threads that control data acquisition. Has to be done in own thread to avoid missing data due to latency
        self.data_handler_I = SimpleDataHandler("Data List I", self.data_list_I, self.digitizer_I_channel)
        self.data_handler_Q = SimpleDataHandler("Data List Q", self.data_list_Q, self.digitizer_Q_channel)
        
        
        self.sleep_time = sleep_time_between_trials / (10**9) #stored in seconds internally
        self.save_path = save_path
        
        
        
    def configureChannels(self):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in volts.'''
        self.AWG_I_channel.configure(amplitude = 1, #volt
                trigger_source = SD1.SD_TriggerModes.SWHVITRIG) #We want to trigger from software
        
        self.AWG_Q_channel.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)

        self.readout_pulse_marker.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)

        self.qubit_marker.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)
 
        self.digitizer_I_channel.configure(points_per_cycle = READOUT_PULSE // 2, #Divide by 2 because bandwidth of digitizer is less than AWG
                                           #Number of data points to acquire at one time, at 2 ns apiece.
                                           
                                           cycles = NUM_TRIALS * NUM_SWEEPS, #Total number of times digitizer will be asked to take data
                                           buffer_time_out = 100, #Time before aborting data read, ms. 
                                           #In general want to be quite large, but not infinite
                                           
                                           trigger_mode = SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                           #Triggered externally by linking to readout pulse
                                           use_buffering = False, #Unnecessary with a Data Handler subprocess
                                           cycles_per_return = 1) #Return and log 1 trial's data at a time. Can increase to bundle data together.
        #time.sleep(0.001)
        self.digitizer_Q_channel.configure(points_per_cycle = READOUT_PULSE // 2,
                                           buffer_time_out = 1000,
                                           cycles = NUM_TRIALS * NUM_SWEEPS,
                                           trigger_mode = SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                           use_buffering = False,
                                           cycles_per_return = 1)
        
        
    def loadAndQueueWaveforms(self, waveforms_I, waveforms_Q, markers_readout,
                              markers_qubit):
        '''Loads the provided waveforms from a pulse sequence to the appropriate modules.
        This should easily link into the PulseSequenceBuilder.
        
        Note that all waveforms should consist of values from -1 to 1 (inclusive) only. Amplitude is set in the configureChannels() method.
        If you accidentally pass in an array containing larger values, the code raises a KeysightError: Invalid Waveform.
        
        Params:
            waveforms_I: A list of numpy arrays representing waveforms (or a 2D array) for the "I" input AWG to the mixer, 
                1 for each unique trial/experiment. Should be the same number of waveforms as num_experiments in the __init__ method. If a 2D array,
                the "inner" array is a waveform, and the "outer" dimension corresponds to the experiment/trial.
            waveforms_Q: Same for the "Q" channel.
            markers_readout: Same for the readout pulse.
            markers_qubit: Same for the qubit pulse.'''
        AWG_module = self.chassis.getModule(6)
        if len(waveforms_I)!=len(waveforms_Q) or len(waveforms_I)!=len(markers_readout) or len(waveforms_I)!=len(markers_qubit):
            raise TypeError("Not all waveform lists are the same length")
            
        self.AWG_module.clearAll() #Remove any waveforms already on the machine
        
        for i in range(len(waveforms_I)):
            #Generate KeysightLib.Waveform objects
            wave_I = key.Waveform(waveforms_I[i], append_zero = True) #Have to include append_zero or the triggers get messed up!
            wave_Q = key.Waveform(waveforms_Q[i], append_zero = True)
            wave_readout = key.Waveform(markers_readout[i], append_zero = True)
            wave_qubit = key.Waveform(markers_qubit[i], append_zero = True)
            
            #Load objects to the modules
            wave_I.loadToModule(AWG_module)
            wave_Q.loadToModule(AWG_module)
            wave_readout.loadToModule(AWG_module)
            wave_qubit.loadToModule(AWG_module)
            
            #Queue the waveforms. Want to set trigger mode to SWHVITRIG to trigger from computer.
            wave_I.queue(self.AWG_I_channel, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            wave_Q.queue(self.AWG_Q_channel, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            wave_readout.queue(self.readout_pulse_marker, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            wave_qubit.queue(self.qubit_marker, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
        
    def run(self):
        print("Experiment starting")
        try:
            #Calls the run() method of the Data Handlers in new thread.
            self.data_handler_I.start()
            self.data_handler_Q.start()
            
            #Start all the channels on the AWG and digitizer modules.
            self.AWG_module.startAll()
            self.DIG_module.startAll()
            
            #Run the experiment
            for i in range(self.num_sweeps):
                print("sweep " + str(i))
                for j in range(self.num_data_points):
                    self.AWG_module.triggerAll()
                    time.sleep(self.sleep_time) #between trials
            print("Done taking data")
            
            self.data_handler_I.stopWhenReady()
            self.data_handler_Q.stopWhenReady()
        except BaseException as e: #Quickly kill everything and risk data loss, mainly in case of keyboard interrupt
            self.data_handler_I.kill()
            self.data_handler_Q.kill()
        finally: #Clean up threads to prevent zombies. If this fails, you have to restart program.            
            self.data_handler_I.join()
            self.data_handler_Q.join()


class SimpleDataHandler(threading.Thread):
    
    def __init__(self, name, data_list, channel):
        threading.Thread.__init__(self, name = name)
        self._stop_flag = False
        self._kill_flag = False
        self._data_list = data_list
        self._channel = channel
    
    def run(self):
        '''Loop that runs in the background while data is being collected.
        Overrides Thread.run(). Exits when kill flag is set to True, or
        if stop flag is set to True and there is no more data.'''
        while not self._kill_flag:
            #data = self._channel.readDataBuffered()
            data = self._channel.readDataQuiet(timeout = 100) #TODO: change? currently 0.5 sec #ms
            
            if type(data) != np.ndarray:
                if self._stop_flag:
                    break
            elif data.size == 0:
                if self._stop_flag:
                    break
            else:
                self._data_list.append(data)
                    
    def kill(self):
        self._kill_flag = True
        
    def stopWhenReady(self):
        self._stop_flag = True
        

'''Various Rabi-experiment specific code. When integrated with PulseSequenceBuilder, will be obsolete.'''       
        
def gaussianSin(x, amplitude, mu, sigma, period = 10):
    '''Returns a gaussian * sin.
    Params:
        x: The point where the function is to be evaluated
        amplitude, mu, sigma: In waveform points (or equivalently ns)
        period: The period of the sin wave, in ns
    Returns:  The gaussia*sin function acting on x.'''
    return amplitude * np.exp(-(x-mu)**2 / (2 * sigma **2)) * np.sin(2 * np.pi / period *(x-mu))

def gaussianCos(x, amplitude, mu, sigma, period = 10):
    #Analogous to gaussianSin
    return amplitude * np.exp(-(x-mu)**2 / (2 * sigma **2)) * np.cos(2 * np.pi / period *(x-mu))

def gaussian(x, amplitude, mu, sigma):
    #Analogous to gaussianSin
    return amplitude * np.exp(-(x-mu)**2 / (2 * sigma **2))

def generateWaveforms():
    '''Generates the waveforms for the Rabi experiment. Assumes pulses are 4 sigma width.
    Will be obsolete once PulseSequenceBuilder is linked in.'''
    print("Generating waveforms")
    I_arr = []
    Q_arr = []
    readout_arr = []
    qubit_arr = []
    for sigma in range(1, SIGMA_STEP*(1+NUM_TRIALS), SIGMA_STEP):
        I_arr.append(np.array([gaussianCos(x, 1, 2*sigma, sigma) for x in range(4*sigma)]))
        Q_arr.append(np.array([gaussianSin(x, 1, 2*sigma, sigma) for x in range(4*sigma)]))
        readout_arr.append(np.append(np.zeros(4*sigma), np.ones(READOUT_PULSE)))
        qubit_arr.append(np.ones(4*sigma))
    return I_arr, Q_arr, readout_arr, qubit_arr


#Code for Rabi experiment!  
if __name__ == "__main__":
    setup = KeysightSingleQubit(sleep_time_between_trials = 0)
    try:
        waveforms_I, waveforms_Q, readout, qubit = generateWaveforms()
        setup.loadAndQueueWaveforms(waveforms_I, waveforms_Q, readout, qubit)
        setup.run()
    finally:
        print(len(setup.data_list_I), setup.data_list_I[0].size, len(setup.data_list_Q), setup.data_list_Q[0].size)
        setup.AWG_module.stopAll()
        setup.AWG_module.clearAll()
        setup.chassis.close()

    
        
        
        
        
        