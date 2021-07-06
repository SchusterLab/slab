# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:30:04 2018

@author: slab
"""
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
    
BASIC VERSION WITH ONLY ONE CHANNEL!
"""

from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
#from slab.experiments.HVIExperiments import HVIExpLib as exp
import time
import numpy as np

class SingleQubitKeysightSetup:
    
    def __init__(self, num_sweeps = 100, save_path = r"C:\Users\slab\Desktop\data",
                 num_data_points = 1000,
                 sleep_time_between_trials = 100000): #sleep_time_between_trials in ns
        chassis = key.KeysightChassis(1,
                                  {6: key.ModuleType.OUTPUT,
                                   7: key.ModuleType.OUTPUT,
                                   8: key.ModuleType.OUTPUT,
                                   9: key.ModuleType.OUTPUT,
                                   10: key.ModuleType.INPUT})
        self.chassis = chassis
        self.AWG_I_channel = chassis.getChannel(6, 1)
        #self.AWG_Q_channel = chassis.getChannel(6, 2)
        #self.readout_pulse_marker = chassis.getChannel(6, 3)
        #self.digitizer_trigger_marker = chassis.getChannel(6, 4)
        self.AWG_module = chassis.getModule(6)
        '''
        self.digitizer_I_channel = chassis.getChannel(10, 1)
        self.digitizer_Q_channel = chassis.getChannel(10, 2)
        self.DIG_module = chassis.getModule(10)'''
        
        self.configureChannels()
        self.num_sweeps = num_sweeps
        self.num_data_points = num_data_points
        '''self.dispatcher = exp.Dispatcher(channels = [self.digitizer_I_channel, self.digitizer_Q_channel])
        saver_I = exp.Saver(self.digitizer_I_channel, name = "Saver I", filepath = save_path, prefix = "I")
        saver_Q = exp.Saver(self.digitizer_Q_channel, name = "Saver Q", filepath = save_path, prefix = "Q")
        self.dispatcher.addWorker(saver_I)
        self.dispatcher.addWorker(saver_Q)'''
        self.sleep_time = sleep_time_between_trials / (10**9)
        
        
        
    def configureChannels(self):
        self.AWG_I_channel.configure()
        '''self.AWG_Q_channel.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)
        self.readout_pulse_marker.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)
        self.digitizer_trigger_marker.configure(
                amplitude=1, trigger_source = SD1.SD_TriggerModes.SWHVITRIG)'''
        '''
        self.digitizer_I_channel.configure(points_per_cycle = 1000,
                                           cycles = 2000 * 200,
                                           buffer_time_out = 1000,
                                           trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
                                           use_buffering = False)
        self.digitizer_Q_channel.configure(points_per_cycle = 1000,
                                           buffer_time_out = 1000,
                                           cycles = 2000 * 200,
                                           trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
                                           use_buffering = False)'''
        
    def loadAndQueueWaveforms(self, waveforms_I):
        AWG_module = self.chassis.getModule(6)
        self.AWG_module.clearAll()
        for i in range(len(waveforms_I)):
            wave_I = key.Waveform(waveforms_I[i], append_zero = True)
            #wave_Q = key.Waveform(waveforms_Q[i], waveform_number = 200 + i, append_zero = True)
            #wave_readout = key.Waveform(markers_readout[i], waveform_number = 400 + i, append_zero = True)
            #wave_trig = key.Waveform(markers_digitizer[i], waveform_number = 600 + i, append_zero = True)
            wave_I.loadToModule(AWG_module)
            #wave_Q.loadToModule(AWG_module)
            #wave_readout.loadToModule(AWG_module)
            #wave_trig.loadToModule(AWG_module)
            wave_I.queue(self.AWG_I_channel, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG,
                         cycles = 1)
            #wave_Q.queue(self.AWG_Q_channel, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            #wave_readout.queue(self.readout_pulse_marker, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            #wave_trig.queue(self.digitizer_trigger_marker, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
        
    def run(self):
        print("Experiment starting")
        #self.dispatcher.takeData()
        self.AWG_module.startAll()
        '''
        self.DIG_module.startChannels(1,2)
        '''
        for i in range(self.num_sweeps):
            print("sweep " + str(i))
            for j in range(self.num_data_points):
                self.AWG_I_channel.trigger()
                time.sleep(self.sleep_time) #200 us
        print("Done taking data")
        #self.dispatcher.stopAllWhenReady()
        self.AWG_module.clearAll()
        self.chassis.close()
        
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x-mu)**2 / (2 * sigma **2))

def gaussianDerivative(x, amplitude, mu, sigma):
    return -amplitude * (x-mu) * gaussian(x, 1, mu, sigma) * np.sqrt(np.e) / sigma / 2 #normalize
        

def generateWaveformsI():
    print("Generating waveforms I")
    arr = []
    for sigma in range(10, 210):
        pulse = np.array([gaussian(x, 1, 500, sigma) for x in range(1000)])
        arr.append(pulse)
    return arr

def generateWaveformsQ():
    print("Generating waveforms Q")
    arr = []
    for sigma in range(1, 201):
        pulse = np.array([gaussianDerivative(x, 1, 500, sigma) for x in range(1000)])
        print(pulse)
        arr.append(pulse)
    return arr

def generateMarkers():
    print("Generating markers")
    marker = np.append(np.zeros(550), np.ones(1000))
    return np.array([marker for i in range(200)])



#Code for Rabi experiment!
setup = SingleQubitKeysightSetup()
#markers = generateMarkers()
setup.loadAndQueueWaveforms(generateWaveformsI())
setup.run()
    
        
        
        
        
        