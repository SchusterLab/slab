# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:39:57 2018

@author: Josie Meyer

Test experiment that extends HVIEnabledQubitExperiment. Mainly to test functionality,
does not do anything scientificalyl interesting. However, it shows how to build an
HVIEnabledQubitExperiment subclass, so I have kept it.

DEBUG: Currently the digitizers do not work. Need to figure out why.
"""
import numpy as np
import slab.experiments.HVIExperiments.HVIExpLib as exp
from slab.instruments.keysight import KeysightLib as key
import os


class TestExperiment(exp.HVIEnabledQubitExperiment):
    '''A test extension of HVIEnabledQubitExperiment. Shows largely how to 
    extend the class and what can be modified. (Note: can also extend post_run()
    analogous to pre_run())).'''
    
    def __init__(self):
        '''Add any data channels, workers, channels to listen'''
        exp.HVIEnabledQubitExperiment.__init__(self, liveplot_enabled = False,
                                           save_in_register = True)
        data_channel = self.getChannel(10, 1)
        self.addChannelToListen(data_channel)
        self.addWorkers(exp.Saver(data_channel, r"C:\Users\slab\keysight_test_data", "testdata",
                                  name = "SAVER"))
    
    def buildHVIPulseSequence(self):
        '''Specify the pulse sequence'''
        return TestPulseSequence()
    
           
    def pre_run(self):
        '''Example pre_run code. Mostly for configuration.'''
        #configure AWG; defaults OK for now
        self.getChannel(6, 1).configure()
        #configure digitizer
        self.getChannel(10, 1).configure(
                points_per_cycle = 5000,
                cycles = 100000,
                use_buffering = False, #must be false
                cycles_per_return = 10)
    

class TestPulseSequence(exp.HVIPulseSequence):
    
    def __init__(self):
        '''Initializes the pulse sequence. Here we only add two waveforms.
        Note that we have specified an HVI file and a .keycfg hardware file.'''
        path = r"C:\_Lib\python\slab\experiments\HVIExperiments"
        exp.HVIPulseSequence.__init__(self, 
            HVI_file = os.path.join(path, r"Test\TestExperimentHVI.HVI"),
            module_config_file = os.path.join(path, r"default_hardware.keycfg"))
        self.addWaveform(key.Waveform(arr = np.ones(1000),
                                                  waveform_number = 1,
                                                  append_zero = True),
                                            modules = [6])
        self.addWaveform(key.Waveform(arr = np.arange(0, 1, .001),
                                                  waveform_number = 2,
                                                  append_zero = True),
                                            modules = [6])
                
        
if __name__ == "__main__":
    exp = TestExperiment()
    exp.go() #will run experiment, just like the Experiment class
        