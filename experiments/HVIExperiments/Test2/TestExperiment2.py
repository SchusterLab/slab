# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:39:57 2018

@author: slab
"""
import numpy as np
import slab.experiments.HVIExperiments.HVIExpLib as exp
from slab.instruments.keysight import KeysightLib as key
import os


class TestExperiment2(exp.HVIEnabledQubitExperiment):
    
    def __init__(self):
        exp.HVIEnabledQubitExperiment.__init__(self, liveplot_enabled = False,
                                           save_in_register = True)
        #data_channel = self.getChannel(10, 1)
        #self.addChannelToListen(data_channel)
        #self.addWorkers(exp.Saver(data_channel, r"C:\Users\slab\keysight_test_data", "testdata",
                                  #name = "SAVER"))
    
    def buildHVIPulseSequence(self):
        return TestPulseSequence2()
    
           
    def pre_run(self):
        #configure AWG; defaults OK
        self.getChannel(6, 1).configure()
        #configure digitizer
        self.getChannel(10, 1).configure(
                points_per_cycle = 5000,
                cycles = 100000,
                use_buffering = False,
                cycles_per_return = 10)
    

class TestPulseSequence2(exp.HVIPulseSequence):
        
    def __init__(self):
        '''Initializes the pulse sequence'''
        path = r"C:\_Lib\python\slab\experiments\HVIExperiments"
        exp.HVIPulseSequence.__init__(self, 
            HVI_file = os.path.join(path, r"Test2\TestExperiment2.HVI"),
            module_config_file = os.path.join(path, r"default_hardware.keycfg"))
        
        '''for i in range(1000):
            pass
        
        
        self.addWaveform(key.Waveform(arr = np.ones(1000),
                                                  waveform_number = 1,
                                                  append_zero = True),
                                            modules = [6])
        self.addWaveform(key.Waveform(arr = np.arange(0, 1, .001),
                                                  waveform_number = 2,
                                                  append_zero = True),
                                            modules = [6])'''
            
        for i, elem in enumerate(make_pulses()):
            self.addWaveform(key.Waveform(arr = elem, waveform_number = i,
                                          append_zero = True), modules = [6])
                
def gaussian(amplitude, mu, sigma, x):
    return amplitude*np.exp(-(x-mu)**2 / (2*sigma**2))

def make_pulses():
    x=np.arange(0,600,1)
    
    return [gaussian(ii/100.,300,100,x) for ii in range(100)]
        

    #print(gaussian(1,50,10,x))
    
    
if __name__ == "__main__":
    exp = TestExperiment2()
    exp.go()

        

#def gaussian_wave(amplitude=1, mu, sigma, num_points=600):
#    return np.array(list(map(gaussian, x)))
        
        