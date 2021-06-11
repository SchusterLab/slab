# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:00:40 2018

@author: slab
"""

import KeysightLib as key
import numpy as np
import keysightSD1 as SD1
import time

waveform_data_list_ramp = [0.01,
0.02,
0.03,
0.04,
0.05,
0.06,
0.07,
0.08,
0.09,
0.1,
0.11,
0.12,
0.13,
0.14,
0.15,
0.16,
0.17,
0.18,
0.19,
0.2,
0.21,
0.22,
0.23,
0.24,
0.25,
0.26,
0.27,
0.28,
0.29,
0.3,
0.31,
0.32,
0.33,
0.34,
0.35,
0.36,
0.37,
0.38,
0.39,
0.4,
0.41,
0.42,
0.43,
0.44,
0.45,
0.46,
0.47,
0.48,
0.49,
0.5,
0.51,
0.52,
0.53,
0.54,
0.55,
0.56,
0.57,
0.58,
0.59,
0.6,
0.61,
0.62,
0.63,
0.64,
0.65,
0.66,
0.67,
0.68,
0.69,
0.7,
0.71,
0.72,
0.73,
0.74,
0.75,
0.76,
0.77,
0.78,
0.79,
0.8,
0.81,
0.82,
0.83,
0.84,
0.85,
0.86,
0.87,
0.88,
0.89,
0.9,
0.91,
0.92,
0.93,
0.94,
0.95,
0.96,
0.97,
0.98,
0.99]


waveform_sin = [.0314, .0627, .0941,
                      .1253, .1564, .1874,
                      .2181, .2487, .2790,
                      .3090, .3387, .3681,
                      .3971, .4258, .4540,
                      .4818, .5090, .5358,
                      .5621, .5878, .6129,
                      .6374, .6613, .6845,
                      .7071, .7290, .7501,
                      .7705, .7902, .8090,
                      .8271, .8443, .8607,
                      .8672, .8190, .9049,
                      .9178, .9298, .9409,
                      .9511, .9603, .9687,
                      .9759, .9823, .9877,
                      .9921, .9956, .9980,
                      .9996, 1.0, .9996,
                      .9980, .9956, .9921,
                      .9877, .9823, .9759,
                      .9687, .9603, .9511,
                      .9409, .9298, .9178,
                      .9049, .8190, .8672,
                      .8607, .8443, .8271,
                      .8090, .7902, .7705,
                      .7501, .7290, .7071,
                      .6845, .6613, .6374,
                      .6129, .5878, .5621,
                      .5358, .5090, .4818,
                      .4540, .4258, .3971,
                      .3681, .3387, .3090,
                      .2790, .2487, .2181,
                      .1874, .1564, .1253,
                      .0941, .0627, .0314, 0]

waveform_data_list_sin = []
for i in waveform_sin:
    waveform_data_list_sin.append(i)
for i in waveform_sin:
    waveform_data_list_sin.append(-i)
  
pulse = np.zeros(100)
for i in range(1,20):
    pulse[i] = 1
    
    
chassis = key.KeysightChassis(1, {6: key.ModuleType.OUTPUT,
                                  7: key.ModuleType.OUTPUT,
                                  8: key.ModuleType.OUTPUT,
                                  9: key.ModuleType.OUTPUT})

module = chassis.getModule(6)
ch1 = chassis.getChannel(6, 1)
ch2 = chassis.getChannel(6, 2)
ch3 = chassis.getChannel(6, 3)

ch1.configure()
ch2.configure()
ch3.configure()

waveform1 = key.Waveform(waveform_data_list_sin, append_zero = True)
waveform1.loadToModule(module)
waveform1.queue(ch1, cycles = key.KeysightConstants.INFINITY,
                trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)

waveform2 = key.Waveform(arr = waveform_data_list_ramp, append_zero = True)
waveform2.loadToModule(module)
waveform2.queue(ch2, cycles = key.KeysightConstants.INFINITY,
                trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)

waveform3 = key.Waveform(arr = pulse)
waveform3.loadToModule(module)
waveform3.queue(ch3, cycles = key.KeysightConstants.INFINITY,
                trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)

ch1.start()
ch2.start()
ch3.start()

ch1.trigger()
ch2.trigger()
ch3.trigger()

time.sleep(5)
ch1.stop()
ch2.stop()
ch3.stop()

chassis.clearAll()
chassis.close()