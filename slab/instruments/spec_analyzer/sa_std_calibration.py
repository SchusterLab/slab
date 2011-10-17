# -*- coding: utf-8 -*-
"""
Single Calibration Profile Generation 
The saved profile will cover the whole known dynamic range of the device. 
So it gives the bounds on frequency and power replacing the need for a profile
on minimal detectable power

Created on Tue Sep 06 12:30:19 2011

@author: Dai
"""
from spectrum_analyzer import *
from instruments import E8257D
import time
import numpy as np

def calibrate(sa, frequency, rf_power, lo, rf):
    """returns [frequency, sa ouput, rf_power] given frequency in Hz, rf_power in dBm"""
    #lo.set_output(False)
    #rf.set_output(False)    
    lo.set_power(sa.lo_power)
    lo.set_frequency(frequency+sa.lo_offset)
    rf.set_frequency(frequency)    
    rf.set_power(rf_power)
    lo.set_output()
    rf.set_output()
    time.sleep(0.2)
    return [frequency, sa.get_avg_power(), rf_power]

def find_min_pwr(sa, frequency, lo, rf, threshold=50.0,
              lower_bound=-80.0, upper_bound=10.0):
    """returns the minimal detectable rf power at given frequency.
    the threshold is the lower bound of the dynamic range of the output from sa"""
    #lo.set_output(False)
    #rf.set_output(False)
    lo.set_power(sa.lo_power)
    lo.set_frequency(frequency+sa.lo_offset)
    rf.set_frequency(frequency)
    lo.set_output()
    
    rp = upper_bound
    lp = lower_bound
    while (rp - lp > 0.2):
        rf.set_power((lp+rp)/2.0)
        rf.set_output(True)
        time.sleep(0.1)
        m = sa.get_avg_power()
        #rf.set_output(False)        
        if (m > 50.0):
            rp = (lp + rp) / 2.0
            #print 'rp = '+str(rp)
        else:
            lp = (lp + rp) / 2.0
            #print 'lp = '+str(lp)
    return (rp+lp)/2.0

def plot_output_power(sa, frequency, lo, rf, powers=range(-50, 10, 10)):
    """plotting sa output over RF power"""
    #lo.set_output(False)
    #rf.set_output(False)
    lo.set_power(sa.lo_power)
    lo.set_frequency(frequency+sa.lo_offset)
    rf.set_frequency(frequency)
    lo.set_output()
    
    op = []
    for pwr in powers:
        rf.set_power(pwr)
        #rf.set_output()
        time.sleep(0.1)
        op.append(sa.get_avg_power())
        #rf.set_output(False)
    #lo.set_output(False)
    plt.plot(np.array(powers), np.array(op))
    plt.autoscale()
    plt.show()

def plot_output_offset(sa, frequency, rf_pwr, lo, rf, offsets=np.arange(10e6,11e6,0.01e6)):
    """plotting the sa output over LO offset"""
    #lo.set_output(False)
    #rf.set_output(False)
    lo.set_power(sa.lo_power)
    lo.set_frequency(frequency)
    lo.set_output()
    rf.set_power(rf_pwr)
    
    op = []
    for of in offsets:
        rf.set_frequency(frequency+of)
        #rf.set_output()
        time.sleep(0.1)
        op.append(sa.get_avg_power())
        #rf.set_output(False)
    #lo.set_output(False)
    plt.plot(np.array(offsets), np.array(op))
    plt.autoscale()
    plt.show()

def remove_frequency(cali_data, frequency):
    for dp in cali_data:
        if dp[0] == frequency:
            cali_data.remove(dp)
    return cali_data  

if __name__ =='__main__':
    import pickle
    import matplotlib.pyplot as plt
    
    rf = E8257D(address='rfgen2.circuitqed.com')
    lo = E8257D(address='rfgen1.circuitqed.com')
    sa = SpectrumAnalyzer(protocol='serial', port=2)

    #listing all the desired frequencies
    fr = np.arange(2.5e9, 7.5e9, 0.1e9)
    
    #finding the minimal detectable powers at certain frequencies
    min_power = []

    for freq in fr:
        nd = [freq, find_min_pwr(sa, freq, lo, rf)]
        min_power.append(nd)
        print nd
    
    cali = []
    
    step = 5.0
    for freq, mp in min_power:
        for pwr in np.arange(mp, 14.0, step):
            nd = calibrate(sa, freq, pwr, lo, rf)
            cali.append(nd)
            print nd
    
    print cali
    pickle.dump(cali, open('10dBm_cali.data', 'w'))
    
    c = np.array(cali)
    c = c.transpose()
    plt.scatter(c[0], c[1])
    plt.show()