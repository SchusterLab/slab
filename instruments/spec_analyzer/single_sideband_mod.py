# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:07:26 2011

@author: Dai
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import pickle

#from instruments import AWG81180A
from slab.instruments import InstrumentManager
from spectrum_analyzer import *
from sa_calibration_manager import *

def set_DC(awg, channel, offset):
    awg.write('inst:sel '+str(channel)+'; function:shape DC;')
    awg.write('inst:sel '+str(channel)+'; DC '+str(offset))

def get_DC(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; DC ?'))
    
def get_output(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; OUTPUT ?'))
    
def set_output(awg, channel, output=True):
    awg.write('inst:sel '+str(channel)+'; OUTPUT '+str(int(output)))

def get_sync(awg):
    return bool(awg.query('inst:coup:stat ?'))

def set_sync(awg, sync=True):
    if sync:
        awg.write('inst:coup:stat on')
    else:
        awg.write('inst:coup:stat off')

def get_power(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; POW ?'))
    
def set_sin(awg, channel, amplitude, frequency, offset=0.0, phase=0.0):
    """@param amplitude DC coupled amplitude at mV"""
    awg.write('inst:sel '+str(channel)+'; volt:ampl '+str(amplitude/1.0e3))
    awg.write('inst:sel '+str(channel)+'; function:shape SIN;')
    awg.write('inst:sel '+str(channel)+'; SIN:phase '+str(phase))
    awg.write('inst:sel '+str(channel)+'; frequency '+str(frequency))
    awg.write('inst:sel '+str(channel)+'; volt:offset'+str(offset))

def get_power_at(sa, lo, frequency):
    lo.set_frequency(frequency+sa.lo_offset)
    return sa.get_avg_power()

if __name__ == '__main__':
    #awg = AWG81180A(name='awg', address='GPIB::04::INSTR')
    im = InstrumentManager("ssm_optimization.cfg" )    
    awg = im['AWG']
    #sa = SpectrumAnalyzer(protocol='serial', port=2)
    #Agilent analog waveform generator     
    #sacm = SACalibrationManager(pickle.load(open('10dBm_cali.data')))
    
    sa = im['SA']
    rf = im['RF']    
    lo = im['LO']
    sacm = SACalibrationManager(pickle.load(open('10dBm_cali.data')))
    
    #arbitrary setup parameters
    c_freq = 7e9
    m_freq = 0.1e9
    m_amp = 0.1
    
    #setup analog waveform generators
    rf.set_frequency(c_freq)
    rf.set_power(-3)
    lo.set_frequency(c_freq+sa.lo_offset)
    lo.set_power(10)
    
    #initialize search range
    d_step = 0.01
    p_step = 1.0
    #a_step = 
    d1s = np.arange(-0.05, 0.05, d_step)
    d2s = np.arange(-0.05, 0.05, d_step)
    p1s = np.arange(-10.0, 10.0, p_step)
    
    
    min_d1 = 0.0
    min_d2 = 0.0
    min_p1 = 0.0
    min_op = 0.0
    #max_op = 200.0
    
    #awg global setup
    #synchronize the two channels
    awg.write('inst:coup:stat on')
    #turn on both channels
    set_output(awg, 1)
    set_output(awg, 2)
    
    #plotting
    #zs = []
    #X = []
    #Y = []    
    
    """search for minimal (d1, d2) config"""
    min_op = 0.0
    while d_step >= 0.001:
        if d_step != 0.01 and d_step >= 0.001:
            d1s = np.arange(min_d1-d_step*9.0, min_d1+d_step*9.0, d_step)
            d2s = np.arange(min_d2-d_step*9.0, min_d2+d_step*9.0, d_step)
            
        for d2 in d2s:
            for d1 in d1s:
                """set the awg ch1 to the optimal p1"""
                set_sin(awg, 1, m_amp, m_freq, d1, 90.0+min_p1)
                set_sin(awg, 2, m_amp, m_freq, d2, 0.0)
                print 'd1 = '+str(d1)+', d2 = '+str(d2)
                time.sleep(0.05)
                #evaluation the configuration
                #lsb_pwr = get_power_at(sa, lo, c_freq-m_freq)
                c_pwr = get_power_at(sa, lo, c_freq)
                #usb_pwr = get_power_at(sa, lo, c_freq+m_freq)
                op = c_pwr
                if min_op == 0.0:
                    min_op = op
                    min_d1 = d1
                    min_d2 = d2
                if op < min_op:
                    min_op = op
                    min_d1 = d1
                    min_d2 = d2
                    #min_p1 = p1
                #if pwr > max_op:
                #    max_op = pwr
                #print 'op = '+str(op)+': lsb_pwr = '+str(lsb_pwr)+', c_pwr = '+str(c_pwr)+', usb_pwr = '+str(usb_pwr)
                print 'c_pwr = '+str(c_pwr)

        #print 'd_step = '+str(d_step)+' p_step = '+str(p_step)+' minimal (d1, d2, p1) = ('+str(min_d1)+', '+str(min_d2)+', '+str(min_p1)+')'
        print 'best (d1, d2) = ('+str(min_d1)+', '+str(min_d2)+') at c_pwr = '+str(min_op)
        d_step *= 0.1    
    
    """search for optimal (p1) config"""
    min_op = 0.0
    while p_step >= 0.1:
        if p_step != 1.0 and p_step >= 0.1:
            p1s = np.arange(min_p1-p_step*9.0, min_p1+p_step*9.0, p_step)
        
        for p1 in p1s:
            set_sin(awg, 1, m_amp, m_freq, min_d1, 90.0 + p1)
            set_sin(awg, 2, m_amp, m_freq, min_d2, 0.0)
            print 'p1 = '+str(p1)
            """awg takes time to adjust"""
            time.sleep(0.05)
            try:
                lsb_pwr = sacm.get_rf_power(c_freq-m_freq, 
                                            get_power_at(sa, lo, c_freq-m_freq))
                #c_pwr = get_power_at(sa, lo, c_freq)
            except OutputOutOfRangeError as e:
                lsb_pwr = e.lower_bound_pwr
            
            try:
                usb_pwr = sacm.get_rf_power(c_freq+m_freq, 
                                            get_power_at(sa, lo, c_freq+m_freq))
            except OutputOutOfRangeError as e:
                usb_pwr = e.lower_bound_pwr            
                    
            """seek to maximize the difference between the lsb and the usb, i.e. down tune"""
            op = -(10.0**(lsb_pwr/10.0) - 10.0**(usb_pwr/10.0))
            if min_op == 0.0:
                min_op = op
                min_p1 = p1
            if op < min_op:
                min_op = op
                min_p1 = p1
            print 'lsb_pwr = '+str(lsb_pwr)
            print 'usb_pwr = '+str(usb_pwr)
        print 'best p1 = '+str(min_p1)
        p_step *= 0.1
    

    """use the optimal config"""        
    set_sin(awg, 1, m_amp, m_freq, min_d1, 90.0+min_p1)
    set_sin(awg, 2, m_amp, m_freq, min_d2, 0.0)
        
    """
    #print out the final result        
    print 'MINIMAL (d1, d2, p1) = ('+str(min_d1)+', '+str(min_d2)+', '+str(min_p1)+') at lsb_pwr = '+str(lsb_pwr)
    
    set_sin(awg, 1, m_amp, m_freq, min_d1, 90.0 + min_p1)
    set_sin(awg, 2, m_amp, m_freq, min_d2, 0.0)
    
    lsb_pwr = get_power_at(sa, lo, c_freq-m_freq)
    c_pwr = get_power_at(sa, lo, c_freq)
    usb_pwr = get_power_at(sa, lo, c_freq+m_freq)
    
    print 'lsb_pwr = '+str(lsb_pwr)+', c_pwr = '+str(c_pwr)+', usb_pwr = '+str(usb_pwr)
    
    
    print 'I on, Q off'
    set_sin(awg, 1, m_amp, m_freq, 0, 90.0)
    set_sin(awg, 2, 0.0, m_freq, 0, 0.0)
    time.sleep(0.05)
    
    lsb_pwr = sacm.get_rf_power(c_freq-m_freq, get_power_at(sa, lo, c_freq-m_freq))
    c_pwr = sacm.get_rf_power(c_freq, get_power_at(sa, lo, c_freq))
    usb_pwr = sacm.get_rf_power(c_freq+m_freq, get_power_at(sa, lo, c_freq+m_freq))
    
    #lsb_pwr = get_power_at(sa, lo, c_freq-m_freq)
    #c_pwr = get_power_at(sa, lo, c_freq)
    #usb_pwr = get_power_at(sa, lo, c_freq+m_freq)    
    
    print 'lsb_pwr = '+str(lsb_pwr)+', c_pwr = '+str(c_pwr)+', usb_pwr = '+str(usb_pwr)
    
    print 'I off, Q on'
    set_sin(awg, 1, 0.0, m_freq, 0, 90.0)
    set_sin(awg, 2, m_amp, m_freq, 0, 0.0)
    time.sleep(0.05)
    
    lsb_pwr = get_power_at(sa, lo, c_freq-m_freq)
    c_pwr = get_power_at(sa, lo, c_freq)
    usb_pwr = get_power_at(sa, lo, c_freq+m_freq)
    
    print 'lsb_pwr = '+str(lsb_pwr)+', c_pwr = '+str(c_pwr)+', usb_pwr = '+str(usb_pwr)
    """
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlabel('Q offset (V)')
    #ax.set_ylabel('I offset (V)')
    #ax.set_zlabel('RF power (dBm)')
    #ax.scatter(np.array(X),np.array(Y),np.array(zs))
    #ax.plot_surface(np.array(X),np.array(Y),np.array(zs))
    