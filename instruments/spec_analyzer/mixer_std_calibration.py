# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:02:40 2011

@author: Dai
"""

import pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
import pickle

#from instruments import AWG81180A
from slab.instruments import InstrumentManager
from .spectrum_analyzer import *
from .sa_calibration_manager import *

from .single_sideband_mod import *


def int2amp(intensity):
    """convert intensity (in dBm) to amplitude (or more precisely sqrt(intensity))"""
    return 10.0**(intensity/20.0)

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
    
    """arbitrary setup parameters"""
    c_freq = 7e9
    m_freq = 0.1e9
    # amplitude in mV
    m_amp = 200.0
    
    """setup analog waveform generators"""
    rf.set_frequency(c_freq)
    # +13dBm LO to drive IQB0618
    rf.set_power(13)
    lo.set_frequency(c_freq+sa.lo_offset)
    lo.set_power(10)
    
    """awg global setup"""
    """synchronize the two channels"""
    awg.write('inst:coup:stat on')
    """turn on both channels"""
    set_output(awg, 1)
    set_output(awg, 2)

    """sweep DC offsets and collect carrier amplitudes"""
#    s_step = 0.001
#    s_start = -0.5
#    s_end = 0.5
#    
#    d1s, d2s = np.meshgrid(np.arange(s_start, s_end, s_step), np.arange(s_start, s_end, s_step))
#    
#    t1 = []
#    cs = []
#    for d2 in d2s:
#        set_sin(awg, 2, m_amp, m_freq, d2[0])
#        for d1 in d1s[0]:
#            set_sin(awg, 1, m_amp, m_freq, d1, 90.0)
#            time.sleep(0.003)
#            """carrier frequency"""
#            try:
#                c = int2amp(sacm.get_rf_power(c_freq-m_freq, get_power_at(sa, lo, c_freq-m_freq)))
#            except OutputOutOfRangeError as e:
#                c = int2amp(e.lower_bound_pwr)
#            
#            t1.append(c)
#            print 'd1, d2, c = '+repr((d1, d2[0], c))
#        cs.append(t1)
#        t1 = []
#    
    
    """sweep phase and collect sidebands amplitudes"""    
    s_step = 0.1
    s_start = -20.0
    s_end = 20.0

    ps = np.arange(s_start, s_end, s_step)

    plsbs = []
    pusbs = []
    i = 0
    for p in ps:
        set_sin(awg, 2, m_amp, m_freq, 0.0, 0.0)
        set_sin(awg, 1, m_amp, m_freq, 0.0, 90.0+p)
        time.sleep(0.1)
        """lower side band"""
        #print 'raw # = '+repr(get_power_at(sa, lo, c_freq-m_freq))
        try:
            plsbs.append(sacm.get_rf_power(c_freq-m_freq, 
                                            get_power_at(sa, lo, c_freq-m_freq)))
        except OutputOutOfRangeError as e:
            plsbs.append(e.lower_bound_pwr)

        """upper side band"""
        #print 'raw # = '+repr(get_power_at(sa, lo, c_freq+m_freq))
        try:
            pusbs.append(sacm.get_rf_power(c_freq+m_freq, 
                                            get_power_at(sa, lo, c_freq+m_freq)))
        except OutputOutOfRangeError as e:
            pusbs.append(e.lower_bound_pwr)
        
        print('p, lsb, usb = '+repr((p, plsbs[i], pusbs[i])))
        i += 1

#    """sweep I, Q and collect sidebands amplitudes"""    
#    s_step = 5.0
#    s_start = 50.0
#    s_end = 200.0
#
#    iis, qqs = np.meshgrid(np.arange(s_start, s_end, s_step), np.arange(s_start, s_end, s_step))
#
#    t1 = []
#    t2 = []
#    lsbs = []
#    usbs = []
#    for q in qqs:
#        set_sin(awg, 2, q[0], m_freq, 0.0, 0.0)
#        for i in iis[0]:
#            set_sin(awg, 1, i, m_freq, 0.0, 90.0)
#            time.sleep(0.01)
#            """lower side band"""
#            #print 'raw # = '+repr(get_power_at(sa, lo, c_freq-m_freq))
#            try:
#                lsb = int2amp(sacm.get_rf_power(c_freq-m_freq, 
#                                                get_power_at(sa, lo, c_freq-m_freq)))
#            except OutputOutOfRangeError as e:
#                lsb = int2amp(e.lower_bound_pwr)
#            t1.append(lsb)
#    
#            """upper side band"""
#            #print 'raw # = '+repr(get_power_at(sa, lo, c_freq+m_freq))
#            try:
#                usb = int2amp(sacm.get_rf_power(c_freq+m_freq, 
#                                                get_power_at(sa, lo, c_freq+m_freq)))
#            except OutputOutOfRangeError as e:
#                usb = int2amp(e.lower_bound_pwr)
#            t2.append(usb)        
#            
#            print 'i, q, lsb, usb = '+repr((i, q[0], lsb, usb))
#    
#        lsbs.append(t1)
#        usbs.append(t2)
#        t1 = []
#        t2 = []    


    """numpified for plotting"""
#    lsbs = np.array(lsbs)    
#    usbs = np.array(usbs)
#    cs = np.array(cs)
    plsbs = np.array(plsbs)
    pusbs = np.array(pusbs)
    
    """plots"""            
    """carrier amplitude over offsets"""
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#    ax.set_xlabel('Q offset (V)')
#    ax.set_ylabel('I offset (V)')
#    ax.set_zlabel('Carrier Amplitude')
#    #ax.scatter(d1, d2, cs)
#    ax.plot_wireframe(d1s, d2s, cs)
#    plt.show()
    
#    """lsb over IQ"""
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#    ax.set_xlabel('Q amplitude')
#    ax.set_ylabel('I amplitude')
#    ax.set_zlabel('LSB Amplitude')
#    #ax.scatter(d1, d2, cs)
#    ax.plot_wireframe(iis, qqs, lsbs)
#    plt.show()
#        
#    """usb over IQ"""
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#    ax.set_xlabel('Q amplitude')
#    ax.set_ylabel('I amplitude')
#    ax.set_zlabel('USB Amplitude')
#    #ax.scatter(d1, d2, cs)
#    ax.plot_wireframe(iis, qqs, usbs)
#    plt.show()

    """lsb over phase"""
#    ax.set_xlabel('phase (deg)')
#    ax.set_ylabel('LSB amplitude')
    plt.plot(ps, plsbs)
    plt.show()
    
    """usb over phase"""
#    ax.set_xlabel('phase (deg)')
#    ax.set_ylabel('LSB amplitude')
    plt.plot(ps, pusbs)
    plt.show()