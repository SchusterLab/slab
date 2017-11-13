# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:21:02 2011

@author: Dai
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import pickle

#from instruments import AWG81180A
from slab.instruments import InstrumentManager
from .spectrum_analyzer import *
from .sa_calibration_manager import *

def set_DC(awg, channel, offset):
    awg.write('inst:sel '+str(channel)+'; function:shape DC; DC '+str(offset))

def get_DC(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; DC ?'))
    
def get_output(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; OUTPUT ?'))
    
def set_output(awg, channel, output=True):
    awg.write('inst:sel '+str(channel)+'; OUTPUT '+str(int(output)))
    
def get_power(awg, channel):
    return float(awg.query('inst:sel '+str(channel)+'; POW ?'))
    
def set_sin(awg, channel, intensity, frequency, phase=0.0):
    awg.write('inst:sel '+str(channel)+'; pow '+str(intensity)+
    '; function:shape SIN; SIN:phase '+str(phase)+'; frequency '+str(frequency))
    
if __name__ == '__main__':
    #awg = AWG81180A(name='awg', address='GPIB::04::INSTR')
    im = InstrumentManager("offset_optimization.cfg" )    
    awg = im['AWG']
    #sa = SpectrumAnalyzer(protocol='serial', port=2)
    #Agilent analog waveform generator     
    #sacm = SACalibrationManager(pickle.load(open('10dBm_cali.data')))
    #sacm = SACalibrationManager(pickle.load(open('10dBm_LMS_cali.data')))
    
    sa = im['SA']
    rf = im['RF']    
    lo = im['LO']
    
    rf.set_frequency(6e9)
    rf.set_power(-3)
    lo.set_frequency(6e9+sa.lo_offset)
    lo.set_power(10)    
    
    step = 0.1    
    xs = np.arange(-1.0, 1.0, step)
    ys = np.arange(-1.0, 1.0, step)
    zs = []
    X = []
    Y = []
    min_d1 = 0.0
    min_d2 = 0.0
    min_op = 200.0
    max_op = 100.0
    
    set_output(awg, 1)
    set_output(awg, 2)
    
    while step >= 0.001:
        if step != 0.1:
            xs = np.arange(min_d1-step*9.0, min_d1+step*9.0, step)
            ys = np.arange(min_d2-step*9.0, min_d2+step*9.0, step)
            
        for d2 in ys:
            for d1 in xs:
                set_DC(awg, 1, d1)
                set_DC(awg, 2, d2)
                print('d1 = '+str(d1)+', d2 = '+str(d2))
                #time.sleep(0.01)
                pwr = sa.get_avg_power()
                if pwr <= min_op:
                    min_op = pwr
                    min_d1 = d1
                    min_d2 = d2
                if pwr > max_op:
                    max_op = pwr
                print('pwr = '+str(pwr))
                try:
                    zs.append(pwr)
                    X.append(d1)
                    Y.append(d2)                
                    
                    #zs.append(sacm.get_rf_power(4.99e9, pwr))
                except OutputOutOfRangeError as e:
                    print(e)
                    #zs.append(sacm.get_rf_power(4.99e9, e.lower_bound))
        
        print('step = '+str(step)+' minimal (d1, d2) = ('+str(min_d1)+', '+str(min_d2)+')')
        step *= 0.1
    #X, Y = np.meshgrid(xs, ys)
    
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlabel('Q offset (V)')
    #ax.set_ylabel('I offset (V)')
    #ax.set_zlabel('RF power (dBm)')
    #ax.scatter(np.array(X),np.array(Y),np.array(zs))
    #ax.plot_surface(np.array(X),np.array(Y),np.array(zs))
    
    zs = [str((pwr-min_op)/(max_op-min_op)) for pwr in zs]
    plt.scatter(X, Y, c=zs)
    plt.show()