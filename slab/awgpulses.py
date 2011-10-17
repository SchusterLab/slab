# -*- coding: utf-8 -*-
"""
Created on Sat Jul 02 12:01:55 2011

@author: Dave
"""

from numpy import *
from guiqwt.pyplot import *

def square_pulse(amp,length):
    return amp*ones(length)

def delay(length):
    return zeros(length)
    
def gauss(amp,sigma,cutoff_length=None):
    if cutoff_length is None: cutoff_length=3*sigma
    return amp*exp(-1.0*arange (-cutoff_length,cutoff_length)**2.0/(2.0*sigma**2.0))
 
def pad_left(arr, length):
    return append(zeros(length-size(arr)),arr)
    
def pad_right(arr,length):
    return append(arr,zeros(length-size(arr)))
    
def ramsey(pi2_amp,pi2_width,wait,pause_before_measure,total_length):
    return pad_left(concatenate((gauss(pi2_amp,pi2_width,cutoff_length=3*pi2_width),
                           delay(wait),
                           gauss(pi2_amp,pi2_width,cutoff_length=3*pi2_width),
                           delay(pause_before_measure))),
                    total_length)

def rabi(amp,width,pause_before_measure,total_length):
    return pad_left(concatenate((gauss(amp,width,cutoff_length=3*width),delay(pause_before_measure))),total_length)

def T1(pi_amp,pi_width,wait,total_length):
    return pad_left(concatenate((gauss(pi_amp,pi_width,cutoff_length=3*pi_width),delay(wait))),total_length)

#def shift_pulse_frequency(pulse, freq,phase):


total_length=200
sigma=3
pi2_amp=10
mdelay=10
pi_amp=20

numexpts=100

ramsey_sweep=array([ramsey(pi2_amp,sigma,d,mdelay,total_length) for d in range (numexpts)])
rabi_sweep=array([rabi(a,sigma,mdelay,total_length) for a in range (numexpts)])
T1_sweep = array([T1(pi_amp,sigma,d,total_length) for d in range (numexpts)])

RRTpulses=hstack((ramsey_sweep,rabi_sweep,T1_sweep))


#pulses=array([co])    
#figure(1)
#plot(square_pulse(10,10))
#plot(gauss(10,3,cutoff_length=10))
#
#figure(2)
#plot(ramsey(10,3,100,10,200))

figure(3)
imshow(RRTpulses)

show()
