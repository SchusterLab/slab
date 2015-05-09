# -*- coding: utf-8 -*-
"""
Created on Sat Jul 02 12:01:55 2011

@author: Dave
"""

from numpy import *

def square_pulse(amp,length):
    return amp*ones(length)

def delay(length):
    return zeros(length)
    
def gauss(amp,sigma,cutoff_length=None):
    if cutoff_length is None: cutoff_length=3*sigma
    return amp*exp(-1.0*arange (-cutoff_length,cutoff_length)**2.0/(2.0*sigma**2.0))

def smooth_square(pulse_center, smooth_time, flat_time, pulse_height,total_length):

    arr=arange(0,total_length)
    pulse = zeros(total_length)

    pulse += pulse_height*(exp(-1.0*(arr-(pulse_center-flat_time/2))**2.0/(2.0*smooth_time**2.0))-exp(-2.0))/(1.0-exp(-2.0))*(arr<=(pulse_center-flat_time/2))*(arr>=(pulse_center-flat_time/2-2*smooth_time))
    pulse += pulse_height*(exp(-1.0*(arr-(pulse_center+flat_time/2))**2.0/(2.0*smooth_time**2.0))-exp(-2.0))/(1.0-exp(-2.0))*(arr>=(pulse_center+flat_time/2))*(arr<=(pulse_center+flat_time/2+2*smooth_time))
    pulse += pulse_height*((arr>=(pulse_center-flat_time/2))*(arr<=(pulse_center+flat_time/2)))
    return pulse

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

def spin_echo(amp,width,pause_between_pulses,total_length):
    return pad_left(concatenate((gauss(amp,width,cutoff_length=4*width),delay(pause_between_pulses-6*width),gauss(amp,2*width,cutoff_length=8*width),delay(pause_between_pulses-4*width+50000))),total_length)    
    
def T1(pi_amp,pi_width,wait,total_length):
    return pad_left(concatenate((gauss(pi_amp,pi_width,cutoff_length=3*pi_width),delay(wait))),total_length)

#def shift_pulse_frequency(pulse, freq,phase):

if __name__=="__main__":
    from guiqwt.pyplot import *
    total_length=400
    sigma=3
    pi2_amp=10
    mdelay=10
    pi_amp=20
    
    numexpts=50
    
    #spin_sweep=array([spin_echo(20,5,a+6*20,total_length) for a in range (numexpts)])
    ramsey_sweep=array([ramsey(pi2_amp,sigma,d,mdelay,total_length) for d in range (numexpts)])
    rabi_sweep=array([rabi(a,sigma,mdelay,total_length) for a in range (numexpts)])
    T1_sweep = array([T1(pi_amp,sigma,d,total_length) for d in range (numexpts)])
    
    #RRTpulses=hstack((ramsey_sweep,rabi_sweep,T1_sweep,spin_sweep))
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
