from qutip import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pd
from numpy import pi, sqrt, array

def cav_response_new(p, x):
    """(p[0]/p[1])/(-1/2*p[0]/p[1] - 1j*(x-p[0])"""
    ### p[0]=center freq, p[1]=kappa
    temp = (p[1])/(p[1] - 1j*(x-p[0]))
    return temp/max(abs(temp))

def erf_t(A, sig, tc, tb, t):
    #A-Amplitude, sig-Gaussian Filter Width, tc-Core Pulse length, tb - zero-amplitude buffer length
    return (A/2)*(sc.special.erf((t-tb)/(sqrt(2)*sig))-sc.special.erf((t-tc-tb)/(sqrt(2)*sig)))

cav_freq = 0 #cavity frequency
Q = 15000 #cavity quality factor
kappa = (7.79e9)/Q
read_width = 600e-9 #readout length  in (seconds)
dt = 1e-9
n_points = int(read_width/dt)

pad_factor = 10
t = dt*np.arange(0, n_points*pad_factor)
ifreq = np.fft.fftfreq(n_points*pad_factor, dt)
freq = np.fft.fftshift(ifreq)

desired_output = erf_t(1, 100e-9, 3200e-9, 1000e-9, t)

#Plot Desired Output
figure(figsize = (10, 6))
plot(t*1e6, desired_output)
xlabel('Time (us)', fontsize =14)
ylabel('Drive amplitude (MHz)', fontsize =14)
title('Cavity response', fontsize =14)

#Generate Opt Input Pulse
desired_output_ifft = fft.fft(desired_output,n_points*pad_factor)/n_points
desired_output_sfft = fft.fftshift(desired_output_ifft) #"sfft" denotes shifted spectrum to center at cav_freq

lorenz_c = cav_response_new([cav_freq,kappa],freq)
lorenz_g = cav_response_new([cav_freq+0.2e6,Q],freq)
lorenz_e = cav_response_new([cav_freq-0.2e6,Q],freq)
lorenz= (lorenz_g*lorenz_e)

input_sfft = desired_output_sfft/lorenz_c

output_sfft = input_sfft*lorenz_c
output_fft = fft.ifftshift(output_sfft)
output_pulse = fft.ifft(output_sfft)

input_fft= fft.ifftshift(input_sfft)
input_pulse = fft.ifft(input_fft)
flip=input_pulse[::-1]

opt_pulse = real(flip/abs(max(flip)))

#Plot Generated Input Pulse
figure(figsize = (10,6))
plot(t*1.0e6,opt_pulse)
xlabel("time (us)", fontsize = 14)
ylabel("Drive Amplitude (MHz)", fontsize = 14)
title('Input Pulse', fontsize = 14)