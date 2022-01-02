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

def IF_window(p,x):
    ### p[0] = center freq, p[1] = window width
    temp = zeros(len(x)) + 1j*zeros(len(x))
    for ii in range(len(x)):
        if x[ii]>(p[0]-p[1]) and x[ii]<(p[0]+p[1]):
            temp[ii] = 1/sqrt(2)*(1+1j)
        else:
            pass
    return temp/max(abs(temp))

def erf_t(A, sig, tc, tb, t):
    #A-Amplitude, sig-Gaussian Filter Width, tc-Core Pulse length, tb - zero-amplitude buffer length
    return (A/2)*(sc.special.erf((t-tb)/(sqrt(2)*sig))-sc.special.erf((t-tc-tb)/(sqrt(2)*sig)))

def optimal_rr(readout_params, flat_top_amp):

    rr_f, rr_q, rr_len, dt, twochi = readout_params

    kappa = rr_f/rr_q
    cav_freq = 0
    n_points = int(rr_len/dt/10)
    pad_factor = 20
    t = dt * np.arange(0, n_points * pad_factor)
    ifreq = np.fft.fftfreq(n_points * pad_factor, dt)
    freq = np.fft.fftshift(ifreq)
    if_band = IF_window([cav_freq, 250e6], freq)
    """Hard cut-off to constrain the BW of the output pulse to be within the AWG BW"""
    #     pulse_len = 1/twochi
    pulse_len = rr_len

    """Find the sigma of rising waveform for a given flat top amplitude"""
    sigarray = np.arange(5e-9, 200e-9, 1.0e-9)
    ratio = []

    for sig in sigarray:
        desired_output = erf_t(1, sig, pulse_len, 100e-9, t)

        desired_output_ifft = np.fft.fft(desired_output, n_points * pad_factor)/n_points
        desired_output_sfft = np.fft.fftshift(desired_output_ifft) #"sfft" denotes shifted spectrum to center at cav_freq

        lorenz_c = cav_response_new([cav_freq, kappa], freq)

        input_sfft = (desired_output_sfft/lorenz_c)*if_band

        output_sfft = input_sfft*lorenz_c
        output_fft = np.fft.ifftshift(output_sfft)
        output_pulse = np.fft.ifft(output_sfft)

        input_fft = np.fft.ifftshift(input_sfft)
        input_pulse = np.fft.ifft(input_fft)

        flip = input_pulse[::-1]

        opt_pulse = np.real(flip)

        max_opt = np.max(opt_pulse)
        mid_opt = opt_pulse[int(len(opt_pulse)/2)]
        ratio.append(mid_opt/max_opt)


    ratio_targ = flat_top_amp
    ratio = np.array(ratio)
    index = np.argmin(abs(ratio-ratio_targ))

    desired_sigma = sigarray[index]
    print(desired_sigma)
    """Returns the sigma"""

    desired_output = erf_t(1, desired_sigma, pulse_len, 100e-9, t)

    #Generate Opt Input Pulse
    desired_output_ifft = np.fft.fft(desired_output,n_points*pad_factor)/n_points
    desired_output_sfft = np.fft.fftshift(desired_output_ifft) #"sfft" denotes shifted spectrum to center at cav_freq

    lorenz_c = cav_response_new([cav_freq, kappa], freq)
    #     lorenz_g = cav_response_new([rr_f + 0.2e6, Q], freq)
    #     lorenz_e = cav_response_new([rr_f - 0.2e6, Q], freq)
    #     lorenz = (lorenz_g * lorenz_e)

    input_sfft = (desired_output_sfft/lorenz_c)*if_band

    output_sfft = input_sfft * lorenz_c
    output_fft = np.fft.ifftshift(output_sfft)
    output_pulse = np.fft.ifft(output_sfft)

    input_fft= np.fft.ifftshift(input_sfft)
    input_pulse = np.fft.ifft(input_fft)
    flip=input_pulse[::-1]

    opt_pulse = np.real(flip)
    opt_pulse = opt_pulse/np.max(opt_pulse)
    #Find Desired Range
    result1 = np.where(opt_pulse < 0.001*flat_top_amp)[0]
    result2 = np.where(opt_pulse[::-1] < 0.0*flat_top_amp)[0]

    #Trim Pulse
    start_trim = result1[np.argmin(abs(np.argmax(opt_pulse)-result1))]
    end_trim = result2[np.argmin(abs(np.argmin(opt_pulse)-result2))]

    #     end_trim = result2[len(result2[0])-1]
    numb = end_trim - start_trim
    rem = numb % 4

    #Check Total Length is Multiple of 4 for AWG
    trim_pulse = opt_pulse[start_trim:end_trim - rem]
    check = len(trim_pulse) % 4
    if check !=0:
        print('Error: The final pulse is not a multiple of 4')

    return trim_pulse

"""Comments for the output pulse

1. Coarsely find the optimal amptidue for a square shape pulse which results in maximum g-e fidelity
2. Use that as the 'flat_top_amp' for the optimal shape

"""



read_params = [7.79016586*1e9, 9737, 3e-6, 1e-9, 1535e3]
s = optimal_rr(read_params, flat_top_amp=0.65)

plt.figure(dpi=300)
plt.plot(s)
plt.xlabel('Time (ns)')
plt.ylabel('AWG amp. (a.u.)')
# plt.xlim(2800, 3300)
plt.show()

print(len(s))