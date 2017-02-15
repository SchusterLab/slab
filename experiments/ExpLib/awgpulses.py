__author__ = 'dave'

import numpy as np


def sideband(t, plus, minus, freq=0, phase=0, offset=False, offset_fit_lin=0,offset_fit_quad=0,t0=0):
    if offset:
        if (not max(plus) == 0):
            time_step = t[1]-t[0]
            freq_calibrated = getFreq(plus,freq,offset_fit_lin,offset_fit_quad);
            freq_integ_array = np.cumsum(freq_calibrated)*time_step
            # np.savetxt('time.out', t, delimiter=',')
        return ( np.cos(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * minus,
             +np.sin(2 * np.pi * (freq_integ_array/1.0e9)+ phase*np.pi/180.0) * plus + np.sin(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * minus)
    else:
        # For ML0218 Mixer
        return ( np.cos(2 * np.pi * (freq/1.0e9 * (t-t0))+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * (t-t0)) + phase*np.pi/180.0) * minus,
             +np.sin(2 * np.pi * (freq/1.0e9 * (t-t0))+ phase*np.pi/180.0) * plus +np.sin(2 * np.pi * (freq/1.0e9 * (t-t0)) + phase*np.pi/180.0) * minus)
        #
        # For IQ0317 Mixer
        # return ( np.cos(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus,
        #      -np.sin(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.sin(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus)



def getFreq(pulse,freq,offset_fit_lin=0,offset_fit_quad=0):
    #time is a point
    #pulse is an array

    abs_pulse = abs(pulse)
    max_a = max(abs_pulse)

    freq_list = freq + (offset_fit_lin*abs_pulse+offset_fit_quad*abs_pulse**2) - (offset_fit_lin*max_a+offset_fit_quad*max_a**2)
    freq_array = np.array(freq_list)

    ## This saves the freq and pulse data for the churpped pulse for analysis
    # if (not max_a == 0):
    #     np.savetxt('freq.out', freq_array, delimiter=',')
    #     np.savetxt('pulse.out', pulse, delimiter=',')
    # print "done"
    return freq_array


def gauss(t, a, t0, sigma):
    if sigma >0:
        return (t >= t0-2*sigma) * (t <= t0+2*sigma)*a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2))
    else:
        return 0*(t-t0)


def dgauss(t, a, t0, sigma):
    return a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2)) * (t - t0) / sigma ** 2


def ramp(t, a, t0, w):
    return a * (t - t0) * (t >= t0) * (t < t0 + w)


def square(t, a, t0, w, sigma=0):
    if sigma>0:
        return a * (
            (t >= t0) * (t < t0 + w) +  # Normal square pulse
            (t >= t0-2*sigma) * (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
            (t >= t0 + w)* (t <= t0+w+2*sigma) * np.exp(-(t - (t0 + w)) ** 2 / (2 * sigma ** 2))  # trailing edge
        )
    else:
        return a * (t >= t0) * (t < t0 + w)


def trapezoid(t, a, t0, w, edge_time=0):
    return a * (
        (t - t0) * (t >= t0) * (t < t0 + edge_time) + (t >= t0 + edge_time) * (t < t0 + edge_time + w) + (
            t0 - t) * (
            t >= t0 + w + edge_time) * (
            t >= t0 + w + 2 * edge_time) )


def get_pulse_span_length(cfg, type, length, ramp_sigma = None):
    if type == "gauss":
        return length * 4 +cfg['spacing'] ## 4 sigma
    if type == "square":
        if ramp_sigma == None:
            return length + 4 * cfg[type]['ramp_sigma'] +cfg['spacing']
        else:
            return length + 4 * ramp_sigma +cfg['spacing']