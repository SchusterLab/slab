__author__ = 'dave'

import numpy as np


def sideband(t, plus, minus, freq=0, phase=0, offset=False, offset_fit_lin=0,offset_fit_quad=0):
    if offset:
        if (not max(plus) == 0):
            time_step = t[1]-t[0]
            freq_calibrated = getFreq(plus,freq,offset_fit_lin,offset_fit_quad);
            freq_integ_array = np.cumsum(freq_calibrated)*time_step
            # np.savetxt('time.out', t, delimiter=',')
        return ( np.cos(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * minus,
             -np.sin(2 * np.pi * (freq_integ_array/1.0e9)+ phase*np.pi/180.0) * plus - np.sin(2 * np.pi * (freq_integ_array/1.0e9) + phase*np.pi/180.0) * minus)
    else:
        # For ML0218 Mixer
        # return ( np.cos(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus,
        #      +np.sin(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus +np.sin(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus)
        #
        # For IQ0317 Mixer
        return ( np.cos(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus,
             -np.sin(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.sin(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus)



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


def linear_ramp(t, start_a, stop_a, t0, w):
    if w > 0:
        return ( (stop_a - start_a) * (t - t0) / w + start_a) * (t >= t0) * (t < t0 + w)
    else:
        return 0*t

def logistic_ramp(t, start_a, stop_a, t0, w):
    # smooth S-shaped ramp using logistic function
    # truncated and rescaled to be continuous
    # bigger r - steeper slope
    if w > 0:
        r = 8.0 / w
        scale = (1 + np.exp(r*w/2.0))/(-1 + np.exp(r*w/2.0))
        return  (((start_a + stop_a * np.exp(r*(t-(t0+w/2.0))* (t >= t0) * (t < t0 + w)))/(1 + np.exp(r*(t-(t0+w/2.0))* (t >= t0) * (t < t0 + w)))-(start_a+stop_a)/2.0) \
                * scale + (start_a+stop_a)/2.0) * (t >= t0) * (t < t0 + w)
    else:
        return 0*t


def square(t, a, t0, w, sigma=0):
    if w > 0:
        if sigma>0:
            return a * (
                (t >= t0) * (t < t0 + w) +  # Normal square pulse
                (t >= t0-2*sigma) * (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
                (t >= t0 + w)* (t <= t0+w+2*sigma) * np.exp(-(t - (t0 + w)) ** 2 / (2 * sigma ** 2))  # trailing edge
            )
        else:
            return a * (t >= t0) * (t < t0 + w)
    else:
        return 0*t

def square_exp(t, a, t0, w, sigma=0, exponent=0):
    if w > 0:
        if sigma>0:
            return a * (
                (t >= t0) * (t < t0 + w) * np.exp( (t >= t0) * (t < t0 + w) * (t-t0) * exponent ) +  # Normal square pulse
                (t >= t0-2*sigma) * (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
                (t >= t0 + w)* (t <= t0+w+2*sigma) * np.exp(-(t - (t0 + w)) ** 2 / (2 * sigma ** 2)) * np.exp( w * exponent ) # trailing edge
            )
        else:
            return a * (t >= t0) * np.exp( (t >= t0) * (t < t0 + w) * (t-t0) * exponent )
    else:
        return 0 * t


def trapezoid(t, a, t0, w, edge_time=0):
    return a * (
        (t - t0) * (t >= t0) * (t < t0 + edge_time) + (t >= t0 + edge_time) * (t < t0 + edge_time + w) + (
            t0 - t) * (
            t >= t0 + w + edge_time) * (
            t >= t0 + w + 2 * edge_time) )


def get_pulse_span_length(cfg, type, length):

    if length > 0:

        if type == "gauss":
            return length * 4 + cfg['spacing'] ## 4 sigma + spacing
        if type == "square" or type == "square_exp":
            return length + 4 * cfg[type]['ramp_sigma'] +cfg['spacing']

        if type in ["ramp", "linear_ramp", "logistic_ramp"]:
            return length
    else:
        return 0.0
