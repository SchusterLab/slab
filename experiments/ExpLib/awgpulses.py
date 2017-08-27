__author__ = 'dave'

import numpy as np
import numexpr as ne


def sideband(t, plus, minus, freq=0, phase=0, offset=False, offset_fit_lin=0,offset_fit_quad=0):
    if freq==0 and phase==0:
        return (plus - minus)

    elif offset:
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
        # return ( np.cos(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus,
        #      -np.sin(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.sin(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus)

        wts = ne.evaluate('2 * (freq/1.0e9 * t)+ phase/180.0') * np.pi
        cosdata = ne.evaluate('cos(wts)')
        sindata = ne.evaluate('sin(wts)')
        # return ne.evaluate('cosdata * (plus - minus)'), ne.evaluate('- sindata * (plus + minus)')
        return ne.evaluate('cosdata * (plus - minus)'), ne.evaluate('- sindata * (plus + minus)')



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
    vv = np.zeros(t.shape)
    if sigma > 0:
        start = np.searchsorted(t, t0 - 2 * sigma, side='left')
        stop = np.searchsorted(t, t0 + 2 * sigma, side='right') # include both ends <=/<=
        t = t[start:stop]
        vv[start:stop] = a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2))
    return vv

def square(t, a, t0, w, sigma=0):
    vv = np.zeros(t.shape)
    if w > 0:
        start = np.searchsorted(t, t0 - 2 * sigma, side='left')
        stop = np.searchsorted(t, t0 + w + 2 * sigma, side='right') # include both ends <=/<=
        t = t[start:stop]
        if sigma > 0:
            vv[start:stop] = a * (
                (t >= t0) * (t < t0 + w) +  # Normal square pulse
                (t >= t0 - 2 * sigma) * (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
                (t >= t0 + w) * (t <= t0 + w + 2 * sigma) * np.exp(-(t - (t0 + w)) ** 2 / (2 * sigma ** 2)) # trailing edge
                )
        else:
            vv[start:stop] = a * (t >= t0) * (t < t0 + w)
    return vv

def linear_ramp(t, start_a, stop_a, t0, w):
    vv = np.zeros(t.shape)
    if w > 0:
        start = np.searchsorted(t, t0, side='left')
        stop = np.searchsorted(t, t0 + w, side='left') # include only left end <=/<
        t = t[start:stop]
        vv[start:stop] = ( (stop_a - start_a) * (t - t0) / w + start_a)
    return vv

def linear_ramp_with_mod(t, start_a, stop_a, t0, w, mod_amp, mod_freq, mod_start_phase):
    vv = np.zeros(t.shape)
    if w > 0:
        start = np.searchsorted(t, t0, side='left')
        stop = np.searchsorted(t, t0 + w, side='left') # include only left end <=/<
        t = t[start:stop]
        vv[start:stop] = ( (stop_a - start_a) * (t - t0) / w + start_a +
                           mod_amp * np.sin( 2*np.pi*(mod_freq/1.0e9)*(t-t0) + mod_start_phase/180.0 )
                           )
    return vv

def logistic_ramp(t, start_a, stop_a, t0, w):
    # smooth S-shaped ramp using logistic function
    # truncated and rescaled to be continuous
    # bigger r - steeper slope
    vv = np.zeros(t.shape)
    if w > 0:
        start = np.searchsorted(t, t0, side='left')
        stop = np.searchsorted(t, t0 + w, side='left') # include only left end <=/<
        t = t[start:stop]
        r = 8.0 / w
        scale = (1 + np.exp(r * w / 2.0)) / (-1 + np.exp(r * w / 2.0))
        vv[start:stop] = (((start_a + stop_a * np.exp(r * (t - (t0 + w / 2.0)) ))
                           / (1 + np.exp(r * (t - (t0 + w / 2.0)) )) - (start_a + stop_a) / 2.0)
                            * scale + (start_a + stop_a) / 2.0)
    return vv



####
## the following are old (unused?) pulses:

def dgauss(t, a, t0, sigma):
    return a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2)) * (t - t0) / sigma ** 2

def ramp(t, a, t0, w):
    return a * (t - t0) * (t >= t0) * (t < t0 + w)

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

        if type in ["ramp", "linear_ramp", "linear_ramp_with_mod", "logistic_ramp"]:
            return length
    else:
        return 0.0

def get_pulse_area(type=None, length=0, a=0, start_a=0, stop_a=0):

    if type in ["linear_ramp", "linear_ramp_with_mod", 'logistic_ramp']:
        return (start_a + stop_a)/2.0*length
    else:
        return 0.0