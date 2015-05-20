__author__ = 'dave'

import numpy as np


def sideband(t, i, q, freq=0, phase=0):
    return ( np.cos(2 * np.pi * (freq * t + phase)) * i - np.sin(2 * np.pi * (freq * t + phase)) * q,
             np.sin(2 * np.pi * (freq * t + phase)) * i + np.cos(2 * np.pi * (freq * t + phase)) * q)


def gauss(t, a, t0, sigma):
    return a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2))


def dgauss(t, a, t0, sigma):
    return a * np.exp(-1.0 * (t - t0) ** 2 / (2 * sigma ** 2)) * (t - t0) / sigma ** 2


def ramp(t, a, t0, w):
    return a * (t - t0) * (t >= t0) * (t < t0 + w)


def square(t, a, t0, w, sigma=0):
    if sigma>0:
        return a * (
            (t >= t0) * (t < t0 + w) +  # Normal square pulse
            (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
            (t >= t0 + w) * np.exp(-(t - t0 + w) ** 2 / (2 * sigma ** 2))  # trailing edge
        )
    else:
        return a * (t >= t0) * (t < t0 + w)


def trapezoid(t, a, t0, w, edge_time=0):
    return a * (
        (t - t0) * (t >= t0) * (t < t0 + edge_time) + (t >= t0 + edge_time) * (t < t0 + edge_time + w) + (
            t0 - t) * (
            t >= t0 + w + edge_time) * (
            t >= t0 + w + 2 * edge_time) )



