import visdom
import numpy as np
from scipy import interpolate

''' Started 7/8/2021 by Meg to address issue of synchronizing pulses on multiple qubits from the pulse backs rather than 
fronts.  Idea is to construct the array, and the t0 and t_array that provide relative phase, and then come back around 
laterto multitply by a cosine with the appropriate frequency. So then pulse chains constructed from the back 
(readout pulse) forward won't have weird issues with relative phase alignment. '''

''' Note that the pulses have a new variable align_side to assert when you're calling them compared to pulses from the 
pulse_classes document!'''

class BackwardsPulse:

    def __init__(self):
        pass

    def plot_backwards_pulse(self):
        vis = visdom.Visdom()
        win = vis.line(
            X=np.arange(0, len(self.pulse_array)) * self.dt,
            Y=self.pulse_array)
        # Think about if pulse array needs to be flipped here

    def generate_backwards_pulse_array(self, t0=0, dt=None):
        if dt is not None:
            self.dt = dt
        if self.dt is None:
            raise ValueError('dt is not specified.')

        self.t0 = t0

        total_length = self.get_backwards_length()
        self.t_array = self.get_backwards_t_array(total_length)

        # Some of these functions like get_pulse_array and get_length are defined in derative classes below.
        # Pulse is the base class. Redundant functions are going to be overwritten by ones in derivative classes.
        self.pulse_array = self.get_backwards_pulse_array()
        self.pulse_array = self.scale_by_cosine(self.pulse_array)

        if self.plot:
            self.plot_backwards_pulse()

    # Be careful with this and consider the orientation of time
    def get_backwards_t_array(self, total_length):
        return np.arange(0, total_length, self.dt) + self.t0

class Gauss_B(BackwardsPulse):
    def __init__(self, max_amp, sigma_len, cutoff_sigma, freq, phase, dt=None, plot=False, align_side='Front'):
        self.max_amp = max_amp
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot
        self.align_side = align_side

        self.t0 = 0

    def get_backwards_pulse_array(self):

        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        # Flip isn't strictly necessary if it's symmetric but good to think ahead
        pulse_array = np.flip(pulse_array, axis=0)
        # Spun cosine part out to separate function below
        # pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array


    # Multiplies by cosine scooted over by phase. Phase is added by default in back alignment so a negative shift.
    # Front aligned cosine gets flipped backwards so that phase shift is from the ultimate pulse front
    def scale_by_cosine(self, pulse_array_in):
        # Phase offset sign is ambiguous -- was positive which corresp. to a negative shift?
        # Keep positive as normal phase for back alignment
        if self.align_side == 'Front':
            # Flip time backwards for the purposes of applying phase to the array
            cos_array = np.cos(2 * np.pi * self.freq * self.t_array + self.phase)
            cos_array = np.flip(cos_array, axis=0)
            pulse_array = pulse_array_in * cos_array
        elif self.align_side == 'Back':
            # Just have phase propagate from late to early time, with the direction of extant t_array
            # Keep
            pulse_array = pulse_array_in * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)
        else:
            print("You have a problem with invoking alignment options in your pulse class, check yourself")
            # Default to back alignment since we're constructing pulses from the back anyways here
            pulse_array = pulse_array_in * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)
        return pulse_array

    def get_backwards_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len

class Square_B(BackwardsPulse):

    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0=0, fix_phase=True,
                 dc_offset=0, dt=None, plot=False, align_side='Front'):

        self.max_amp = max_amp
        self.flat_len = flat_len
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot
        self.fix_phase = fix_phase
        self.dc_offset = dc_offset
        self.align_side = align_side

        self.t0 = 0

    def get_backwards_pulse_array(self):

        t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
        t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len
        t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

        pulse_array = self.max_amp * (
                    (self.t_array >= t_flat_start) * (
                    self.t_array < t_flat_end) +  # Normal square pulse
                    (self.t_array >= self.t0) * (self.t_array < t_flat_start) * np.exp(
                -1.0 * (self.t_array - (t_flat_start)) ** 2 / (
                        2 * self.ramp_sigma_len ** 2)) +  # leading gaussian edge
                    (self.t_array >= t_flat_end) * (
                            self.t_array <= t_end) * np.exp(
                -1.0 * (self.t_array - (t_flat_end)) ** 2 / (
                        2 * self.ramp_sigma_len ** 2))  # trailing edge
            )

        pulse_array = np.flip(pulse_array, axis=0)

        return pulse_array

    def scale_by_cosine(self, pulse_array_in):

        if self.align_side == 'Front':
            if not self.fix_phase:
                cos_array = np.cos(
                    2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)
                cos_array = np.flip(cos_array, axis=0)
                pulse_array = pulse_array_in * cos_array
            else:
                cos_array = np.cos(2 * np.pi * (self.freq + self.dc_offset) * (
                        self.t_array - self.phase_t0) + self.phase - 2 * np.pi * (self.dc_offset) * (
                                                              self.t_array - self.t_array[0]))
                cos_array = np.flip(cos_array, axis=0)
                pulse_array = pulse_array_in * cos_array
        elif self.align_side == 'Back':
            # Just keep the same
            if not self.fix_phase:
                pulse_array = pulse_array_in * np.cos(
                    2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)
            else:
                pulse_array = pulse_array_in * np.cos(2 * np.pi * (self.freq + self.dc_offset) * (
                        self.t_array - self.phase_t0) + self.phase - 2 * np.pi * (self.dc_offset) * (
                                                              self.t_array - self.t_array[0]))
        else:
            if not self.fix_phase:
                pulse_array = pulse_array_in * np.cos(
                    2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)
            else:
                pulse_array = pulse_array_in * np.cos(2 * np.pi * (self.freq + self.dc_offset) * (
                        self.t_array - self.phase_t0) + self.phase - 2 * np.pi * (self.dc_offset) * (
                                                              self.t_array - self.t_array[0]))

        return pulse_array

    def get_backwards_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len


class Ones_B(BackwardsPulse):
    def __init__(self, time, dt=None, plot=False):
        self.time = time
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_backwards_pulse_array(self):
        pulse_array = np.ones_like(self.t_array)
        return pulse_array
    # Nothing about this has to be changed or flipped ?
    # Time is a length of the array in ones if you check in sequences

    def get_backwards_length(self):
        return self.time


class Idle_B(BackwardsPulse):
    def __init__(self, time, dt=None, plot=False):
        self.time = time
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_backwards_pulse_array(self):
        pulse_array = np.zeros_like(self.t_array)

        return pulse_array

    def get_backwards_length(self):
        return self.time