import visdom
import numpy as np
from scipy import interpolate

''' Started 7/8/2021 by Meg to address issue of simultaneous readout of qubits being fed different pulse chains. 
Idea is to construct the array, and the t0 and t_array that provide relative phase, and then come back around later
to multitply by a cosine with the appropriate frequency. So then pulse chains constructed from the back (readout pulse) 
forward won't have weird issues with relative phase alignment. 

It remains to be seen if we'll need to flip sequences ordering around too in order to be consistent about the 
construction of pulse chains. Start with pulse classes and work outwards.'''


class BackwardsPulse:

    def __init__(self):
        pass

    def plot_backwards_pulse(self):
        vis = visdom.Visdom()
        win = vis.line(
            X=np.arange(0, len(self.pulse_array)) * self.dt,
            Y=self.pulse_array)

    # Outcome is that sequencer.append calls this and sticks pulse array on a chain of pulses assigned to a specific channel
    # Think about orientation and inserts/flips out at sequencer.append
    # Unless asymm pulses in which think about it in pulse classes below first after sequencer.append gets rewritten
    # Excepting relative phase, and issues if pulses are asymmetric

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
    def __init__(self, max_amp, sigma_len, cutoff_sigma, freq, phase, dt=None, plot=False):
        self.max_amp = max_amp
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_backwards_pulse_array(self):

        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        # Spun this out to separate function below
        # pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array

    def scale_by_cosine(self, pulse_array_in):
        pulse_array = pulse_array_in * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)
        return pulse_array

    def get_backwards_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len

class Square_B(BackwardsPulse):

    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0=0, fix_phase=True,
                 dc_offset=0, dt=None, plot=False):

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

        return pulse_array

    def scale_by_cosine(self, pulse_array_in):

        if not self.fix_phase:
            pulse_array = pulse_array_in * np.cos(2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)
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