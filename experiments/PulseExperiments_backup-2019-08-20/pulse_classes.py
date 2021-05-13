import visdom
import numpy as np
from scipy import interpolate


class Pulse:
    def __init__(self):
        pass


    def plot_pulse(self):
        vis = visdom.Visdom()
        win = vis.line(
            X=np.arange(0, len(self.pulse_array)) * self.dt,
            Y=self.pulse_array
        )

    def generate_pulse_array(self, t0=0, dt=None):
        if dt is not None:
            self.dt = dt
        if self.dt is None:
            raise ValueError('dt is not specified.')

        self.t0 = t0

        total_length = self.get_length()
        self.t_array = self.get_t_array(total_length)

        self.pulse_array = self.get_pulse_array()

        if self.plot:
            self.plot_pulse()

    def get_t_array(self, total_length):
        return np.arange(0, total_length, self.dt) + self.t0


class Gauss(Pulse):
    def __init__(self, max_amp, sigma_len, cutoff_sigma, freq, phase, dt=None, plot=False):
        self.max_amp = max_amp
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot

        self.t0 = 0


    def get_pulse_array(self):

        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array

    def get_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len


class Square(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, dt=None, plot=False):
        self.max_amp = max_amp
        self.flat_len = flat_len
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

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

        pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)

        return pulse_array

    def get_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len


class DRAG(Pulse):
    def __init__(self, A, beta, sigma_len, cutoff_sigma, freq, phase, dt=None, plot=False):
        self.A = A
        self.beta = beta
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        t_center = self.t0 + 2 * self.sigma_len

        pulse_array_x = self.A * np.exp(
            -1.0 * (self.t_array - t_center) ** 2 / (2 * self.sigma_len ** 2))
        pulse_array_y = self.beta * (-(self.t_array - t_center) / (self.sigma_len ** 2)) * self.A * np.exp(
            -1.0 * (self.t_array - t_center) ** 2 / (2 * self.sigma_len ** 2))

        pulse_array = pulse_array_x * np.cos(2 * np.pi * self.freq * self.t_array + self.phase) + \
                      - pulse_array_y * np.sin(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array

    def get_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len


class ARB(Pulse):
    def __init__(self, A_list, B_list, len, freq, phase, dt=None, plot=False):
        self.A_list = np.pad(A_list, (1, 1), 'constant', constant_values=(0, 0))
        self.B_list = np.pad(B_list, (1, 1), 'constant', constant_values=(0, 0))
        self.len = len
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):
        t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list))
        t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list))

        tck_A = interpolate.splrep(t_array_A_list, self.A_list)
        tck_B = interpolate.splrep(t_array_B_list, self.B_list)

        pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
        pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

        pulse_array = pulse_array_x * np.cos(2 * np.pi * self.freq * self.t_array + self.phase) + \
                      - pulse_array_y * np.sin(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array

    def get_length(self):
        return self.len


class ARB_freq_a(Pulse):
    def __init__(self, A_list, B_list, len, freq_a_fit, phase, delta_freq = 0, dt=None, plot=False):
        self.A_list = A_list
        self.B_list = B_list
        self.len = len
        self.freq_a_fit = freq_a_fit
        self.phase = phase
        self.delta_freq = delta_freq
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):
        t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list))
        t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list))

        # # spline
        # tck_A = interpolate.splrep(t_array_A_list, self.A_list)
        # tck_B = interpolate.splrep(t_array_B_list, self.B_list)
        #
        # pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
        # pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

        # linear interpolation
        pulse_array_x = np.interp(self.t_array,t_array_A_list,self.A_list)
        pulse_array_y = np.interp(self.t_array,t_array_B_list,self.B_list)

        freq_array_x = self.freq_a_fit(pulse_array_x) + self.delta_freq
        freq_array_y = self.freq_a_fit(pulse_array_y) + self.delta_freq

        # print(freq_array_x)
        # print(freq_array_y)
        # print(self.dt)

        phase_array_x = 2*np.pi*np.cumsum(freq_array_x*self.dt) + self.phase
        phase_array_y = 2*np.pi*np.cumsum(freq_array_y*self.dt) + self.phase

        pulse_array = pulse_array_x * np.cos(phase_array_x) + \
                      - pulse_array_y * np.sin(phase_array_y)

        return pulse_array

    def get_length(self):
        return self.len


class Ones(Pulse):
    def __init__(self, time, dt=None, plot=False):
        self.time = time
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):
        pulse_array = np.ones_like(self.t_array)

        return pulse_array

    def get_length(self):
        return self.time


class Idle(Pulse):
    def __init__(self, time, dt=None, plot=False):
        self.time = time
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):
        pulse_array = np.zeros_like(self.t_array)

        return pulse_array

    def get_length(self):
        return self.time


if __name__ == "__main__":
    vis = visdom.Visdom()
    vis.close()

    gauss = Gauss(max_amp=0.1, sigma_len=2, cutoff_sigma=2, freq=1, phase=0, dt=0.1, plot=True)
    gauss.generate_pulse_array()
    gauss = Gauss(max_amp=0.1, sigma_len=2, cutoff_sigma=2, freq=1, phase=np.pi / 2, dt=0.1, plot=True)
    gauss.generate_pulse_array()

    # test_pulse = Pulse(np.arange(0,10),0.1)