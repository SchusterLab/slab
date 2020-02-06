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


class Square(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, fix_phase = True,dc_offset = 0, dt=None, plot=False):
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

        if not self.fix_phase:pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * (self.t_array - self.phase_t0) + self.phase)
        else:pulse_array = pulse_array * np.cos(2 * np.pi * (self.freq + self.dc_offset) * (self.t_array - self.phase_t0) + self.phase -2 * np.pi * (self.dc_offset)*(self.t_array - self.t_array[0]) )
        return pulse_array

    def get_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len


class Double_Square(Pulse):
    def __init__(self, max_amp1,max_amp2,flat_len1,delay,flat_len2, ramp_sigma_len, cutoff_sigma, freq1, freq2, phase1=0,
                 phase2=0, phase_t0 = 0, fix_phase = False,dc_offset = 0, dt=None, plot=False,doubling_trick = False):
        self.max_amp1 = max_amp1
        self.flat_len1 = flat_len1
        self.max_amp2 = max_amp2
        self.flat_len2 = flat_len2
        self.delay = delay
        self.fix_phase = fix_phase
        self.dc_offset = dc_offset
        self.doubling_trick = doubling_trick

        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq1 = freq1
        self.phase1 = phase1
        self.freq2 = freq2
        self.phase2 = phase2

        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        t_flat_start1 = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
        t_flat_end1 = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len1
        t_end1 = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len1

        t_start2 = t_end1 + self.delay
        t_flat_start2 = t_start2 + self.cutoff_sigma * self.ramp_sigma_len
        t_flat_end2 = t_start2 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len2
        t_end2 = t_start2 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len2


        pulse_array1 = self.max_amp1 * (
            (self.t_array >= t_flat_start1) * (
                self.t_array < t_flat_end1) +  # Normal square pulse
            (self.t_array >= self.t0) * (self.t_array < t_flat_start1) * np.exp(
                -1.0 * (self.t_array - (t_flat_start1)) ** 2 / (
                    2 * self.ramp_sigma_len ** 2)) +  # leading gaussian edge
            (self.t_array >= t_flat_end1) * (
                self.t_array <= t_end1) * np.exp(
                -1.0 * (self.t_array - (t_flat_end1)) ** 2 / (
                    2 * self.ramp_sigma_len ** 2))  # trailing edge
        )

        pulse_array2 = self.max_amp2 * (
            (self.t_array >= t_flat_start2) * (
                self.t_array < t_flat_end2) +  # Normal square pulse
            (self.t_array >= t_start2) * (self.t_array < t_flat_start2) * np.exp(
                -1.0 * (self.t_array - (t_flat_start2)) ** 2 / (
                    2 * self.ramp_sigma_len ** 2)) +  # leading gaussian edge
            (self.t_array >= t_flat_end2) * (
                self.t_array <= t_end2) * np.exp(
                -1.0 * (self.t_array - (t_flat_end2)) ** 2 / (
                    2 * self.ramp_sigma_len ** 2))  # trailing edge
        )

        if not self.doubling_trick:
            if not self.fix_phase:
                pulse_array = pulse_array1 * np.cos(2 * np.pi * (self.freq1) * (self.t_array - self.phase_t0) + self.phase1)+ pulse_array2 * np.cos(2 * np.pi * self.freq2 * (self.t_array - self.phase_t0) + self.phase2)
            else:
                pulse_array = pulse_array1 * np.cos(2 * np.pi * (self.freq1 + self.dc_offset) * (self.t_array- self.phase_t0) + self.phase1  -2 * np.pi * (self.dc_offset)*(self.t_array - self.t_array[0])) + pulse_array2 * np.cos(2 * np.pi * self.freq2 * (self.t_array - self.phase_t0) + self.phase2)
        else:
            pulse_array = pulse_array1 * np.cos(2 * np.pi * (self.freq1 + self.dc_offset) * (self.t_array - self.phase_t0) + self.phase1 - 2 * np.pi * (self.dc_offset) * (self.t_array - self.t_array[0])) + \
                          pulse_array2 * np.cos(2 * np.pi * 2*(self.freq1 + self.dc_offset) * (self.t_array - self.phase_t0) + self.phase2 +
                                                2*np.pi*(self.freq2 - 2*(self.freq1 + self.dc_offset))*(self.t_array-self.t_array[0]-self.flat_len1-self.delay -  2 * self.cutoff_sigma * self.ramp_sigma_len))
        return pulse_array

    def get_length(self):
        return self.flat_len1 + self.flat_len2 + self.delay +  4 * self.cutoff_sigma * self.ramp_sigma_len


class Square_two_tone(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq1, freq2, phase1 = 0, phase2 = 0, phase_t0 = 0, dt=None, plot=False):
        self.max_amp = max_amp
        self.flat_len = flat_len
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq1 = freq1
        self.freq2 = freq2
        self.phase1 = phase1
        self.phase2 = phase2
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

        pulse_array = pulse_array * (np.cos(2 * np.pi * self.freq1 * (self.t_array - self.phase_t0) + self.phase1) + np.cos(2 * np.pi * self.freq2*(self.t_array - self.phase_t0) + self.phase2))/2.0

        return pulse_array

    def get_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len

class Square_multitone(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freqs, phases = [0], phase_t0 = 0, dt=None, plot=False):
        self.max_amp = max_amp
        self.flat_len = flat_len
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freqs = freqs
        self.phases = phases
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
        s = np.zeros(len(self.t_array))
        for ii,freq in enumerate(self.freqs):s+=np.cos(2 * np.pi * freq * (self.t_array - self.phase_t0) + self.phases[ii])
        pulse_array = pulse_array * (s)
        return pulse_array

    def get_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len

class Square_with_blockade(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0,blockade_amp=0.0, blockade_pulse_type='square', blockade_freqs = [0],dt=None, plot=False):

        self.max_amp = max_amp
        self.flat_len = flat_len
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot

        self.blockade_amp = blockade_amp
        self.blockade_pulse_type = blockade_pulse_type
        self.blockade_freqs = blockade_freqs
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
        pulse_array = pulse_array * np.cos(2 * np.pi * self.freq*self.t_array + self.phase)

        if self.blockade_pulse_type.lower() == "gauss":
            self.sigma_len =  (2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len)/4.0
            blockade_pulse_array = self.blockade_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i] * self.t_array)
        elif self.blockade_pulse_type.lower() == "square":
            blockade_pulse_array = self.blockade_amp * (
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
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i]*self.t_array)
        else:
            print("blockade pulse not recognized, not blockading")

        return pulse_array

    def get_length(self):
        return self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len

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

        # pulse_array = self.max_amp * np.exp(
        #     -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)

        return pulse_array

    def get_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len


class Gauss_multitone(Pulse):
    def __init__(self, max_amp, sigma_len, cutoff_sigma, freqs, phases, dt=None, plot=False):
        self.max_amp = max_amp
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freqs = freqs
        self.phases = phases
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        # pulse_array = self.max_amp * np.exp(
        #     -1.0 * (self.t_array - (self.t0 + 2 * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))

        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        
        s = np.zeros(len(self.t_array))
        for ii,freq in enumerate(self.freqs):s+=np.cos(2 * np.pi * freq * (self.t_array - self.phase_t0) + self.phases[ii])
        pulse_array = pulse_array * (s)

        return pulse_array

    def get_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len


class Gauss_with_blockade(Pulse):
    def __init__(self, max_amp, sigma_len, cutoff_sigma, freq, phases, blockade_amp=0.0, blockade_pulse_type='square',
                 blockade_freqs=[0], dt=None, plot=False):
        self.max_amp = max_amp
        self.sigma_len = sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot
        self.blockade_amp = blockade_amp
        self.blockade_pulse_type = blockade_pulse_type
        self.blockade_freqs = blockade_freqs
        self.t0 = 0
        # default for square blockade pulse
        self.ramp_sigma_len = 0.001
        self.flat_len = 2 * self.cutoff_sigma * self.sigma_len - 2 * self.cutoff_sigma * self.ramp_sigma_len

    def get_pulse_array(self):
        pulse_array = self.max_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
        pulse_array = pulse_array * np.cos(2 * np.pi * self.freq * self.t_array + self.phase)

        if self.blockade_pulse_type.lower() == "gauss":
            blockade_pulse_array = self.blockade_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i] * self.t_array)
        elif self.blockade_pulse_type.lower() == "square":
            t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
            t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

            t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

            blockade_pulse_array = self.blockade_amp * (
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
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i]*self.t_array)
        else:
            print("blockade pulse not recognized, not blockading")

        return pulse_array

    def get_length(self):
        return 2 * self.cutoff_sigma * self.sigma_len


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
    def __init__(self, A_list, B_list, len, freq, phase, dt=None, plot=False, scale=1.0):
        self.A_list = np.pad(A_list, (1,1 ), 'constant', constant_values=(0, 0))
        self.B_list = np.pad(B_list, (1,1 ), 'constant', constant_values=(0, 0))

        self.len = len
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot
        self.scale = scale

        self.t0 = 0

    def get_pulse_array(self):
        t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list))
        t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list))

        tck_A = interpolate.splrep(t_array_A_list, self.A_list)
        tck_B = interpolate.splrep(t_array_B_list, self.B_list)
        
        # print (self.t_array[0],self.t_array[-1])

        pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
        pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

        # print ("freq = ", self.freq)

        pulse_array = (pulse_array_x * np.cos(2 * np.pi * self.freq * self.t_array + self.phase) + \
                      - pulse_array_y * np.sin(2 * np.pi * self.freq * self.t_array + self.phase)) * self.scale

        return pulse_array

    def get_length(self):
        return self.len


class ARB_with_blockade(Pulse):
    def __init__(self, A_list, B_list, len, freq, phase, blockade_amp=0.0, blockade_pulse_type='square',
                 blockade_freqs=[0], dt=None, plot=False, scale=1.0):
        self.A_list = np.pad(A_list, (1, 1), 'constant', constant_values=(0, 0))
        self.B_list = np.pad(B_list, (1, 1), 'constant', constant_values=(0, 0))

        self.len = len
        self.freq = freq
        self.phase = phase
        self.dt = dt
        self.plot = plot
        self.scale = scale

        self.blockade_amp = blockade_amp
        self.blockade_pulse_type = blockade_pulse_type
        self.blockade_freqs = blockade_freqs
        self.t0 = 0

        # blockade pulse default params
        self.cutoff_sigma = 2
        self.sigma_len = self.len / (2 * self.cutoff_sigma)
        self.ramp_sigma_len = 0.001
        self.flat_len = self.len - - 2 * self.cutoff_sigma * self.ramp_sigma_len

    def get_pulse_array(self):
        t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list))
        t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list))

        tck_A = interpolate.splrep(t_array_A_list, self.A_list)
        tck_B = interpolate.splrep(t_array_B_list, self.B_list)

        # print(self.t_array[0], self.t_array[-1])

        pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
        pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

        # print("freq = ", self.freq)

        pulse_array = (pulse_array_x * np.cos(2 * np.pi * self.freq * self.t_array + self.phase) + \
                       - pulse_array_y * np.sin(2 * np.pi * self.freq * self.t_array + self.phase)) * self.scale

        if self.blockade_pulse_type.lower() == "gauss":
            blockade_pulse_array = self.blockade_amp * np.exp(
            -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (2 * self.sigma_len ** 2))
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i] * self.t_array)
        elif self.blockade_pulse_type.lower() == "square":
            t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
            t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

            t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

            blockade_pulse_array = self.blockade_amp * (
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
            for i in range(len(self.blockade_freqs)):
                pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i]*self.t_array)
        else:
            print("blockade pulse not recognized, not blockading")

        return pulse_array

    def get_length(self):
        return self.len


class ARB_Sum(Pulse):
    def __init__(self, A_list_list, B_list_list, len, freq_list, phase_list, dt=None, plot=False, scale_list=[1.0]):
        self.A_list_list  = [np.pad(A_list, (1, 1), 'constant', constant_values=(0, 0)) for A_list in A_list_list]
        self.B_list_list = [np.pad(B_list, (1, 1), 'constant', constant_values=(0, 0)) for B_list in B_list_list]
        self.len = len
        self.freq_list = freq_list
        self.phase_list = phase_list
        self.dt = dt
        self.plot = plot
        self.t0 = 0
        self.scale_list = scale_list

    def get_pulse_array(self):
        pulse_array = np.zeros(len(self.t_array))
        for ii in range(len(self.A_list_list)):
            t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list_list[ii]))
            t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list_list[ii]))

            tck_A = interpolate.splrep(t_array_A_list, self.A_list_list[ii])
            tck_B = interpolate.splrep(t_array_B_list, self.B_list_list[ii])

            pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
            pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

            pulse_array += (pulse_array_x * np.cos(2*np.pi*self.freq_list[ii] * self.t_array + self.phase_list[ii]) - \
                           pulse_array_y * np.sin(2*np.pi*self.freq_list[ii] * self.t_array + self.phase_list[ii])) * \
                           self.scale_list[ii]

        return pulse_array

    def get_length(self):
        return self.len


class ARB_Sum_with_blockade(Pulse):
    def __init__(self, A_list_list, B_list_list, len, freq_list, phase_list, blockade_amp=0.0, blockade_pulse_type='square',
                 blockade_freqs=[0], dt=None, plot=False, scale_list=[1.0]):
        self.A_list_list  = [np.pad(A_list, (1, 1), 'constant', constant_values=(0, 0)) for A_list in A_list_list]
        self.B_list_list = [np.pad(B_list, (1, 1), 'constant', constant_values=(0, 0)) for B_list in B_list_list]
        self.len = len
        self.freq_list = freq_list
        self.phase_list = phase_list
        self.dt = dt
        self.plot = plot
        self.t0 = 0
        self.scale_list = scale_list

        self.blockade_amp = blockade_amp
        self.blockade_pulse_type = blockade_pulse_type
        self.blockade_freqs = blockade_freqs

        # blockade pulse default params
        self.cutoff_sigma = 2
        self.sigma_len = self.len / (2 * self.cutoff_sigma)
        self.ramp_sigma_len = 0.001
        self.flat_len = self.len - - 2 * self.cutoff_sigma * self.ramp_sigma_len

    def get_pulse_array(self):
        pulse_array = np.zeros(len(self.t_array))
        for ii in range(len(self.A_list_list)):
            t_array_A_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.A_list_list[ii]))
            t_array_B_list = np.linspace(self.t_array[0], self.t_array[-1], num=len(self.B_list_list[ii]))

            tck_A = interpolate.splrep(t_array_A_list, self.A_list_list[ii])
            tck_B = interpolate.splrep(t_array_B_list, self.B_list_list[ii])

            pulse_array_x = interpolate.splev(self.t_array, tck_A, der=0)
            pulse_array_y = interpolate.splev(self.t_array, tck_B, der=0)

            pulse_array += (pulse_array_x * np.cos(2*np.pi*self.freq_list[ii] * self.t_array + self.phase_list[ii]) - \
                           pulse_array_y * np.sin(2*np.pi*self.freq_list[ii] * self.t_array + self.phase_list[ii])) * \
                           self.scale_list[ii]

            if self.blockade_pulse_type.lower() == "gauss":
                blockade_pulse_array = self.blockade_amp * np.exp(
                    -1.0 * (self.t_array - (self.t0 + self.cutoff_sigma * self.sigma_len)) ** 2 / (
                                2 * self.sigma_len ** 2))
                for i in range(len(self.blockade_freqs)):
                    pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i] * self.t_array)
            elif self.blockade_pulse_type.lower() == "square":
                t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
                t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

                t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len

                blockade_pulse_array = self.blockade_amp * (
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
                for i in range(len(self.blockade_freqs)):
                    pulse_array += blockade_pulse_array * np.cos(2 * np.pi * self.blockade_freqs[i] * self.t_array)
            else:
                print("blockade pulse not recognized, not blockading")

        return pulse_array

    def get_length(self):
        return self.len


class ARB_freq_a(Pulse):
    def __init__(self, A_list, B_list, len, freq_a_fit, phase, delta_freq=0, dt=None, plot=False):
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