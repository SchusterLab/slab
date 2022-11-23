import visdom
import numpy as np
from scipy import interpolate
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\_Lib\python\slab\experiments\PulseExperiments')
import analyze_qudit




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


class opt_single_qudit(Pulse):
    def __init__(self, Omega, total_len, Ns, ratio_ge, ratio_ef, alpha, dt=None, plot=False):
        self.Omega = Omega
        self.total_len = total_len
        self.Ns = Ns
        self.ratio_ge = ratio_ge
        self.ratio_ef = ratio_ef
        self.alpha = alpha
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        pulser = analyze_qudit.Pulser(self.Omega, self.t_array, self.Ns)
        # rescaling alpha
        alpha_scaled = []
        for i in range(len(self.alpha)):
            if np.mod(i, 4) < 2:
                alpha_scaled.append(self.alpha[i] / self.ratio_ge)
            else:
                alpha_scaled.append(self.alpha[i] / self.ratio_ef)
        alpha_scaled = np.array(alpha_scaled)
        # advancing phase
        # alpha_scaled1 = []
        # for i in range(len(alpha_scaled)):
        #     if np.mod(i, 4) == 0:
        #         alpha_scaled1.append(
        #             alpha_scaled[i] * np.cos(self.Omega[0][0] * self.t0 * np.pi * 2) - alpha_scaled[i + 1] * np.sin(
        #                 self.Omega[0][0] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 1:
        #         alpha_scaled1.append(
        #             alpha_scaled[i - 1] * np.sin(self.Omega[0][0] * self.t0 * np.pi * 2) + alpha_scaled[i] * np.cos(
        #                 self.Omega[0][0] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 2:
        #         alpha_scaled1.append(
        #             alpha_scaled[i] * np.cos(self.Omega[0][1] * self.t0 * np.pi * 2) - alpha_scaled[i + 1] * np.sin(
        #                 self.Omega[0][1] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 3:
        #         alpha_scaled1.append(
        #             alpha_scaled[i - 1] * np.sin(self.Omega[0][1] * self.t0 * np.pi * 2) + alpha_scaled[i] * np.cos(
        #                 self.Omega[0][1] * self.t0 * np.pi * 2))
        # alpha_scaled1 = np.array(alpha_scaled1)
        # f = pulser.params_to_controls(alpha_scaled1)
        # print([[self.Omega[0][0] * self.t0 * np.pi * 2, self.Omega[0][1] * self.t0 * np.pi * 2]])
        f = pulser.params_to_controls(alpha_scaled, phase_shift=[np.pi/2, np.pi/2])
        # print(f)

        return f[0]

    def get_length(self):
        return self.total_len


class opt_single_qudit4(Pulse):
    def __init__(self, Omega, total_len, Ns, ratio_ge, ratio_ef, ratio_fh, alpha, dt=None, plot=False):
        self.Omega = Omega
        self.total_len = total_len
        self.Ns = Ns
        self.ratio_ge = ratio_ge
        self.ratio_ef = ratio_ef
        self.ratio_fh = ratio_fh
        self.alpha = alpha
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        pulser = analyze_qudit.Pulser(self.Omega, self.t_array, self.Ns)
        # rescaling alpha
        alpha_scaled = []
        for i in range(len(self.alpha)):
            if np.mod(i, 6) < 2:
                alpha_scaled.append(self.alpha[i] / self.ratio_ge)
            elif np.mod(i, 6) < 4:
                alpha_scaled.append(self.alpha[i] / self.ratio_ef)
            else:
                alpha_scaled.append(self.alpha[i] / self.ratio_fh)
        alpha_scaled = np.array(alpha_scaled)
        # advancing phase
        # alpha_scaled1 = []
        # for i in range(len(alpha_scaled)):
        #     if np.mod(i, 4) == 0:
        #         alpha_scaled1.append(
        #             alpha_scaled[i] * np.cos(self.Omega[0][0] * self.t0 * np.pi * 2) - alpha_scaled[i + 1] * np.sin(
        #                 self.Omega[0][0] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 1:
        #         alpha_scaled1.append(
        #             alpha_scaled[i - 1] * np.sin(self.Omega[0][0] * self.t0 * np.pi * 2) + alpha_scaled[i] * np.cos(
        #                 self.Omega[0][0] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 2:
        #         alpha_scaled1.append(
        #             alpha_scaled[i] * np.cos(self.Omega[0][1] * self.t0 * np.pi * 2) - alpha_scaled[i + 1] * np.sin(
        #                 self.Omega[0][1] * self.t0 * np.pi * 2))
        #     if np.mod(i, 4) == 3:
        #         alpha_scaled1.append(
        #             alpha_scaled[i - 1] * np.sin(self.Omega[0][1] * self.t0 * np.pi * 2) + alpha_scaled[i] * np.cos(
        #                 self.Omega[0][1] * self.t0 * np.pi * 2))
        # alpha_scaled1 = np.array(alpha_scaled1)
        # f = pulser.params_to_controls(alpha_scaled1)
        # print([[self.Omega[0][0] * self.t0 * np.pi * 2, self.Omega[0][1] * self.t0 * np.pi * 2]])
        f = pulser.params_to_controls(alpha_scaled, phase_shift=[np.pi/2*0, np.pi/2*0, np.pi/2*0])
        # print(f)

        return f[0]

    def get_length(self):
        return self.total_len


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


class Square_multitone(Pulse):
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, dt=None, plot=False):
        self.max_amp = np.array(max_amp)
        self.flat_len = np.array(flat_len)
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = np.array(freq)
        self.phase = np.array(phase)
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        multone_pulse_arr = 0*self.t_array
        t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len

        for ii in range(self.freq.size):
            t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]
            t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]

            pulse_array = self.max_amp[ii] * (
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

            multone_pulse_arr = multone_pulse_arr + pulse_array * np.cos(2 * np.pi * self.freq[ii] * (self.t_array - self.phase_t0) + self.phase[ii])

        if max(multone_pulse_arr)>1.0: print('WARNING: Max value exceeded 1.0')
        return multone_pulse_arr

    def get_length(self):
        return max(self.flat_len + 2 * self.cutoff_sigma * self.ramp_sigma_len) # max of all pulses


class Multitone_EDG(Pulse):
    ''' Different rows are applied simultaneously
            Author: Ziqian Li, 2021/06/15'''
    def __init__(self, max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, shapes, nos = [], repeat=1, phase_t0 = 0, dt=None, plot=False):
        self.max_amp = np.array(max_amp)
        self.flat_len = np.array(flat_len)
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freq = np.array(freq)
        self.phase = np.array(phase)
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot
        self.shapes = np.array(shapes)    # shape=["cos/tanh", alpha, f, A, tg, gamma(if tanh is used)]
        self.nos = np.array(nos)
        self.repeat = repeat

        self.t0 = 0

        self.maxlen = 0
        #  caculating max pulse length
        for ii in range(len(self.shapes)):
            if self.maxlen<float(self.shapes[ii][4]):
                self.maxlen = float(self.shapes[ii][4])
        self.maxlen = self.maxlen*float(self.repeat)

        if self.maxlen<np.max(self.flat_len)+self.ramp_sigma_len*self.cutoff_sigma:
            self.maxlen = np.max(self.flat_len)+self.ramp_sigma_len*self.cutoff_sigma

    def get_pulse_array(self):

        multone_pulse_arr = 0*self.t_array
        pulse_array1 = 0 * self.t_array
        pulse_array2 = 0*self.t_array
        t_flat_start = self.t0 + self.cutoff_sigma * self.ramp_sigma_len
        edg_counts = -1

        for ii in range(self.freq.size):

            if ii in self.nos:
                edg_counts += 1
                self.tg = float(self.shapes[edg_counts][4])
                self.shape = self.shapes[edg_counts][0]
                self.alpha = float(self.shapes[edg_counts][1])
                self.f = float(self.shapes[edg_counts][2])
                self.A = float(self.shapes[edg_counts][3])

                for kk in range(self.repeat):
                    t_begin = self.t0 + (self.tg+self.dt)*(kk)
                    t_end = self.t0 + (self.tg)*(kk+1) + self.dt*kk

                    if self.shape == "tanh":
                        self.gamma = float(self.shapes[edg_counts][5])

                        pulse_array2 += (self.t_array <= t_end) * (self.t_array >= t_begin) * self.A * (
                                    np.tanh(self.gamma * (self.t_array - t_begin) / self.tg) - np.tanh(self.gamma * (
                                        (self.t_array - t_begin) / self.tg - np.ones_like(
                                    self.t_array))) - np.ones_like(self.t_array) * np.tanh(self.gamma)) ** 2

                        # pulse_array2 = (self.t_array <= t_end) * (self.t_array >= t_begin) * self.A * ( - np.tanh(self.gamma) * np.ones_like(self.t_array)) ** 2
                    else:
                        pulse_array2 = (self.t_array <= t_end) * (self.t_array >= t_begin) * self.A/2 * (np.ones_like(self.t_array) - np.cos(2 * np.pi * (self.t_array-t_begin)/self.tg))

                    pulse_array1 = (self.t_array <= t_end) * (self.t_array >= t_begin) * self.max_amp[ii] * (
                            np.ones_like(self.t_array) + self.alpha*np.sin((self.t_array-t_begin)*2*np.pi*self.f)
                    )

                    multone_pulse_arr = multone_pulse_arr + pulse_array1 * pulse_array2 * np.cos(
                        2 * np.pi * self.freq[ii] * (self.t_array) + self.phase[ii])



            else:
                t_flat_end = self.t0 + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]
                t_end = self.t0 + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]

                pulse_array = self.max_amp[ii] * (
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

                multone_pulse_arr = multone_pulse_arr + pulse_array * np.cos(2 * np.pi * self.freq[ii] * (self.t_array - self.phase_t0) + self.phase[ii])

        if max(multone_pulse_arr)>1.0: print('WARNING: Max value exceeded 1.0')
        return multone_pulse_arr

    def get_length(self):
        return self.maxlen # max of all pulses



class Square_multitone_sequential(Pulse):
    ''' Different rows are applied sequentially
        Author: Tanay Roy, 26 Jun 2020'''
    def __init__(self, max_amps, flat_lens, ramp_sigma_len, cutoff_sigma, freqs, phases, phase_t0 = 0, dt=None, plot=False):
        self.max_amps = np.array(max_amps)
        self.flat_lens = np.array(flat_lens)
        self.ramp_sigma_len = ramp_sigma_len
        self.cutoff_sigma = cutoff_sigma
        self.freqs = np.array(freqs)
        self.phases = np.array(phases)
        self.phase_t0 = phase_t0
        self.dt = dt
        self.plot = plot

        self.t0 = 0

    def get_pulse_array(self):

        multone_pulse_arr = 0*self.t_array
        pulse_start = self.t0

        for jj,freq in enumerate(self.freqs):
            self.freq = freq
            self.max_amp = self.max_amps[jj]
            self.flat_len = self.flat_lens[jj]
            self.phase = self.phases[jj]

            t_flat_start = pulse_start + self.cutoff_sigma * self.ramp_sigma_len

            for ii in range(self.freq.size):

                t_flat_end = pulse_start + self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]
                t_end = pulse_start + 2 * self.cutoff_sigma * self.ramp_sigma_len + self.flat_len[ii]

                pulse_array = self.max_amp[ii] * (
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

                multone_pulse_arr += pulse_array * np.cos(2 * np.pi * self.freq[ii] * (self.t_array - self.phase_t0) + self.phase[ii])

            pulse_start += 2 * self.cutoff_sigma * self.ramp_sigma_len + max(self.flat_len)

        if max(multone_pulse_arr)>1.0: print('WARNING: Max value exceeded 1.0')
        return multone_pulse_arr

    def get_length(self):
        # sum of max of all pulse blocks
        return sum(np.amax(self.flat_lens, axis=1) + 2 * self.cutoff_sigma * self.ramp_sigma_len)


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