from slab.dsfit import*
import os
from numpy import*
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
import json
import time
import glob

class PostExperimentAnalyzeAndSave:

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, data_path, experiment_name, data, P = 'Q',
                 phi=0, cont_data_file=None, cont_name="cont_v0", save=False):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.data_path = os.path.join(data_path, 'data/')
        self.save = save

        self.exptname = experiment_name
        self.data
        self.P = P
        self.mag = []
        self.phase = []

        self.cont_data_file = cont_data_file
        self.cont_slab_file = None
        self.phi = phi
        self.expt_nb = None
        self.time = None
        self.raw_I_mean = None
        self.raw_Q_mean = None
        self.p = []


        if self.save:
            if cont_data_file == None:
                self.cont_data_file = os.path.join(self.data_path, get_next_filename(self.data_path, cont_name, suffix='.h5'))
            else:
                self.cont_data_file = cont_data_file
                print("cont data file successfully passed")
            self.cont_slab_file = SlabFile(self.cont_data_file)


        eval('self.' + experiment_name)()

    def current_file_index(self, prefix=''):
        """Searches directories for files of the form *_prefix* and returns current number
            in the series"""

        dirlist = glob.glob(os.path.join(self.data_path, '*_' + prefix + '*'))
        dirlist.sort()
        try:
            ii = int(os.path.split(dirlist[-1])[-1].split('_')[0])
        except:
            ii = 0
        return ii

    def iq_rot(self):
        """Digitially rotates IQdata by phi, calcualting phase as np.unrwap(np.arctan2(Q, I))
        :selfparam I: I data from h5 file
        :selfparam Q: Q data from h5 file
        :selfparam phi: iq rotation desired (in degrees)
        :returns: sets self.I, self.Q
        """
        self.phi = self.phi * np.pi / 180  # convert to radians
        phase = np.unwrap(np.arctan2(self.Q, self.I))
        self.Q = self.Q / np.sin(phase) * np.sin(phase + self.phi)
        self.I = self.I / np.cos(phase) * np.cos(phase + self.phi)

    def iq_process(self):
        """Converts digitial data to voltage data, rotates iq, subtracts off mean, calculates mag and phase
        :param data: data from h5 file, [["I", "Q"]], [["qbA_I", "qbA_Q"],["qbB_I", "qbB_Q"]
        :param raw_I: I data from h5 file
        :param raw_Q: Q data from h5 file
        :param on_qbs: homodyne setups being used
        :param ran: range of DAC. If set to -1, doesn't convert data to voltage
        :returns: sets self: I, Q, mag, phase
        """

        raw_I = self.data[0][0]
        raw_Q = self.data[0][1]
        ran = self.hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']

        self.raw_I_mean = mean(array(raw_I).flatten())
        self.raw_Q_mean = mean(array(raw_Q).flatten())
        self.I = array(raw_I).flatten() - self.raw_I_mean
        self.Q = array(raw_Q).flatten() - self.raw_Q_mean


        # divide by 2**15 to convert from bits to voltage, *ran to get right voltage range
        if ran > 0:
            self.I = self.I / 2 ** 15 * ran
            self.Q = self.Q / 2 ** 15 * ran

        # calculate mag and phase
        phase = np.unwrap(np.arctan2(self.Q, self.I))
        mag = np.sqrt(np.square(self.I) + np.square(self.Q))
        self.mag = mag
        self.phase = phase

        # IQ rotate
        self.iq_rot()

    def get_params(hardware_cfg, experiment_cfg, quantum_device_cfg, on_qb):
        """puts all config info into a dictionary, so that if cfg files change we just need to change this file
        rather than all files
        :returns: dict params
        """

        params = {}
        params['ran'] = hardware_cfg['awg_info']['keysight_pxi']['digtzr_vpp_range']
        params['readout_params'] = quantum_device_cfg['readout'][on_qb]
        params['readout_freq'] = params['readout_params']["freq"]
        params['dig_atten_qb'] = quantum_device_cfg['powers'][on_qb]['drive_digital_attenuation']
        params['dig_atten_rd'] = quantum_device_cfg['powers'][on_qb]['readout_drive_digital_attenuation']
        params['read_lo_pwr'] = quantum_device_cfg['powers'][on_qb]['readout_drive_lo_powers']
        params['qb_lo_pwr'] = quantum_device_cfg['powers'][on_qb]['drive_lo_powers']
        params['qb_freq'] = quantum_device_cfg['qubit'][on_qb]['freq']
        return params

    def pulse_probe_iq(self):
        print("Starting pulse probe analysis")
        expt_params = self.experiment_cfg[self.exptname]
        params = self.get_params(self.hardware_cfg, self.experiment_cfg, self.quantum_device_cfg, expt_params[
            'on_qubits'][0])

        nu_q = params['qb_freq']

        self.iq_process()
        f = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))] + nu_q
        exp_nb = self.current_file_index(prefix=self.exptname)
        try:
            p = fitlor(f, np.square(self.mag), showfit=False) #returns [offset,amplitude,center,hwhm]
            print("pulse probe fit worked!")
        except:
            print("Pulse probe fit failed on exp", exp_nb)
            p = [0, 0, 0, 0]

        pulse_probe_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]
        if self.save:
            with self.cont_slab_file as file:
                file.append_line('pulse_probe_iq_meta', pulse_probe_meta)
                file.append_line('pulse_probe_iq_fit', p)
                print("appended line correctly")
        self.p = p

    def rabi(self):
        print("Starting rabi analysis")
        expt_params = self.experiment_cfg[self.exptname]

        self.iq_process()

        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
        exp_nb = self.current_file_index(prefix=self.exptname)
        pulse_type = expt_params['pulse_type']

        try:
            # p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
            if self.P=='Q':
                p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
            else:
                p = fitdecaysin(t, self.I, showfit=False, fitparams=None, domain=None)
            print("rabi fit worked!")
        except:
            print("rabi fit failed on exp", exp_nb)
            p = [0, 0, 0, 0, 0]

        rabi_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]
        if self.save:
            with self.cont_slab_file as file:
                file.append_line('rabi_meta', rabi_meta)
                file.append_line('rabi_fit', p)
                print("appended line correctly")
        self.p = p


    def t1(self):
        print("Starting t1 analysis")
        expt_params = self.experiment_cfg[self.exptname]

        self.iq_process()
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
        exp_nb = self.current_file_index(prefix=self.exptname)
        t = t / 1000  # convert to us
        try:
            #exponential decay (p[0]+p[1]*exp(-(x-p[2])/p[3])
            if self.P=='Q':
                p = fitexp(t,self.Q,fitparams=None, domain=None, showfit=False)
            else:
                p = fitexp(t,self.Q,fitparams=None, domain=None, showfit=False)
            print("t1 fit worked!")
        except:
            print("t1 fit failed on exp", exp_nb)
            p = [0, 0, 0, 0]

        t1_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]

        if save:
            with self.cont_slab_file as file:
                file.append_line('t1_meta', t1_meta)
                file.append_line('t1_fit', p)
                print("appended line correctly")
        self.p=p

    def ramsey(self):
        print("Starting ramsey analysis")
        expt_params = self.experiment_cfg[self.exptname]

        self.iq_process()
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
        t = t/1000 #convert to us
        exp_nb = self.current_file_index(prefix=self.exptname)

        try:
            #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
            if self.P=='Q':
                p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
            else:
                p = fitdecaysin(t, self.Q, showfit=False, fitparams=None, domain=None)
            print("ramsey fit worked!")
        except:
            print("ramsey fit failed on exp", exp_nb)
            p = [0, 0, 0, 0, 0]

        if self.save:
            ramsey_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]
            with self.cont_slab_file as file:
                file.append_line('ramsey_meta', ramsey_meta)
                file.append_line('ramsey_fit', p)
                print("appended line correctly")
        self.p = p

    def echo(self):
        print("Starting echo analysis")
        expt_params = self.experiment_cfg[self.exptname]

        self.iq_process()
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
        t = t/1000 #convert to us
        exp_nb = self.current_file_index(prefix=self.exptname)

        try:
            #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
            if self.P=='Q':
                p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
            else:
                p = fitdecaysin(t, self.Q, showfit=False, fitparams=None, domain=None)
            print("echo fit worked!")
        except:
            print("echo fit failed on exp", exp_nb)
            p = [0, 0, 0, 0, 0]

        if self.save:
            echo_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]
            with self.cont_slab_file as file:
                file.append_line('echo_meta', echo_meta)
                file.append_line('echo_fit', p)
                print("appended line correctly")
        self.p = p


    def save_cfg_info(self, f):
            f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
            f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
            f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
            f.close()


def temperature_q(nu, rat):
    Kb = 1.38e-23
    h = 6e-34
    return h * nu / (Kb * log(1 / rat))

def occupation_q(nu, T):
    Kb = 1.38e-23
    h = 6e-34
    T = T * 1e-3
    return 1 / (exp(h * nu / (Kb * T)) + 1)


# class PostExperiment:

#     def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, data, P = 'Q', show = True):
#         self.quantum_device_cfg = quantum_device_cfg
#         self.experiment_cfg = experiment_cfg
#         self.hardware_cfg = hardware_cfg
#
#         self.exptname = experiment_name
#         self.I = I
#         self.Q = Q
#         self.P = P
#         self.show = show
#         self.p=[]
#
#         #try:eval('self.' + experiment_name)()
#         #except:print("No post experiment analysis yet")
#         eval('self.' + experiment_name)()
#
#     def check_sync(self):
#         expt_num=0
#         readout_window = self.quantum_device_cfg["readout"]["window"]
#         dt_dig = self.hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
#         data_1 = self.I
#         data_2 = self.Q
#         fig = plt.figure(figsize=(12, 4))
#         ax = fig.add_subplot(131, title='I')
#         plt.imshow(data_1, aspect='auto')
#         ax.set_xlabel('Digitizer bins')
#         ax.set_ylabel('Experiment number')
#         ax2 = fig.add_subplot(132, title='Q')
#         plt.imshow(data_2, aspect='auto')
#         ax2.set_xlabel('Digitizer bins')
#         ax2.set_ylabel('Experiment number')
#         ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
#         ax3.plot(np.arange(data_1[0].size*dt_dig, step=dt_dig), data_1[expt_num])
#         ax3.plot(np.arange(data_1[0].size*dt_dig, step=dt_dig), data_2[expt_num])
#         ax3.axvspan(readout_window[0], readout_window[1], alpha=0.2, color='b')
#         ax3.set_xlabel('Time (ns)')
#         ax3.set_ylabel('Signal')
#         fig.tight_layout()
#         plt.show()
#
#     def resonator_spectroscopy(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         I = self.I.flatten() - mean(self.I.flatten())
#         Q = self.Q.flatten() - mean(self.Q.flatten())
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(I))]
#         mag = sqrt(I ** 2 + Q ** 2)
#         p = fitlor(f, mag ** 2, showfit=False)
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, I, 'b.-', label='I')
#             ax.plot(f, Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             ax2 = ax.twinx()
#             ax2.plot(f, mag, 'k.-', alpha=0.3)
#
#             ax2.plot(f, sqrt(lorfunc(p, f)), 'k--')
#             ax2.axvline(p[2], color='r', linestyle='dashed')
#             fig.tight_layout()
#             plt.show()
#         else:pass
#
#         print("Resonant frequency from fitting mag squared: ", p[2], "GHz")
#         print("Resonant frequency from I peak : ", f[argmax(abs(I))], "GHz")
#
#     def pulse_probe_iq(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         self.I = self.I - mean(self.I)
#         self.Q = self.Q - mean(self.Q)
#
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, self.I, 'b.-', label='I')
#             ax.plot(f, self.Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             p = fitlor(f, eval('self.'+self.P), showfit=False)
#             ax.plot(f, lorfunc(p, f), 'k--')
#             ax.axvline(p[2], color='g', linestyle='dashed')
#             plt.show()
#         else:p = fitlor(f, eval('self.'+self.P), showfit=False)
#
#         print("Qubit frequency = ", p[2], "GHz")
#         print("Pulse probe width = ", p[3] * 1e3, "MHz")
#         print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')
#         self.p=p
#
#     def rabi(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t[2:], P[2:], showfit=True)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#             ax.axvline(t_pi, color='k', linestyle='dashed')
#             ax.axvline(t_half_pi, color='k', linestyle='dashed')
#
#             plt.show()
#
#         else:
#             p = fitdecaysin(t, P, showfit=False)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#         print("Half pi length =", t_half_pi, "ns")
#         print("pi length =", t_pi, "ns")
#         print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
#         print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
#               amp * (t_half_pi) / float(int(t_half_pi) + 1))
#
#     def t1(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (us)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitexp(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitexp(t, P, showfit=False)
#         self.p = p
#         print("T1 =", p[3], "us")
#
#     def ramsey(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#         nu_q_new = nu_q + offset
#
#         print("Original qubit frequency choice =", nu_q, "GHz")
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested qubit frequency choice =", nu_q_new, "GHz")
#         print("T2* =", p[3], "ns")
#
#     def echo(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#
#         print("Echo experiment: CP = ",expt_cfg['cp'],"CPMG = ",expt_cfg['cpmg'])
#         print ("Number of echoes = ",expt_cfg['echo_times'])
#         print("T2 =",p[3],"ns")
#
#     def pulse_probe_ef_iq(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']
#
#         self.I = self.I - mean(self.I)
#         self.Q = self.Q - mean(self.Q)
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q + alpha
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, self.I, 'b.-', label='I')
#             ax.plot(f, self.Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             p = fitlor(f, eval('self.'+self.P), showfit=False)
#             ax.plot(f, lorfunc(p, f), 'k--')
#             ax.axvline(p[2], color='g', linestyle='dashed')
#             plt.show()
#         else:p = fitlor(f, eval('self.'+self.P), showfit=False)
#
#
#         print ("ef frequency = ",p[2],"GHz")
#         print ("Expected anharmonicity =",p[2]-nu_q,"GHz")
#         print ("ef pulse probe width = ",p[3]*1e3,"MHz")
#         print ("Estimated ef pi pulse time: ",1/(sqrt(2)*2*p[3]),'ns' )
#
#     def ef_rabi(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t[2:], P[2:], showfit=True)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#             ax.axvline(t_pi, color='k', linestyle='dashed')
#             ax.axvline(t_half_pi, color='k', linestyle='dashed')
#
#             plt.show()
#
#         else:
#             p = fitdecaysin(t, P, showfit=False)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#         print("Half pi length =", t_half_pi, "ns")
#         print("pi length =", t_pi, "ns")
#         print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
#         print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
#               amp * (t_half_pi) / float(int(t_half_pi) + 1))
#
#     def ef_t1(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (us)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitexp(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitexp(t, P, showfit=False)
#
#         print("T1 =", p[3], "us")
#
#     def ef_ramsey(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#         alpha_new = alpha + offset
#
#         print("Original alpha choice =", alpha, "GHz")
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested alpha = ", alpha_new, "GHz")
#         print("T2* =", p[3], "ns")
#
#     def ef_echo(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'],expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#
#         print("Echo experiment: CP = ",expt_cfg['cp'],"CPMG = ",expt_cfg['cpmg'])
#         print ("Number of echoes = ",expt_cfg['echo_times'])
#         print("T2 =",p[3],"ns")
#
#     def pulse_probe_fh_iq(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']
#
#         self.I = self.I - mean(self.I)
#         self.Q = self.Q - mean(self.Q)
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q + 2 * alpha
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, self.I, 'b.-', label='I')
#             ax.plot(f, self.Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             p = fitlor(f, eval('self.' + self.P), showfit=False)
#             ax.plot(f, lorfunc(p, f), 'k--')
#             ax.axvline(p[2], color='g', linestyle='dashed')
#             plt.show()
#         else:
#             p = fitlor(f, eval('self.' + self.P), showfit=False)
#
#         print("fh frequency = ", p[2], "GHz")
#         print("Expected anharmonicity 2 =", p[2] - nu_q - alpha, "GHz")
#         print("ef pulse probe width = ", p[3] * 1e3, "MHz")
#         print("Estimated ef pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')
#
#     def fh_rabi(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t[2:], P[2:], showfit=True)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#             ax.axvline(t_pi, color='k', linestyle='dashed')
#             ax.axvline(t_half_pi, color='k', linestyle='dashed')
#
#             plt.show()
#
#         else:
#             p = fitdecaysin(t, P, showfit=False)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#         print("Half pi length =", t_half_pi, "ns")
#         print("pi length =", t_pi, "ns")
#         print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
#         print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
#               amp * (t_half_pi) / float(int(t_half_pi) + 1))
#
#     def fh_ramsey(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']
#         alpha_fh = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity_fh']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#         alpha_new = alpha_fh + offset
#
#         print("Original alpha_fh choice =", alpha_fh, "GHz")
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested alpha_fh = ", alpha_new, "GHz")
#         print("T2* =", p[3], "ns")
#
#     def histogram(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         if expt_cfg['singleshot']:
#             a_num = expt_cfg['acquisition_num']
#             ns = expt_cfg['num_seq_sets']
#             numbins = expt_cfg['numbins']
#
#             I = self.I
#             Q = self.Q
#
#             colors = ['r', 'b', 'g']
#             labels = ['g', 'e', 'f']
#             titles = ['-I', '-Q']
#
#             IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3],1)
#             IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[2::3], Q.T.flatten()[2::3]
#
#             fig = plt.figure(figsize=(12, 12))
#             ax = fig.add_subplot(421, title='Averaged over acquisitions')
#
#             for j in range(ns):
#                 for ii in range(3):
#                     ax.plot(IQs[2 * ii], IQs[2 * ii + 1], 'o', color=colors[ii])
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#             ax = fig.add_subplot(422, title='Mean and std all')
#
#             for ii in range(3):
#                 ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
#                             yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#             for kk in range(6):
#                 ax = fig.add_subplot(4, 2, kk + 3, title=self.exptname + titles[kk % 2])
#                 ax.hist(IQsss[kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
#                 ax.legend()
#                 ax.set_xlabel('Value'+titles[kk % 2])
#                 ax.set_ylabel('Number')
#
#
#             for ii,i in enumerate(['I','Q']):
#                 sshg, ssbinsg = np.histogram(IQsss[ii], bins=numbins)
#                 sshe, ssbinse = np.histogram(IQsss[ii+2], bins=numbins)
#                 fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()
#
#                 print("Single shot g-e readout fidility from channel ", i, " = ", fid)
#
#         else:
#             fig = plt.figure(figsize=(7, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             for ii in range(expt_cfg['num_seq_sets']):
#                 if ii is 0:
#                     ax.plot(self.I[0::3], self.Q[0::3], 'ro', label='g')
#                     ax.plot(self.I[1::3], self.Q[1::3], 'bo', label='e')
#                     ax.plot(self.I[2::3], self.Q[2::3], 'go', label='f')
#                 else:
#                     ax.plot(self.I[0::3], self.Q[0::3], 'ro')
#                     ax.plot(self.I[1::3], self.Q[1::3], 'bo')
#                     ax.plot(self.I[2::3], self.Q[2::3], 'go')
#                 ax.legend()
#
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#         fig.tight_layout()
#         plt.show()
#
#     def sideband_histogram(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         if expt_cfg['singleshot']:
#             a_num = expt_cfg['acquisition_num']
#             ns = expt_cfg['num_seq_sets']
#             numbins = expt_cfg['numbins']
#
#             I = self.I
#             Q = self.Q
#
#             colors = ['r', 'b', 'g']
#             labels = ['g', 'e', 'f']
#             titles = ['-I', '-Q']
#
#             IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3],1)
#             IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[2::3], Q.T.flatten()[2::3]
#
#             fig = plt.figure(figsize=(12, 12))
#             ax = fig.add_subplot(421, title='Averaged over acquisitions')
#
#             for j in range(ns):
#                 for ii in range(3):
#                     ax.plot(IQs[2 * ii], IQs[2 * ii + 1], 'o', color=colors[ii])
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#             ax = fig.add_subplot(422, title='Mean and std all')
#
#             for ii in range(3):
#                 ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
#                             yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#             for kk in range(6):
#                 ax = fig.add_subplot(4, 2, kk + 3, title=self.exptname + titles[kk % 2])
#                 ax.hist(IQsss[kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
#                 ax.legend()
#                 ax.set_xlabel('Value'+titles[kk % 2])
#                 ax.set_ylabel('Number')
#
#
#             for ii,i in enumerate(['I','Q']):
#                 sshg, ssbinsg = np.histogram(IQsss[ii], bins=numbins)
#                 sshe, ssbinse = np.histogram(IQsss[ii+2], bins=numbins)
#                 fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()
#
#                 print("Single shot g-e readout fidility from channel ", i, " = ", fid)
#
#         else:
#             fig = plt.figure(figsize=(7, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             for ii in range(expt_cfg['num_seq_sets']):
#                 if ii is 0:
#                     ax.plot(self.I[0::3], self.Q[0::3], 'ro', label='g')
#                     ax.plot(self.I[1::3], self.Q[1::3], 'bo', label='e')
#                     ax.plot(self.I[2::3], self.Q[2::3], 'go', label='f')
#                 else:
#                     ax.plot(self.I[0::3], self.Q[0::3], 'ro')
#                     ax.plot(self.I[1::3], self.Q[1::3], 'bo')
#                     ax.plot(self.I[2::3], self.Q[2::3], 'go')
#                 ax.legend()
#
#             ax.set_xlabel('I')
#             ax.set_ylabel('Q')
#
#         fig.tight_layout()
#         plt.show()
#
#     def qubit_temperature(self):
#         expt_cfg = self.experiment_cfg['ef_rabi']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:]
#         contrast = []
#
#         fig = plt.figure(figsize=(14, 7))
#         ax = fig.add_subplot(111, title=self.exptname)
#         for i in range(2):
#             ax.plot(t, P[i], 'bo-', label='ge_pi = ' + str(i is 0))
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#
#             ax.legend()
#
#             p = fitdecaysin(t[2:], P[i][2:], showfit=True)
#             contrast.append(p[0])
#
#             ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
#             ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')
#
#             print("Half pi length =", 1 / (4 * p[1]), "ns")
#             print("pi length =", 1 / (2 * p[1]), "ns")
#
#         if self.show:
#             plt.show()
#
#         ratio = abs(contrast[1] / contrast[0])
#         print("Qubit Temp:", 1e3 * temperature_q(nu_q * 1e9, ratio), " mK")
#         print("Contrast ratio:",ratio)
#         print("Qubit Excited State Occupation:", occupation_q(nu_q, 1e3 * temperature_q(nu_q, ratio)))
#
#     def sideband_rabi(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t[2:], P[2:], showfit=True)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#             ax.axvline(t_pi, color='k', linestyle='dashed')
#             ax.axvline(t_half_pi, color='k', linestyle='dashed')
#
#             plt.show()
#
#         else:
#             p = fitdecaysin(t, P, showfit=False)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#         print("Half pi length =", t_half_pi, "ns")
#         print("pi length =", t_pi, "ns")
#         print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
#         print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
#               amp * (t_half_pi) / float(int(t_half_pi) + 1))
#
#     def sideband_rabi_freq_scan(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         sideband_freq = expt_cfg['freq']
#         P = eval('self.' + self.P)
#         df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         freqs = sideband_freq + df
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(freqs, P, 'o-', label=self.P)
#             ax.set_xlabel('Sideband freq (GHz)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             plt.show()
#
#         else:
#             pass
#
#     def sideband_t1(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (us)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitexp(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitexp(t, P, showfit=False)
#
#         print("T1 =", p[3], "us")
#
#     def sideband_ramsey(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#         suggested_dc_offset = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['f0g1_dc_offset'] + offset
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested dc offset =", suggested_dc_offset * 1e3, "MHz")
#         print("T2 =", p[3], "ns")
#
#     def sideband_rabi_two_tone_freq_scan(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         sideband_freq = expt_cfg['center_freq']+expt_cfg['offset_freq']
#         P = eval('self.' + self.P)
#         df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         freqs = df
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(freqs, P, 'o-', label=self.P)
#             ax.set_xlabel('Detuning (GHz)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             plt.show()
#
#         else:
#             pass
#
#     def sideband_rabi_two_tone(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#         amp = expt_cfg['amp']
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t[2:], P[2:], showfit=True)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#             ax.axvline(t_pi, color='k', linestyle='dashed')
#             ax.axvline(t_half_pi, color='k', linestyle='dashed')
#
#             plt.show()
#
#         else:
#             p = fitdecaysin(t, P, showfit=False)
#             t_pi = 1 / (2 * p[1])
#             t_half_pi = 1 / (4 * p[1])
#
#         print("Half pi length =", t_half_pi, "ns")
#         print("pi length =", t_pi, "ns")
#         print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
#         print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
#               amp * (t_half_pi) / float(int(t_half_pi) + 1))
#
#     def sideband_pulse_probe_iq(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         self.I = self.I - mean(self.I)
#         self.Q = self.Q - mean(self.Q)
#
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, self.I, 'b.-', label='I')
#             ax.plot(f, self.Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             p = fitlor(f, eval('self.' + self.P), showfit=False)
#             ax.plot(f, lorfunc(p, f), 'k--')
#             ax.axvline(p[2], color='g', linestyle='dashed')
#             plt.show()
#         else:
#             p = fitlor(f, eval('self.' + self.P), showfit=False)
#
#         print("Qubit frequency = ", p[2], "GHz")
#         print("Pulse probe width = ", p[3] * 1e3, "MHz")
#         print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')
#
#     def sideband_chi_ge_calibration(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#
#         suggested_chi_shift = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_e'] + offset
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested ge chi shift = 2 pi x",suggested_chi_shift* 1e3, "MHz")
#         print("T2 =", p[3], "ns")
#
#     def sideband_chi_ef_calibration(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#
#         suggested_chi_shift = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_ef'] + offset
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested ef chi shift = 2 pi x",suggested_chi_shift* 1e3, "MHz")
#         print("T2 =", p[3], "ns")
#
#     def sideband_chi_gf_calibration(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         ramsey_freq = expt_cfg['ramsey_freq']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         P = eval('self.'+self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitdecaysin(t, P, showfit=True)
#             plt.show()
#
#         else:p = fitdecaysin(t, P, showfit=False)
#
#         offset = ramsey_freq - p[1]
#
#         suggested_chi_shift = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_f'] + offset
#         print("Offset freq =", offset * 1e3, "MHz")
#         print("Suggested f chi shift = 2 pi x",suggested_chi_shift* 1e3, "MHz")
#         print("T2 =", p[3], "ns")
#
#
#     def sideband_reset_qubit_temperature(self):
#         expt_cfg = self.experiment_cfg['sideband_transmon_reset']
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:]
#         contrast = []
#
#         fig = plt.figure(figsize=(14, 7))
#         ax = fig.add_subplot(111, title=self.exptname)
#         for i in range(2):
#             ax.plot(t, P[i], 'bo-', label='ge_pi = ' + str(i is 0))
#             ax.set_xlabel('Time (ns)')
#             ax.set_ylabel(self.P)
#
#             ax.legend()
#
#             p = fitdecaysin(t[2:], P[i][2:], showfit=True)
#             contrast.append(p[0])
#
#             ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
#             ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')
#
#             print("Half pi length =", 1 / (4 * p[1]), "ns")
#             print("pi length =", 1 / (2 * p[1]), "ns")
#
#         if self.show:
#             plt.show()
#
#         ratio = abs(contrast[1] / contrast[0])
#         print("Qubit Temp:", 1e3 * temperature_q(nu_q * 1e9, ratio), " mK")
#         print("Contrast ratio:",ratio)
#         print("Qubit Excited State Occupation:", occupation_q(nu_q, 1e3 * temperature_q(nu_q, ratio)))
#
#
#     def qp_pumping_t1(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))] / 1e3
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Time (us)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             p = fitexp(t, P, showfit=True)
#             plt.show()
#
#         else:
#             p = fitexp(t, P, showfit=False)
#
#         print("T1 =", p[3], "us")
#
#     def sideband_parity_measurement(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         P = eval('self.' + self.P)
#         t = arange(expt_cfg['num_expts'])
#
#         if self.show:
#
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(t, P, 'o-', label=self.P)
#             ax.set_xlabel('Experiment number)')
#             ax.set_ylabel(self.P)
#             ax.legend()
#             plt.show()
#
#     def sideband_cavity_photon_number(self):
#         expt_cfg = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
#
#         self.I = self.I - mean(self.I)
#         self.Q = self.Q - mean(self.Q)
#
#         f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q
#
#         if self.show:
#             fig = plt.figure(figsize=(14, 7))
#             ax = fig.add_subplot(111, title=self.exptname)
#             ax.plot(f, self.I, 'b.-', label='I')
#             ax.plot(f, self.Q, 'r.-', label='Q')
#             ax.set_xlabel('Freq(GHz)')
#             ax.set_ylabel('I/Q')
#             ax.legend()
#             p = fitlor(f, eval('self.'+self.P), showfit=False)
#             ax.plot(f, lorfunc(p, f), 'k--')
#             ax.axvline(p[2], color='g', linestyle='dashed')
#             plt.show()
#         else:p = fitlor(f, eval('self.'+self.P), showfit=False)
#
#         print("Qubit frequency = ", p[2], "GHz")
#         print("Pulse probe width = ", p[3] * 1e3, "MHz")
#         print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')
#
#
#
#     def save_cfg_info(self, f):
#             f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
#             f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
#             f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
#             f.close()
#

# class PostExperimentAnalyze:
#
#     def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, data_path, experiment_name, data, P = 'Q',
#                  phi=0):
#         self.quantum_device_cfg = quantum_device_cfg
#         self.experiment_cfg = experiment_cfg
#         self.hardware_cfg = hardware_cfg
#         self.data_path = os.path.join(data_path, 'data/')
#
#         self.exptname = experiment_name
#         self.I = I
#         self.Q = Q
#         self.P = P
#         self.mag = []
#         self.phase = []
#
#         self.phi = phi
#         self.expt_nb = None
#         self.time = None
#         self.raw_I_mean = None
#         self.raw_Q_mean = None
#         self.p = []
#
#         eval('self.' + experiment_name)()
#
#     def current_file_index(self, prefix=''):
#         """Searches directories for files of the form *_prefix* and returns current number
#             in the series"""
#
#         dirlist = glob.glob(os.path.join(self.data_path, '*_' + prefix + '*'))
#         dirlist.sort()
#         try:
#             ii = int(os.path.split(dirlist[-1])[-1].split('_')[0])
#         except:
#             ii = 0
#         return ii
#
#     def iq_rot(self):
#         """Digitially rotates IQdata by phi, calcualting phase as np.unrwap(np.arctan2(Q, I))
#         :selfparam I: I data from h5 file
#         :selfparam Q: Q data from h5 file
#         :selfparam phi: iq rotation desired (in degrees)
#         :returns: sets self.I, self.Q
#         """
#         self.phi = self.phi * np.pi / 180  # convert to radians
#         phase = np.unwrap(np.arctan2(self.Q, self.I))
#         self.Q = self.Q / np.sin(phase) * np.sin(phase + self.phi)
#         self.I = self.I / np.cos(phase) * np.cos(phase + self.phi)
#
#     def iq_process(self):
#         """Converts digitial data to voltage data, rotates iq, subtracts off mean, calculates mag and phase
#         :param raw_I: I data from h5 file
#         :param raw_Q: Q data from h5 file
#         :param ran: range of DAC. If set to -1, doesn't convert data to voltage
#         :returns: sets self: I, Q, mag, phase
#         """
#         raw_I = self.I
#         raw_Q = self.Q
#         ran = self.hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
#
#         self.raw_I_mean = mean(array(raw_I).flatten())
#         self.raw_Q_mean = mean(array(raw_Q).flatten())
#         self.I = array(raw_I).flatten() - self.raw_I_mean
#         self.Q = array(raw_Q).flatten() - self.raw_Q_mean
#
#
#         # divide by 2**15 to convert from bits to voltage, *ran to get right voltage range
#         if ran > 0:
#             self.I = self.I / 2 ** 15 * ran
#             self.Q = self.Q / 2 ** 15 * ran
#
#         # calculate mag and phase
#         phase = np.unwrap(np.arctan2(self.Q, self.I))
#         mag = np.sqrt(np.square(self.I) + np.square(self.Q))
#         self.mag = mag
#         self.phase = phase
#
#         # IQ rotate
#         self.iq_rot()
#
#     def pulse_probe_iq(self):
#         print("Starting pulse probe analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#         nu_q = self.quantum_device_cfg['qubit'][expt_params['on_qubits'][0]]['freq']
#
#         self.iq_process()
#         f = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))] + nu_q
#         exp_nb = self.current_file_index(prefix=self.exptname)
#         try:
#             p = fitlor(f, np.square(self.mag), showfit=False) #returns [offset,amplitude,center,hwhm]
#             print("pulse probe fit worked!")
#         except:
#             print("Pulse probe fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0]
#
#         self.p = p
#
#     def rabi(self):
#         print("Starting rabi analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         exp_nb = self.current_file_index(prefix=self.exptname)
#         pulse_type = expt_params['pulse_type']
#         #if pulse_type == "gauss":
#         #   t = t * 4
#         try:
#             #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
#             p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
#             print("rabi fit worked!")
#         except:
#             print("rabi fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0, 0]
#         self.p = p
#
#     def rabi_lmfit(self):
#         print("Starting rabi analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         exp_nb = self.current_file_index(prefix=self.exptname)
#         pulse_type = expt_params['pulse_type']
#         #if pulse_type == "gauss":
#         #   t = t * 4
#         try:
#             #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
#             p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
#
#             print("rabi fit worked!")
#         except:
#             print("rabi fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0, 0]
#         self.p = p
#
#     '''
#     def rabi(self):
#         print("Starting rabi analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         exp_nb = self.current_file_index(prefix=self.exptname)
#         fitparams = [0.03, 1/5800, 270, 40000, 0]
#         pulse_type = expt_params['pulse_type']
#         if pulse_type == "gauss":
#             t = t * 4
#         try:
#             #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
#             p = fitdecaysin(t,self.Q,showfit=False, fitparams=fitparams, domain=None)
#             print("rabi fit worked!")
#         except:
#             print("rabi fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0, 0]
#
#         rabi_meta = [exp_nb, time.time(), self.raw_I_mean, self.raw_Q_mean]
#         with self.cont_slab_file as file:
#             file.append_line('rabi_meta', rabi_meta)
#             file.append_line('rabi_fit', p)
#             print("appended line correctly")
#     '''
#
#     def t1(self):
#         print("Starting t1 analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         exp_nb = self.current_file_index(prefix=self.exptname)
#         t = t / 1000  # convert to us
#         try:
#             #exponential decay (p[0]+p[1]*exp(-(x-p[2])/p[3])
#             p = fitexp(t,self.Q,fitparams=None, domain=None, showfit=False)
#             print("t1 fit worked!")
#         except:
#             print("t1 fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0]
#
#         self.p=p
#
#     def ramsey(self):
#         print("Starting ramsey analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         t = t/1000 #convert to us
#         exp_nb = self.current_file_index(prefix=self.exptname)
#
#         try:
#             #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
#             p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
#             print("ramsey fit worked!")
#         except:
#             print("ramsey fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0, 0]
#
#         self.p = p
#     def echo(self):
#         print("Starting echo analysis")
#         expt_params = self.experiment_cfg[self.exptname]
#
#         self.iq_process()
#         t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(self.I))]
#         t = t/1000 #convert to us
#         exp_nb = self.current_file_index(prefix=self.exptname)
#
#         try:
#             #p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]
#             p = fitdecaysin(t,self.Q,showfit=False, fitparams=None, domain=None)
#             print("echo fit worked!")
#         except:
#             print("echo fit failed on exp", exp_nb)
#             p = [0, 0, 0, 0, 0]
#
#         self.p = p
#
#
#     def save_cfg_info(self, f):
#             f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
#             f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
#             f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
#             f.close()
