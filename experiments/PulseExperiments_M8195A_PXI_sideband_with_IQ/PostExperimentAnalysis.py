from slab.dsfit import*
import matplotlib.pyplot as plt
from numpy import*

class PostExperiment:

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, I , Q, P = 'Q', show=True):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.exptname = experiment_name
        self.I = I
        self.Q = Q
        self.P = P
        self.show = show

        # eval('self.' + experiment_name)()
        try:
            temp = eval('self.' + experiment_name)()
        except:
            print("No post experiment analysis yet")

    def resonator_spectroscopy(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        I = self.I.flatten() - mean(self.I.flatten())
        Q = self.Q.flatten() - mean(self.Q.flatten())
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(I))]
        mag = sqrt(I ** 2 + Q ** 2)
        p = fitlor(f, mag ** 2, showfit=False)

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, I, 'b.-', label='I')
            ax.plot(f, Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            ax2 = ax.twinx()
            ax2.plot(f, mag, 'k.-', alpha=0.3)

            ax2.plot(f, sqrt(lorfunc(p, f)), 'k--')
            ax2.axvline(p[2], color='r', linestyle='dashed')
            fig.tight_layout()
            plt.show()
        else:pass

        print("Resonant frequency from fitting mag squared: ", p[2], "GHz")
        print("Resonant frequency from I peak : ", f[argmax(abs(I))], "GHz")

    def pulse_probe_iq(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)

        print("Qubit frequency = ", p[2], "GHz")
        print("Pulse probe width = ", p[3] * 1e3, "MHz")
        print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')

    def rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        dt = 0.0625
        print("suggested_pi_length = ", (int(t_pi / dt) + 1) * dt, "suggested_pi_amp = ",amp * (t_pi) / float((int(t_pi / dt) + 1) * dt))
        print("suggested_half_pi_length = ", (int(t_half_pi / dt) + 1) * dt, "suggested_piby2_amp = ",amp * (t_half_pi) / float((int(t_half_pi / dt) + 1) * dt))

    def t1(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (us)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitexp(t, P, showfit=True)
            plt.show()

        else:p = fitexp(t, P, showfit=False)

        print("T1 =", p[3], "us")

    def ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        nu_q_new = nu_q + offset

        print("Original qubit frequency choice =", nu_q, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested qubit frequency choice =", nu_q_new, "GHz")
        print("T2* =", p[3], "ns")

    def echo(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]

        print("Echo experiment: CP = ",expt_cfg['cp'],"CPMG = ",expt_cfg['cpmg'])
        print ("Number of echoes = ",expt_cfg['echo_times'])
        print("T2 =",p[3],"ns")

    def pulse_probe_ef_iq(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q + alpha

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)


        print ("ef frequency = ",p[2],"GHz")
        print ("Expected anharmonicity =",p[2]-nu_q,"GHz")
        print ("ef pulse probe width = ",p[3]*1e3,"MHz")
        print ("Estimated ef pi pulse time: ",1/(sqrt(2)*2*p[3]),'ns' )

    def ef_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        dt = 0.0625
        print("suggested_pi_length = ", (int(t_pi / dt) + 1) * dt, "suggested_pi_amp = ",
              amp * (t_pi) / float((int(t_pi / dt) + 1) * dt))
        print("suggested_half_pi_length = ", (int(t_half_pi / dt) + 1) * dt, "suggested_piby2_amp = ",
              amp * (t_half_pi) / float((int(t_half_pi / dt) + 1) * dt))

    def ef_t1(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (us)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitexp(t, P, showfit=True)
            plt.show()

        else:p = fitexp(t, P, showfit=False)

        print("T1 =", p[3], "us")

    def ef_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        alpha_new = alpha + offset

        print("Original alpha choice =", alpha, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested alpha = ", alpha_new, "GHz")
        print("T2* =", p[3], "ns")

    def gf_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        alpha_new = alpha + offset

        print("Original alpha choice =", alpha, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested alpha = ", alpha_new, "GHz")
        print("T2* =", p[3], "ns")

    def ef_echo(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'],expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]

        print("Echo experiment: CP = ",expt_cfg['cp'],"CPMG = ",expt_cfg['cpmg'])
        print ("Number of echoes = ",expt_cfg['echo_times'])
        print("T2 =",p[3],"ns")

    def pulse_probe_fh_iq(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q + 2 * alpha

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.' + self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:
            p = fitlor(f, eval('self.' + self.P), showfit=False)

        print("fh frequency = ", p[2], "GHz")
        print("Expected anharmonicity 2 =", p[2] - nu_q - alpha, "GHz")
        print("ef pulse probe width = ", p[3] * 1e3, "MHz")
        print("Estimated ef pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')

    def fh_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def fh_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']
        alpha_fh = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity_fh']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        alpha_new = alpha_fh + offset

        print("Original alpha_fh choice =", alpha_fh, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested alpha_fh = ", alpha_new, "GHz")
        print("T2* =", p[3], "ns")

    def histogram(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ran = self.hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
        readout_length = self.quantum_device_cfg['readout']['length']
        window = self.quantum_device_cfg['readout']['window']
        atten = self.quantum_device_cfg['readout']['dig_atten']

        findfid = expt_cfg['findfid']

        if findfid == 'ef':
            ff0, ff1, title = 1, 2, 'e/f'
        elif findfid == 'gf':
            ff0, ff1, title = 0, 2, 'g/f'
        elif findfid == 'gh':
            ff0, ff1, title = 0, 3, 'g/h'
        elif findfid == 'eh':
            ff0, ff1, title = 1, 3, 'e/h'
        elif findfid == 'fh':
            ff0, ff1, title = 2, 3, 'e/h'
        else:
            ff0, ff1, title = 0, 1, 'g/e'

        if expt_cfg['include_h'] and expt_cfg['show_h']:seq = 4
        else:seq=3


        if expt_cfg['singleshot']:
            a_num = expt_cfg['acquisition_num']
            sample = expt_cfg['sample']
            ns = expt_cfg['num_seq_sets']
            numbins = expt_cfg['numbins']
            rancut = expt_cfg['rancut']

            I = self.I
            Q = self.Q

            I, Q = I/2**15*ran,Q/2**15*ran

            colors = ['r', 'b', 'g','orange']
            labels = ['g', 'e', 'f','h']
            titles = ['-I', '-Q']

            if expt_cfg['include_h']:
                IQs = mean(I[::4], 1), mean(Q[::4], 1), mean(I[1::4], 1), mean(Q[1::4], 1), mean(I[2::4], 1), mean(
                    Q[2::4], 1), mean(I[3::4], 1), mean(Q[3::4], 1)
                IQsss = I.T.flatten()[0::4], Q.T.flatten()[0::4], I.T.flatten()[1::4], Q.T.flatten()[1::4], \
                        I.T.flatten()[2::4], Q.T.flatten()[2::4], I.T.flatten()[3::4], Q.T.flatten()[3::4]
            else:
                IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3],1)
                IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[
                                                                                                            2::3], Q.T.flatten()[
                                                                                                                   2::3]

            fig = plt.figure(figsize=(12,16))

            ax = fig.add_subplot(421, title='length,window = ' + str(readout_length) + ',' + str(window))
            x0g, y0g = mean(IQsss[2 * ff0][::int(a_num / sample)]), mean(IQsss[2 * ff0 + 1][::int(a_num / sample)])
            x0e, y0e = mean(IQsss[2 * ff1][::int(a_num / sample)]), mean(IQsss[2 * ff1 + 1][::int(a_num / sample)])
            phi = arctan((y0e - y0g) / (x0e - x0g))
            for ii in range(seq):
                ax.plot(IQsss[2 * ii][:], IQsss[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

            ax.set_xlabel('I (V)')
            ax.set_ylabel('Q (V)')
            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

            ax = fig.add_subplot(422)

            for ii in range(seq):
                ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
                            yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            ax.set_xlim(x0g - ran / rancut, x0g + ran/rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran/rancut)

            if expt_cfg['include_h']:
                IQsssrot = (I.T.flatten()[0::4] * cos(phi) + Q.T.flatten()[0::4] * sin(phi),
                            -I.T.flatten()[0::4] * sin(phi) + Q.T.flatten()[0::4] * cos(phi),
                            I.T.flatten()[1::4] * cos(phi) + Q.T.flatten()[1::4] * sin(phi),
                            -I.T.flatten()[1::4] * sin(phi) + Q.T.flatten()[1::4] * cos(phi),
                            I.T.flatten()[2::4] * cos(phi) + Q.T.flatten()[2::4] * sin(phi),
                            -I.T.flatten()[2::4] * sin(phi) + Q.T.flatten()[2::4] * cos(phi),
                            I.T.flatten()[3::4] * cos(phi) + Q.T.flatten()[3::4] * sin(phi),
                            -I.T.flatten()[3::4] * sin(phi) + Q.T.flatten()[3::4] * cos(phi))
            else:
                IQsssrot = (I.T.flatten()[0::3] * cos(phi) + Q.T.flatten()[0::3] * sin(phi),
                            -I.T.flatten()[0::3] * sin(phi) + Q.T.flatten()[0::3] * cos(phi),
                            I.T.flatten()[1::3] * cos(phi) + Q.T.flatten()[1::3] * sin(phi),
                            -I.T.flatten()[1::3] * sin(phi) + Q.T.flatten()[1::3] * cos(phi),
                            I.T.flatten()[2::3] * cos(phi) + Q.T.flatten()[2::3] * sin(phi),
                            -I.T.flatten()[2::3] * sin(phi) + Q.T.flatten()[2::3] * cos(phi))

            ax = fig.add_subplot(423, title='rotated')
            x0g, y0g = mean(IQsssrot[2 * ff0][:]), mean(IQsssrot[2 * ff0 + 1][:])
            x0e, y0e = mean(IQsssrot[2 * ff1][:]), mean(IQsssrot[2 * ff1 + 1][:])

            for ii in range(seq):
                ax.plot(IQsssrot[2 * ii][:], IQsssrot[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

            ax.set_xlabel('I (V)')
            ax.set_ylabel('Q (V)')
            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

            ax = fig.add_subplot(424)

            for ii in range(seq):
                ax.errorbar(mean(IQsssrot[2 * ii]), mean(IQsssrot[2 * ii + 1]), xerr=std(IQsssrot[2 * ii]),
                            yerr=std(IQsssrot[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

            ax = fig.add_subplot(4, 2, 5, title='I')
            ax.hist(IQsssrot[2 * ff0], bins=numbins, alpha=0.75, color=colors[ff0])
            ax.hist(IQsssrot[2 * ff1], bins=numbins, alpha=0.75, color=colors[ff1])
            ax.set_xlabel('I' + '(V)')
            ax.set_ylabel('Number')
            ax.legend()
            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)

            ax = fig.add_subplot(4, 2, 6, title='Q')
            ax.hist(IQsssrot[2 * ff0 + 1], bins=numbins, alpha=0.75, color=colors[ff0])
            ax.hist(IQsssrot[2 * ff1 + 1], bins=numbins, alpha=0.75, color=colors[ff1])
            ax.set_xlim(y0g - ran / rancut, y0g + ran / rancut)
            ax.set_xlabel('Q' + '(V)')
            ax.set_ylabel('Number')
            ax.legend()


            for ii, i in enumerate(['I', 'Q']):
                sshg, ssbinsg = np.histogram(IQsss[ii + 2 * ff0], bins=numbins)
                sshe, ssbinse = np.histogram(IQsss[ii + 2 * ff1], bins=numbins)
                fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

                print("Single shot readout fidility from channel ", i, " = ", fid)
            print('---------------------------')

            for ii, i in enumerate(['I', 'Q']):
                if ii is 0:
                    lims = [x0g - ran / rancut, x0g + ran / rancut]
                else:
                    lims = [y0g - ran / rancut, y0g + ran / rancut]
                sshg, ssbinsg = np.histogram(IQsssrot[ii + 2 * ff0], bins=numbins, range=lims)
                sshe, ssbinse = np.histogram(IQsssrot[ii + 2 * ff1], bins=numbins, range=lims)
                fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

                print("Single shot readout fidility from channel ", i, " after rotation = ", fid)
                print("Optimal angle =", phi)

                ax = fig.add_subplot(4, 2, 7 + ii)
                ax.plot(ssbinse[:-1], cumsum(sshg) / sshg.sum(), color=colors[ff0])
                ax.plot(ssbinse[:-1], cumsum(sshe) / sshg.sum(), color=colors[ff1])
                ax.plot(ssbinse[:-1], np.abs(cumsum(sshe) - cumsum(sshg)) / sshg.sum(), color='k')
                if ii == 0:
                    ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
                else:
                    ax.set_xlim(y0g - ran / rancut, y0g + ran / rancut)
                ax.set_xlabel(titles[ii] + '(V)')
                ax.set_ylabel('$F$')
            print('---------------------------')

            fig.tight_layout()
            plt.show()

        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            for ii in range(expt_cfg['num_seq_sets']):
                if ii is 0:
                    ax.plot(self.I[0::3], self.Q[0::3], 'ro', label='g')
                    ax.plot(self.I[1::3], self.Q[1::3], 'bo', label='e')
                    ax.plot(self.I[2::3], self.Q[2::3], 'go', label='f')
                else:
                    ax.plot(self.I[0::3], self.Q[0::3], 'ro')
                    ax.plot(self.I[1::3], self.Q[1::3], 'bo')
                    ax.plot(self.I[2::3], self.Q[2::3], 'go')
                ax.legend()

            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            fig.tight_layout()
            plt.show()

    def sideband_histogram(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        if expt_cfg['singleshot']:
            a_num = expt_cfg['acquisition_num']
            ns = expt_cfg['num_seq_sets']
            numbins = expt_cfg['numbins']

            I = self.I
            Q = self.Q

            colors = ['r', 'b', 'g']
            labels = ['g', 'e', 'f']
            titles = ['-I', '-Q']

            IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3],1)
            IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[2::3], Q.T.flatten()[2::3]

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(421, title='Averaged over acquisitions')

            for j in range(ns):
                for ii in range(3):
                    ax.plot(IQs[2 * ii], IQs[2 * ii + 1], 'o', color=colors[ii])
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            ax = fig.add_subplot(422, title='Mean and std all')

            for ii in range(3):
                ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
                            yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            for kk in range(6):
                ax = fig.add_subplot(4, 2, kk + 3, title=self.exptname + titles[kk % 2])
                ax.hist(IQsss[kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
                ax.legend()
                ax.set_xlabel('Value'+titles[kk % 2])
                ax.set_ylabel('Number')


            for ii,i in enumerate(['I','Q']):
                sshg, ssbinsg = np.histogram(IQsss[ii], bins=numbins)
                sshe, ssbinse = np.histogram(IQsss[ii+2], bins=numbins)
                fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

                print("Single shot g-e readout fidility from channel ", i, " = ", fid)

        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            for ii in range(expt_cfg['num_seq_sets']):
                if ii is 0:
                    ax.plot(self.I[0::3], self.Q[0::3], 'ro', label='g')
                    ax.plot(self.I[1::3], self.Q[1::3], 'bo', label='e')
                    ax.plot(self.I[2::3], self.Q[2::3], 'go', label='f')
                else:
                    ax.plot(self.I[0::3], self.Q[0::3], 'ro')
                    ax.plot(self.I[1::3], self.Q[1::3], 'bo')
                    ax.plot(self.I[2::3], self.Q[2::3], 'go')
                ax.legend()

            ax.set_xlabel('I')
            ax.set_ylabel('Q')

        fig.tight_layout()
        plt.show()

    def qubit_temperature(self):
        expt_cfg = self.experiment_cfg['ef_rabi']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:]
        contrast = []

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, title=self.exptname)
        for i in range(2):
            ax.plot(t, P[i], 'bo-', label='ge_pi = ' + str(i is 0))
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)

            ax.legend()

            p = fitdecaysin(t[2:], P[i][2:], showfit=True)
            contrast.append(p[0])

            ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
            ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')

            print("Half pi length =", 1 / (4 * p[1]), "ns")
            print("pi length =", 1 / (2 * p[1]), "ns")

        if self.show:
            plt.show()

        ratio = abs(contrast[1] / contrast[0])
        print("Qubit Temp:", 1e3 * temperature_q(nu_q * 1e9, ratio), " mK")
        print("Contrast ratio:",ratio)
        print("Qubit Excited State Occupation:", occupation_q(nu_q, 1e3 * temperature_q(nu_q, ratio)))

    def histogram_sweep(self):
        expt_cfg = self.experiment_cfg['histogram']
        a_num = expt_cfg['acquisition_num']
        ns = expt_cfg['num_seq_sets']
        numbins = expt_cfg['numbins']
        swp_cfg = self.experiment_cfg['histogram_sweep']
        attens = np.arange(swp_cfg['atten_start'], swp_cfg['atten_stop'], swp_cfg['atten_step'])
        freqs = np.arange(swp_cfg['freq_start'], swp_cfg['freq_stop'], swp_cfg['freq_step'])
        sweep_amp = swp_cfg['sweep_amp']
        colors = ['r', 'b', 'g']
        labels = ['g', 'e', 'f']
        titles = ['-I', '-Q']

        if sweep_amp:x = attens[:]
        else:x = freqs[:]

        if expt_cfg['singleshot']:

            I = self.I.flatten().reshape(len(x),3*ns,a_num)
            Q = self.Q.flatten().reshape(len(x),3*ns,a_num)

            IQsss = array([(i.T.flatten()[0::3], Q[ii].T.flatten()[0::3],
                            i.T.flatten()[1::3], Q[ii].T.flatten()[1::3],
                            i.T.flatten()[2::3], Q[ii].T.flatten()[2::3]) for ii, i in enumerate(I)])

            Is = mean(I, axis=2).reshape(len(x), ns, 3)
            Qs = mean(Q, axis=2).reshape(len(x), ns, 3)

            fidsI,fidsQ  = [],[]

            for k, f in enumerate(x):
                for ii, i in enumerate(['I', 'Q']):
                    sshg, ssbinsg = np.histogram(IQsss[k][ii], bins=numbins)
                    sshe, ssbinse = np.histogram(IQsss[k][ii + 2], bins=numbins)
                    fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / (sshg.sum() / 2.0 + sshe.sum() / 2.0))).max()
                    if ii == 0:fidsI.append(fid)
                    else:fidsQ.append(fid)

            arg = argmax(eval('fids' + chan))
            fig = plt.figure(figsize=(15, 15))

            ax = fig.add_subplot(521, title=expt_name + '-averaged')
            for j in range(ns):
                for ii in range(3):
                    ax.plot(Is[arg][j][ii], Qs[arg][j][ii], 'o', color=colors[ii])
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            ax = fig.add_subplot(522, title=expt_name + '- mean and std ')

            for ii in range(3):
                ax.errorbar(mean(IQsss[arg][2 * ii]), mean(IQsss[arg][2 * ii + 1]), xerr=std(IQsss[arg][2 * ii]),
                            yerr=std(IQsss[arg][2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            for kk in range(6):
                ax = fig.add_subplot(5, 2, kk + 3, title=expt_name + titles[kk % 2])
                ax.hist(IQsss[arg][kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
                ax.set_xlabel('Value' + titles[kk % 2])
                ax.set_ylabel('Number')
                ax.legend()

            ax = fig.add_subplot(515, title='Single shot readout fidelity')
            ax.plot(x, fidsI, 'bo-', label='I')
            ax.plot(x, fidsQ, 'ro-', label='Q')
            axvline(x[arg])
            ax.legend()
            if sweep_amp:
                ax.set_xlabel('dig atten (dB)')
                print('Optimal readout amplitude =  ', x[arg], 'dB')
            else:
                ax.set_xlabel('Freq (GHz)')
                print('Optimal readout freq =  ', x[arg], 'GHz')
            ax.set_ylabel('Single shot readout fidelity')
            fig.tight_layout()

        else:
            print ("Set singleshot to True")

        fig.tight_layout()
        plt.show()

    def sideband_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def sideband_rabi_freq_scan(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        sideband_freq = expt_cfg['freq']
        P = eval('self.' + self.P)
        df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        freqs = sideband_freq + df
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(freqs, P, 'o-', label=self.P)
            ax.set_xlabel('Sideband freq (GHz)')
            ax.set_ylabel(self.P)
            ax.legend()
            plt.show()

        else:
            pass

    def sideband_t1(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]/1e3

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (us)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitexp(t, P, showfit=True)
            plt.show()

        else:p = fitexp(t, P, showfit=False)

        print("T1 =", p[3], "us")

    def sideband_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        suggested_dc_offset = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['f0g1_dc_offset']['mode_index'] + offset
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested dc offset =", suggested_dc_offset * 1e3, "MHz")
        print("T2 =", p[3], "ns")

    def sideband_pi_pi_offset(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        phase = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(phase, P, 'o-', label=self.P)
            ax.set_xlabel('Phase (rad)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitsin3(t, P, showfit=True)
            plt.show()

        else:p = fitsin3(t, P, showfit=False)
        offset_phase = p[2]-np.pi/2.0
        print("pi pi offset phase =? ", offset_phase / 2, "rad")
        ax.axvline(offset_phase,color='k',linestyle='dashed')

    def sideband_rabi_two_tone_freq_scan(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        sideband_freq = expt_cfg['center_freq']+expt_cfg['offset_freq']
        P = eval('self.' + self.P)
        df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        freqs = df
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(freqs, P, 'o-', label=self.P)
            ax.set_xlabel('Detuning (GHz)')
            ax.set_ylabel(self.P)
            ax.legend()
            plt.show()

        else:
            pass

    def sideband_rabi_two_tone(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def sideband_pulse_probe_iq(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.' + self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:
            p = fitlor(f, eval('self.' + self.P), showfit=False)

        print("Qubit frequency = ", p[2], "GHz")
        print("Pulse probe width = ", p[3] * 1e3, "MHz")
        print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')

    def sideband_chi_ge_calibration(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]

        suggested_chi_shift = (2 *self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_e'][expt_cfg['mode_index']] + offset) / 2.0
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested ge chi shift = 2 pi x", suggested_chi_shift * 1e3, "MHz")
        print("T2 =", p[3], "ns")

    def sideband_chi_ef_calibration(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]

        suggested_chi_shift = (2 * quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_ef'] + offset) / 2.0
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested ge chi shift = 2 pi x", suggested_chi_shift * 1e3, "MHz")
        print("T2 =", p[3], "ns")

    def sideband_chi_gf_calibration(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]

        suggested_chi_shift = (2 * quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['chiby2pi_f'] + offset) / 2.0
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested ge chi shift = 2 pi x", suggested_chi_shift * 1e3, "MHz")
        print("T2 =", p[3], "ns")

    def sideband_reset_qubit_temperature(self):
        expt_cfg = self.experiment_cfg['sideband_transmon_reset']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:]
        contrast = []

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, title=self.exptname)
        for i in range(2):
            ax.plot(t, P[i], 'bo-', label='ge_pi = ' + str(i is 0))
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)

            ax.legend()

            p = fitdecaysin(t[2:], P[i][2:], showfit=True)
            contrast.append(p[0])

            ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
            ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')

            print("Half pi length =", 1 / (4 * p[1]), "ns")
            print("pi length =", 1 / (2 * p[1]), "ns")

        if self.show:
            plt.show()

        ratio = abs(contrast[1] / contrast[0])
        print("Qubit Temp:", 1e3 * temperature_q(nu_q * 1e9, ratio), " mK")
        print("Contrast ratio:",ratio)
        print("Qubit Excited State Occupation:", occupation_q(nu_q, 1e3 * temperature_q(nu_q, ratio)))

    def qp_pumping_t1(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))] / 1e3

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (us)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitexp(t, P, showfit=True)
            plt.show()

        else:
            p = fitexp(t, P, showfit=False)

        print("T1 =", p[3], "us")

    def sideband_parity_measurement(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['num_expts'])

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Experiment number)')
            ax.set_ylabel(self.P)
            ax.legend()
            plt.show()

    def sideband_cavity_photon_number(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)

        print("Qubit frequency = ", p[2], "GHz")
        print("Pulse probe width = ", p[3] * 1e3, "MHz")
        print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')

    def direct_cavity_spectroscopy(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_c = self.quantum_device_cfg['cavity'][expt_cfg['on_cavities'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_c

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)

        print("Cavity frequency = ", p[2], "GHz")
        print("Pulse probe width = ", p[3] * 1e3, "MHz")

    def cavity_drive_pulse_probe_iq(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)

        print("Qubit frequency = ", p[2], "GHz")
        print("Pulse probe width = ", p[3] * 1e3, "MHz")
        print("Estimated pi pulse time: ", 1 / (sqrt(2) * 2 * p[3]), 'ns')

    def sideband_pi_pi_offset(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        phase = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        p = fitsin3(phase[:], P[:], showfit=False)
        offset_phase = (p[2] + pi / 2)
        print("pi pi offset phase = ", offset_phase, "rad")

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(phase, P, 'bo--', markersize=10.0, label=show)
            ax.set_xlabel('$\Phi$ (rad)')
            ax.set_ylabel(show)
            ax.legend()
            p = fitsin3(phase[:], P[:], showfit=True)
            ax.axvline(offset_phase, color='k', linestyle='dashed')
            plt.show()

    def wigner_tomography_test_sideband_only(self):


        expt_cfg = self.experiment_cfg['wigner_tomography_test_sideband_only']

        time = expt_cfg['cavity_pulse_len']
        print("Cavity pulse time = ", time, "ns")

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title='Wigner tomograph for state =  ' + str(expt_cfg['state']))

            P = eval('self.' + self.P)


            W = -2 / pi * (2 * P - 1)
            r, t = meshgrid(amps, phases)

            x = r * np.cos(t)
            y = r * np.sin(t)
            plt.pcolormesh(x, y, W, cmap='RdBu')

            ax.set_xlabel('$\\propto Re(\\alpha$)')
            ax.set_ylabel('$\\propto Im(\\alpha$)')
            plt.colorbar()
            plt.show()

    def wigner_tomography_sideband_only_phase_sweep(self):


        expt_cfg = self.experiment_cfg['wigner_tomography_test_sideband_only']
        swp_cfg = self.experiment_cfg['wigner_tomography_sideband_only_phase_sweep']

        time = expt_cfg['cavity_pulse_len']
        print("Cavity pulse time = ", time, "ns")

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title='Wigner tomograph for state =  ' + str(expt_cfg['state']))

            P = eval('self.' + self.P)
            phases = arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])[:len(P)]
            amps = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:]

            W = -2 / pi * (2 * P - 1)
            r, t = meshgrid(amps, phases)

            x = r * np.cos(t)
            y = r * np.sin(t)
            plt.pcolormesh(x, y, W, cmap='RdBu')

            ax.set_xlabel('$\\propto Re(\\alpha$)')
            ax.set_ylabel('$\\propto Im(\\alpha$)')
            plt.colorbar()
            plt.show()

    def sideband_transmon_ge_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def sideband_transmon_pulse_probe_ef(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q + alpha

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)


        print ("ef frequency = ",p[2],"GHz")
        print ("Expected anharmonicity =",p[2]-nu_q,"GHz")
        print ("ef pulse probe width = ",p[3]*1e3,"MHz")
        print ("Estimated ef pi pulse time: ",1/(sqrt(2)*2*p[3]),'ns' )

    def sideband_transmon_pulse_probe_ge(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(self.I))] + nu_q

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(f, self.I, 'b.-', label='I')
            ax.plot(f, self.Q, 'r.-', label='Q')
            ax.set_xlabel('Freq(GHz)')
            ax.set_ylabel('I/Q')
            ax.legend()
            p = fitlor(f, eval('self.'+self.P), showfit=False)
            ax.plot(f, lorfunc(p, f), 'k--')
            ax.axvline(p[2], color='g', linestyle='dashed')
            plt.show()
        else:p = fitlor(f, eval('self.'+self.P), showfit=False)


        print ("ge frequency = ",p[2],"GHz")
        print ("gr pulse probe width = ",p[3]*1e3,"MHz")
        print ("Estimated ge pi pulse time: ",1/(sqrt(2)*2*p[3]),'ns' )

    def sideband_transmon_ef_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def sideband_transmon_ge_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        nu_q_new = nu_q + offset

        print("Original qubit frequency choice =", nu_q, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested qubit frequency choice =", nu_q_new, "GHz")
        print("T2* =", p[3], "ns")

    def sideband_transmon_ef_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity']

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        alpha_new = alpha + offset

        print("Original alpha choice =", alpha, "GHz")
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested alpha = ", alpha_new, "GHz")
        print("T2* =", p[3], "ns")

    def sideband_f0g1rabi_freq_scan(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        sideband_freq = expt_cfg['freq']
        P = eval('self.' + self.P)
        df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        freqs = sideband_freq + df
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(freqs, P, 'o-', label=self.P)
            ax.set_xlabel('Sideband freq (GHz)')
            ax.set_ylabel(self.P)
            ax.legend()
            plt.show()

        else:
            pass

    def sideband_f0g1rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

    def sideband_f0g1ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)

        offset = ramsey_freq - p[1]
        suggested_dc_offset = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]][
                                  'f0g1_dc_offset'] + offset
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested dc offset =", suggested_dc_offset * 1e3, "MHz")
        print("T2 =", p[3], "ns")

    def sideband_f0g1_pi_pi_offset(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        phase = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        p = fitsin3(phase[:], P[:], showfit=False)
        offset_phase = (p[2] + pi / 2)
        print("pi pi offset phase = ", offset_phase, "rad")

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(phase, P, 'bo--', markersize=10.0, label=show)
            ax.set_xlabel('$\Phi$ (rad)')
            ax.set_ylabel(show)
            ax.legend()
            p = fitsin3(phase[:], P[:], showfit=True)
            ax.axvline(offset_phase, color='k', linestyle='dashed')
            plt.show()

    def wigner_tomography_2d_sideband_alltek2(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)


        W = - 1 / np.pi * (2 * P - 1)

        x = arange(expt_cfg['startx'], expt_cfg['stopx'], expt_cfg['stepx'])
        y = arange(expt_cfg['starty'], expt_cfg['stopy'], expt_cfg['stepy'])

        W2d = W.reshape(len(x), len(y))

        if self.show:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, title='$\\psi =$ ' + expt_cfg['state'])

            plt.pcolormesh(x, y, W2d, cmap='RdBu')
            clim(-1/np.pi,1/np.pi)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y[0], y[-1])

            ax.set_xlabel('$\\propto Re(\\alpha$)')
            ax.set_ylabel('$\\propto Im(\\alpha$)')

            plt.colorbar()
            plt.show()

    def cavity_drive_ramsey(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        ramsey_freq = expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        nu_c = self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][expt_cfg['mode_index']]

        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[1:], P[1:], showfit=False)
            p = fitdecaysin(t[:], P[:], fitparams=p, showfit=True)
            plt.show()
        else:
            p = fitdecaysin(t[1:], P[1:], showfit=False)
            p = fitdecaysin(t, P, showfit=False)
        offset = ramsey_freq - p[1]
        suggested_dc_offset = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['f0g1_dc_offset'][expt_cfg['mode_index']] + offset
        print("Offset freq =", offset * 1e3, "MHz")
        print("Suggested dc offset =", suggested_dc_offset * 1e3, "MHz")
        print("T2 =", p[3], "ns")
        nu_c_new = nu_c + offset
        return nu_c_new

    def weak_rabi(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.' + self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        amp = expt_cfg['amp']
        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t[2:], P[2:], showfit=True)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            plt.show()

        else:
            p = fitdecaysin(t, P, showfit=False)
            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])

        print("Half pi length =", t_half_pi, "ns")
        print("pi length =", t_pi, "ns")
        dt = 0.0625
        print("suggested_pi_length = ", (int(t_pi / dt) + 1) * dt, "suggested_pi_amp = ",
              amp * (t_pi) / float((int(t_pi / dt) + 1) * dt))
        print("suggested_half_pi_length = ", (int(t_half_pi / dt) + 1) * dt, "suggested_piby2_amp = ",
              amp * (t_half_pi) / float((int(t_half_pi / dt) + 1) * dt))

    def save_cfg_info(self, f):
        f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
        f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
        f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
        f.close()

    def optimal_control_test(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        P = eval('self.'+self.P)
        t = arange(0.0, 1.0, expt_cfg['steps'])[:(len(P))]
        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Evolution')
            ax.set_ylabel(self.P)
            ax.legend()
            plt.show()
        print("Final state: ", P[-1])

    def cavity_spectroscopy_resolved_qubit_pulse(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_c = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_cavities']]['cavity_freqs'][expt_cfg['mode_index']]
        amp = expt_cfg['cavity_amp']
        length = expt_cfg['cavity_pulse_len']

        P = eval('self.'+self.P)
        df = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:(len(P))]
        freqs = nu_c + df

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title='$\\tau$ = ' + str(length / 1e3) + ' us')
            ax.plot(freqs, P, 'bo-', label='a = ' + str(amp), linewidth=3)
            p = fitlor(freqs, -P, showfit=False)
            ax.plot(freqs, -lorfunc(p, freqs), 'r-')
            ax.set_xlabel('Cavity freq (GHz)')
            ax.set_ylabel('$P_0$')
            ax.axvline(p[2], color='r', linestyle='dashed', alpha=0.25)
            ax.legend()
            ax.set_ylim(-0.1, 1.1)
            plt.show()
        else:
            p = fitlor(freqs, -P, showfit=False)
        print("Resonance frequency = ", p[2])
        print("Width = ", p[3] * 1e3, "MHz")
        print("========================================")
        return p[2]


    def photon_number_resolved_qubit_spectroscopy(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_c = self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][expt_cfg['mode_index']]
        nu_q = self.quantum_device_cfg['qubit']['1']['freq']
        chi = self.quantum_device_cfg['flux_pulse_info']['1']['chiby2pi_e'][expt_cfg['mode_index']]
        if chi == 0:
            chi = -0.3e-3
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:] + nu_q
        I = self.I
        N = 3

        p = 0 * ones(3 * N + 1)
        for ii in range(N):
            p[3 * ii + 1] = 1
            p[3 * ii + 2] = nu_q + 2 * chi * ii
            p[3 * ii + 3] = 0.0001

        def gaussfuncsum(p, x):
            """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
            y = 0
            for ii in range(N):
                y = y + p[3 * ii + 1] * exp(-(x - (p[3 * ii + 2])) ** 2 / 2 / p[3 * ii + 3] ** 2)
            return y

        def fitgausssum(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False, label="",
                        debug=False):
            """fit lorentzian:
                returns [offset,amplitude,center,hwhm]"""
            if domain is not None:
                fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
            else:
                fitdatax = xdata
                fitdatay = ydata
            if fitparams is None:
                fitparams = 0 * ones(3 * N + 1)
                fitparams[0] = (fitdatay[0] + fitdatay[-1]) / 2.
                fitparams[1] = max(fitdatay) - min(fitdatay)
                fitparams[2] = fitdatax[np.argmax(fitdatay)]
                fitparams[3] = (max(fitdatax) - min(fitdatax)) / 10.
            if debug == True: print(fitparams)
            p1 = fitgeneral(fitdatax, fitdatay, gaussfuncsum, fitparams, domain=None, showfit=showfit,
                            showstartfit=showstartfit,
                            label=label)
            p1[3] = abs(p1[3])
            return p1

        p = fitgausssum(f, I, fitparams=p, showfit=False)
        pfit = p
        Y = abs(pfit[2::3])
        X = abs(pfit[1::3])
        nus = -sort(-Y)
        chi_n = nus[1] / 2 - nus[0] / 2

        if self.show:
            fig = plt.figure(figsize=(14,7))
            ax = fig.add_subplot(111, title='mode = ' + str(expt_cfg['mode_index']))
            ax.plot(f, I, 'b.--')
            for ii in range(6):
                ax.axvline(nu_q + 2 * ii * chi_n, color='g', linestyle='dashed', alpha=0.25)
            ax.set_xlabel('Freq (GHz)')
            ax.set_ylabel('$P_e$')
            ax.set_ylim(0.0, 1.0)
            plt.show()
        print("chi = ", chi_n)
        return chi_n


    def blockade_experiments_cavity_spectroscopy(self):
        expt_cfg = self.experiment_cfg[self.exptname]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        nu_c = self.quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][expt_cfg['mode_index']]
        print("use weak drive for dressing:", expt_cfg['use_weak_drive_for_dressing'])
        print("use weak drive for probe:", expt_cfg['use_weak_drive_for_probe'])
        print("dressing amp:", expt_cfg['dressing_amp'])
        print("cavity drive amp:", expt_cfg['cavity_amp'])
        print("cavity awg scales :", self.hardware_cfg['awg_info']['m8195a']['amplitudes'])
        print("prep state before blockade = ", expt_cfg['prep_state_before_blockade'])
        P = eval('self.'+self.P)
        f = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:] * 1e3

        if self.show:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title="$\\nu_c$ = " + str(around(nu_c, 3)) + " GHz")
            ax.plot(f, P, 'o--', markersize=7)
            p = fitlor(f, -P, showfit=True)
            ax.plot(f, -lorfunc(p, f), 'k--')
            axvline(p[2], color='r', linestyle='dashed')
            ax.set_xlabel('$\\delta\\nu$ (MHz)')
            ax.set_ylabel('$P_{n}$')
            ax.set_ylim(0, 1)
            plt.show()
        else:
            p = fitlor(f, -P, showfit=False)

        print("offset = ", p[2])
        print("new cavity offset freq (MHz) = ", p[2] + expt_cfg['cavity_offset_freq'] * 1e3)
        print("new cavity freq = ", nu_c + p[2] * 1e-3)
        return p[2] / 1e3 + expt_cfg['cavity_offset_freq']


    def sequential_multitone_blockaded_cavity_rabi_vary_probe_level(self):
        swp_cfg = self.experiment_cfg[self.exptname]
        use_weak_cav = experiment_cfg['blockade_experiments_cavity_spectroscopy']['weak_cavity']
        if use_weak_cav:
            expt_cfg = self.experiment_cfg['multitone_blockaded_weak_cavity_rabi']
        else:
            expt_cfg = self.experiment_cfg['multitone_blockaded_cavity_rabi']
        nu_c = self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][expt_cfg['mode_index']]
        nu_q = self.quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['freq']
        print("=============== file num = ", f, " ==========================")
        print("use weak drive for dressing:", expt_cfg['use_weak_drive_for_dressing'])
        print("use weak drive for probe:", expt_cfg['use_weak_drive_for_probe'])
        print("dressing amp:", expt_cfg['dressing_amp'])
        print("cavity drive amp:", expt_cfg['cavity_amp'])
        print("cavity awg scales :", hardware_cfg['awg_info']['m8195a']['amplitudes'])
        P = eval('self.'+self.P)
        t = arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])[:] / 1e3
        ns = arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])[:]
        fitlist = [0,1]
        pi_lens = []
        if self.show:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, title="$\\nu_c$ = " + str(around(nu_c, 3)) + " GHz")
            for ii, pp in enumerate(P):
                ax.plot(t, pp[:len(t)], 'o--', label='$P_{%s}$' % (int(ns[ii])), markersize=7)
                if ii in fitlist:
                    f = fitdecaysin(t, pp, fitparams=[0.5, 1 / (2 * 12.0), 0, 10.0, 0.5], showfit=False)
                    ax.plot(t, decaysin(append(f, 0), t), 'r--')
                    ax.axvline(1 / f[1] / 2, color='k', linestyle='dashed')
                    print("pi length = ", 1 / f[1] / 2.0)
                    pi_lens.append(1 / f[1] / 2.0)
            ax.set_xlabel('Time ($\\mu$s)')
            ax.set_ylabel('$P_{n}$')
            ax.set_xlim(0, 56)
            ax.set_ylim(0, 1)
            ax.axhline(0.08)
            plt.show()
        else:
            for ii, pp in enumerate(P):
                if ii in fitlist:
                    f = fitdecaysin(t, pp, fitparams=[0.5, 1 / (2 * 12.0), 0, 10.0, 0.5], showfit=False)
                    print("pi length = ", 1 / f[1] / 2.0)
                    pi_lens.append(1 / f[1] / 2.0)
        return pi_lens


def temperature_q(nu, rat):
    Kb = 1.38e-23
    h = 6e-34
    return h * nu / (Kb * log(1 / rat))

def occupation_q(nu, T):
    Kb = 1.38e-23
    h = 6e-34
    T = T * 1e-3
    return 1 / (exp(h * nu / (Kb * T)) + 1)
