from slab.dsfit import*
import matplotlib.pyplot as plt
from numpy import*

class PostExperiment:

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, I , Q, P = 'Q', show = True):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.exptname = experiment_name
        self.I = I
        self.Q = Q
        self.P = P
        self.show = show

        try:eval('self.' + experiment_name)()
        except:print("No post experiment analysis yet")
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
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_piby2_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

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
        print("suggested_pi_length = ", int(t_pi) + 1, "suggested_pi_amp = ", amp * (t_pi) / float(int(t_pi) + 1))
        print("suggested_half_pi_length = ", int(t_half_pi) + 1, "suggested_half_pi_amp = ",
              amp * (t_half_pi) / float(int(t_half_pi) + 1))

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

    def histogram(self):
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
