from slab.dsfit import*
import matplotlib.pyplot as plt
from numpy import*

class PostExperiment:

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, I , Q, P, show):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.expt_cfg = self.experiment_cfg[experiment_name]
        self.exptname = experiment_name
        self.I = I
        self.Q = Q
        self.P = P
        self.show = show

        eval('self.' + experiment_name)()

    def resonator_spectroscopy(self):
        pass

    def resonator_spectroscopy_sweep(self):
        pass

    def pulse_probe_iq(self):

        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)

        f = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(self.I))] + nu_q

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
        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
            ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        print("half pi length =", 1 / (4 * p[1]), "ns")
        print("pi length =", 1 / (2 * p[1]), "ns")

    def t1(self):
        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]/1e3

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
        ramsey_freq = self.expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

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
        ramsey_freq = self.expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

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

        print("Echo experiment: CP = ",self.expt_cfg['cp'],"CPMG = ",self.expt_cfg['cpmg'])
        print ("Number of echoes = ",self.expt_cfg['echo_times'])
        print("T2 =",p[3],"ns")

    def pulse_probe_ef_iq(self):
        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['anharmonicity']

        self.I = self.I - mean(self.I)
        self.Q = self.Q - mean(self.Q)
        f = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(self.I))] + nu_q + alpha

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
        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

        if self.show:

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, title=self.exptname)
            ax.plot(t, P, 'o-', label=self.P)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(self.P)
            ax.legend()
            p = fitdecaysin(t, P, showfit=True)
            ax.axvline(1 / (2 * p[1]), color='k', linestyle='dashed')
            ax.axvline(1 / (4 * p[1]), color='k', linestyle='dashed')
            plt.show()

        else:p = fitdecaysin(t, P, showfit=False)

        print("half pi length =", 1 / (4 * p[1]), "ns")
        print("pi length =", 1 / (2 * p[1]), "ns")

    def ef_t1(self):
        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]/1e3

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
        ramsey_freq = self.expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']
        alpha = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['anharmonicity']

        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

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

        ramsey_freq = self.expt_cfg['ramsey_freq']
        nu_q = self.quantum_device_cfg['qubit'][self.expt_cfg['on_qubits'][0]]['freq']

        P = eval('self.'+self.P)
        t = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])[:(len(P))]

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

        print("Echo experiment: CP = ",self.expt_cfg['cp'],"CPMG = ",self.expt_cfg['cpmg'])
        print ("Number of echoes = ",self.expt_cfg['echo_times'])
        print("T2 =",p[3],"ns")

    def histogram(self):
        pass

