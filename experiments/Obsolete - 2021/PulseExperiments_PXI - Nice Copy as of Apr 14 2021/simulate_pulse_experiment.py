
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from qutip import *
import matplotlib
from tqdm import tqdm
import json

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)


from slab.datamanagement import SlabFile
from slab.dataanalysis import get_next_filename
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperiment
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.dsfit import*



class Simulate_Multimode_Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, multimode_params=None, RWA=False, dtsim=1.0,
                 sequences=None, name=None):

        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.params = multimode_params
        self.dtsim = dtsim

        self.I = None
        self.Q = None

        self.JC_hamiltonian(RWA)

    def transmon_hamiltonian(self, ng):

        Ej, Ec, N = self.params["Ej"], self.params["Ec"], self.params["N"]
        m = np.diag(4 * Ec * (arange(-N, N + 1) - ng) ** 2) - Ej / 2.0 * (
        (np.diag(np.ones(2 * N), -1) + np.diag(np.ones(2 * N), 1)))

        return Qobj(m)

    def plot_transmon_energies(self, ng_vec, ylim=(0, 10)):

        energies = array([self.transmon_hamiltonian(ng).eigenstates()[0] for ng in ng_vec])
        print("Qubit frequency = %s" % (energies.T[1][0] - energies.T[0][0]))
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))

        for n in range(len(energies[0, :])):
            axes.plot(ng_vec, (energies[:, n] - energies[:, 0]))
        axes.set_ylim(ylim[0], ylim[1])
        axes.set_xlim(ng_vec[0], ng_vec[-1])
        axes.set_xlabel(r'$n_g$', fontsize=18)
        axes.set_ylabel(r'$E_n$', fontsize=18)

        return fig, axes

    def JC_hamiltonian(self, RWA=False):

        N_q, N_r, N_m = self.params["truncation"]
        omega_r, omega_m = 2 * pi * array(self.params["nus"])
        omega_gr, omega_gm = 2 * pi * array(self.params["gs"])

        energies = self.transmon_hamiltonian(0).eigenstates()[0]  # energies in GHz
        omega_q = 2 * pi * Qobj(diag(energies[0:N_q] - energies[0]))  # converting to angular frequency

        self.ar = tensor(destroy(N_r), qeye(N_m), qeye(N_q))
        self.am = tensor(qeye(N_r), destroy(N_m), qeye(N_q))
        self.b = tensor(qeye(N_r), qeye(N_m), destroy(N_q))

        if RWA:
            self.H = omega_r * self.ar.dag() * self.ar + omega_m * self.am.dag() * self.am + tensor(qeye(N_r),
                                                                                                    qeye(N_m),
                                                                                                    omega_q) + omega_gr * (
            self.ar.dag() * self.b + self.ar * self.b.dag()) + omega_gm * (
            self.am.dag() * self.b + self.am * self.b.dag())
        else:
            self.H = omega_r * self.ar.dag() * self.ar + omega_m * self.am.dag() * self.am + tensor(qeye(N_r),
                                                                                                    qeye(N_m),
                                                                                                    omega_q) + omega_gr * (
            self.ar.dag() + self.ar) * (self.b.dag() + self.b) + omega_gm * (self.am.dag() + self.am) * (
            self.b.dag() + self.b)

        self.jc_energies, self.jc_vectors = self.H.eigenstates()[0], self.H.eigenstates()[1]
        self.nr_vec = array([expect(self.ar.dag() * self.ar, jc_vector) for jc_vector in self.jc_vectors])
        self.nm_vec = array([expect(self.am.dag() * self.am, jc_vector) for jc_vector in self.jc_vectors])
        self.nq_vec = array([expect(self.b.dag() * self.b, jc_vector) for jc_vector in self.jc_vectors])

        return self.H

    def JC_spectrum_states(self):

        level_number = arange(len(self.jc_energies)) + 1
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.plot(level_number, self.jc_energies / (2 * pi), 'k.--', markersize=10)
        ax.set_xlabel('Level number')
        ax.set_ylabel('Frequency (GHz)')
        ax2 = ax.twinx()
        ax2.plot(level_number, self.nr_vec, 'rs', label='$n_{r}$', alpha=0.5)
        ax2.plot(level_number, self.nm_vec, 'gs', label='$n_{m}$', alpha=0.5)
        ax2.plot(level_number, self.nq_vec, 'bd', label='$n_{q}$', alpha=0.5)
        ax2.set_yticks(arange(5))
        ax2.set_ylabel('$n_{r,q}$', color='k')
        ax2.tick_params('y', colors='k')
        ax2.legend()
        print("|q r>")
        for ii in arange(len(level_number)):
            print("|%s,%s,%s> : %s GHz" % (
            int(round(self.nq_vec[ii])), int(round(self.nr_vec[ii])), int(round(self.nm_vec[ii])),
            around(self.jc_energies[ii] / (2 * pi), 10)))

    def find_freq(self, nq, nr, nm):
        return self.jc_energies[argmin(abs(self.nr_vec - nr) + abs(self.nm_vec - nm) + abs(self.nq_vec - nq))] / (
        2 * pi)

    def projector(self, nq, nr, nm):
        ind = argmin(abs(self.nr_vec - nr) + abs(self.nm_vec - nm) + abs(self.nq_vec - nq))
        return self.jc_vectors[ind] * self.jc_vectors[ind].dag()

    def state(self, nq, nr, nm):
        ind = argmin(abs(self.nr_vec - nr) + abs(self.nm_vec - nm) + abs(self.nq_vec - nq))
        return self.jc_vectors[ind]

    def Udress(self):
        N_q, N_r, N_m = self.params["truncation"]
        op = 0

        for ii, nq in enumerate(self.nq_vec):
            nq_r = int(round(nq))
            nr_r = int(round(self.nr_vec[ii]))
            nm_r = int(round(self.nm_vec[ii]))
            op += self.jc_vectors[ii] * tensor(basis(N_r, nr_r), basis(N_m, nm_r), basis(N_q, nq_r)).dag()

        return op

    def lindblad_dissipator(self):

        kappa_q, kappa_r, kappa_m = 1 / (array(self.params["T1s"]))
        n_thq, n_thr, n_thm = self.params["nths"]

        c_ops = kappa_q * (1 + n_thq) * lindblad_dissipator(self.b) + kappa_q * (n_thq) * lindblad_dissipator(
            self.b.dag()) + kappa_r * (1 + n_thr) * lindblad_dissipator(self.ar) + kappa_r * (
        n_thr) * lindblad_dissipator(self.ar.dag()) \
                + kappa_m * (1 + n_thm) * lindblad_dissipator(self.am) + kappa_m * (n_thm) * lindblad_dissipator(
            self.am.dag())

        return c_ops

    def H_t(self):

        def Hd_coeff(t, args):
            xi = args['xi']
            nud = args['nud']
            return 2 * pi * xi * sin((2 * pi * nud) * t)

        Hd = (self.ar + self.ar.dag())
        return [self.H, [Hd, Hd_coeff]]

    def rabisquaremesolve(self, tlist=np.linspace(0, 300.0, 601), xi=0.1, psi0=None, nu_d=0):

        print("Drive frequency - %s GHz" % (nu_d))
        print("Drive amplitude - %s GHz" % (xi))

        c_ops = self.lindblad_dissipator()

        ard = self.ar.transform(self.Udress())
        amd = self.am.transform(self.Udress())
        bd = self.b.transform(self.Udress())

        output = mesolve(self.H_t(), psi0, tlist, c_ops, [ard.dag() * ard, amd.dag() * amd, bd.dag() * bd],
                         args={'xi': xi, 'nud': nu_d}, progress_bar=True, options=Odeoptions(nsteps=12000))

        return output

    def sequenceslist(self, sequences, waveform_channels):
        wv = []
        for channel in waveform_channels:
            if not channel == None:
                wv.append(sequences[channel])
            else:
                wv.append(np.zeros_like(sequences[waveform_channels[0]]))
        return wv

    def get_qubit_iq_waveforms(self, name, sequences):

        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        pxi_sequences = {}
        for channel in pxi_waveform_channels:
            pxi_sequences[channel] = sequences[channel]
        wv = self.sequenceslist(pxi_sequences, pxi_waveform_channels)
        return wv

    def get_sideband_waveforms(self, name, sequences):

        if 'sideband' in name:
            tek2_waveform_channels = self.hardware_cfg['awg_info']['tek70001a']['waveform_channels']
            tek2_waveforms = [sequences[channel] for channel in tek2_waveform_channels]
            return tek2_waveforms
        else:
            pass

    def get_charge_pulse(self, name, sequences):
        amps = self.params['amp_cal']

        nu = self.quantum_device_cfg['qubit']['1']['freq'] - self.quantum_device_cfg['pulse_info']['1']['iq_freq']
        readout = int((self.quantum_device_cfg['readout']['length'] + 175) / self.dtsim)
        pad = int(590 / self.dtsim)
        cut = int(1500 / self.dtsim)
        tek2_delay = int(self.hardware_cfg['channels_delay']['tek2_trig'] / self.dtsim)

        I, Q = amps[0] * (array(self.get_qubit_iq_waveforms(name, sequences)).T[pad - tek2_delay:-readout]).T[:2]
        t = arange(len(I.T)) * self.dtsim
        R = [i * cos(2 * pi * nu * t) + Q[ii] * sin(2 * pi * nu * t) for ii, i in enumerate(I)]

        if 'sideband' in name:
            SB = self.get_sideband_waveforms(name, sequences)[0]
            tr = array(self.get_qubit_iq_waveforms(name, sequences))[4]
            for ii, y in enumerate(tr): tr[ii][cut:cut + len(SB[0])] = SB[ii]
            tr = tr.T[pad - tek2_delay:-readout].T
            R = [r + amps[1] * array(tr[ii]) for ii, r in enumerate(R)]

        return t, array(R)

    def psb_mesolve(self, name, sequences, seq_list=None):

        Hd = (self.ar + self.ar.dag())
        psi0 = self.state(0, 0, 0)
        c_ops = self.lindblad_dissipator()

        tlist, cp = self.get_charge_pulse(name, sequences)

        if seq_list is None: seq_list = arange(len(cp))

        def make_cp_func(seq_num):
            def _function(t, args=None):
                time_id = int(t / self.dtsim)
                if time_id >= len(cp[seq_num]):
                    return 0
                else:
                    return 2 * pi * cp[seq_num][time_id]

            return _function

        output = []
        for ii in tqdm(seq_list):
            H_t = [self.H, [Hd, make_cp_func(ii)]]
            output.append(mesolve(H_t, psi0, tlist, c_ops, [self.projector(1, 0, 0)], progress_bar=True,
                                  options=Odeoptions(nsteps=12000)))

        return output

    def post_analysis(self, experiment_name, P='Q', show=False, check_sync=False):
        if check_sync:
            pass
        else:
            PA = PostExperiment(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, experiment_name,
                                self.I, self.Q, P, show)