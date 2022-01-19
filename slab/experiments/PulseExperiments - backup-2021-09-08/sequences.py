try:
    from .sequencer import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, Square_multitone, Square_multitone_sequential, DRAG, ARB_freq_a, Multitone_EDG
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, Square_multitone, Square_multitone_sequential, DRAG, ARB_freq_a, Multitone_EDG
# from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom
import os
import pickle
import random

import copy

class PulseSequences:
    # channels and awgs

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']


        # pulse params

        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']}

        self.qubit_ef_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq']+self.quantum_device_cfg['qubit']['1']['anharmonicity'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']+self.quantum_device_cfg['qubit']['2']['anharmonicity']}

        self.qubit_ff_freq = {
            "1": self.quantum_device_cfg['qubit']['1']['freq'] + self.quantum_device_cfg['qubit']['1']['anharmonicity'] + self.quantum_device_cfg['qubit']['ff1_disp'],
            "2": self.quantum_device_cfg['qubit']['2']['freq'] + self.quantum_device_cfg['qubit']['2']['anharmonicity'] + self.quantum_device_cfg['qubit']['ff2_disp']}

        self.qubit_gf_freq = {
            "1": self.quantum_device_cfg['qubit']['1']['freq'] + self.quantum_device_cfg['qubit']['gf1_disp'],
            "2": self.quantum_device_cfg['qubit']['2']['freq'] + self.quantum_device_cfg['qubit']['fg2_disp']}

        self.qubit_ef_e_freq = {
            "1": self.quantum_device_cfg['qubit']['1']['freq'] + self.quantum_device_cfg['qubit']['1'][
                'anharmonicity'] + self.quantum_device_cfg['qubit']['ef1_disp'],
            "2": self.quantum_device_cfg['qubit']['2']['freq'] + self.quantum_device_cfg['qubit']['2'][
                'anharmonicity'] + self.quantum_device_cfg['qubit']['fe2_disp']}

        self.qubit_ee_freq = self.quantum_device_cfg['qubit']['1']['freq']+self.quantum_device_cfg['qubit']['2']['freq']+self.quantum_device_cfg['qubit']['qq_disp']

        self.pulse_info = self.quantum_device_cfg['pulse_info']

        self.flux_pulse_info = self.quantum_device_cfg['flux_pulse_info']

        self.charge_port = {"1": self.quantum_device_cfg['qubit']['1']['charge_port'],
                             "2": self.quantum_device_cfg['qubit']['2']['charge_port']}



        self.qubit_pi = {
        "1": Square(max_amp=self.pulse_info['1']['pi_amp'], flat_len=self.pulse_info['1']['pi_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_freq["1"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_freq["2"], phase=0)}

        self.qubit_half_pi = {
        "1": Square(max_amp=self.pulse_info['1']['half_pi_amp'], flat_len=self.pulse_info['1']['half_pi_len'],
                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_freq["1"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['half_pi_amp'], flat_len=self.pulse_info['2']['half_pi_len'],
                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_freq["2"], phase=0)}

        # self.qubit_pi = {
        # "1": Gauss(max_amp=self.pulse_info['1']['pi_amp'], sigma_len=self.pulse_info['1']['pi_len'], cutoff_sigma=2,
        #            freq=self.qubit_freq["1"], phase=0, plot=False),
        # "2": Gauss(max_amp=self.pulse_info['2']['pi_amp'], sigma_len=self.pulse_info['2']['pi_len'], cutoff_sigma=2,
        #            freq=self.qubit_freq["2"], phase=0, plot=False)}
        #
        # self.qubit_half_pi = {
        # "1": Gauss(max_amp=self.pulse_info['1']['half_pi_amp'], sigma_len=self.pulse_info['1']['half_pi_len'],
        #            cutoff_sigma=2, freq=self.qubit_freq["1"], phase=0, plot=False),
        # "2": Gauss(max_amp=self.pulse_info['2']['half_pi_amp'], sigma_len=self.pulse_info['2']['half_pi_len'],
        #            cutoff_sigma=2, freq=self.qubit_freq["2"], phase=0, plot=False)}

        self.qubit_ef_pi = {
        "1": Square(max_amp=self.pulse_info['1']['pi_ef_amp'], flat_len=self.pulse_info['1']['pi_ef_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq["1"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['pi_ef_amp'], flat_len=self.pulse_info['2']['pi_ef_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq["2"], phase=0)}

        self.qubit_ef_half_pi = {
        "1": Square(max_amp=self.pulse_info['1']['half_pi_ef_amp'], flat_len=self.pulse_info['1']['half_pi_ef_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq["1"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['half_pi_ef_amp'], flat_len=self.pulse_info['2']['half_pi_ef_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq["2"], phase=0)}

        self.qubit_ff_pi = {
            "1": Square(max_amp=self.pulse_info['1']['pi_ff_amp'], flat_len=self.pulse_info['1']['pi_ff_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_ff_freq["1"], phase=0),
            "2": Square(max_amp=self.pulse_info['2']['pi_ff_amp'], flat_len=self.pulse_info['2']['pi_ff_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_ff_freq["2"], phase=0)}

        self.qubit_ef_e_pi = {
            "1": Square(max_amp=self.pulse_info['1']['pi_ef_e_amp'], flat_len=self.pulse_info['1']['pi_ef_e_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_ef_e_freq["1"], phase=0),
            "2": Square(max_amp=self.pulse_info['2']['pi_ef_e_amp'], flat_len=self.pulse_info['2']['pi_ef_e_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_ef_e_freq["2"], phase=0)}


        self.flux_pulse = {
        "1": Square(max_amp=self.flux_pulse_info['1']['pulse_amp'], flat_len=self.flux_pulse_info['1']['pulse_len'],
                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2,
                    freq=self.flux_pulse_info['1']['freq'], phase=np.pi/180*self.flux_pulse_info['1']['pulse_phase']),
        "2": Square(max_amp=self.flux_pulse_info['2']['pulse_amp'], flat_len=self.flux_pulse_info['2']['pulse_len'],
                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2,
                    freq=self.flux_pulse_info['2']['freq'], phase=np.pi/180*self.flux_pulse_info['2']['pulse_phase'])}


        # self.qubit_ef_pi = {
        # "1": Gauss(max_amp=self.pulse_info['1']['pi_ef_amp'], sigma_len=self.pulse_info['1']['pi_ef_len'], cutoff_sigma=2,
        #            freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        # "2": Gauss(max_amp=self.pulse_info['2']['pi_ef_amp'], sigma_len=self.pulse_info['2']['pi_ef_len'], cutoff_sigma=2,
        #            freq=self.qubit_ef_freq["2"], phase=0, plot=False)}
        #
        # self.qubit_ef_half_pi = {
        # "1": Gauss(max_amp=self.pulse_info['1']['half_pi_ef_amp'], sigma_len=self.pulse_info['1']['half_pi_ef_len'],
        #            cutoff_sigma=2, freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        # "2": Gauss(max_amp=self.pulse_info['2']['half_pi_ef_amp'], sigma_len=self.pulse_info['2']['half_pi_ef_len'],
        #            cutoff_sigma=2, freq=self.qubit_ef_freq["2"], phase=0, plot=False)}

        self.qubit_ee_pi = {
        "1": Square(max_amp=self.pulse_info['1']['pi_ee_amp'], flat_len=self.pulse_info['1']['pi_ee_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ee_freq-self.qubit_freq["2"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ee_freq-self.qubit_freq["1"], phase=0)}

        self.qubit_ee_half_pi = {
        "1": Square(max_amp=self.pulse_info['1']['half_pi_ee_amp'], flat_len=self.pulse_info['1']['half_pi_ee_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ee_freq-self.qubit_freq["2"], phase=0),
        "2": Square(max_amp=self.pulse_info['2']['half_pi_ee_amp'], flat_len=self.pulse_info['2']['half_pi_ee_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ee_freq-self.qubit_freq["1"], phase=0)}

        self.qubit_gf_pi = {
            "1": Square(max_amp=self.pulse_info['1']['pi_gf_amp'], flat_len=self.pulse_info['1']['pi_gf_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_gf_freq["1"], phase=0),
            "2": Square(max_amp=self.pulse_info['2']['pi_gf_amp'], flat_len=self.pulse_info['2']['pi_gf_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_gf_freq["2"], phase=0)}

        self.qubit_gf_half_pi = {
            "1": Square(max_amp=self.pulse_info['1']['half_pi_gf_amp'], flat_len=self.pulse_info['1']['half_pi_gf_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_gf_freq["1"], phase=0),
            "2": Square(max_amp=self.pulse_info['2']['half_pi_gf_amp'], flat_len=self.pulse_info['2']['half_pi_gf_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                        cutoff_sigma=2, freq=self.qubit_gf_freq["2"], phase=0)}

        self.multimodes = self.quantum_device_cfg['multimodes']

        self.mm_sideband_pi = {"1":[], "2":[]}

        for qubit_id in ["1","2"]:
            for mm_id in range(len(self.multimodes[qubit_id]['freq'])):
                self.mm_sideband_pi[qubit_id].append(
                                     Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id],
                                            flat_len=self.multimodes[qubit_id]['pi_len'][mm_id],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                            plot=False))


        self.communication = self.quantum_device_cfg['communication']
        self.sideband_cooling = self.quantum_device_cfg['sideband_cooling']


        with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/1_100kHz.pkl'), 'rb') as f:
            freq_a_p_1 = pickle.load(f)

        with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/2_100kHz.pkl'), 'rb') as f:
            freq_a_p_2 = pickle.load(f)

        with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/1_ef_100kHz.pkl'), 'rb') as f:
            ef_freq_a_p_1 = pickle.load(f)

        with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/2_ef_100kHz.pkl'), 'rb') as f:
            ef_freq_a_p_2 = pickle.load(f)

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)

        A_list_1 = self.communication['1']['pi_amp'] * np.ones_like(gauss_envelop)
        A_list_2 = self.communication['2']['pi_amp'] * np.ones_like(gauss_envelop)

        self.communication_flux_pi = {
            "1": ARB_freq_a(A_list = A_list_1, B_list = np.zeros_like(A_list_1), len=self.communication['1']['pi_len'], freq_a_fit = freq_a_p_1, phase = 0),
            "2": ARB_freq_a(A_list = A_list_2, B_list = np.zeros_like(A_list_2), len=self.communication['2']['pi_len'], freq_a_fit = freq_a_p_2, phase = 0)
        }

        A_list_1_v2 = self.communication['1']['pi_amp_v2'] * np.ones_like(gauss_envelop)
        A_list_2_v2 = self.communication['2']['pi_amp_v2'] * np.ones_like(gauss_envelop)

        self.communication_flux_pi_v2 = {
            "1": ARB_freq_a(A_list = A_list_1_v2, B_list = np.zeros_like(A_list_1_v2), len=self.communication['1']['pi_len_v2'], freq_a_fit = freq_a_p_1, phase = 0),
            "2": ARB_freq_a(A_list = A_list_2_v2, B_list = np.zeros_like(A_list_2_v2), len=self.communication['2']['pi_len_v2'], freq_a_fit = freq_a_p_2, phase = 0)
        }

        A_list_ef_1 = self.communication['1']['ef_pi_amp'] * np.ones_like(gauss_envelop)
        A_list_ef_2 = self.communication['2']['ef_pi_amp'] * np.ones_like(gauss_envelop)

        self.communication_flux_ef_pi = {
            "1": ARB_freq_a(A_list = A_list_ef_1, B_list = np.zeros_like(A_list_ef_1), len=self.communication['1']['ef_pi_len'], freq_a_fit = ef_freq_a_p_1, phase = 0),
            "2": ARB_freq_a(A_list = A_list_ef_2, B_list = np.zeros_like(A_list_ef_2), len=self.communication['2']['ef_pi_len'], freq_a_fit = ef_freq_a_p_2, phase = 0)
        }

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)

        A_list_1h = self.communication['1']['half_transfer_amp'] * np.ones_like(gauss_envelop)
        A_list_2h = self.communication['2']['half_transfer_amp'] * np.ones_like(gauss_envelop)

        self.communication_flux_half_transfer = {
            "1": ARB_freq_a(A_list = A_list_1h, B_list = np.zeros_like(A_list_1h), len=self.communication['1']['half_transfer_len'], freq_a_fit = freq_a_p_1, phase = 0),
            "2": ARB_freq_a(A_list = A_list_2h, B_list = np.zeros_like(A_list_2h), len=self.communication['2']['half_transfer_len'], freq_a_fit = freq_a_p_2, phase = 0)
        }


        self.readout_sideband = {
            "1": Square(max_amp=self.quantum_device_cfg['heterodyne']['1']['sideband_amp'],
                                            flat_len=self.quantum_device_cfg['heterodyne']['1']['length'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.multimodes['1']['freq'][0] - self.quantum_device_cfg['heterodyne']['1']['sideband_detune'], phase=0,
                                            plot=False),
            "2": Square(max_amp=self.quantum_device_cfg['heterodyne']['2']['sideband_amp'],
                                            flat_len=self.quantum_device_cfg['heterodyne']['2']['length'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.multimodes['2']['freq'][0] - self.quantum_device_cfg['heterodyne']['2']['sideband_detune'], phase=0,
                                            plot=False)
        }


    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg)


    def readout(self, sequencer, on_qubits=None, sideband = False):
        if on_qubits == None:
            on_qubits = ["1", "2"]

        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('alazar_trig')

        # get readout time to be integer multiple of 5ns (
        # 5ns is the least common multiple between tek1 dt (1/1.2 ns) and alazar dt (1 ns)
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5

        sequencer.append_idle_to_time('alazar_trig', readout_time_5ns_multiple)
        sequencer.sync_channels_time(self.channels)

        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for qubit_id in on_qubits:
            sequencer.append('hetero%s_I' % qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                    flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'], phase=0,
                                    phase_t0=readout_time_5ns_multiple))
            sequencer.append('hetero%s_Q' % qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                    flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'],
                                    phase=np.pi / 2 + heterodyne_cfg[qubit_id]['phase_offset'], phase_t0=readout_time_5ns_multiple))
            if sideband:
                sequencer.append('flux%s'%qubit_id,self.readout_sideband[qubit_id])
            sequencer.append('readout%s_trig' % qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))

        sequencer.append('alazar_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['alazar']))

        return readout_time

    def sideband_rabi(self, sequencer):
        # sideband rabi time domain
        rabi_freq = self.expt_cfg['freq']

        if self.expt_cfg['Gaussian'][0]:
            for rabi_amp in np.arange(self.expt_cfg['Gaussian'][2], self.expt_cfg['Gaussian'][3], self.expt_cfg['Gaussian'][4]):
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:

                    for ge_pi_id in self.expt_cfg['ge_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for ef_pi_id in self.expt_cfg['ef_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_ef_pi[ef_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for flux_line_id in self.expt_cfg['flux_line']:
                        sequencer.append('flux%s' % flux_line_id,
                                        Gauss(max_amp=rabi_amp, sigma_len=self.expt_cfg['Gaussian'][1], cutoff_sigma=2, freq=rabi_freq, phase=0))



                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()
        else:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:

                    for ge_pi_id in self.expt_cfg['ge_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for ef_pi_id in self.expt_cfg['ef_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_ef_pi[ef_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for flux_line_id in self.expt_cfg['flux_line']:
                        sequencer.append('flux%s' %flux_line_id, Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                     cutoff_sigma=2, freq=rabi_freq, phase=0, plot=False))

                    # sequencer.append('charge%s' %self.charge_port[qubit_id], Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                    #             ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=rabi_freq, phase=0))

                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        return sequencer.complete(self,plot=False)

    def sideband_rabi_drive_both_flux_old(self, sequencer):
        # sideband rabi time domain, drive both VSLQ flux loops with different amplitudes and phases
        rabi_freq = self.expt_cfg['freq']

        if self.expt_cfg['Gaussian'][0]: # Gaussian part not fixed yet
            for rabi_amp in np.arange(self.expt_cfg['Gaussian'][2], self.expt_cfg['Gaussian'][3], self.expt_cfg['Gaussian'][4]):
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:

                    for ge_pi_id in self.expt_cfg['ge_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for ef_pi_id in self.expt_cfg['ef_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_ef_pi[ef_pi_id])
                        else:
                            sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for flux_line_id in self.expt_cfg['flux_line']:
                        sequencer.append('flux%s' % flux_line_id,
                                        Gauss(max_amp=rabi_amp, sigma_len=self.expt_cfg['Gaussian'][1], cutoff_sigma=2, freq=rabi_freq, phase=0))



                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()
        else:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)

                for index,qubit_id in enumerate(self.expt_cfg['on_qubits']):

                    for ge_pi_id in self.expt_cfg['ge_pi']: # Modified by Tanay
                        if len(self.expt_cfg.get('flux_pi')) == 0:
                            sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])
                        else:
                            sequencer.append('flux%s' %self.expt_cfg['flux_pi'][index], self.qubit_pi[ge_pi_id])


                    sequencer.sync_channels_time(self.channels)

                    for ef_pi_id in self.expt_cfg['ef_pi']: # Modified by Tanay
                        if len(self.expt_cfg.get('flux_pi')) == 0:
                            sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                        else:
                            sequencer.append('flux%s' %self.expt_cfg['flux_pi'][index], self.qubit_ef_pi[ef_pi_id])

                    sequencer.sync_channels_time(self.channels)

                    for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id, Square(max_amp=self.expt_cfg['amp'][index], flat_len=rabi_len,
                                                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                                        cutoff_sigma=2, freq=rabi_freq, phase=self.expt_cfg['phase'][index], plot=False))

                    # sequencer.append('charge%s' %self.charge_port[qubit_id], Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                    #             ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=rabi_freq, phase=0))

                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        return sequencer.complete(self,plot=False)


    def sideband_rabi_drive_both_flux(self, sequencer):
        # sideband rabi time domain, drive both VSLQ flux loops with different amplitudes and phases
        rabi_freq = self.expt_cfg['freq']

        if self.expt_cfg['Gaussian'][0]: # Gaussian part not fixed yet
            for rabi_amp in np.arange(self.expt_cfg['Gaussian'][2], self.expt_cfg['Gaussian'][3], self.expt_cfg['Gaussian'][4]):
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:

                    for ge_pi_id in self.expt_cfg['ge_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                            sequencer.sync_channels_time(self.channels)
                        else:
                            sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])
                            sequencer.sync_channels_time(self.channels)

                    for ef_pi_id in self.expt_cfg['ef_pi']:
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_ef_pi[ef_pi_id])
                            sequencer.sync_channels_time(self.channels)
                        else:
                            sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                            sequencer.sync_channels_time(self.channels)

                    for ge_both_pi_id in self.expt_cfg['ge_both_pi']:  #  only when ef_pi doesn't exist.
                        if self.expt_cfg['flux_ge_pi']:
                            sequencer.append('flux1', self.qubit_ee_pi[ge_both_pi_id])
                            sequencer.sync_channels_time(self.channels)
                        else:
                            sequencer.append('charge%s' %self.charge_port[ge_both_pi_id], self.qubit_ee_pi[ge_both_pi_id])
                            sequencer.sync_channels_time(self.channels)

                    if self.expt_cfg['pre_pulse']:
                        # Initial pulse before Rabi
                        # State preparation
                        pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                        sequencer.append('charge1',
                                         Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                     flat_lens=pre_pulse_info['times_prep'],
                                                                     ramp_sigma_len=
                                                                     self.quantum_device_cfg['flux_pulse_info'][
                                                                         '1'][
                                                                         'ramp_sigma_len'],
                                                                     cutoff_sigma=2,
                                                                     freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                     phases=np.pi / 180 * np.array(
                                                                         pre_pulse_info['charge1_phases_prep']),
                                                                     plot=False))
                        sequencer.append('charge2',
                                         Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                     flat_lens=pre_pulse_info['times_prep'],
                                                                     ramp_sigma_len=
                                                                     self.quantum_device_cfg['flux_pulse_info'][
                                                                         '1'][
                                                                         'ramp_sigma_len'],
                                                                     cutoff_sigma=2,
                                                                     freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                     phases=np.pi / 180 * np.array(
                                                                         pre_pulse_info['charge2_phases_prep']),
                                                                     plot=False))
                        sequencer.append('flux1',
                                         Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                     flat_lens=pre_pulse_info['times_prep'],
                                                                     ramp_sigma_len=
                                                                     self.quantum_device_cfg['flux_pulse_info'][
                                                                         '1'][
                                                                         'ramp_sigma_len'],
                                                                     cutoff_sigma=2,
                                                                     freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                     phases=np.pi / 180 * np.array(
                                                                         pre_pulse_info['flux1_phases_prep']),
                                                                     plot=False))
                        sequencer.append('flux2',
                                         Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                     flat_lens=pre_pulse_info['times_prep'],
                                                                     ramp_sigma_len=
                                                                     self.quantum_device_cfg['flux_pulse_info'][
                                                                         '1'][
                                                                         'ramp_sigma_len'],
                                                                     cutoff_sigma=2,
                                                                     freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                     phases=np.pi / 180 * np.array(
                                                                         pre_pulse_info['flux2_phases_prep']),
                                                                     plot=False))

                        sequencer.sync_channels_time(self.channels)

                    for flux_line_id in self.expt_cfg['flux_line']:
                        sequencer.append('flux%s' % flux_line_id,
                                        Gauss(max_amp=rabi_amp, sigma_len=self.expt_cfg['Gaussian'][1], cutoff_sigma=2, freq=rabi_freq, phase=0))

                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        else:
            # Modified by Tanay
            # For flat-top pulses, has Gaussian ramp
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)

                # for index, qubit_id in enumerate(self.expt_cfg['on_qubits']):
                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])
                    sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                    sequencer.sync_channels_time(self.channels)

                for ge_both_pi_id in self.expt_cfg['ge_both_pi']:  # only when ef_pi doesn't exist.
                    if self.expt_cfg['flux_ge_pi']:
                        sequencer.append('flux1', self.qubit_ee_pi[ge_both_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_both_pi_id], self.qubit_ee_pi[ge_both_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for flux_pulse_id in self.expt_cfg['flux_pulse']:
                    sequencer.append('flux%s' % flux_pulse_id, self.flux_pulse[flux_pulse_id])

                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    # State preparation
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge1_phases_prep']),
                                                                 plot=False))
                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge2_phases_prep']),
                                                                 plot=False))
                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))
                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                    sequencer.sync_channels_time(self.channels)


                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    sequencer.append('flux%s' %flux_line_id,
                                     Square(max_amp=self.expt_cfg['amp'][index], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=rabi_freq, phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        return sequencer.complete(self,plot=False)



    def multitone_error_divisible_gate(self, sequencer):
        # Modified by Ziqian
        # in each braket, the amplitude, frequency, phases determines a simultaneous multitone send through flux line 1 or 2
        # pulse shape: ['cos\tanh', alpha, f, A, tg, Gamma (if needed)]
        tones = max(len(self.expt_cfg['freqs'][0]),len(self.expt_cfg['freqs'][-1]))
        freqs = self.expt_cfg['freqs']
        time_length = self.expt_cfg['time_length']
        amps = (self.expt_cfg['amps'])
        phases = (self.expt_cfg['phases']) # phase should be in degrees
        shape = self.expt_cfg['shape']

        time_length[int(self.expt_cfg['edg_line'])-1][self.expt_cfg['edg_no']] = shape[4]  # replace the EDG tone length with the EDG shape variable 'tg'

        # Modified by Ziqian
        # For flat-top pulses, has Gaussian ramp for other tones, analytic expression for EDG part
        # Sweeping the iteration time, perform readout at the end
        cycle_no = 0
        for repeat_time in np.arange(self.expt_cfg['start_no'], self.expt_cfg['stop_no'], self.expt_cfg['step_no']):
            sequencer.new_sequence(self)

            for index, qubit_id in enumerate(self.expt_cfg['on_qubits']):

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before EDG
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge2_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['flux1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['flux2_phases_prep']),
                                                                 plot=False))

                    sequencer.sync_channels_time(self.channels)

                for flux_pulse_id in self.expt_cfg['flux_pulse']:
                    sequencer.append('flux%s' %flux_pulse_id, self.flux_pulse[flux_pulse_id])

                sequencer.sync_channels_time(self.channels)


                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    if self.expt_cfg['edg_line'] == int(flux_line_id):
                        sequencer.append('flux%s' %flux_line_id,
                                         Multitone_EDG(max_amp=np.array(amps[index]),
                                            flat_len=time_length*tones, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=np.array(freqs[index]), phase=np.pi/180*(np.array(phases[index])), shapes=[shape], nos=[self.expt_cfg['edg_no']], repeat=repeat_time, plot=False))
                    else:
                        sequencer.append('flux%s' % flux_line_id,
                                         Square_multitone(max_amp=np.array(amps[index]),
                                                       flat_len=time_length * tones, ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=np.array(freqs[index]),
                                                       phase=np.pi / 180 * (np.array(phases[index])), plot=False))

                sequencer.sync_channels_time(self.channels)

                #  inverse to the initial state

                if self.expt_cfg['inverse_rotation']:
                    # Inverse pulse after EDG
                    inverse_rotation_info = self.quantum_device_cfg['inverse_rotation']
                    replace_no = inverse_rotation_info['replace_no']
                    inverse_rotation_info['times_prep'][replace_no[0]][replace_no[1]] = inverse_rotation_info['inverse_tlist'][cycle_no]
                    cycle_no += 1
                    if cycle_no>len(inverse_rotation_info['inverse_tlist'])-1:
                        cycle_no = 0
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['charge1_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['charge1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['charge2_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['charge2_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux1',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['flux1_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['flux1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['flux1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux2',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['flux2_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['flux2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['flux2_phases_prep']),
                                                                 plot=False))

                    sequencer.sync_channels_time(self.channels)


                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        return sequencer.complete(self,plot=False)


    def multitone_sideband_rabi_drive_both_flux(self, sequencer):
        # Modified by Ziqian
        # in each braket, the amplitude, frequency, phases determines a simultaneous multitone send through flux line 1 or 2
        # sideband rabi time domain, drive both VSLQ flux loops with mutitones having different amplitudes and phases
        tones = max(len(self.expt_cfg['freqs'][0]),len(self.expt_cfg['freqs'][1]))
        freqs = self.expt_cfg['freqs']
        amps = (self.expt_cfg['amps'])
        phases = (self.expt_cfg['phases']) # phase should be in degrees

        if self.expt_cfg['Gaussian'][0]: # Gaussian part not fixed yet
            pass

        else:
            # Modified by Tanay
            # For flat-top pulses, has Gaussian ramp
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge2_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['flux1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['flux2_phases_prep']),
                                                                 plot=False))

                    sequencer.sync_channels_time(self.channels)


                for index, qubit_id in enumerate(self.expt_cfg['on_qubits']):



                    for flux_pulse_id in self.expt_cfg['flux_pulse']:
                        sequencer.append('flux%s' %flux_pulse_id, self.flux_pulse[flux_pulse_id])

                    sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    sequencer.append('flux%s' %flux_line_id,
                                     Square_multitone(max_amp=np.array(amps[index]),
                                        flat_len=[rabi_len]*tones, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=np.array(freqs[index]), phase=np.pi/180*(np.array(phases[index])), plot=False))
                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['post_pulse']:
                    # Post pulse after Rabi
                    post_pulse_info = self.quantum_device_cfg['post_pulse_info']
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=post_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=post_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=post_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     post_pulse_info['charge1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=post_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=post_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=post_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     post_pulse_info['charge2_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux1',
                                     Square_multitone_sequential(max_amps=post_pulse_info['flux1_amps_prep'],
                                                                 flat_lens=post_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=post_pulse_info['flux1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     post_pulse_info['flux1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux2',
                                     Square_multitone_sequential(max_amps=post_pulse_info['flux2_amps_prep'],
                                                                 flat_lens=post_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=post_pulse_info['flux2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     post_pulse_info['flux2_phases_prep']),
                                                                 plot=False))

                    sequencer.sync_channels_time(self.channels)

                self.readout(sequencer, self.expt_cfg['on_qubits'])

                sequencer.end_sequence()

        return sequencer.complete(self,plot=False)



    def sideband_rabi_drive_both_flux_LO(self, sequencer):
        # sideband rabi time domain
        # drive both VSLQ flux loops with different amplitudes and phases using SignalCore as LO and AWG is IQ
        rabi_freq = 0

        # For flat-top pulses, has Gaussian ramp
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for index, qubit_id in enumerate(self.expt_cfg['on_qubits']):

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                sequencer.sync_channels_time(self.channels)

                for flux_pulse_id in self.expt_cfg['flux_pulse']:
                    sequencer.append('flux%s' %flux_pulse_id, self.flux_pulse[flux_pulse_id])

                sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    sequencer.append('flux%s' %flux_line_id, Square(max_amp=self.expt_cfg['amp'][index], flat_len=rabi_len,
                                                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                                    cutoff_sigma=2, freq=rabi_freq, phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))


            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)



    def sideband_rabi_2_freq(self, sequencer):
        # sideband rabi time domain
        rabi_freq_list = [self.expt_cfg['freq_1'], self.expt_cfg['freq_2']]
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s'%qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=rabi_freq_list[int(qubit_id)-1], phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)

    def sideband_rabi_freq(self, sequencer):
        # sideband rabi freq sweep
        rabi_len = self.expt_cfg['pulse_len']

        if self.expt_cfg["around"] == "comm":
            amp = self.expt_cfg['amp']

            qubit_id = self.expt_cfg['on_qubits'][0]

            with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_500kHz.pkl' %qubit_id), 'rb') as f:
                freq_a_p = pickle.load(f)

            center_freq = freq_a_p(amp)
            print("center freq: %s" %center_freq)
            freq_array = np.arange(center_freq-self.expt_cfg['freq_range'],center_freq+self.expt_cfg['freq_range'],self.expt_cfg['step'])

        elif self.expt_cfg["around"] == "mm":

            qubit_id = self.expt_cfg['on_qubits'][0]
            mm_freq_list = self.quantum_device_cfg['multimodes'][qubit_id]['freq']
            freq_list_all = []
            for mm_freq in mm_freq_list:
                freq_list_all += [np.arange(mm_freq-self.expt_cfg['freq_range'],mm_freq+self.expt_cfg['freq_range'],self.expt_cfg['step'])]

            freq_array = np.hstack(np.array(freq_list_all))

            # print(freq_array)
        else:
            freq_array = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

        for rabi_freq in freq_array:
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s'%qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=rabi_freq, phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)

    def ef_sideband_rabi_freq(self, sequencer):
        # sideband rabi freq sweep
        rabi_len = self.expt_cfg['pulse_len']

        if self.expt_cfg["around"] == "comm":
            amp = self.expt_cfg['amp']

            qubit_id = self.expt_cfg['on_qubits'][0]

            with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_ef_1MHz.pkl' %qubit_id), 'rb') as f:
                freq_a_p = pickle.load(f)

            center_freq = freq_a_p(amp)
            print("center freq: %s" %center_freq)
            freq_array = np.arange(center_freq-self.expt_cfg['freq_range'],center_freq+self.expt_cfg['freq_range'],self.expt_cfg['step'])

        elif self.expt_cfg["around"] == "mm":

            qubit_id = self.expt_cfg['on_qubits'][0]
            mm_freq_list = self.quantum_device_cfg['multimodes'][qubit_id]['ef_freq']
            freq_list_all = []
            for mm_freq in mm_freq_list:
                freq_list_all += [np.arange(mm_freq-self.expt_cfg['freq_range'],mm_freq+self.expt_cfg['freq_range'],self.expt_cfg['step'])]

            freq_array = np.hstack(np.array(freq_list_all))

            # print(freq_array)
        else:
            freq_array = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

        for rabi_freq in freq_array:
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s'%qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=rabi_freq, phase=0,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                sequencer.append('charge%s'%qubit_id, self.qubit_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)


    def pulse_probe(self, sequencer):
        # pulse_probe sequences

        for qubit_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:


                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_ge_pi']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_ge_pi']:
                        sequencer.append('flux1', self.qubit_ef_pi[ef_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:

                    sequencer.append('flux1',
                                     Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                            ramp_sigma_len=2, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                            phase_t0=0))
                else:

                    sequencer.append('charge%s' %self.charge_port[qubit_id],
                                     Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                            ramp_sigma_len=2, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                            phase_t0=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def pulse_probe_while_flux_backup(self, sequencer):
        # pulse_probe while flux driving

        for qubit_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:


                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)


                sequencer.append('flux%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['flux_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=2, cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=0,
                                        phase_t0=0))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=2, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                        phase_t0=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def pulse_probe_while_flux(self, sequencer):
        # pulse_probe while flux driving

        for qubit_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:


                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)


                sequencer.append('flux%s' % self.expt_cfg['flux_line'][0],
                                 Square(max_amp=self.expt_cfg['flux_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=2, cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=0,
                                        phase_t0=0))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=2, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                        phase_t0=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def pulse_probe_through_flux(self, sequencer):
        # pulse_probe through flux driving only, added by Tanay; charge, flux_pulse not tested

        for flux_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            # Apply pulses through charge drives
            for ge_pi_id in self.expt_cfg['ge_pi']:
                sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

            sequencer.sync_channels_time(self.channels)

            for ef_pi_id in self.expt_cfg['ef_pi']:
                sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_pi[ef_pi_id])

            sequencer.sync_channels_time(self.channels)

            # Apply pulses through flux drive
            for flux_pulse_id in self.expt_cfg['flux_pulse']:
                sequencer.append('flux%s' %flux_pulse_id, self.flux_pulse[flux_pulse_id])

            sequencer.sync_channels_time(self.channels)

            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                sequencer.append('flux%s' %flux_line_id,
                                 Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][flux_line_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=flux_freq, phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

            sequencer.sync_channels_time(self.channels)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def rabi(self, sequencer):
        # rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['use_freq'][0]:
                    drive_freq = self.expt_cfg['use_freq'][1]
                    drive_port = self.expt_cfg['use_freq'][-1]
                else:
                    drive_freq = self.qubit_freq[qubit_id]
                    drive_port = 'charge%s' %self.charge_port[qubit_id]

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:

                    sequencer.append('flux1',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=drive_freq, phase=np.pi/180*self.expt_cfg['phase']))
                else:
                    sequencer.append(drive_port,
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=drive_freq, phase=np.pi/180*self.expt_cfg['phase']))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rabi_while_flux_old(self, sequencer):
        # rabi experiment while flux driving

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                # sequencer.append('flux%s' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['flux_amp'], flat_len=rabi_len,
                #                         ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                #                         cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=0))

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rabi_while_flux_backup(self, sequencer):
        # rabi experiment while flux driving

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge1_phases_prep']),
                                                                            plot=False))

                    sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge2_phases_prep']),
                                                                            plot=False))

                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))

                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                    sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rabi_while_flux(self, sequencer):
        # rabi experiment while flux driving

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ef_pi_id, self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for ge_pi2_id in self.expt_cfg['ge_pi2']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi2_id, self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi2_id], self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    repeat = pre_pulse_info['repeat']
                    for repeat_i in range(repeat):
                        sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge1_phases_prep']),
                                                                                plot=False))
                        sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge2_phases_prep']),
                                                                                plot=False))
                        sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux1_phases_prep']),
                                                                              plot=False))
                        sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux2_phases_prep']),
                                                                              plot=False))

                    sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rabi_while_flux_gf(self, sequencer):
        # rabi experiment while flux driving

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ef_pi_id, self.qubit_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ef_pi_id, self.qubit_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    repeat = pre_pulse_info['repeat']
                    for repeat_i in range(repeat):
                        sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge1_phases_prep']),
                                                                                plot=False))
                        sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge2_phases_prep']),
                                                                                plot=False))
                        sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux1_phases_prep']),
                                                                              plot=False))
                        sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux2_phases_prep']),
                                                                              plot=False))

                    sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rabi_while_flux_charge_phase(self, sequencer):
        # rabi experiment while flux driving

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge1_phases_prep']),
                                                                            plot=False))

                    sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge2_phases_prep']),
                                                                            plot=False))

                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))

                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                    sequencer.sync_channels_time(self.channels)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=self.expt_cfg['charge_phase']/180*np.pi))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def multitone_rabi_while_flux(self, sequencer):
        # multitone rabi experiment while flux driving NOT TESTED, check "tones" thing

        tones = len(self.expt_cfg['freqs'])
        freqs = self.expt_cfg['freqs']
        amps = np.array(self.expt_cfg['amps'])
        phases = np.array(self.expt_cfg['phases']) # phase should be in degrees

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)



            for ge_pi_id in self.expt_cfg['ge_pi']:
                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                    sequencer.sync_channels_time(self.channels)
                else:
                    sequencer.append('charge%s'%ge_pi_id, self.qubit_pi[ge_pi_id])
                    sequencer.sync_channels_time(self.channels)

            if self.expt_cfg['pre_pulse']:
                # Initial pulse before Rabi
                pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                repeat = pre_pulse_info['repeat']
                for repeat_i in range(repeat):
                    sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge1_phases_prep']),
                                                                            plot=False))
                    sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                            flat_lens=pre_pulse_info['times_prep'],
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                pre_pulse_info['charge2_phases_prep']),
                                                                            plot=False))
                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))
                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                sequencer.sync_channels_time(self.channels)

            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    sequencer.append('flux%s' %flux_line_id,
                                     Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.expt_cfg['flux_freq'], phase=np.pi/180*self.expt_cfg['flux_phase'][index], plot=False))

            for index, charge_id in enumerate(self.expt_cfg['on_qubits']):
                sequencer.append('charge%s' % charge_id,
                                 Square_multitone(max_amp=np.array(amps[index]),
                                                  flat_len=[rabi_len] * tones,
                                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                                      'ramp_sigma_len'],
                                                  cutoff_sigma=2, freq=np.array(freqs[index]),
                                                  phase=np.pi / 180 * (np.array(phases[index])), plot=False))

            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def vacuum_rabi(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for iq_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            if self.expt_cfg['pre_pulse']:
                # Initial pulse before Rabi
                # State preparation
                pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                sequencer.append('charge1',
                                 Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                             flat_lens=pre_pulse_info['times_prep'],
                                                             ramp_sigma_len=
                                                             self.quantum_device_cfg['flux_pulse_info'][
                                                                 '1'][
                                                                 'ramp_sigma_len'],
                                                             cutoff_sigma=2,
                                                             freqs=pre_pulse_info['charge1_freqs_prep'],
                                                             phases=np.pi / 180 * np.array(
                                                                 pre_pulse_info['charge1_phases_prep']),
                                                             plot=False))
                sequencer.append('charge2',
                                 Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                             flat_lens=pre_pulse_info['times_prep'],
                                                             ramp_sigma_len=
                                                             self.quantum_device_cfg['flux_pulse_info'][
                                                                 '1'][
                                                                 'ramp_sigma_len'],
                                                             cutoff_sigma=2,
                                                             freqs=pre_pulse_info['charge2_freqs_prep'],
                                                             phases=np.pi / 180 * np.array(
                                                                 pre_pulse_info['charge2_phases_prep']),
                                                             plot=False))
                sequencer.append('flux1',
                                 Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                             flat_lens=pre_pulse_info['times_prep'],
                                                             ramp_sigma_len=
                                                             self.quantum_device_cfg['flux_pulse_info'][
                                                                 '1'][
                                                                 'ramp_sigma_len'],
                                                             cutoff_sigma=2,
                                                             freqs=pre_pulse_info['flux1_freqs_prep'],
                                                             phases=np.pi / 180 * np.array(
                                                                 pre_pulse_info['flux1_phases_prep']),
                                                             plot=False))
                sequencer.append('flux2',
                                 Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                             flat_lens=pre_pulse_info['times_prep'],
                                                             ramp_sigma_len=
                                                             self.quantum_device_cfg['flux_pulse_info'][
                                                                 '1'][
                                                                 'ramp_sigma_len'],
                                                             cutoff_sigma=2,
                                                             freqs=pre_pulse_info['flux2_freqs_prep'],
                                                             phases=np.pi / 180 * np.array(
                                                                 pre_pulse_info['flux2_phases_prep']),
                                                             plot=False))

                sequencer.sync_channels_time(self.channels)


            sequencer.append('alazar_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['alazar']))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('hetero%s_I' % qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                        flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq, phase=0,
                                        phase_t0=0))
                sequencer.append('hetero%s_Q' % qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                        flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq,
                                        phase=np.pi / 2, phase_t0=0))
                sequencer.append('readout%s_trig' % qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    # def histogram(self, sequencer):
    #     # vacuum rabi sequences
    #     heterodyne_cfg = self.quantum_device_cfg['heterodyne']
    #
    #     for ii in range(50):
    #
    #         # no pi pulse (g state)
    #         sequencer.new_sequence(self)
    #
    #         self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
    #         sequencer.end_sequence()
    #
    #         # with pi pulse (e state)
    #         sequencer.new_sequence(self)
    #
    #         for qubit_id in self.expt_cfg['on_qubits']:
    #             if self.expt_cfg['flux_probe']:
    #                 sequencer.append('flux1', self.qubit_pi[qubit_id])
    #             else:
    #                 sequencer.append('charge1', self.qubit_pi[qubit_id])
    #         self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
    #         sequencer.end_sequence()
    #
    #         # with pi pulse and ef pi pulse (f state)
    #         sequencer.new_sequence(self)
    #
    #         for qubit_id in self.expt_cfg['on_qubits']:
    #             if self.expt_cfg['flux_probe']:
    #                 sequencer.append('flux1', self.qubit_pi[qubit_id])
    #                 sequencer.append('flux1', self.qubit_ef_pi[qubit_id])
    #             else:
    #                 sequencer.append('charge1', self.qubit_pi[qubit_id])
    #                 sequencer.append('charge1', self.qubit_ef_pi[qubit_id])
    #         self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
    #         sequencer.end_sequence()
    #
    #     return sequencer.complete(self, plot=False)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for ii in range(50):

            for state in self.expt_cfg['states']:

                sequencer.new_sequence(self)

                # ge state
                if state == 'ge':

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi['2'])
                    else:
                        sequencer.append('charge%s' %self.charge_port['2'], self.qubit_pi['2'])

                # eg state
                if state == 'eg':

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi['1'])
                    else:
                        sequencer.append('charge%s' %self.charge_port['1'], self.qubit_pi['1'])

                # ee state
                if state == 'ee':

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi['2'])
                        sequencer.append('flux1', self.qubit_ee_pi['1'])
                    else:
                        sequencer.append('charge%s' %self.charge_port['2'], self.qubit_pi['2'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge%s' %self.charge_port['2'], self.qubit_ee_pi['1'])

                # gf state
                if state == 'gf':

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi['2'])
                        sequencer.append('flux1', self.qubit_ef_pi['2'])
                    else:
                        sequencer.append('charge%s' %self.charge_port['2'], self.qubit_pi['2'])
                        sequencer.append('charge%s' %self.charge_port['2'], self.qubit_ef_pi['2'])


                # fg state
                if state == 'fg':

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi['1'])
                        sequencer.append('flux1', self.qubit_ef_pi['1'])
                    else:
                        sequencer.append('charge%s' %self.charge_port['1'], self.qubit_pi['1'])
                        sequencer.append('charge%s' %self.charge_port['1'], self.qubit_ef_pi['1'])


                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def histogram_arb_charge_drive(self, sequencer):
        # Author: Tanay 19 Feb 2020
        ''' Histogram between gg and state after prepared by an arbitrary charge drive'''
        amps = np.array(self.expt_cfg['amps'])
        freqs = np.array(self.expt_cfg['freqs'])
        phases = np.array(self.expt_cfg['phases'])

        for ii in range(self.expt_cfg['scatter_points']):

            for state in self.expt_cfg['states']:

                sequencer.new_sequence(self)

                if state == 'arb':
                    for index, charge_line_id in enumerate(self.expt_cfg['charge_line']):
                        sequencer.append('charge%s' %charge_line_id, Square_multitone(max_amp=self.expt_cfg['amp_factor'][index]*amps,
                                        flat_len=self.expt_cfg['times'], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freqs, phase=np.pi/180*(self.expt_cfg['phase_offset'][index] + phases), plot=False))
                    # "phase_offset" is between two flux lines

                self.readout(sequencer, self.expt_cfg['on_cavities'], sideband=False)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def histogram_sideband_flux_LO(self, sequencer):
        # Author: Tanay 19 Feb 2020
        # "singleshot" must be set to true
        amps = np.array(self.expt_cfg['amps'])
        freqs = np.array(self.expt_cfg['freqs'])
        phases = np.array(self.expt_cfg['phases'])

        for ii in range(self.expt_cfg['scatter_points']):

            for state in self.expt_cfg['states']:

                sequencer.new_sequence(self)

                if state == 'Bell':
                    for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id, Square_multitone(max_amp=self.expt_cfg['amp_factor'][index]*amps,
                                        flat_len=self.expt_cfg['times'], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freqs, phase=np.pi/180*(self.expt_cfg['phase_offset'][index] + phases), plot=False))
                    # "phase_offset" is between two flux lines

                self.readout(sequencer, self.expt_cfg['on_cavities'], sideband=False)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def histogram_multitone_charge_flux_drive_LO(self, sequencer):
        # Author: Tanay 19 Feb 2020 :NOT CHECKED YET
        ''' Histogram between gg and state after prepared by an arbitrary charge and flux drive'''

        for ii in range(self.expt_cfg['scatter_points']):

            for state in self.expt_cfg['states']:

                sequencer.new_sequence(self)

                if state == 'arb':
                    sequencer.append('charge1', Square_multitone(max_amp=self.expt_cfg['charge1_amps'],
                                    flat_len=self.expt_cfg['times'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.expt_cfg['charge1_freqs'],
                                    phase=np.pi/180*np.array(self.expt_cfg['charge1_phases']), plot=False))

                    sequencer.append('charge2', Square_multitone(max_amp=self.expt_cfg['charge2_amps'],
                                    flat_len=self.expt_cfg['times'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.expt_cfg['charge2_freqs'],
                                    phase=np.pi/180*np.array(self.expt_cfg['charge2_phases']), plot=False))

                    sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps'],
                                    flat_len=self.expt_cfg['times'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs'],
                                    phase=np.pi/180*np.array(self.expt_cfg['flux1_phases']), plot=False))

                    sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps'],
                                    flat_len=self.expt_cfg['times'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs'],
                                    phase=np.pi/180*np.array(self.expt_cfg['flux2_phases']), plot=False))

                self.readout(sequencer, self.expt_cfg['on_cavities'], sideband=False)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def rb_both(self, sequencer):
        # Author: Ziqian 2021/07/14
        ''' simultaneous single qubit randomized benchmarking '''

        ## generate sequences of random pulses
        ## 1:Z,   2:X, 3:Y
        ## 4:Z/2, 5:X/2, 6:Y/2
        ## 7:-Z/2, 8:-X/2, 9:-Y/2
        ## 0:I

        gate_list1 = []
        gate_list2 = []
        for ii in range(self.expt_cfg['depth']):
            gate_list1.append(random.randint(0, 9))
            gate_list2.append(random.randint(0, 9))
        # gate_list = [9,7,4,1,9]
        # gate_list = [5,0,0,3,7]
        # gate_list = [4, 0, 7, 7, 6, 3, 5, 9, 2, 5]
        # gate_list = [4,0,7,7,6]
        # gate_list = [5,4]
        # gate_list1 = [9,0,3,3,4]
        # gate_list2 = [6,8,1,4,7]
        # gate_list1 = [7, 5, 7, 7, 9, 6, 7, 7, 4, 0]
        # gate_list2 = [3, 4, 1, 5, 9, 7, 5, 2, 2, 3]
        #
        # gate_list1 = [7, 5, 7, 7, 9, 6, 7, 0, 0, 0]
        # gate_list2 = [3, 4, 1, 5, 9, 7, 5, 0, 0, 0]
        # gate_list1 = [0,0,0,0,0,0,0,0,0,0]
        self.quantum_device_cfg['rb_gate']['rb_list'].append(gate_list1)
        self.quantum_device_cfg['rb_gate']['rb_list'].append(gate_list2)


        print('gate_list1:', gate_list1)
        print('gate_list2:', gate_list2)

        freq_q1 = self.quantum_device_cfg['qubit']['1']['freq']
        amp_pi1 = self.quantum_device_cfg['pulse_info']['1']['pi_amp']
        amp_hpi1 = self.quantum_device_cfg['pulse_info']['1']['half_pi_amp']
        pi_len1 = self.quantum_device_cfg['pulse_info']['1']['pi_len']
        hpi_len1 = self.quantum_device_cfg['pulse_info']['1']['half_pi_len']

        freq_q2 = self.quantum_device_cfg['qubit']['2']['freq']
        amp_pi2 = self.quantum_device_cfg['pulse_info']['2']['pi_amp']
        amp_hpi2 = self.quantum_device_cfg['pulse_info']['2']['half_pi_amp']
        pi_len2 = self.quantum_device_cfg['pulse_info']['2']['pi_len']
        hpi_len2 = self.quantum_device_cfg['pulse_info']['2']['half_pi_len']

        ## Calculate inverse rotation
        matrix_ref = {}
        # Z, X, Y, -Z, -X, -Y
        matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['1'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['3'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['4'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0]])
        matrix_ref['5'] = np.matrix([[0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0]])
        matrix_ref['6'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['7'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0]])
        matrix_ref['8'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0]])
        matrix_ref['9'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])

        a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
        anow1 = a0
        anow2 = a0
        for i in gate_list1:
            anow1 = np.dot(matrix_ref[str(i)], anow1)
        for i in gate_list2:
            anow2 = np.dot(matrix_ref[str(i)], anow2)
        anow1 = np.matrix.tolist(anow1.T)[0]
        anow2 = np.matrix.tolist(anow2.T)[0]
        max_index1 = anow1.index(max(anow1))
        max_index2 = anow2.index(max(anow2))
        # print(gate_list)
        print(max_index1, max_index2)
        # effective inverse is pi phase+the same operation


        ## apply pulse accordingly
        sequencer.new_sequence(self)
        universal_phase1 = 0
        universal_phase2 = 0
        q1_info = self.quantum_device_cfg['rb_gate']
        for ii in range(self.expt_cfg['depth']):
            #  Single qubit gate first
            gate_name1 = gate_list1[ii]
            gate_name2 = gate_list2[ii]
            if gate_name1 == 0:
                pass
            if gate_name1 == 1:
                universal_phase1 += np.pi
                # if universal_phase < -2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name1 == 2:
                sequencer.append('charge1',
                                 Square(max_amp=amp_pi1, flat_len=pi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=0+universal_phase1))

            if gate_name1 == 3:
                sequencer.append('charge1',
                                 Square(max_amp=amp_pi1, flat_len=pi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=-np.pi/2 + universal_phase1))

            if gate_name1 == 4:
                universal_phase1 += np.pi/2
                # if universal_phase < 2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name1 == 5:
                sequencer.append('charge1',
                                 Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=0 + universal_phase1))
            if gate_name1 == 6:
                sequencer.append('charge1',
                                 Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=-np.pi/2 + universal_phase1))
            if gate_name1 == 7:
                universal_phase1 -= np.pi/2
                # if universal_phase > 2*np.pi:
                #     universal_phase = universal_phase - 2*np.pi

            if gate_name1 == 8:
                sequencer.append('charge1',
                                 Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=-np.pi + universal_phase1))

            if gate_name1 == 9:
                sequencer.append('charge1',
                                 Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q1,
                                        phase=np.pi/2 + universal_phase1))

            # on Q2
            if gate_name2 == 0:
                pass
            if gate_name2 == 1:
                universal_phase2 += np.pi
                # if universal_phase < -2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name2 == 2:
                sequencer.append('charge2',
                                 Square(max_amp=amp_pi2, flat_len=pi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=0+universal_phase2))

            if gate_name2 == 3:
                sequencer.append('charge2',
                                 Square(max_amp=amp_pi2, flat_len=pi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=-np.pi/2 + universal_phase2))

            if gate_name2 == 4:
                universal_phase2 += np.pi/2
                # if universal_phase < 2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name2 == 5:
                sequencer.append('charge2',
                                 Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=0 + universal_phase2))
            if gate_name2 == 6:
                sequencer.append('charge2',
                                 Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=-np.pi/2 + universal_phase2))
            if gate_name2 == 7:
                universal_phase2 -= np.pi/2
                # if universal_phase > 2*np.pi:
                #     universal_phase = universal_phase - 2*np.pi

            if gate_name2 == 8:
                sequencer.append('charge2',
                                 Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=-np.pi + universal_phase2))

            if gate_name2 == 9:
                sequencer.append('charge2',
                                 Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q2,
                                        phase=np.pi/2 + universal_phase2))


            sequencer.sync_channels_time(self.channels)
        # inverse of the rotation
        # on Q1
        if max_index1==1:  # X
            sequencer.append('charge1',
                             Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q1,
                                    phase=np.pi / 2 + universal_phase1))
        if max_index1==2:  # Y
            sequencer.append('charge1',
                             Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q1,
                                    phase=0 + universal_phase1))
        if max_index1==3:  #-Z
            sequencer.append('charge1',
                             Square(max_amp=amp_pi1, flat_len=pi_len1,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q1,
                                    phase=0 + universal_phase1))

        if max_index1==4:  # -X
            sequencer.append('charge1',
                             Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q1,
                                    phase=-np.pi / 2 + universal_phase1))
        if max_index1==5:  # -Y
            sequencer.append('charge1',
                             Square(max_amp=amp_hpi1, flat_len=hpi_len1,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q1,
                                    phase=np.pi + universal_phase1))

        # on Q2
        if max_index2 == 1:  # X
            sequencer.append('charge2',
                             Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q2,
                                    phase=np.pi / 2 + universal_phase2))
        if max_index2 == 2:  # Y
            sequencer.append('charge2',
                             Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q2,
                                    phase=0 + universal_phase2))
        if max_index2 == 3:  # -Z
            sequencer.append('charge2',
                             Square(max_amp=amp_pi2, flat_len=pi_len2,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q2,
                                    phase=0 + universal_phase2))

        if max_index2 == 4:  # -X
            sequencer.append('charge2',
                             Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q2,
                                    phase=-np.pi / 2 + universal_phase2))
        if max_index2 == 5:  # -Y
            sequencer.append('charge2',
                             Square(max_amp=amp_hpi2, flat_len=hpi_len2,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q2,
                                    phase=np.pi + universal_phase2))

        sequencer.sync_channels_time(self.channels)
        # Readout of the RB sequence:
        self.readout(sequencer, ['1','2'], sideband=False)
        sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, ['1','2'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration
        for qubit_id in ['2', '1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, ['1','2'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration
        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
            'ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',
                         Square(max_amp=self.pulse_info['2']['pi_ee_amp'],
                                flat_len=self.pulse_info['2']['pi_ee_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                cutoff_sigma=2,
                                freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        self.readout(sequencer, ['1','2'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def rb(self, sequencer):
        # Author: Ziqian 2021/06/29
        ''' single qubit randomized benchmarking '''

        ## generate sequences of random pulses
        ## 1:Z,   2:X, 3:Y
        ## 4:Z/2, 5:X/2, 6:Y/2
        ## 7:-Z/2, 8:-X/2, 9:-Y/2
        ## 0:I

        gate_list = []
        for ii in range(self.expt_cfg['depth']):
            gate_list.append(random.randint(0, 9))
        # gate_list = [9,7,4,1,9]
        # gate_list = [5,0,0,3,7]
        # gate_list = [4, 0, 7, 7, 6, 3, 5, 9, 2, 5]
        # gate_list = [4,0,7,7,6]
        # gate_list = [3, 6, 8, 4, 8, 8]
        self.quantum_device_cfg['rb_gate']['rb_list'].append(gate_list)


        print('gate_list:', gate_list)
        on_qubit = self.expt_cfg['on_qubits']
        # if on_qubit == '1':
        #     freq_q = self.quantum_device_cfg['qubit']['1']['freq']
        #     amp_pi = self.quantum_device_cfg['pulse_info']['1']['pi_amp']
        #     amp_hpi = self.quantum_device_cfg['pulse_info']['1']['half_pi_amp']
        #     pi_len = self.quantum_device_cfg['pulse_info']['1']['pi_len']
        #     hpi_len = self.quantum_device_cfg['pulse_info']['1']['half_pi_len']
        # else:
        #     freq_q = self.quantum_device_cfg['qubit']['2']['freq']
        #     amp_pi = self.quantum_device_cfg['pulse_info']['2']['pi_amp']
        #     amp_hpi = self.quantum_device_cfg['pulse_info']['2']['half_pi_amp']
        #     pi_len = self.quantum_device_cfg['pulse_info']['2']['pi_len']
        #     hpi_len = self.quantum_device_cfg['pulse_info']['2']['half_pi_len']

        ## Calculate inverse rotation
        matrix_ref = {}
        # Z, X, Y, -Z, -X, -Y
        matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['1'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['3'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['4'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0]])
        matrix_ref['5'] = np.matrix([[0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0]])
        matrix_ref['6'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['7'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0]])
        matrix_ref['8'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0]])
        matrix_ref['9'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])

        a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
        anow = a0
        for i in gate_list:
            anow = np.dot(matrix_ref[str(i)], anow)
        anow1 = np.matrix.tolist(anow.T)[0]
        max_index = anow1.index(max(anow1))
        print(gate_list)
        print(max_index)
        # effective inverse is pi phase+the same operation


        ## apply pulse accordingly
        sequencer.new_sequence(self)
        universal_phase = 0
        q1_info = self.quantum_device_cfg['rb_gate']
        gate_symbol = ['+z','+x','+y','+z/2', '+x/2', '+y/2', '-z/2', '-x/2', '-y/2']
        for ii in range(self.expt_cfg['depth']):
            # print(universal_phase)
            #  Single qubit gate first
            gate_name = gate_list[ii]
            if gate_name == 0:
                pass
            else:
                if self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['drive']:

                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_amps_prep'],
                                                                 flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_phases_prep'])+universal_phase,
                                                                 plot=False))
                    sequencer.append('charge2',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge2_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge2_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'charge2_phases_prep']) + universal_phase,
                                         plot=False))
                    sequencer.append('flux1',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux1_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux1_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'flux1_phases_prep']) + universal_phase,
                                         plot=False))
                    sequencer.append('flux2',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux2_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux2_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'flux2_phases_prep']) + universal_phase,
                                         plot=False))
                    universal_phase += self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                           'vz_phase'] / 180 * np.pi
                else:   # only VZ gate
                    universal_phase += self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['vz_phase']/180*np.pi

            sequencer.sync_channels_time(self.channels)
        # inverse of the rotation
        inverse_gate_symbol = ['-y/2', '+x/2', '+x', '+y/2', '-x/2']
        if max_index == 0:
            pass
        else:
            if self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]]['drive']:

                sequencer.append('charge1',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge1_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge1_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'charge1_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('charge2',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge2_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge2_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'charge2_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('flux1',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux1_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                         'flux1_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                             'flux1_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('flux2',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux2_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux2_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'flux2_phases_prep']) + universal_phase,
                                     plot=False))
                universal_phase += self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                       'vz_phase'] / 180 * np.pi
            else:  # only VZ gate
                universal_phase += self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]]['vz_phase'] / 180 * np.pi

        sequencer.sync_channels_time(self.channels)
        # Readout of the RB sequence:
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # histogram
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % on_qubit, self.qubit_pi[on_qubit])
            pi_lenth = self.pulse_info[on_qubit]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rb_interleaved(self, sequencer):
        # Author: Ziqian 2021/06/29
        ''' single qubit randomized benchmarking '''

        ## generate sequences of random pulses
        ## 1:Z,   2:X, 3:Y
        ## 4:Z/2, 5:X/2, 6:Y/2
        ## 7:-Z/2, 8:-X/2, 9:-Y/2
        ## 0:I

        gate_list = []
        for ii in range(self.expt_cfg['depth']):
            gate_list.append(random.randint(0, 9))
            gate_list.append(int(self.expt_cfg['interleaved']))
        # gate_list = [9,7,4,1,9]
        # gate_list = [5,0,0,3,7]
        # gate_list = [4, 0, 7, 7, 6, 3, 5, 9, 2, 5]
        # gate_list = [4,0,7,7,6]
        # gate_list = [3, 6, 8, 4, 8, 8]
        self.quantum_device_cfg['rb_gate']['rb_list'].append(gate_list)


        print('gate_list:', gate_list)
        on_qubit = self.expt_cfg['on_qubits']
        # if on_qubit == '1':
        #     freq_q = self.quantum_device_cfg['qubit']['1']['freq']
        #     amp_pi = self.quantum_device_cfg['pulse_info']['1']['pi_amp']
        #     amp_hpi = self.quantum_device_cfg['pulse_info']['1']['half_pi_amp']
        #     pi_len = self.quantum_device_cfg['pulse_info']['1']['pi_len']
        #     hpi_len = self.quantum_device_cfg['pulse_info']['1']['half_pi_len']
        # else:
        #     freq_q = self.quantum_device_cfg['qubit']['2']['freq']
        #     amp_pi = self.quantum_device_cfg['pulse_info']['2']['pi_amp']
        #     amp_hpi = self.quantum_device_cfg['pulse_info']['2']['half_pi_amp']
        #     pi_len = self.quantum_device_cfg['pulse_info']['2']['pi_len']
        #     hpi_len = self.quantum_device_cfg['pulse_info']['2']['half_pi_len']

        ## Calculate inverse rotation
        matrix_ref = {}
        # Z, X, Y, -Z, -X, -Y
        matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['1'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['3'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['4'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0]])
        matrix_ref['5'] = np.matrix([[0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0]])
        matrix_ref['6'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['7'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0]])
        matrix_ref['8'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0]])
        matrix_ref['9'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])

        a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
        anow = a0
        for i in gate_list:
            anow = np.dot(matrix_ref[str(i)], anow)
        anow1 = np.matrix.tolist(anow.T)[0]
        max_index = anow1.index(max(anow1))
        print(gate_list)
        print(max_index)
        # effective inverse is pi phase+the same operation


        ## apply pulse accordingly
        sequencer.new_sequence(self)
        universal_phase = 0
        q1_info = self.quantum_device_cfg['rb_gate']
        gate_symbol = ['+z','+x','+y','+z/2', '+x/2', '+y/2', '-z/2', '-x/2', '-y/2']
        for ii in range(self.expt_cfg['depth']):
            # print(universal_phase)
            #  Single qubit gate first
            gate_name = gate_list[ii]
            if gate_name == 0:
                pass
            else:
                if self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['drive']:

                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_amps_prep'],
                                                                 flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge1_phases_prep'])+universal_phase,
                                                                 plot=False))
                    sequencer.append('charge2',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge2_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['charge2_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'charge2_phases_prep']) + universal_phase,
                                         plot=False))
                    sequencer.append('flux1',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux1_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux1_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'flux1_phases_prep']) + universal_phase,
                                         plot=False))
                    sequencer.append('flux2',
                                     Square_multitone_sequential(
                                         max_amps=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux2_amps_prep'],
                                         flat_lens=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['flux2_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]][
                                                 'flux2_phases_prep']) + universal_phase,
                                         plot=False))
                    universal_phase += self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                           'vz_phase'] / 180 * np.pi
                else:   # only VZ gate
                    universal_phase += self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name-1]]['vz_phase']/180*np.pi

            sequencer.sync_channels_time(self.channels)
        # inverse of the rotation
        inverse_gate_symbol = ['-y/2', '+x/2', '+x', '+y/2', '-x/2']
        if max_index == 0:
            pass
        else:
            if self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]]['drive']:

                sequencer.append('charge1',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge1_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge1_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'charge1_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('charge2',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge2_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'charge2_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'charge2_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('flux1',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux1_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                         'flux1_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][gate_symbol[gate_name - 1]][
                                             'flux1_phases_prep']) + universal_phase,
                                     plot=False))
                sequencer.append('flux2',
                                 Square_multitone_sequential(
                                     max_amps=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux2_amps_prep'],
                                     flat_lens=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'times_prep'],
                                     ramp_sigma_len=
                                     self.quantum_device_cfg['flux_pulse_info'][
                                         '1'][
                                         'ramp_sigma_len'],
                                     cutoff_sigma=2,
                                     freqs=self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                         'flux2_freqs_prep'],
                                     phases=np.pi / 180 * np.array(
                                         self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                             'flux2_phases_prep']) + universal_phase,
                                     plot=False))
                universal_phase += self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]][
                                       'vz_phase'] / 180 * np.pi
            else:  # only VZ gate
                universal_phase += self.quantum_device_cfg['rb'][on_qubit][inverse_gate_symbol[max_index - 1]]['vz_phase'] / 180 * np.pi

        sequencer.sync_channels_time(self.channels)
        # Readout of the RB sequence:
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # histogram
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % on_qubit, self.qubit_pi[on_qubit])
            pi_lenth = self.pulse_info[on_qubit]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def rb1(self, sequencer):
        # Author: Ziqian 2021/06/29
        ''' single qubit randomized benchmarking '''

        ## generate sequences of random pulses
        ## 1:Z,   2:X, 3:Y
        ## 4:Z/2, 5:X/2, 6:Y/2
        ## 7:-Z/2, 8:-X/2, 9:-Y/2
        ## 0:I

        gate_list = []
        for ii in range(self.expt_cfg['depth']):
            gate_list.append(random.randint(0, 9))
        # gate_list = [9,7,4,1,9]
        # gate_list = [5,0,0,3,7]
        # gate_list = [4, 0, 7, 7, 6, 3, 5, 9, 2, 5]
        # gate_list = [4,0,7,7,6]
        gate_list = [3, 6, 8, 4, 8, 8]
        self.quantum_device_cfg['rb _gate']['rb_list'].append(gate_list)


        print('gate_list:', gate_list)
        on_qubit = self.expt_cfg['on_qubits']
        if on_qubit == '1':
            freq_q = self.quantum_device_cfg['qubit']['1']['freq']
            amp_pi = self.quantum_device_cfg['pulse_info']['1']['pi_amp']
            amp_hpi = self.quantum_device_cfg['pulse_info']['1']['half_pi_amp']
            pi_len = self.quantum_device_cfg['pulse_info']['1']['pi_len']
            hpi_len = self.quantum_device_cfg['pulse_info']['1']['half_pi_len']
        else:
            freq_q = self.quantum_device_cfg['qubit']['2']['freq']
            amp_pi = self.quantum_device_cfg['pulse_info']['2']['pi_amp']
            amp_hpi = self.quantum_device_cfg['pulse_info']['2']['half_pi_amp']
            pi_len = self.quantum_device_cfg['pulse_info']['2']['pi_len']
            hpi_len = self.quantum_device_cfg['pulse_info']['2']['half_pi_len']

        ## Calculate inverse rotation
        matrix_ref = {}
        # Z, X, Y, -Z, -X, -Y
        matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['1'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0]])
        matrix_ref['3'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['4'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0]])
        matrix_ref['5'] = np.matrix([[0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0]])
        matrix_ref['6'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])
        matrix_ref['7'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0]])
        matrix_ref['8'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                     [0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0]])
        matrix_ref['9'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1]])

        a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
        anow = a0
        for i in gate_list:
            anow = np.dot(matrix_ref[str(i)], anow)
        anow1 = np.matrix.tolist(anow.T)[0]
        max_index = anow1.index(max(anow1))
        print(gate_list)
        print(max_index)
        # effective inverse is pi phase+the same operation


        ## apply pulse accordingly
        sequencer.new_sequence(self)
        universal_phase = 0
        q1_info = self.quantum_device_cfg['rb_gate']
        for ii in range(self.expt_cfg['depth']):
            print(universal_phase)
            #  Single qubit gate first
            gate_name = gate_list[ii]
            if gate_name == 0:
                pass
            if gate_name == 1:
                universal_phase += np.pi
                # if universal_phase < -2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name == 2:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_pi, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=0+universal_phase))

            if gate_name == 3:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_pi, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=-np.pi/2 + universal_phase))

            if gate_name == 4:
                universal_phase += np.pi/2
                # if universal_phase < 2*np.pi:
                #     universal_phase = universal_phase + 2*np.pi

            if gate_name == 5:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_hpi, flat_len=hpi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=0 + universal_phase))
            if gate_name == 6:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_hpi, flat_len=hpi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=-np.pi/2 + universal_phase))
            if gate_name == 7:
                universal_phase -= np.pi/2
                # if universal_phase > 2*np.pi:
                #     universal_phase = universal_phase - 2*np.pi

            if gate_name == 8:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_hpi, flat_len=hpi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=-np.pi + universal_phase))

            if gate_name == 9:
                sequencer.append('charge%s' % on_qubit,
                                 Square(max_amp=amp_hpi, flat_len=hpi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=freq_q,
                                        phase=np.pi/2 + universal_phase))


            sequencer.sync_channels_time(self.channels)
        # inverse of the rotation
        if max_index==1:  # X
            sequencer.append('charge%s' % on_qubit,
                             Square(max_amp=amp_hpi, flat_len=hpi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q,
                                    phase=np.pi / 2 + universal_phase))
        if max_index==2:  # Y
            sequencer.append('charge%s' % on_qubit,
                             Square(max_amp=amp_hpi, flat_len=hpi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q,
                                    phase=0 + universal_phase))
        if max_index==3:  #-Z
            sequencer.append('charge%s' % on_qubit,
                             Square(max_amp=amp_pi, flat_len=pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q,
                                    phase=0 + universal_phase))

        if max_index==4:  # -X
            sequencer.append('charge%s' % on_qubit,
                             Square(max_amp=amp_hpi, flat_len=hpi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q,
                                    phase=-np.pi / 2 + universal_phase))
        if max_index==5:  # -Y
            sequencer.append('charge%s' % on_qubit,
                             Square(max_amp=amp_hpi, flat_len=hpi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq_q,
                                    phase=np.pi + universal_phase))

        sequencer.sync_channels_time(self.channels)
        # Readout of the RB sequence:
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # histogram
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % on_qubit, self.qubit_pi[on_qubit])
            pi_lenth = self.pulse_info[on_qubit]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)



    def t1(self, sequencer):
        # t1 sequences

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            if self.expt_cfg['pre_pulse']:
                # Initial pulse before t1
                pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                repeat = pre_pulse_info['repeat']
                for repeat_i in range(repeat):
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge1_phases_prep']),
                                                                 plot=False))
                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge2_phases_prep']),
                                                                 plot=False))
                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))
                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                sequencer.sync_channels_time(self.channels)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['use_freq_amp_pi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_pi'][-1]
                    freq = self.expt_cfg['use_freq_amp_pi'][1]
                    amp = self.expt_cfg['use_freq_amp_pi'][2]
                    pi_len = self.expt_cfg['use_freq_amp_pi'][3]
                    self.qubit_pi_new = {
                                        "1": Square(max_amp=amp, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}
                else:
                    self.qubit_pi_new = self.qubit_pi
                    charge_port = 'charge%s' %self.charge_port[qubit_id]


                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_pi_new[qubit_id])
                    sequencer.append('flux1', Idle(time=t1_len))
                else:
                    sequencer.append(charge_port, self.qubit_pi_new[qubit_id])
                    sequencer.append(charge_port, Idle(time=t1_len))

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def t1_while_flux(self, sequencer):
        # t1 sequences
        tones = max(len(self.expt_cfg['flux_freq'][0]), len(self.expt_cfg['flux_freq'][1]))
        flux_freqs = self.expt_cfg['flux_freq']
        flux_amps = (self.expt_cfg['flux_amp'])
        flux_phases = (self.expt_cfg['flux_phase'])  # phase should be in degrees

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            if self.expt_cfg['pre_pulse']:
                # Initial pulse before t1
                pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                repeat = pre_pulse_info['repeat']
                for repeat_i in range(repeat):
                    sequencer.append('charge1',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge1_phases_prep']),
                                                                 plot=False))
                    sequencer.append('charge2',
                                     Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                 flat_lens=pre_pulse_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     pre_pulse_info['charge2_phases_prep']),
                                                                 plot=False))
                    sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux1_phases_prep']),
                                                                          plot=False))
                    sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                          flat_lens=pre_pulse_info['times_prep'],
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              pre_pulse_info['flux2_phases_prep']),
                                                                          plot=False))

                sequencer.sync_channels_time(self.channels)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['use_freq_amp_pi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_pi'][-1]
                    freq = self.expt_cfg['use_freq_amp_pi'][1]
                    amp = self.expt_cfg['use_freq_amp_pi'][2]
                    pi_len = self.expt_cfg['use_freq_amp_pi'][3]
                    self.qubit_pi_new = {
                                        "1": Square(max_amp=amp, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}
                else:
                    self.qubit_pi_new = self.qubit_pi
                    charge_port = 'charge%s' %self.charge_port[qubit_id]


                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_pi_new[qubit_id])
                    sequencer.append('flux1', Idle(time=t1_len))
                else:
                    sequencer.append(charge_port, self.qubit_pi_new[qubit_id])
                    sequencer.sync_channels_time(self.channels)
                    sequencer.append(charge_port, Idle(time=t1_len))

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    sequencer.append('flux%s' % flux_line_id,
                                     Square_multitone(max_amp=np.array(flux_amps[index]),
                                                      flat_len=[t1_len] * tones,
                                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                                          'ramp_sigma_len'],
                                                      cutoff_sigma=2, freq=np.array(flux_freqs[index]),
                                                      phase=np.pi / 180 * (np.array(flux_phases[index])), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def t1_while_flux_LO(self, sequencer):
        # Flux-drive os on between pi pulse and readout, use Signalcore as LO
        # Set flux_LO -> 0.0 if not want to use signalcore

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['use_freq_amp_pi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_pi'][-1]
                    freq = self.expt_cfg['use_freq_amp_pi'][1]
                    amp = self.expt_cfg['use_freq_amp_pi'][2]
                    pi_len = self.expt_cfg['use_freq_amp_pi'][3]
                    self.qubit_pi_new = {
                                        "1": Square(max_amp=amp, flat_len=pi_len,
                                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                                    cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=pi_len,
                                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                                    cutoff_sigma=2, freq=freq, phase=0)}
                else:
                    self.qubit_pi_new = self.qubit_pi
                    charge_port = 'charge%s' %self.charge_port[qubit_id]


                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_pi_new[qubit_id])
                    sequencer.sync_channels_time(self.channels)
                    sequencer.append('flux1', Idle(time=t1_len))
                else:
                    sequencer.append(charge_port, self.qubit_pi_new[qubit_id])
                    sequencer.sync_channels_time(self.channels)
                    sequencer.append(charge_port, Idle(time=t1_len))

            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id,
                                         Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=t1_len,
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                                phase=np.pi/180*self.expt_cfg['flux_phase'][index], plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def ef_t1(self, sequencer):
        # t1 for the e and f level

        for ef_t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux%s' % qubit_id, self.qubit_pi[qubit_id])
                    sequencer.append('flux%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.append('flux%s' % qubit_id, Idle(time=ef_t1_len))
                else:
                    sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id, Idle(time=ef_t1_len))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_rabi(self, sequencer):
        # ef rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:

                    sequencer.append('flux1',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=0))
                else:
                    sequencer.append('charge1',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            ramsey_phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_pi[qubit_id])
                    sequencer.append('flux1', self.qubit_ef_half_pi[qubit_id])
                    sequencer.append('flux1', Idle(time=ramsey_len))
                    sequencer.append('flux1',
                                     Square(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_ef_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=ramsey_phase))
                    sequencer.append('flux1', self.qubit_pi[qubit_id])
                else:
                    sequencer.append('charge1', self.qubit_pi[qubit_id])
                    sequencer.append('charge1', self.qubit_ef_half_pi[qubit_id])
                    sequencer.append('charge1', Idle(time=ramsey_len))
                    sequencer.append('charge1',
                                     Square(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_ef_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=ramsey_phase))
                    sequencer.append('charge1', self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_echo(self, sequencer):
        # ef echo sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_half_pi[qubit_id])
                for echo_id in range(self.expt_cfg['echo_times']):
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['pi_ef_len'], cutoff_sigma=2,
                                       freq=self.qubit_ef_freq[qubit_id], phase=0.5*np.pi, plot=False))
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ramsey(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                charge_port = 'charge%s' %self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}


                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_half_pi[qubit_id])
                    sequencer.append('flux1', Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                    sequencer.append('flux1', ramsey_2nd_pulse)
                else:
                    sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                    sequencer.append(charge_port, Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                    sequencer.append(charge_port, ramsey_2nd_pulse)

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def t1rho(self, sequencer): # Added by Tanay
        # mid_phase is the phase of the middle pulse in degree.

        for t1rho_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for index, qubit_id in enumerate(self.expt_cfg['on_qubits']):

                charge_port = 'charge%s' %self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}


                for ge_pi_id in self.expt_cfg['ge_pi']:
                    if len(self.expt_cfg.get('flux_pi')) == 0:
                        sequencer.append('charge%s' %self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('flux%s' %self.expt_cfg['flux_pi'][index], self.qubit_pi[ge_pi_id])

                sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if len(self.expt_cfg.get('flux_pi')) == 0:
                        sequencer.append('charge%s' %self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                    else:
                        sequencer.append('flux%s' %self.expt_cfg['flux_pi'][index], self.qubit_ef_pi[ef_pi_id])

                sequencer.sync_channels_time(self.channels)
                sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                sequencer.append(charge_port, Square(max_amp=self.expt_cfg['amp'], flat_len=t1rho_len,
                                                     ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                     cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=np.pi/180*self.expt_cfg['mid_phase'],
                                                     plot=False))
                sequencer.append(charge_port, self.qubit_half_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ramsey_while_flux_old(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                sequencer.append('charge1', self.qubit_half_pi[qubit_id])
                sequencer.append('charge1', Idle(time=ramsey_len))
                ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                sequencer.append('charge1', ramsey_2nd_pulse)

                sequencer.append('flux%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['flux_amp'], flat_len=2*half_pi_len+ramsey_len+2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len']*2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def t2_while_flux(self, sequencer):
        # t2 sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                charge_port = 'charge%s' % self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0),
                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0)}

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ef_pi_id, self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for ge_pi2_id in self.expt_cfg['ge_pi2']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ge_pi2_id, self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi2_id], self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    repeat = pre_pulse_info['repeat']
                    for repeat_i in range(repeat):
                        sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge1_phases_prep']),
                                                                                plot=False))
                        sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge2_phases_prep']),
                                                                                plot=False))
                        sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux1_phases_prep']),
                                                                              plot=False))
                        sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux2_phases_prep']),
                                                                              plot=False))

                    sequencer.sync_channels_time(self.channels)


                sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                sequencer.append(charge_port, Idle(time=ramsey_len))
                ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                sequencer.append(charge_port, ramsey_2nd_pulse)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    # Don't use two qubit charge drives together
                    sequencer.append('flux%s' %flux_line_id,
                                     Square(max_amp=self.expt_cfg['flux_amp'][index],
                                            flat_len=2 * self.pulse_info[qubit_id]['half_pi_len'] + ramsey_len + 2 *
                                                     self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'] * 2,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                            phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))
            sequencer.sync_channels_time(self.channels)
            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def multitone_ramsey_while_flux(self, sequencer):
        # ramsey sequences while driving all other tones
        # By Ziqian

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                charge_port = 'charge%s' % self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0),
                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0)}

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ef_pi_id, self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for ge_pi2_id in self.expt_cfg['ge_pi2']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ge_pi2_id, self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi2_id], self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    repeat = pre_pulse_info['repeat']
                    for repeat_i in range(repeat):
                        sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge1_phases_prep']),
                                                                                plot=False))
                        sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge2_phases_prep']),
                                                                                plot=False))
                        sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux1_phases_prep']),
                                                                              plot=False))
                        sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux2_phases_prep']),
                                                                              plot=False))

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_half_pi[qubit_id])
                    sequencer.append('flux1', Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append('flux1', ramsey_2nd_pulse)
                else:
                    sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                    sequencer.sync_channels_time(self.channels)
                    ramsey_len1 = copy.deepcopy(self.expt_cfg['charge1_amps_a'])
                    for ii in range(len(self.expt_cfg['charge1_amps_a'])):
                        for jj in range(len(self.expt_cfg['charge1_amps_a'][ii])):
                            ramsey_len1[ii][jj] = ramsey_len
                    # print(self.expt_cfg)
                    sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_a'],
                                                                            flat_lens=ramsey_len1,
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=self.expt_cfg['charge1_freqs_a'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                self.expt_cfg['charge1_phases_a']),
                                                                            plot=False))
                    sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_a'],
                                                                            flat_lens=ramsey_len1,
                                                                            ramp_sigma_len=
                                                                            self.quantum_device_cfg['flux_pulse_info'][
                                                                                '1'][
                                                                                'ramp_sigma_len'],
                                                                            cutoff_sigma=2,
                                                                            freqs=self.expt_cfg['charge2_freqs_a'],
                                                                            phases=np.pi / 180 * np.array(
                                                                                self.expt_cfg['charge2_phases_a']),
                                                                            plot=False))
                    sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_a'],
                                                                          flat_lens=ramsey_len1,
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=self.expt_cfg['flux1_freqs_a'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              self.expt_cfg['flux1_phases_a']),
                                                                          plot=False))
                    sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_a'],
                                                                          flat_lens=ramsey_len1,
                                                                          ramp_sigma_len=
                                                                          self.quantum_device_cfg['flux_pulse_info'][
                                                                              '1'][
                                                                              'ramp_sigma_len'],
                                                                          cutoff_sigma=2,
                                                                          freqs=self.expt_cfg['flux2_freqs_a'],
                                                                          phases=np.pi / 180 * np.array(
                                                                              self.expt_cfg['flux2_phases_a']),
                                                                          plot=False))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append(charge_port, ramsey_2nd_pulse)


            sequencer.sync_channels_time(self.channels)
            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def ramsey_while_flux(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                charge_port = 'charge%s' % self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0),
                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0)}

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                for ef_pi_id in self.expt_cfg['ef_pi']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ef_pi_id, self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ef_pi_id], self.qubit_ef_pi[ef_pi_id])
                        sequencer.sync_channels_time(self.channels)

                for ge_pi2_id in self.expt_cfg['ge_pi2']:
                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux%s' % ge_pi2_id, self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi2_id], self.qubit_gf_pi[ge_pi2_id])
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['pre_pulse']:
                    # Initial pulse before Rabi
                    pre_pulse_info = self.quantum_device_cfg['pre_pulse_info']
                    repeat = pre_pulse_info['repeat']
                    for repeat_i in range(repeat):
                        sequencer.append('charge1', Square_multitone_sequential(max_amps=pre_pulse_info['charge1_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge1_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge1_phases_prep']),
                                                                                plot=False))
                        sequencer.append('charge2', Square_multitone_sequential(max_amps=pre_pulse_info['charge2_amps_prep'],
                                                                                flat_lens=pre_pulse_info['times_prep'],
                                                                                ramp_sigma_len=
                                                                                self.quantum_device_cfg['flux_pulse_info'][
                                                                                    '1'][
                                                                                    'ramp_sigma_len'],
                                                                                cutoff_sigma=2,
                                                                                freqs=pre_pulse_info['charge2_freqs_prep'],
                                                                                phases=np.pi / 180 * np.array(
                                                                                    pre_pulse_info['charge2_phases_prep']),
                                                                                plot=False))
                        sequencer.append('flux1', Square_multitone_sequential(max_amps=pre_pulse_info['flux1_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux1_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux1_phases_prep']),
                                                                              plot=False))
                        sequencer.append('flux2', Square_multitone_sequential(max_amps=pre_pulse_info['flux2_amps_prep'],
                                                                              flat_lens=pre_pulse_info['times_prep'],
                                                                              ramp_sigma_len=
                                                                              self.quantum_device_cfg['flux_pulse_info'][
                                                                                  '1'][
                                                                                  'ramp_sigma_len'],
                                                                              cutoff_sigma=2,
                                                                              freqs=pre_pulse_info['flux2_freqs_prep'],
                                                                              phases=np.pi / 180 * np.array(
                                                                                  pre_pulse_info['flux2_phases_prep']),
                                                                              plot=False))

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_half_pi[qubit_id])
                    sequencer.append('flux1', Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append('flux1', ramsey_2nd_pulse)
                else:
                    sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                    sequencer.append(charge_port, Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append(charge_port, ramsey_2nd_pulse)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    # Don't use two qubit charge drives together
                    sequencer.append('flux%s' %flux_line_id,
                                     Square(max_amp=self.expt_cfg['flux_amp'][index],
                                            flat_len=2 * self.pulse_info[qubit_id]['half_pi_len'] + ramsey_len + 2 *
                                                     self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'] * 2,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                            phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))
            sequencer.sync_channels_time(self.channels)
            self.readout(sequencer, self.expt_cfg['on_cavity'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def ramsey_while_flux_during_idle(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                charge_port = 'charge%s' % self.charge_port[qubit_id]
                if self.expt_cfg['use_freq_amp_halfpi'][0]:
                    charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
                    freq = self.expt_cfg['use_freq_amp_halfpi'][1]
                    amp = self.expt_cfg['use_freq_amp_halfpi'][2]
                    half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
                    self.qubit_half_pi = {
                        "1": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0),
                        "2": Square(max_amp=amp, flat_len=half_pi_len,
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2, freq=freq, phase=0)}

                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['flux_probe']:
                    sequencer.append('flux1', self.qubit_half_pi[qubit_id])
                    sequencer.append('flux1', Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append('flux1', ramsey_2nd_pulse)
                else:
                    sequencer.append(charge_port, self.qubit_half_pi[qubit_id])
                    sequencer.append(charge_port, Idle(time=ramsey_len))
                    ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                    ramsey_2nd_pulse.phase = 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']
                    sequencer.append(charge_port, ramsey_2nd_pulse)

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                    # Don't use two qubit charge drives together
                    sequencer.append('flux%s' % flux_line_id,
                                     Square(max_amp=0, flat_len=self.pulse_info[qubit_id]['half_pi_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                            phase=np.pi / 180 * self.expt_cfg['phase'][index], plot=False))

                    sequencer.append('flux%s' %flux_line_id,
                                     Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=ramsey_len -
                                            4*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                            phase=np.pi/180*self.expt_cfg['phase'][index], plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def sideband_ramsey(self, sequencer):
        # calibrating sideband freqs

        freq = self.expt_cfg['freq_amp_halfpi'][0]
        amp = self.expt_cfg['freq_amp_halfpi'][1]
        half_pi_len = self.expt_cfg['freq_amp_halfpi'][2]
        self.sideband_half_pi = {
                            "1": Square(max_amp=amp, flat_len=half_pi_len,
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0),
                            "2": Square(max_amp=amp, flat_len=half_pi_len,
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2, freq=freq, phase=0)}


        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            self.sideband_idle = {
                    "1": Square(max_amp=amp, flat_len=ramsey_len,
                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.expt_cfg['idle_freq'], phase=0),
                    "2": Square(max_amp=amp, flat_len=ramsey_len,
                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.expt_cfg['idle_freq'], phase=0)}

            for qubit_id in self.expt_cfg['on_qubits']:


                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        sequencer.append('charge1', self.qubit_pi[ge_pi_id])
                        sequencer.sync_channels_time(self.channels)

                sequencer.append('flux1', self.sideband_half_pi[qubit_id])
                sequencer.append('flux1', self.sideband_idle[qubit_id])
                ramsey_2nd_pulse = copy.copy(self.sideband_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                sequencer.append('flux1', ramsey_2nd_pulse)

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def half_pi_sweep(self, sequencer):
        # ramsey sequences
        for halflength in np.arange(self.expt_cfg['start_pi_length'], self.expt_cfg['stop_pi_length'],
                                    self.expt_cfg['step_pi_length']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                # prepare initial state
                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' % ge_pi_id, self.qubit_pi[ge_pi_id])
                    sequencer.sync_channels_time(self.channels)

                # Apply two half pi pulses
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.expt_cfg['amp'][0], flat_len=halflength,
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'][0], phase=np.pi/180*self.expt_cfg['pulse_phase'][0]))
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.expt_cfg['amp'][1], flat_len=halflength,
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'][1], phase=np.pi/180*self.expt_cfg['pulse_phase'][1]))

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def half_pi_sweep_phase(self, sequencer):
        # ramsey sequences
        for halfphase in np.arange(self.expt_cfg['start_pi_phase'], self.expt_cfg['stop_pi_phase'],
                                    self.expt_cfg['step_pi_phase']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                # prepare initial state
                for ge_pi_id in self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' % ge_pi_id, self.qubit_pi[ge_pi_id])
                    sequencer.sync_channels_time(self.channels)

                # Apply two half pi pulses
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.expt_cfg['amp'][0], flat_len=self.expt_cfg['pulse_length'][0],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'][0], phase=0))
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.expt_cfg['amp'][1], flat_len=self.expt_cfg['pulse_length'][1],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'][1], phase=halfphase*np.pi/180))

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def iswap_calibration(self, sequencer):
        # Author: Ziqian Li Jul 14th 2021
        # sweeping the phase between iswap and charge lines to minimize the gate error

        for phase in np.arange(self.expt_cfg['phase_start'], self.expt_cfg['phase_stop'], self.expt_cfg['phase_step']):
            sequencer.new_sequence(self)

            qubit_id = self.expt_cfg['on_qubits']
            if self.expt_cfg['U']=='pi':
                if qubit_id == '1':
                    sequencer.append('charge1', self.qubit_half_pi['1'])
                    sequencer.append('charge2', self.qubit_pi['2'])
                else:
                    sequencer.append('charge1', self.qubit_pi['1'])
                    sequencer.append('charge2', self.qubit_half_pi['2'])
            elif self.expt_cfg['U']=='I':
                if qubit_id=='1':
                    sequencer.append('charge1', self.qubit_half_pi['1'])
                else:
                    sequencer.append('charge2', self.qubit_half_pi['2'])
            else:
                sequencer.append('charge1', self.qubit_half_pi['1'])
                sequencer.append('charge2', self.qubit_half_pi['2'])

            sequencer.sync_channels_time(self.channels)

            # add iswap and -iswap

            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                sequencer.append('flux%s' % flux_line_id,
                                 Square(max_amp=self.expt_cfg['iswap_amp'][index], flat_len=self.expt_cfg['iswap_time'][index],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['iswap_freq'][index],
                                        phase=np.pi / 180 * self.expt_cfg['iswap_phase'][index], plot=False))
            sequencer.sync_channels_time(self.channels)
            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                sequencer.append('flux%s' % flux_line_id,
                                 Square(max_amp=self.expt_cfg['iswap_amp'][index], flat_len=self.expt_cfg['iswap_time'][index],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['iswap_freq'][index],
                                        phase=np.pi / 180 * self.expt_cfg['iswap_phase'][index]+np.pi, plot=False))
            sequencer.sync_channels_time(self.channels)
            # final Xpi/2 rotation with phase shifted

            sequencer.append('charge%s' % qubit_id,
                             Square(max_amp=self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_amp'],
                                    flat_len=self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.quantum_device_cfg['qubit'][qubit_id]['freq'], phase=np.pi / 180 * phase))
            sequencer.sync_channels_time(self.channels)


            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def iswap_calibration_2(self, sequencer):
        # Author: Ziqian Li Jul 14th 2021
        # sweeping the phase between iswap and charge lines to minimize the gate error

        for phase in np.arange(self.expt_cfg['phase_start'], self.expt_cfg['phase_stop'], self.expt_cfg['phase_step']):
            sequencer.new_sequence(self)

            qubit_id = self.expt_cfg['on_qubits']
            sequencer.append('charge1', self.qubit_half_pi['1'])
            sequencer.append('charge2', self.qubit_half_pi['2'])

            sequencer.sync_channels_time(self.channels)

            # add iswap, sweeping its phase

            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                sequencer.append('flux%s' % flux_line_id,
                                 Square(max_amp=self.expt_cfg['iswap_amp'][index], flat_len=self.expt_cfg['iswap_time'][index],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['iswap_freq'][index],
                                        phase=np.pi / 180 * phase, plot=False))
            sequencer.sync_channels_time(self.channels)
            # final Xpi/2 rotation and Ypi/2

            sequencer.append('charge1',
                             Square(max_amp=self.quantum_device_cfg['pulse_info']['1']['half_pi_amp'],
                                    flat_len=self.quantum_device_cfg['pulse_info']['1']['half_pi_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.quantum_device_cfg['qubit']['1']['freq'], phase=self.expt_cfg['phase_adjust1']*np.pi/180))
            sequencer.append('charge2',
                             Square(max_amp=self.quantum_device_cfg['pulse_info']['2']['half_pi_amp'],
                                    flat_len=self.quantum_device_cfg['pulse_info']['2']['half_pi_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2, freq=self.quantum_device_cfg['qubit']['2']['freq'], phase=(self.expt_cfg['phase_adjust2']-90)*np.pi/180))
            sequencer.sync_channels_time(self.channels)


            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomography_test(self, sequencer):
        # testing two qubit tomography
        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]

        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y']]


        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            for pi_id in self.expt_cfg['1st_pulse']['pi']:
                sequencer.append('charge%s' %self.charge_port[pi_id], self.qubit_pi[pi_id])
            for half_pi_id in self.expt_cfg['1st_pulse']['half_pi']:
                sequencer.append('charge%s' %self.charge_port[half_pi_id], self.qubit_half_pi[half_pi_id])
            for pi_ef_id in self.expt_cfg['1st_pulse']['pi_ef']:
                sequencer.append('charge%s' %self.charge_port[pi_ef_id], self.qubit_ef_pi[pi_ef_id])
            for half_pi_ef_id in self.expt_cfg['1st_pulse']['half_pi_ef']:
                sequencer.append('charge%s' %self.charge_port[half_pi_ef_id], self.qubit_ef_half_pi[half_pi_ef_id])
            for pi_ee_id in self.expt_cfg['1st_pulse']['pi_ee']:
                sequencer.append('charge%s' % ( int(pi_ee_id) %2 + 1), self.qubit_ee_pi[pi_ee_id])
            for half_pi_ee_id in self.expt_cfg['1st_pulse']['half_pi_ee']:
                sequencer.append('charge%s' % ( int(half_pi_ee_id) %2 + 1), self.qubit_ee_half_pi[half_pi_ee_id])

            sequencer.sync_channels_time(self.channels)

            for pi_id in self.expt_cfg['2nd_pulse']['pi']:
                sequencer.append('charge%s' %self.charge_port[pi_id], self.qubit_pi[pi_id])
            for half_pi_id in self.expt_cfg['2nd_pulse']['half_pi']:
                sequencer.append('charge%s' %self.charge_port[half_pi_id], self.qubit_half_pi[half_pi_id])
            for pi_ef_id in self.expt_cfg['2nd_pulse']['pi_ef']:
                sequencer.append('charge%s' %self.charge_port[pi_ef_id], self.qubit_ef_pi[pi_ef_id])
            for half_pi_ef_id in self.expt_cfg['2nd_pulse']['half_pi_ef']:
                sequencer.append('charge%s' %self.charge_port[half_pi_ef_id], self.qubit_ef_half_pi[half_pi_ef_id])
            for pi_ee_id in self.expt_cfg['2nd_pulse']['pi_ee']:
                sequencer.append('charge%s' % ( int(pi_ee_id) %2 + 1), self.qubit_ee_pi[pi_ee_id])
            for half_pi_ee_id in self.expt_cfg['2nd_pulse']['half_pi_ee']:
                sequencer.append('charge%s' % ( int(half_pi_ee_id) %2 + 1), self.qubit_ee_half_pi[half_pi_ee_id])

            sequencer.sync_channels_time(self.channels)

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['1'])
                m_pulse1.phase = 0 + measurement_phase
                m_pulse2.phase = 0 + measurement_phase
                sequencer.append('charge1', m_pulse1)
                sequencer.append('charge2', m_pulse2)
            elif qubit_1_measure == 'Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['1'])
                m_pulse1.phase = np.pi/2 + measurement_phase
                m_pulse2.phase = np.pi/2 + measurement_phase
                sequencer.append('charge1', m_pulse1)
                sequencer.append('charge2', m_pulse2)
            elif qubit_1_measure == '-X':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['1'])
                m_pulse1.phase = -np.pi + measurement_phase
                m_pulse2.phase = -np.pi + measurement_phase
                sequencer.append('charge1', m_pulse1)
                sequencer.append('charge2', m_pulse2)
            elif qubit_1_measure == '-Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['1'])
                m_pulse1.phase = -np.pi/2 + measurement_phase
                m_pulse2.phase = -np.pi/2 + measurement_phase
                sequencer.append('charge1', m_pulse1)
                sequencer.append('charge2', m_pulse2)

            sequencer.sync_channels_time(self.channels)

            if qubit_2_measure == 'X':
                m_pulse1 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['2'])
                m_pulse1.phase = 0 + measurement_phase
                m_pulse2.phase = 0 + measurement_phase
                sequencer.append('charge2', m_pulse1)
                sequencer.append('charge1', m_pulse2)
            elif qubit_2_measure == 'Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['2'])
                m_pulse1.phase = np.pi/2 + measurement_phase
                m_pulse2.phase = np.pi/2 + measurement_phase
                sequencer.append('charge2', m_pulse1)
                sequencer.append('charge1', m_pulse2)
            elif qubit_2_measure == '-X':
                m_pulse1 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['2'])
                m_pulse1.phase = -np.pi + measurement_phase
                m_pulse2.phase = -np.pi + measurement_phase
                sequencer.append('charge2', m_pulse1)
                sequencer.append('charge1', m_pulse2)
            elif qubit_2_measure == '-Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2 = copy.copy(self.qubit_ee_half_pi['2'])
                m_pulse1.phase = -np.pi/2 + measurement_phase
                m_pulse2.phase = -np.pi/2 + measurement_phase
                sequencer.append('charge2', m_pulse1)
                sequencer.append('charge1', m_pulse2)

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def tomo_1q_multitone_resonator_corrected(self, sequencer):
        # Author: Ziqian June 24th 2021
        ''' Single qubit tomography in the presence of multitone charge and flux drive, corrected resonator frequency shift'''

        measurement_pulse = ['I', 'X', 'Y']
        if self.expt_cfg['correction']:
            # change readout frequency accordingly
            self.quantum_device_cfg['heterodyne']['1']['lo_freq'] = self.expt_cfg['R1']
            self.quantum_device_cfg['heterodyne']['2']['lo_freq'] = self.expt_cfg['R2']

        for qubit_1_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge1_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge2_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux1_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux2_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']

            for qubit_id in self.expt_cfg['on_qubits']:
                measurement_phase = 0

                if qubit_1_measure == 'X':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Y':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                half_pi_lenth = self.tomo_pulse_info[qubit_id]['half_pi_len']

            if len(self.expt_cfg['on_qubits']) > 1:
                half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'], self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_1_measure != "I":  # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # Np flux drive for g,e calibration pulses
        # qubit g
        sequencer.new_sequence(self)
        # State preparation, with detuned frequency but same amplitude
        # correcting frequencies
        c1_f = []
        c2_f = []
        f1_f = []
        f2_f = []
        for ii in range(len(self.expt_cfg['charge1_freqs_prep'])):
            c1_fp = []
            c2_fp = []
            f1_fp = []
            f2_fp = []
            for jj in range(len(self.expt_cfg['charge1_freqs_prep'][ii])):
                c1_fp.append(self.expt_cfg['charge1_freqs_prep'][ii][jj]+self.expt_cfg['charge1_freq_detune'][ii][jj])
                c2_fp.append(self.expt_cfg['charge2_freqs_prep'][ii][jj] + self.expt_cfg['charge2_freq_detune'][ii][jj])
                f1_fp.append(self.expt_cfg['flux1_freqs_prep'][ii][jj] + self.expt_cfg['flux1_freq_detune'][ii][jj])
                f2_fp.append(self.expt_cfg['flux2_freqs_prep'][ii][jj] + self.expt_cfg['flux2_freq_detune'][ii][jj])

            c1_f.append(c1_fp)
            c2_f.append(c2_fp)
            f1_f.append(f1_fp)
            f2_f.append(f2_fp)
        print(c1_f)

        if self.expt_cfg['correction']:

            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=c1_f,
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=c2_f,
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=f1_f,
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=f2_f,
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)
        if self.expt_cfg['correction']:
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=c1_f,
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=c2_f,
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=f1_f,
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=f2_f,
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

        sequencer.sync_channels_time(self.channels)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def tomo_1q_multitone_charge_flux_drive_gef(self, sequencer):
        # Author: Ziqian 7 Jul 2021
        ''' Single qubit three level tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'Xge', 'Yge', 'Pge', 'Xef', 'Yef', 'PgeXef', 'PgeYef', 'Pgf']

        for qubit_1_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge1_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge2_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux1_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux2_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
                self.tomo_pulse_info['1']['freq_ef'] = self.qubit_ef_freq['1']
                self.tomo_pulse_info['2']['freq_ef'] = self.qubit_ef_freq['2']
            # print(self.tomo_pulse_info)

            for qubit_id in self.expt_cfg['on_qubits']:
                measurement_phase = 0

                if qubit_1_measure == 'Xge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Yge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Pge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Pgf':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)




                elif qubit_1_measure == 'Xef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Yef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'PgeXef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse2)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)


                elif qubit_1_measure == 'PgeYef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse2)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                      phase=-np.pi/2 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)



            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # Np flux drive for g,e calibration pulses
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            # pi_lenth = self.pulse_info[qubit_id]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit f
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
            # pi_lenth = self.pulse_info[qubit_id]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def tomo_1q_multitone_charge_flux_drive(self, sequencer):
        # Author: Tanay 8 Jul 2020
        ''' Single qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']

        for qubit_1_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge1_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge2_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux1_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux2_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']

            for qubit_id in self.expt_cfg['on_qubits']:
                measurement_phase = 0

                if qubit_1_measure == 'X':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Y':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info[qubit_id]['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                half_pi_lenth = self.tomo_pulse_info[qubit_id]['half_pi_len']

            if len(self.expt_cfg['on_qubits']) > 1:
                half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'], self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_1_measure != "I":  # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # Np flux drive for g,e calibration pulses
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_1q_multitone_charge_flux_drive_gate(self, sequencer):
        # Author: Tanay 8 Jul 2020
        ''' Single qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']

        for qubit_1_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge1_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge1_phases_prep']),
                                                                    plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                                                    flat_lens=self.expt_cfg['times_prep'],
                                                                    ramp_sigma_len=
                                                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                        'ramp_sigma_len'],
                                                                    cutoff_sigma=2,
                                                                    freqs=self.expt_cfg['charge2_freqs_prep'],
                                                                    phases=np.pi / 180 * np.array(
                                                                        self.expt_cfg['charge2_phases_prep']),
                                                                    plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux1_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                                                  flat_lens=self.expt_cfg['times_prep'],
                                                                  ramp_sigma_len=
                                                                  self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                      'ramp_sigma_len'],
                                                                  cutoff_sigma=2,
                                                                  freqs=self.expt_cfg['flux2_freqs_prep'],
                                                                  phases=np.pi / 180 * np.array(
                                                                      self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']

            for qubit_id in self.expt_cfg['on_qubits']:
                measurement_phase = 0

                if qubit_1_measure == 'X':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.expt_cfg['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                elif qubit_1_measure == 'Y':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['half_pi_amp'],
                                      flat_len=self.expt_cfg['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2, freq=self.tomo_pulse_info[qubit_id]['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                half_pi_lenth = self.expt_cfg['half_pi_len']

            if len(self.expt_cfg['on_qubits']) > 1:
                half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'], self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_1_measure != "I":  # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # Np flux drive for g,e calibration pulses
        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_1q_multitone_charge_flux_drive_old(self, sequencer):
        # Author: Tanay 9 May 2020
        ''' Single qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']

        for qubit_1_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone(max_amp=self.expt_cfg['charge1_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['charge1_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone(max_amp=self.expt_cfg['charge2_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['charge2_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            for qubit_id in self.expt_cfg['on_qubits']:
                measurement_phase = 0

                if qubit_1_measure == 'X':
                    m_pulse1 = copy.copy(self.qubit_half_pi[qubit_id])
                    m_pulse1.phase = np.pi + measurement_phase
                    sequencer.append('charge%s' % qubit_id, m_pulse1)
                elif qubit_1_measure == 'Y':
                    m_pulse1 = copy.copy(self.qubit_half_pi[qubit_id])
                    m_pulse1.phase = -np.pi / 2 + measurement_phase
                    sequencer.append('charge%s' % qubit_id, m_pulse1)

                half_pi_lenth = self.pulse_info[qubit_id]['half_pi_len']

            if len(self.expt_cfg['on_qubits']) > 1:
                half_pi_lenth = max(self.pulse_info['1']['half_pi_len'],self.pulse_info['2']['half_pi_len'])

            if qubit_1_measure != "I": # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit g
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit e
        sequencer.new_sequence(self)

        for qubit_id in self.expt_cfg['on_qubits']:
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux1_phases_tomo']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux2_phases_tomo']), plot=False))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_old(self, sequencer):
        # Author: Tanay 20 May 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone(max_amp=self.expt_cfg['charge1_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['charge1_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone(max_amp=self.expt_cfg['charge2_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['charge2_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_prep'],
                            flat_len=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_prep'],
                            phase=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse1.phase = np.pi + measurement_phase
                sequencer.append('charge1', m_pulse1)
            elif qubit_1_measure == 'Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse1.phase = -np.pi / 2 + measurement_phase
                sequencer.append('charge1', m_pulse1)

            if qubit_2_measure == 'X':
                m_pulse2 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2.phase = np.pi + measurement_phase
                sequencer.append('charge2', m_pulse2)
            elif qubit_2_measure == 'Y':
                m_pulse2 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2.phase = -np.pi / 2 + measurement_phase
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.pulse_info['1']['half_pi_len'],self.pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        for qubit_id in self.expt_cfg['on_qubits']:
        # for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux1_phases_tomo']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4*self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2,
                        freq=self.qubit_freq["2"]+self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        pi_lenth = self.pulse_info['1']['pi_len'] + self.pulse_info['2']['pi_len'] + \
                   4*self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'] # cutoff_sigma=2

        sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                   flat_len=
                                                   [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                   ramp_sigma_len=
                                                   self.quantum_device_cfg['flux_pulse_info']['1'][
                                                       'ramp_sigma_len'],
                                                   cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                   phase=np.pi / 180 * np.array(
                                                       self.expt_cfg['flux1_phases_tomo']), plot=False))

        sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                   flat_len=
                                                   [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                   ramp_sigma_len=
                                                   self.quantum_device_cfg['flux_pulse_info']['1'][
                                                       'ramp_sigma_len'],
                                                   cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                   phase=np.pi / 180 * np.array(
                                                       self.expt_cfg['flux2_phases_tomo']), plot=False))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_old2(self, sequencer):
        # Author: Tanay 26 Jun 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse1.phase = np.pi + measurement_phase
                sequencer.append('charge1', m_pulse1)
            elif qubit_1_measure == 'Y':
                m_pulse1 = copy.copy(self.qubit_half_pi['1'])
                m_pulse1.phase = -np.pi / 2 + measurement_phase
                sequencer.append('charge1', m_pulse1)

            if qubit_2_measure == 'X':
                m_pulse2 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2.phase = np.pi + measurement_phase
                sequencer.append('charge2', m_pulse2)
            elif qubit_2_measure == 'Y':
                m_pulse2 = copy.copy(self.qubit_half_pi['2'])
                m_pulse2.phase = -np.pi / 2 + measurement_phase
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.pulse_info['1']['half_pi_len'],self.pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        for qubit_id in self.expt_cfg['on_qubits']:
        # for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            pi_lenth = self.pulse_info[qubit_id]['pi_len']

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux1_phases_tomo']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4*self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'], cutoff_sigma=2,
                        freq=self.qubit_freq["2"]+self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        pi_lenth = self.pulse_info['1']['pi_len'] + self.pulse_info['2']['pi_len'] + \
                   4*self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'] # cutoff_sigma=2

        sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                   flat_len=
                                                   [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                   ramp_sigma_len=
                                                   self.quantum_device_cfg['flux_pulse_info']['1'][
                                                       'ramp_sigma_len'],
                                                   cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                   phase=np.pi / 180 * np.array(
                                                       self.expt_cfg['flux1_phases_tomo']), plot=False))

        sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                   flat_len=
                                                   [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                   ramp_sigma_len=
                                                   self.quantum_device_cfg['flux_pulse_info']['1'][
                                                       'ramp_sigma_len'],
                                                   cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                   phase=np.pi / 180 * np.array(
                                                       self.expt_cfg['flux2_phases_tomo']), plot=False))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_old3(self, sequencer):
        # Author: Tanay 4 Jul 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi / 2 + measurement_phase)
                sequencer.append('charge1', m_pulse1)

            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        for qubit_id in self.expt_cfg['on_qubits']:
        # for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            if use_tomo_pulse_info:
                calib_pulse = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=0)
                pi_lenth = self.tomo_pulse_info[qubit_id]['pi_len']
                sequencer.append('charge%s' % qubit_id, calib_pulse)

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [pi_lenth] * len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            else:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        if use_tomo_pulse_info:
            calib_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                 flat_len=self.tomo_pulse_info['1']['pi_len'],
                                 ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                 cutoff_sigma=2,
                                 freq=self.tomo_pulse_info['1']['freq'],
                                 phase=0)
            # pi on qubit1
            sequencer.append('charge1', calib_pulse1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the tomo_pulse_info defined frequency
            calib_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=0)
            sequencer.append('charge2', calib_pulse2)

            pi_lenth_sum = self.tomo_pulse_info['1']['pi_len'] + self.tomo_pulse_info['2']['pi_len'] + \
                       4 * self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len']  # cutoff_sigma=2

            sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth_sum] * len(self.expt_cfg['flux1_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux1_phases_tomo']), plot=False))

            sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                       flat_len=
                                                       [pi_lenth_sum] * len(self.expt_cfg['flux2_amps_tomo']),
                                                       ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                       phase=np.pi / 180 * np.array(
                                                           self.expt_cfg['flux2_phases_tomo']), plot=False))

        else: # use_tomo_pulse_info = False
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_old4(self, sequencer):
        # Author: Tanay 4 Jul 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi / 2 + measurement_phase)
                sequencer.append('charge1', m_pulse1)

            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        for qubit_id in self.expt_cfg['on_qubits']:
        # for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
            'ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',
                         Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                cutoff_sigma=2,
                                freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)



    def tomo_2q_multitone_charge_flux_drive_2021(self, sequencer):
        # Author: Tanay Ziqian 30 Jul 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                self.quantum_device_cfg['flux_pulse_info']['1'][
                                    'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
            'ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',
                         Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                cutoff_sigma=2,
                                freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)



    def edg_xeb(self, sequencer):
        # Author: Ziqian 2021/06/17
        ''' Two-qubit XEB '''

        ## generate sequences of random pulses
        ## 0: X/2, 1: Y/2, 2: X/2Y/2, 3:-X/2Y/2, 4:X/2-Y/2, 5:-X/2-Y/2, 6:Idle,  -1: target two qubit gates
        gate_list_q1 = []
        for ii in range(self.expt_cfg['repeat']):
            gate_list_q1.append(random.randint(0,6))
        # gate_list_q1=[1]
        self.quantum_device_cfg['rb_gate']['gate_list_q1'].append(gate_list_q1)

        gate_list_q2 = []
        for ii in range(self.expt_cfg['repeat']):
            gate_list_q2.append(random.randint(0, 6))
        # gate_list_q2 = [6]
        self.quantum_device_cfg['rb_gate']['gate_list_q2'].append(gate_list_q2)

        print('Q1:',gate_list_q1)
        print('Q2:',gate_list_q2)

        ## Calculate inverse rotation


        ## apply pulse accordingly
        sequencer.new_sequence(self)
        q1_info = self.quantum_device_cfg['rb_gate']
        for ii in range(self.expt_cfg['repeat']):
            #  Single qubit gate first
            gate_name = gate_list_q1[ii]

            if gate_name == 0:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfx']['amp'], flat_len=q1_info['Q1_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfx']['freq'], phase=q1_info['Q1_halfx']['phase']/180*np.pi))
            if gate_name == 1:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfy']['amp'], flat_len=q1_info['Q1_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfy']['freq'], phase=q1_info['Q1_halfy']['phase']/180*np.pi))

            if gate_name == 2:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfx']['amp'], flat_len=q1_info['Q1_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfx']['freq'], phase=q1_info['Q1_halfx']['phase']))
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfy']['amp'], flat_len=q1_info['Q1_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfy']['freq'],
                                        phase=q1_info['Q1_halfy']['phase']/180*np.pi))

            if gate_name == 3:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfx']['amp'], flat_len=q1_info['Q1_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfx']['freq'],
                                        phase=q1_info['Q1_halfx']['phase']/180*np.pi+np.pi))
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfy']['amp'], flat_len=q1_info['Q1_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfy']['freq'],
                                        phase=q1_info['Q1_halfy']['phase']/180*np.pi))

            if gate_name == 4:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfx']['amp'], flat_len=q1_info['Q1_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfx']['freq'],
                                        phase=q1_info['Q1_halfx']['phase'] / 180 * np.pi))
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfy']['amp'], flat_len=q1_info['Q1_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfy']['freq'],
                                        phase=q1_info['Q1_halfy']['phase'] / 180 * np.pi + np.pi))

            if gate_name == 5:
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfx']['amp'], flat_len=q1_info['Q1_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfx']['freq'],
                                        phase=q1_info['Q1_halfx']['phase'] / 180 * np.pi + np.pi))
                sequencer.append('charge1',
                                 Square(max_amp=q1_info['Q1_halfy']['amp'], flat_len=q1_info['Q1_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q1_halfy']['freq'],
                                        phase=q1_info['Q1_halfy']['phase'] / 180 * np.pi + np.pi))

            # for the second qubit
            if self.expt_cfg['sequential_single_gate']:
                sequencer.sync_channels_time(self.channels)

            gate_name = gate_list_q2[ii]
            if gate_name == 0:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfx']['amp'], flat_len=q1_info['Q2_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfx']['freq'],
                                        phase=q1_info['Q2_halfx']['phase'] / 180 * np.pi))
            if gate_name == 1:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfy']['amp'], flat_len=q1_info['Q2_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfy']['freq'],
                                        phase=q1_info['Q2_halfy']['phase'] / 180 * np.pi))

            if gate_name == 2:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfx']['amp'], flat_len=q1_info['Q2_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfx']['freq'],
                                        phase=q1_info['Q2_halfx']['phase']))
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfy']['amp'], flat_len=q1_info['Q2_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfy']['freq'],
                                        phase=q1_info['Q2_halfy']['phase'] / 180 * np.pi))

            if gate_name == 3:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfx']['amp'], flat_len=q1_info['Q2_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfx']['freq'],
                                        phase=q1_info['Q2_halfx']['phase'] / 180 * np.pi + np.pi))
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfy']['amp'], flat_len=q1_info['Q2_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfy']['freq'],
                                        phase=q1_info['Q2_halfy']['phase'] / 180 * np.pi))

            if gate_name == 4:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfx']['amp'], flat_len=q1_info['Q2_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfx']['freq'],
                                        phase=q1_info['Q2_halfx']['phase'] / 180 * np.pi))
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfy']['amp'], flat_len=q1_info['Q2_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfy']['freq'],
                                        phase=q1_info['Q2_halfy']['phase'] / 180 * np.pi + np.pi))

            if gate_name == 5:
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfx']['amp'], flat_len=q1_info['Q2_halfx']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfx']['freq'],
                                        phase=q1_info['Q2_halfx']['phase'] / 180 * np.pi + np.pi))
                sequencer.append('charge2',
                                 Square(max_amp=q1_info['Q2_halfy']['amp'], flat_len=q1_info['Q2_halfy']['len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=q1_info['Q2_halfy']['freq'],
                                        phase=q1_info['Q2_halfy']['phase'] / 180 * np.pi + np.pi))

            # Followed by a two qubit gate
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg['edg_on']:
                edg_info = q1_info['edg']
                tones = max(len(edg_info['freqs'][0]), len(edg_info['freqs'][-1]))
                for index, flux_line_id in enumerate(edg_info['flux_line']):
                    if (edg_info['edg_line'] == int(flux_line_id)):
                        sequencer.append('flux%s' % flux_line_id,
                                         Multitone_EDG(max_amp=np.array(edg_info['amps'][index]),
                                                       flat_len=edg_info['time_length'] * tones, ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=np.array(edg_info['freqs'][index]),
                                                       phase=np.pi / 180 * (np.array(edg_info['phases'][index])),
                                                       shapes=[edg_info['shape']], nos=[edg_info['edg_no']],
                                                       repeat=1, plot=False))
                    else:
                        sequencer.append('flux%s' % flux_line_id,
                                         Square_multitone(max_amp=np.array(edg_info['amps'][index]),
                                                          flat_len=edg_info['time_length'] * tones, ramp_sigma_len=
                                                          self.quantum_device_cfg['flux_pulse_info']['1'][
                                                              'ramp_sigma_len'],
                                                          cutoff_sigma=2, freq=np.array(edg_info['freqs'][index]),
                                                          phase=np.pi / 180 * (np.array(edg_info['phases'][index])),
                                                          plot=False))
            else:
                nonedg_info = q1_info['non_edg']
                # State preparation
                sequencer.append('charge1', Square_multitone_sequential(max_amps=nonedg_info['charge1_amps_prep'],
                                                                        flat_lens=nonedg_info['times_prep'],
                                                                        ramp_sigma_len=
                                                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                            'ramp_sigma_len'],
                                                                        cutoff_sigma=2,
                                                                        freqs=nonedg_info['charge1_freqs_prep'],
                                                                        phases=np.pi / 180 * np.array(
                                                                            nonedg_info['charge1_phases_prep']),
                                                                        plot=False))

                sequencer.append('charge2', Square_multitone_sequential(max_amps=nonedg_info['charge2_amps_prep'],
                                                                        flat_lens=nonedg_info['times_prep'],
                                                                        ramp_sigma_len=
                                                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                            'ramp_sigma_len'],
                                                                        cutoff_sigma=2,
                                                                        freqs=nonedg_info['charge2_freqs_prep'],
                                                                        phases=np.pi / 180 * np.array(
                                                                            nonedg_info['charge2_phases_prep']),
                                                                        plot=False))

                sequencer.append('flux1', Square_multitone_sequential(max_amps=nonedg_info['flux1_amps_prep'],
                                                                      flat_lens=nonedg_info['times_prep'],
                                                                      ramp_sigma_len=
                                                                      self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                          'ramp_sigma_len'],
                                                                      cutoff_sigma=2,
                                                                      freqs=nonedg_info['flux1_freqs_prep'],
                                                                      phases=np.pi / 180 * np.array(
                                                                          nonedg_info['flux1_phases_prep']),
                                                                      plot=False))

                sequencer.append('flux2', Square_multitone_sequential(max_amps=nonedg_info['flux2_amps_prep'],
                                                                      flat_lens=nonedg_info['times_prep'],
                                                                      ramp_sigma_len=
                                                                      self.quantum_device_cfg['flux_pulse_info']['1'][
                                                                          'ramp_sigma_len'],
                                                                      cutoff_sigma=2,
                                                                      freqs=nonedg_info['flux2_freqs_prep'],
                                                                      phases=np.pi / 180 * np.array(
                                                                          nonedg_info['flux2_phases_prep']),
                                                                      plot=False))

            sequencer.sync_channels_time(self.channels)



        ## Readout of the RB sequence:
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # qubit ee for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                              flat_len=self.tomo_pulse_info['1']['pi_len'],
                              ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                              cutoff_sigma=2,
                              freq=self.tomo_pulse_info['1']['freq'],
                              phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_gef(self, sequencer):
        # Author: Ziqian 08 Jul 2021
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'Xge', 'Yge', 'Pge', 'Xef', 'Yef', 'PgeXef', 'PgeYef', 'Pgf']
        if self.expt_cfg['default']:
            print('Preparing state: ', self.expt_cfg['default_state'])
        else:
            print('Default preparation not being used')

        for qubit_1_measure in measurement_pulse:

            for qubit_2_measure in measurement_pulse:

                sequencer.new_sequence(self)

                # State preparation
                if self.expt_cfg['default']:
                    if self.expt_cfg['default_state']=='gg':
                        # print('Preparing state: |gg>')
                        pass
                    if self.expt_cfg['default_state'] == 'ge':
                        # print('Preparing state: |ge>')
                        sequencer.append('charge2', self.qubit_pi['2'])
                    if self.expt_cfg['default_state'] == 'eg':
                        # print('Preparing state: |eg>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                    if self.expt_cfg['default_state'] == 'fg':
                        # print('Preparing state: |fg>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                        sequencer.append('charge2', self.qubit_ef_pi['1'])
                    if self.expt_cfg['default_state'] == 'gf':
                        # print('Preparing state: |gf>')
                        sequencer.append('charge1', self.qubit_pi['2'])
                        sequencer.append('charge2', self.qubit_ef_pi['2'])
                    if self.expt_cfg['default_state'] == 'ee':
                        # print('Preparing state: |ee>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ee_pi['2'])
                    if self.expt_cfg['default_state'] == 'ef':
                        # print('Preparing state: |ef>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ee_pi['2'])
                        sequencer.append('charge2', self.qubit_ef_e_pi['2'])
                    if self.expt_cfg['default_state'] == 'fe':
                        # print('Preparing state: |fe>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ee_pi['2'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ef_e_pi['1'])
                    if self.expt_cfg['default_state'] == 'ff':
                        # print('Preparing state: |ff>')
                        sequencer.append('charge1', self.qubit_pi['1'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ee_pi['2'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ef_e_pi['1'])
                        sequencer.sync_channels_time(self.channels)
                        sequencer.append('charge2', self.qubit_ef_e_pi['2'])
                    if self.expt_cfg['default_state']=='g(g+e)':
                        # print('Preparing state: |g(g+e)>')
                        sequencer.append('charge2', self.qubit_half_pi['2'])
                    if self.expt_cfg['default_state']=='(g+e)g':
                        # print('Preparing state: |(g+e)g>')
                        sequencer.append('charge1', self.qubit_half_pi['1'])
                    if self.expt_cfg['default_state']=='g(g-f)':
                        # print('Preparing state: |g(g-f)>')
                        sequencer.append('charge2', self.qubit_half_pi['2'])
                        sequencer.append('charge2', self.qubit_ef_pi['2'])
                    if self.expt_cfg['default_state']=='(g-f)g':
                        # print('Preparing state: |(g-f)g>')
                        sequencer.append('charge1', self.qubit_half_pi['1'])
                        sequencer.append('charge1', self.qubit_ef_pi['1'])
                else:
                    # print('Default preparation not being used')
                    sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                                    flat_lens=self.expt_cfg['times_prep'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                                    phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

                    sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                                    flat_lens=self.expt_cfg['times_prep'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                                    phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

                    sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                                    flat_lens=self.expt_cfg['times_prep'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                                    phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

                    sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                                    flat_lens=self.expt_cfg['times_prep'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                    cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                                    phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

                sequencer.sync_channels_time(self.channels)

                # Tomographic pulses

                apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

                use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
                if use_tomo_pulse_info:
                    self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
                else:
                    self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                    self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                    self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
                    self.tomo_pulse_info['1']['freq_ef'] = self.qubit_ef_freq['1']
                    self.tomo_pulse_info['2']['freq_ef'] = self.qubit_ef_freq['2']
                measurement_phase = 0

                if qubit_1_measure == 'Xge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'Yge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'Pge':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'Xef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_ef_len'] + 4 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'Yef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_ef_len'] + 4 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'PgeXef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)

                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_ef_len'] + self.tomo_pulse_info['1']['pi_len'] +8 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'PgeYef':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['1']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge1', m_pulse1)

                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['half_pi_ef_len'] + self.tomo_pulse_info['1']['pi_len'] +8 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential

                elif qubit_1_measure == 'Pgf':
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['1']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge1', m_pulse1)
                    m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['1']['pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['1']['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge1', m_pulse1)

                    if apply_sequential_tomo_pulse:
                        idle_time = self.tomo_pulse_info['1']['pi_ef_len'] + self.tomo_pulse_info['1']['pi_len'] +8 * \
                                        self.quantum_device_cfg['flux_pulse_info']['1'][
                                            'ramp_sigma_len']
                        sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


                if qubit_2_measure == 'Xge':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'Yge':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'Pge':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'Xef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'Yef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'PgeXef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'PgeYef':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['2']['half_pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq_ef'],
                                      phase=-np.pi/2*1.0 + measurement_phase)
                    sequencer.append('charge2', m_pulse2)

                elif qubit_2_measure == 'Pgf':
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_amp'],
                                      flat_len=self.tomo_pulse_info['2']['pi_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)
                    m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['pi_ef_amp'],
                                      flat_len=self.tomo_pulse_info['2']['pi_ef_len'],
                                      ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                      cutoff_sigma=2,
                                      freq=self.tomo_pulse_info['2']['freq_ef'],
                                      phase=np.pi + measurement_phase)
                    sequencer.append('charge2', m_pulse2)


                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # sequence: [gg, ge, eg, gf, fg, ee, ef, fe, ff]
        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # gf and fg
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_ef_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_ef_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq_ef'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # qubit ee for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                               flat_len=self.tomo_pulse_info['1']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['1']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * \
                        self.quantum_device_cfg['flux_pulse_info']['1'][
                            'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'],
                                    flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'],
                                    phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ef and fe for calibration

        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', h_pulsee1)
                # Idle time for qubit 2
                idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                    'ramp_sigma_len']
                sequencer.append('charge2', Idle(idle_time))
                # pi on qubit2 at the shifted frequency
                h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                                   flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                                   ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                   cutoff_sigma=2,
                                   freq=self.tomo_pulse_info['zz2']['freq'],
                                   phase=np.pi + measurement_phase)
                sequencer.append('charge2', h_pulsee2)
                # Idle time for qubit 1
                idle_time = self.tomo_pulse_info['zz2']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                    'ramp_sigma_len']
                sequencer.append('charge1', Idle(idle_time))
                # ef pi on qubit_id with shifted frequency
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_e_pi[qubit_id])
                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                # pi on qubit1
                sequencer.append('charge1', self.qubit_pi['1'])
                # Idle time for qubit 2
                idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                    'ramp_sigma_len']
                sequencer.append('charge2', Idle(idle_time))
                # pi on qubit2 with at the shifted frequency
                sequencer.append('charge2',
                                 Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                        cutoff_sigma=2,
                                        freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))
                # Idle time for qubit 1
                idle_time = self.pulse_info['2']['pi_ee_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                    'ramp_sigma_len']
                sequencer.append('charge1', Idle(idle_time))
                # ef pi on qubit_id with shifted frequency
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_e_pi[qubit_id])
                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()



        # qubit ff for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                               flat_len=self.tomo_pulse_info['1']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['1']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)
            # Idle time for qubit 1
            idle_time = self.tomo_pulse_info['zz2']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge1', Idle(idle_time))
            # ef pi on qubit_id with shifted frequency
            sequencer.append('charge1', self.qubit_ef_e_pi['1'])
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_ef_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            sequencer.append('charge2', self.qubit_ff_pi['2'])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'],
                                    flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2'][
                                        'ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'],
                                    phase=0))
            # Idle time for qubit 1
            idle_time = self.pulse_info['2']['pi_ee_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge1', Idle(idle_time))
            # ef pi on qubit_id with shifted frequency
            sequencer.append('charge1', self.qubit_ef_e_pi['1'])
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_ef_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            sequencer.append('charge2', self.qubit_ff_pi['2'])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)



    def tomo_2q_multitone_charge_flux_drive(self, sequencer):
        # Author: Tanay Ziqian 30 Jul 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                self.quantum_device_cfg['flux_pulse_info']['1'][
                                    'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # qubit ee for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                              flat_len=self.tomo_pulse_info['1']['pi_len'],
                              ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                              cutoff_sigma=2,
                              freq=self.tomo_pulse_info['1']['freq'],
                              phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_tomo_angle(self, sequencer):
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0
            rel_y_angle = self.expt_cfg['rel_y_angle'] # in degree

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                self.quantum_device_cfg['flux_pulse_info']['1'][
                                    'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi/2 + np.pi/180*rel_y_angle + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2 + np.pi/180*rel_y_angle +measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # qubit ee for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                              flat_len=self.tomo_pulse_info['1']['pi_len'],
                              ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                              cutoff_sigma=2,
                              freq=self.tomo_pulse_info['1']['freq'],
                              phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_for_edg(self, sequencer):
        # Author: Ziqian 30 Jul 2021
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Perform Error divisible gate
            # in each braket, the amplitude, frequency, phases determines a simultaneous multitone send through flux line 1 or 2
            # pulse shape: ['cos\tanh', alpha, f, A, tg, Gamma (if needed)]
            tones = max(len(self.expt_cfg['freqs'][0]), len(self.expt_cfg['freqs'][-1]))
            freqs = self.expt_cfg['freqs']
            time_length = self.expt_cfg['time_length']
            amps = (self.expt_cfg['amps'])
            phases = (self.expt_cfg['phases'])  # phase should be in degrees
            shape = self.expt_cfg['shape']
            repeat_time = self.expt_cfg['repeat']
            if self.expt_cfg['edg_on']:
                time_length[int(self.expt_cfg['edg_line']) - 1][self.expt_cfg['edg_no']] = shape[
                    4]  # replace the EDG tone length with the EDG shape variable 'tg'

            # Modified by Ziqian
            # For flat-top pulses, has Gaussian ramp for other tones, analytic expression for EDG part
            # Sweeping the iteration time, perform readout at the end



            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                if self.expt_cfg['edg_on']:
                    if (self.expt_cfg['edg_line'] == int(flux_line_id)):
                        sequencer.append('flux%s' % flux_line_id,
                                         Multitone_EDG(max_amp=np.array(amps[index]),
                                                       flat_len=time_length * tones, ramp_sigma_len=
                                                       self.quantum_device_cfg['flux_pulse_info']['1'][
                                                           'ramp_sigma_len'],
                                                       cutoff_sigma=2, freq=np.array(freqs[index]),
                                                       phase=np.pi / 180 * (np.array(phases[index])),
                                                       shapes=[shape], nos=[self.expt_cfg['edg_no']],
                                                       repeat=repeat_time, plot=False))
                    else:
                        sequencer.append('flux%s' % flux_line_id,
                                         Square_multitone(max_amp=np.array(amps[index]),
                                                          flat_len=time_length * tones, ramp_sigma_len=
                                                          self.quantum_device_cfg['flux_pulse_info']['1'][
                                                              'ramp_sigma_len'],
                                                          cutoff_sigma=2, freq=np.array(freqs[index]),
                                                          phase=np.pi / 180 * (np.array(phases[index])),
                                                          plot=False))
                else:
                    for kkk in range(repeat_time):
                        sequencer.append('flux%s' % flux_line_id,
                                         Square_multitone(max_amp=np.array(amps[index]),
                                                          flat_len=time_length * tones, ramp_sigma_len=
                                                          self.quantum_device_cfg['flux_pulse_info']['1'][
                                                              'ramp_sigma_len'],
                                                          cutoff_sigma=2, freq=np.array(freqs[index]),
                                                          phase=np.pi / 180 * (np.array(phases[index])),
                                                          plot=False))

            sequencer.sync_channels_time(self.channels)


            #  inverse to the initial state

            if self.expt_cfg['inverse_rotation']:
                # Inverse pulse after EDG
                inverse_rotation_info = self.quantum_device_cfg['inverse_rotation']
                replace_no = inverse_rotation_info['replace_no']
                inverse_rotation_info['times_prep'][replace_no[0]][replace_no[1]] = \
                inverse_rotation_info['inverse_tlist'][(repeat_time-1) % len(inverse_rotation_info['inverse_tlist'])]

                if inverse_rotation_info['times_prep'][replace_no[0]][replace_no[1]] > 0.0:

                    sequencer.append('charge1',
                                     Square_multitone_sequential(
                                         max_amps=inverse_rotation_info['charge1_amps_prep'],
                                         flat_lens=inverse_rotation_info['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=inverse_rotation_info['charge1_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             inverse_rotation_info['charge1_phases_prep']),
                                         plot=False))

                    sequencer.append('charge2',
                                     Square_multitone_sequential(
                                         max_amps=inverse_rotation_info['charge2_amps_prep'],
                                         flat_lens=inverse_rotation_info['times_prep'],
                                         ramp_sigma_len=
                                         self.quantum_device_cfg['flux_pulse_info'][
                                             '1'][
                                             'ramp_sigma_len'],
                                         cutoff_sigma=2,
                                         freqs=inverse_rotation_info['charge2_freqs_prep'],
                                         phases=np.pi / 180 * np.array(
                                             inverse_rotation_info['charge2_phases_prep']),
                                         plot=False))

                    sequencer.append('flux1',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['flux1_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['flux1_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['flux1_phases_prep']),
                                                                 plot=False))

                    sequencer.append('flux2',
                                     Square_multitone_sequential(max_amps=inverse_rotation_info['flux2_amps_prep'],
                                                                 flat_lens=inverse_rotation_info['times_prep'],
                                                                 ramp_sigma_len=
                                                                 self.quantum_device_cfg['flux_pulse_info'][
                                                                     '1'][
                                                                     'ramp_sigma_len'],
                                                                 cutoff_sigma=2,
                                                                 freqs=inverse_rotation_info['flux2_freqs_prep'],
                                                                 phases=np.pi / 180 * np.array(
                                                                     inverse_rotation_info['flux2_phases_prep']),
                                                                 plot=False))


                sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                self.quantum_device_cfg['flux_pulse_info']['1'][
                                    'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['1']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = self.tomo_pulse_info['1']['half_pi_len'] + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=self.tomo_pulse_info['2']['half_pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(self.tomo_pulse_info['1']['half_pi_len'],self.tomo_pulse_info['2']['half_pi_len'])

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        if use_tomo_pulse_info:
            for qubit_id in ['2', '1']:
                sequencer.new_sequence(self)
                h_pulse1 = Square(max_amp=self.tomo_pulse_info[qubit_id]['pi_amp'],
                                  flat_len=self.tomo_pulse_info[qubit_id]['pi_len'],
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info[qubit_id]['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge%s' % qubit_id, h_pulse1)

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()
        else:
            for qubit_id in ['2','1']:
                sequencer.new_sequence(self)
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

                self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
                sequencer.end_sequence()

        # qubit ee for calibration

        if use_tomo_pulse_info:
            sequencer.new_sequence(self)
            h_pulsee1 = Square(max_amp=self.tomo_pulse_info['1']['pi_amp'],
                              flat_len=self.tomo_pulse_info['1']['pi_len'],
                              ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                              cutoff_sigma=2,
                              freq=self.tomo_pulse_info['1']['freq'],
                              phase=np.pi + measurement_phase)
            sequencer.append('charge1', h_pulsee1)
            # Idle time for qubit 2
            idle_time = self.tomo_pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            h_pulsee2 = Square(max_amp=self.tomo_pulse_info['zz2']['pi_amp'],
                               flat_len=self.tomo_pulse_info['zz2']['pi_len'],
                               ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                               cutoff_sigma=2,
                               freq=self.tomo_pulse_info['zz2']['freq'],
                               phase=np.pi + measurement_phase)
            sequencer.append('charge2', h_pulsee2)

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()
        else:
            sequencer.new_sequence(self)
            # pi on qubit1
            sequencer.append('charge1', self.qubit_pi['1'])
            # Idle time for qubit 2
            idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
                'ramp_sigma_len']
            sequencer.append('charge2', Idle(idle_time))
            # pi on qubit2 with at the shifted frequency
            sequencer.append('charge2',
                             Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                    ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                    cutoff_sigma=2,
                                    freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def tomo_2q_multitone_charge_flux_drive_gate(self, sequencer):
        # Author: Tanay Ziqian 30 Jul 2020
        ''' Two-qubit tomography in the presence of multitone charge and flux drive'''

        measurement_pulse = ['I', 'X', 'Y']
        measurement_pulse = [['I', 'I'], ['I', 'X'], ['I', 'Y'], ['X', 'I'], ['X', 'X'], ['X', 'Y'], ['Y', 'I'],
                             ['Y', 'X'], ['Y', 'Y']]

        if self.expt_cfg['tomo_qubit'] == 1:
            gate2 = self.quantum_device_cfg["pulse_info"]["2"]["half_pi_len"]
            gate1 = self.expt_cfg["tomo_gate"]
        else:
            gate1 = self.quantum_device_cfg["pulse_info"]["1"]["half_pi_len"]
            gate2 = self.expt_cfg["tomo_gate"]

        for qubit_measure in measurement_pulse:

            sequencer.new_sequence(self)

            # State preparation
            sequencer.append('charge1', Square_multitone_sequential(max_amps=self.expt_cfg['charge1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge1_phases_prep']), plot=False))

            sequencer.append('charge2', Square_multitone_sequential(max_amps=self.expt_cfg['charge2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['charge2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['charge2_phases_prep']), plot=False))

            sequencer.append('flux1', Square_multitone_sequential(max_amps=self.expt_cfg['flux1_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux1_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux1_phases_prep']), plot=False))

            sequencer.append('flux2', Square_multitone_sequential(max_amps=self.expt_cfg['flux2_amps_prep'],
                            flat_lens=self.expt_cfg['times_prep'],
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freqs=self.expt_cfg['flux2_freqs_prep'],
                            phases=np.pi/180*np.array(self.expt_cfg['flux2_phases_prep']), plot=False))

            sequencer.sync_channels_time(self.channels)

            # Tomographic pulses

            apply_sequential_tomo_pulse = self.expt_cfg['sequential_tomo_pulse']

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            use_tomo_pulse_info = self.expt_cfg['use_tomo_pulse_info']
            if use_tomo_pulse_info:
                self.tomo_pulse_info = self.quantum_device_cfg['tomo_pulse_info']
            else:
                self.tomo_pulse_info = self.quantum_device_cfg['pulse_info']
                self.tomo_pulse_info['1']['freq'] = self.qubit_freq['1']
                self.tomo_pulse_info['2']['freq'] = self.qubit_freq['2']
            measurement_phase = 0

            if qubit_1_measure == 'X':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=gate1,
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = gate1 + 4 * \
                                self.quantum_device_cfg['flux_pulse_info']['1'][
                                    'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time))  # Make q1 and q2 pulses sequential

            elif qubit_1_measure == 'Y':
                m_pulse1 = Square(max_amp=self.tomo_pulse_info['1']['half_pi_amp'],
                                  flat_len=gate1,
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['1']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge1', m_pulse1)
                if apply_sequential_tomo_pulse:
                    idle_time = gate1 + 4 * \
                                    self.quantum_device_cfg['flux_pulse_info']['1'][
                                        'ramp_sigma_len']
                    sequencer.append('charge2', Idle(idle_time)) # Make q1 and q2 pulses sequential


            if qubit_2_measure == 'X':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=gate2,
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=np.pi + measurement_phase)
                sequencer.append('charge2', m_pulse2)

            elif qubit_2_measure == 'Y':
                m_pulse2 = Square(max_amp=self.tomo_pulse_info['2']['half_pi_amp'],
                                  flat_len=gate2,
                                  ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                  cutoff_sigma=2,
                                  freq=self.tomo_pulse_info['2']['freq'],
                                  phase=-np.pi/2*1.0 + measurement_phase)
                sequencer.append('charge2', m_pulse2)


            half_pi_lenth = max(gate1,gate2)

            if qubit_measure != ["I","I"]: # Flux pulse is not needed for Z measurement

                sequencer.append('flux1', Square_multitone(max_amp=self.expt_cfg['flux1_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth]*len(self.expt_cfg['flux1_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux1_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux1_phases_tomo']), plot=False))

                sequencer.append('flux2', Square_multitone(max_amp=self.expt_cfg['flux2_amps_tomo'],
                                                           flat_len=
                                                           [half_pi_lenth] * len(self.expt_cfg['flux2_amps_tomo']),
                                                           ramp_sigma_len=
                                                           self.quantum_device_cfg['flux_pulse_info']['1'][
                                                               'ramp_sigma_len'],
                                                           cutoff_sigma=2, freq=self.expt_cfg['flux2_freqs_tomo'],
                                                           phase=np.pi / 180 * np.array(
                                                               self.expt_cfg['flux2_phases_tomo']), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit gg for calibration
        sequencer.new_sequence(self)
        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        # qubits ge and eg for calibration

        # for qubit_id in self.expt_cfg['on_qubits']:
        for qubit_id in ['2','1']:
            sequencer.new_sequence(self)
            sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
            sequencer.end_sequence()

        # qubit ee for calibration

        sequencer.new_sequence(self)
        # pi on qubit1
        sequencer.append('charge1', self.qubit_pi['1'])
        # Idle time for qubit 2
        idle_time = self.pulse_info['1']['pi_len'] + 4 * self.quantum_device_cfg['flux_pulse_info']['1'][
            'ramp_sigma_len']
        sequencer.append('charge2', Idle(idle_time))
        # pi on qubit2 with at the shifted frequency
        sequencer.append('charge2',
                         Square(max_amp=self.pulse_info['2']['pi_ee_amp'], flat_len=self.pulse_info['2']['pi_ee_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                                cutoff_sigma=2,
                                freq=self.qubit_freq["2"] + self.quantum_device_cfg['qubit']['qq_disp'], phase=0))

        self.readout(sequencer, self.expt_cfg['on_qubits'], sideband=False)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def half_pi_phase(self, sequencer):
        # ramsey sequences

        for phase in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                half_pi = copy.copy(self.qubit_half_pi[qubit_id])
                half_pi.phase = phase
                sequencer.append('charge%s' % qubit_id, half_pi)
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def echo(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                for echo_id in range(self.expt_cfg['echo_times']):
                    # Idle time before pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        # sequencer.append('charge%s' % qubit_id,
                        #          Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'],
                        #                sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                        #                freq=self.qubit_freq[qubit_id], phase=0.5*np.pi, plot=False))
                        ## Case: Square top pulse
                        cpmg_pulse = copy.copy(self.qubit_pi[qubit_id])
                        cpmg_pulse.phase = 0.5*np.pi
                        sequencer.append('charge%s' % qubit_id, cpmg_pulse)
                    # Idle time after pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                # sequencer.append('charge%s' % qubit_id,
                #                  Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                #                        sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                #                        freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                sequencer.append('charge%s' % qubit_id, ramsey_2nd_pulse)
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def echo2_backup(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                    sequencer.sync_channels_time(self.channels)
                for echo_id in range(self.expt_cfg['echo_times']):
                    # Idle time before pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        # sequencer.append('charge%s' % qubit_id,
                        #          Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'],
                        #                sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                        #                freq=self.qubit_freq[qubit_id], phase=0.5*np.pi, plot=False))
                        ## Case: Square top pulse
                        cpmg_pulse = copy.copy(self.qubit_pi[qubit_id])
                        cpmg_pulse.phase = 0.5*np.pi
                        sequencer.append('charge%s' % qubit_id, cpmg_pulse)
                    # Idle time after pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                # sequencer.append('charge%s' % qubit_id,
                #                  Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                #                        sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                #                        freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                sequencer.append('charge%s' % qubit_id, ramsey_2nd_pulse)
            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                # Don't use two qubit charge drives together
                sequencer.append('flux%s' % flux_line_id,
                                 Square(max_amp=self.expt_cfg['flux_amp'][index],
                                        flat_len=2 * self.pulse_info[qubit_id]['half_pi_len'] + ramsey_len + 2 *
                                                 self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                     'ramp_sigma_len'] * 2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                        phase=np.pi / 180 * self.expt_cfg['phase'][index], plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def echo2(self, sequencer):
        # ramsey sequences
        if self.expt_cfg['use_freq_amp_halfpi'][0]:
            charge_port = self.expt_cfg['use_freq_amp_halfpi'][-1]
            freq = self.expt_cfg['use_freq_amp_halfpi'][1]
            amp = self.expt_cfg['use_freq_amp_halfpi'][2]
            half_pi_len = self.expt_cfg['use_freq_amp_halfpi'][3]
            self.qubit_half_pi = {
                "1": Square(max_amp=amp, flat_len=half_pi_len,
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['1']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=freq, phase=0),
                "2": Square(max_amp=amp, flat_len=half_pi_len,
                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info']['2']['ramp_sigma_len'],
                            cutoff_sigma=2, freq=freq, phase=0)}

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                length_flux = 0
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                for ge_pi_id in self.expt_cfg['ge_pi']:

                    if self.expt_cfg['flux_probe']:
                        sequencer.append('flux1', self.qubit_pi[ge_pi_id])
                    else:
                        sequencer.append('charge%s' % self.charge_port[ge_pi_id], self.qubit_pi[ge_pi_id])

                sequencer.sync_channels_time(self.channels)
                for echo_id in range(self.expt_cfg['echo_times']):
                    # Idle time before pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    length_flux +=ramsey_len/(float(2*self.expt_cfg['echo_times']))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                        length_flux += self.qubit_pi[qubit_id].get_length()
                    elif self.expt_cfg['cpmg']:
                        # sequencer.append('charge%s' % qubit_id,
                        #          Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'],
                        #                sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                        #                freq=self.qubit_freq[qubit_id], phase=0.5*np.pi, plot=False))
                        ## Case: Square top pulse
                        cpmg_pulse = copy.copy(self.qubit_pi[qubit_id])
                        cpmg_pulse.phase = 0.5*np.pi
                        sequencer.append('charge%s' % qubit_id, cpmg_pulse)
                        length_flux += cpmg_pulse.get_length()
                    # Idle time after pi pulse
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    length_flux += ramsey_len / (float(2 * self.expt_cfg['echo_times']))
                # sequencer.append('charge%s' % qubit_id,
                #                  Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                #                        sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                #                        freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                ramsey_2nd_pulse = copy.copy(self.qubit_half_pi[qubit_id])
                ramsey_2nd_pulse.phase = 2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq']
                sequencer.append('charge%s' % qubit_id, ramsey_2nd_pulse)
            for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                # Don't use two qubit charge drives together
                sequencer.append('flux%s' % flux_line_id,
                                 Square(max_amp=self.expt_cfg['flux_amp'][index],
                                        flat_len=length_flux - 2 *
                                                 self.quantum_device_cfg['flux_pulse_info']['1'][
                                                     'ramp_sigma_len'] * 2,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['flux_freq'],
                                        phase=np.pi / 180 * self.expt_cfg['phase'][index], plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def communication_rabi(self, sequencer):
        # mm rabi sequences


        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if qubit_id in self.expt_cfg['pi_pulse']:
                    sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])

                if "freq_a" in self.expt_cfg["use_fit"]:

                    with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%qubit_id), 'rb') as f:
                        freq_a_p = pickle.load(f)

                    # freq_a_p = np.poly1d(np.load(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s.npy'%qubit_id)))
                    freq = freq_a_p(self.communication[qubit_id]['pi_amp'])
                else:
                    freq = self.communication[qubit_id]['freq']

                # if False:
                #     sequencer.append('flux%s'%qubit_id,
                #                  Square(max_amp=self.communication[qubit_id]['pi_amp'], flat_len=rabi_len,
                #                         ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2,
                #                         freq=freq, phase=0,
                #                         plot=False))
                # elif True:
                flux_pulse = self.communication_flux_pi[qubit_id]
                flux_pulse.len = rabi_len
                sequencer.append('flux%s'%qubit_id,flux_pulse)


            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def photon_transfer(self, sequencer, **kwargs):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            # if "freq_a" in self.expt_cfg["use_fit"]:
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%sender_id), 'rb') as f:
            #         freq_a_p_send = pickle.load(f)
            #
            #     freq_send = freq_a_p_send(self.communication[sender_id]['pi_amp'])
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%receiver_id), 'rb') as f:
            #         freq_a_p_rece = pickle.load(f)
            #
            #     freq_rece = freq_a_p_rece(self.communication[receiver_id]['pi_amp'])
            # else:
            #     freq_send = self.communication[sender_id]['freq']
            #     freq_rece = self.communication[receiver_id]['freq']

            if self.expt_cfg['rece_delay'] < 0:
                sequencer.append('flux%s'%sender_id,
                                 Idle(time=abs(self.expt_cfg['rece_delay'])))

            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = rabi_len
            # flux_pulse.delta_freq = 0.001
            if 'send_A_list' in kwargs:
                flux_pulse.A_list = kwargs['send_A_list']
            sequencer.append('flux%s'%sender_id,flux_pulse)

            if self.expt_cfg['rece_delay'] > 0:
                sequencer.append('flux%s'%receiver_id,
                                 Idle(time=self.expt_cfg['rece_delay']))

            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = rabi_len
            # flux_pulse.plot = True
            if 'rece_A_list' in kwargs:
                flux_pulse.A_list = kwargs['rece_A_list']
            sequencer.append('flux%s'%receiver_id,flux_pulse)


            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def photon_transfer_delay(self, sequencer, **kwargs):
        # mm rabi sequences

        for delay in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            rabi_len = 400
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            # if "freq_a" in self.expt_cfg["use_fit"]:
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%sender_id), 'rb') as f:
            #         freq_a_p_send = pickle.load(f)
            #
            #     freq_send = freq_a_p_send(self.communication[sender_id]['pi_amp'])
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%receiver_id), 'rb') as f:
            #         freq_a_p_rece = pickle.load(f)
            #
            #     freq_rece = freq_a_p_rece(self.communication[receiver_id]['pi_amp'])
            # else:
            #     freq_send = self.communication[sender_id]['freq']
            #     freq_rece = self.communication[receiver_id]['freq']

            if delay < 0:
                sequencer.append('flux%s'%sender_id,
                                 Idle(time=abs(delay)))

            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = rabi_len
            # flux_pulse.delta_freq = 0.001
            if 'send_A_list' in kwargs:
                flux_pulse.A_list = kwargs['send_A_list']
            sequencer.append('flux%s'%sender_id,flux_pulse)

            if delay > 0:
                sequencer.append('flux%s'%receiver_id,
                                 Idle(time=delay))

            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = rabi_len
            # flux_pulse.plot = True
            if 'rece_A_list' in kwargs:
                flux_pulse.A_list = kwargs['rece_A_list']
            sequencer.append('flux%s'%receiver_id,flux_pulse)


            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def heralding_only_ge_test(self, sequencer, **kwargs):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])



            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = rabi_len
            sequencer.append('flux%s'%sender_id,flux_pulse)


            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%receiver_id,flux_pulse)

            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])

            sequencer.sync_channels_time(['charge%s' % receiver_id, 'flux%s' % receiver_id])
            sequencer.append('charge%s' % receiver_id, self.qubit_ef_pi[receiver_id])

            sequencer.sync_channels_time(self.channels)

            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%sender_id,flux_pulse)


            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%receiver_id,flux_pulse)

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def heralding_only_ge_tomography(self, sequencer, **kwargs):
        # mm rabi sequences
        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]

        for qubit_measure in measurement_pulse:
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])



            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = 70
            sequencer.append('flux%s'%sender_id,flux_pulse)


            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%receiver_id,flux_pulse)

            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])

            sequencer.sync_channels_time(['charge%s' % receiver_id, 'flux%s' % receiver_id])
            sequencer.append('charge%s' % receiver_id, self.qubit_ef_pi[receiver_id])

            sequencer.sync_channels_time(self.channels)

            flux_pulse = self.communication_flux_pi[sender_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%sender_id,flux_pulse)


            flux_pulse = self.communication_flux_pi[receiver_id]
            flux_pulse.len = 180
            sequencer.append('flux%s'%receiver_id,flux_pulse)


            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])

            sequencer.sync_channels_time(self.channels)

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            if qubit_1_measure == 'X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == 'Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)

            if qubit_2_measure == 'X':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == 'Y':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-X':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-Y':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_send_ge_rece_photon_transfer(self, sequencer, **kwargs):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            # if "freq_a" in self.expt_cfg["use_fit"]:
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_ef_100kHz.pkl'%sender_id), 'rb') as f:
            #         freq_a_p_send = pickle.load(f)
            #
            #     freq_send = freq_a_p_send(self.communication[sender_id]['ef_pi_amp'])
            #
            #     with open(os.path.join(self.quantum_device_cfg['fit_path'],'comm_sideband/%s_100kHz.pkl'%receiver_id), 'rb') as f:
            #         freq_a_p_rece = pickle.load(f)
            #
            #     freq_rece = freq_a_p_rece(self.communication[receiver_id]['pi_amp'])
            # else:
            #     freq_send = self.communication[sender_id]['freq']
            #     freq_rece = self.communication[receiver_id]['freq']

            if self.expt_cfg['rece_delay'] < 0:
                sequencer.append('flux%s'%sender_id,
                                 Idle(time=abs(self.expt_cfg['rece_delay'])))

            flux_pulse = copy.copy(self.communication_flux_ef_pi[sender_id])
            flux_pulse.len = rabi_len
            # flux_pulse.delta_freq = 0.001
            if 'send_A_list' in kwargs:
                flux_pulse.A_list = kwargs['send_A_list']
            sequencer.append('flux%s'%sender_id,flux_pulse)

            if self.expt_cfg['rece_delay'] > 0:
                sequencer.append('flux%s'%receiver_id,
                                 Idle(time=self.expt_cfg['rece_delay']))

            flux_pulse = copy.copy(self.communication_flux_pi_v2[receiver_id])
            flux_pulse.len = rabi_len
            # flux_pulse.plot = True
            if 'rece_A_list' in kwargs:
                flux_pulse.A_list = kwargs['rece_A_list']
            sequencer.append('flux%s'%receiver_id,flux_pulse)

            sequencer.sync_channels_time(self.channels)

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % receiver_id, self.qubit_ef_pi[receiver_id])

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def heralding_protocol_test(self, sequencer, **kwargs):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            rabi_pulse = copy.copy(self.qubit_pi[sender_id])
            rabi_pulse.sigma_len = rabi_len

            sequencer.append('charge%s' % sender_id, rabi_pulse)
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            sequencer.append('flux%s'%sender_id,self.communication_flux_ef_pi[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            sequencer.sync_channels_time(self.channels)

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % receiver_id, self.qubit_ef_pi[receiver_id])

            sequencer.sync_channels_time(self.channels)
            sequencer.append('flux%s'%sender_id,self.communication_flux_ef_pi[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def heralding_protocol_one(self, sequencer, **kwargs):
        # mm rabi sequences

        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]

        for qubit_measure in measurement_pulse:
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_half_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])

            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            sequencer.append('flux%s'%sender_id,self.communication_flux_pi_v2[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            sequencer.sync_channels_time(self.channels)
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[receiver_id])

            sequencer.sync_channels_time(self.channels)

            sequencer.append('flux%s'%sender_id,self.communication_flux_pi_v2[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            sequencer.sync_channels_time(self.channels)

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            if qubit_1_measure == 'X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == 'Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)

            if qubit_2_measure == 'X':
                m_pulse = copy.copy(self.qubit_half_pi['2'])
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == 'Y':
                m_pulse = copy.copy(self.qubit_half_pi['2'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-X':
                m_pulse = copy.copy(self.qubit_half_pi['2'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-Y':
                m_pulse = copy.copy(self.qubit_half_pi['2'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def heralding_protocol_tomography(self, sequencer, **kwargs):
        # mm rabi sequences

        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]

        for qubit_measure in measurement_pulse:
            sequencer.new_sequence(self)

            sender_id = self.communication['sender_id']
            receiver_id = self.communication['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_half_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

            sequencer.append('flux%s'%sender_id,self.communication_flux_ef_pi[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            sequencer.sync_channels_time(self.channels)

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])
            sequencer.append('charge%s' % receiver_id, self.qubit_ef_pi[receiver_id])

            sequencer.sync_channels_time(self.channels)
            sequencer.append('flux%s'%sender_id,self.communication_flux_ef_pi[sender_id])

            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi_v2[receiver_id])

            sequencer.sync_channels_time(self.channels)

            qubit_1_measure = qubit_measure[0]
            qubit_2_measure = qubit_measure[1]

            if qubit_1_measure == 'X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == 'Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-X':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '1', m_pulse)
            elif qubit_1_measure == '-Y':
                m_pulse = copy.copy(self.qubit_half_pi['1'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '1', m_pulse)

            if qubit_2_measure == 'X':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == 'Y':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-X':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = -np.pi
                sequencer.append('charge%s' % '2', m_pulse)
            elif qubit_2_measure == '-Y':
                m_pulse = copy.copy(self.qubit_ef_half_pi['2'])
                m_pulse.phase = -np.pi/2
                sequencer.append('charge%s' % '2', m_pulse)

            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def photon_transfer_arb(self, sequencer, **kwargs):
        # mm rabi sequences

        for repeat_id in range(self.expt_cfg['repeat']):

            for expt_id in range(kwargs['sequence_num']):
                sequencer.new_sequence(self)

                sender_id = self.quantum_device_cfg['communication']['sender_id']
                receiver_id = self.quantum_device_cfg['communication']['receiver_id']


                sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
                sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])


                flux_pulse = self.communication_flux_pi[sender_id]
                flux_pulse.len = kwargs['send_len'][expt_id]
                # flux_pulse.plot = True if repeat_id == 0 else False
                if 'send_A_list' in kwargs:
                    flux_pulse.A_list = kwargs['send_A_list'][expt_id]
                if 'delta_freq_send' in kwargs:
                    flux_pulse.delta_freq = kwargs['delta_freq_send'][expt_id]
                sequencer.append('flux%s'%sender_id,flux_pulse)

                flux_pulse = self.communication_flux_pi[receiver_id]
                flux_pulse.len = kwargs['rece_len'][expt_id]
                # flux_pulse.plot = True if repeat_id == 0 else False
                if 'rece_A_list' in kwargs:
                    flux_pulse.A_list = kwargs['rece_A_list'][expt_id]
                if 'delta_freq_rece' in kwargs:
                    flux_pulse.delta_freq = kwargs['delta_freq_rece'][expt_id]
                sequencer.append('flux%s'%receiver_id,flux_pulse)


                self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_rabi(self, sequencer):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_ef_rabi(self, sequencer):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['ef_pi_amp'][mm_id], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.multimodes[qubit_id]['ef_freq'][mm_id], phase=0,
                                        plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_t1(self, sequencer):
        # multimode t1 sequences

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=t1_len))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_ramsey(self, sequencer):
        # mm rabi sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes'][qubit_id]['dc_offset'][mm_id]),
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def sideband_dc_offset(self, sequencer):
        # sideband dc offset sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                if self.expt_cfg['ge']:

                    sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('flux%s'%qubit_id,
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                            plot=False))
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id,
                                     Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes']['dc_offset'][mm_id]), plot=False))

                else:

                    sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('flux%s'%qubit_id,
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                            plot=False))
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('charge%s' % qubit_id,
                                     Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes']['dc_offset'][mm_id]), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def rabi_transfer(self, sequencer):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            rabi_pulse = copy.copy(self.qubit_pi[sender_id])
            rabi_pulse.sigma_len = rabi_len

            sequencer.append('charge%s' % sender_id, rabi_pulse)
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
            sequencer.append('flux%s'%sender_id,self.communication_flux_pi[sender_id])
            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi[receiver_id])


            self.readout(sequencer)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def bell_entanglement_by_ef(self, sequencer):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)



            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            ef_rabi_pulse = copy.copy(self.qubit_ef_pi[sender_id])
            ef_rabi_pulse.sigma_len = rabi_len
            sequencer.append('charge%s' % sender_id, ef_rabi_pulse)
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
            sequencer.append('flux%s'%sender_id,self.communication_flux_pi[sender_id])
            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi[receiver_id])

            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id])
            # sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])

            self.readout(sequencer)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def bell_entanglement_by_ef_tomography(self, sequencer):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        measurement_pulse = ['I', 'X', 'Y']

        for qubit_1_measure in measurement_pulse:
            for qubit_2_measure in measurement_pulse:
                sequencer.new_sequence(self)

                sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
                ef_rabi_pulse = copy.copy(self.qubit_ef_pi[sender_id])
                ef_rabi_pulse.sigma_len = 14
                sequencer.append('charge%s' % sender_id, ef_rabi_pulse)
                sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
                sequencer.append('flux%s'%sender_id,self.communication_flux_pi[sender_id])
                sequencer.append('flux%s'%receiver_id,self.communication_flux_pi[receiver_id])

                sequencer.sync_channels_time(self.channels)
                # sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
                sequencer.append('charge%s' % sender_id, self.qubit_ef_pi[sender_id])


                if qubit_1_measure == 'X':
                    x_pulse = copy.copy(self.qubit_half_pi['1'])
                    sequencer.append('charge%s' % '1', x_pulse)
                elif qubit_1_measure == 'Y':
                    y_pulse = copy.copy(self.qubit_half_pi['1'])
                    y_pulse.phase = np.pi/2
                    sequencer.append('charge%s' % '1', y_pulse)

                if qubit_2_measure == 'X':
                    x_pulse = copy.copy(self.qubit_half_pi['2'])
                    sequencer.append('charge%s' % '2', x_pulse)
                elif qubit_2_measure == 'Y':
                    y_pulse = copy.copy(self.qubit_half_pi['2'])
                    y_pulse.phase = np.pi/2
                    sequencer.append('charge%s' % '2', y_pulse)

                self.readout(sequencer)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def bell_entanglement_by_half_sideband_tomography(self, sequencer, **kwargs):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]


        for expt_id in range(kwargs.get('sequence_num',1)):
            for qubit_measure in measurement_pulse:
                sequencer.new_sequence(self)

                sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
                sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

                send_flux_pulse = copy.copy(self.communication_flux_half_transfer[sender_id])
                if 'send_len' in kwargs:
                    send_flux_pulse.len = kwargs['send_len'][expt_id]
                if 'send_A_list' in kwargs:
                    send_flux_pulse.A_list = kwargs['send_A_list'][expt_id]

                # send_flux_pulse.phase = -1.82712+0.0580-0.3383 + 0.04741-0.026
                sequencer.append('flux%s'%sender_id,send_flux_pulse)


                receiver_flux_pulse = copy.copy(self.communication_flux_half_transfer[receiver_id])
                if 'rece_len' in kwargs:
                    receiver_flux_pulse.len = kwargs['rece_len'][expt_id]
                if 'rece_A_list' in kwargs:
                    receiver_flux_pulse.A_list = kwargs['rece_A_list'][expt_id]
                sequencer.append('flux%s'%receiver_id,receiver_flux_pulse)

                qubit_1_measure = qubit_measure[0]
                qubit_2_measure = qubit_measure[1]

                measurement_phase = -1.82712+0.0580-0.3383 + 0.04741-0.026 - 0.2054 + 0.12731 + 0.0087 - 0.87632224

                sequencer.sync_channels_time(['charge1', 'flux1'])
                if qubit_1_measure == 'X':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = 0 + measurement_phase
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == 'Y':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = np.pi/2 + measurement_phase
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == '-X':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = -np.pi + measurement_phase
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == '-Y':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = -np.pi/2 + measurement_phase
                    sequencer.append('charge%s' % '1', m_pulse)

                sequencer.sync_channels_time(['charge2', 'flux2'])
                if qubit_2_measure == 'X':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == 'Y':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = np.pi/2
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == '-X':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = -np.pi
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == '-Y':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = -np.pi/2
                    sequencer.append('charge%s' % '2', m_pulse)

                self.readout(sequencer)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ghz_entanglement_by_half_sideband_tomography(self, sequencer, **kwargs):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]


        for expt_id in range(kwargs.get('sequence_num',1)):
            for qubit_measure in measurement_pulse:
                sequencer.new_sequence(self)

                sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
                sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])

                send_flux_pulse = copy.copy(self.communication_flux_half_transfer[sender_id])
                if 'send_len' in kwargs:
                    send_flux_pulse.len = kwargs['send_len'][expt_id]
                if 'send_A_list' in kwargs:
                    send_flux_pulse.A_list = kwargs['send_A_list'][expt_id]
                sequencer.append('flux%s'%sender_id,send_flux_pulse)
                sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id])
                sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])

                receiver_flux_pulse = copy.copy(self.communication_flux_half_transfer[receiver_id])
                if 'rece_len' in kwargs:
                    receiver_flux_pulse.len = kwargs['rece_len'][expt_id]
                if 'rece_A_list' in kwargs:
                    receiver_flux_pulse.A_list = kwargs['rece_A_list'][expt_id]
                sequencer.append('flux%s'%receiver_id,receiver_flux_pulse)



                sequencer.sync_channels_time(self.channels)

                qubit_1_measure = qubit_measure[0]
                qubit_2_measure = qubit_measure[1]

                if qubit_1_measure == 'X':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == 'Y':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = np.pi/2
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == '-X':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = -np.pi
                    sequencer.append('charge%s' % '1', m_pulse)
                elif qubit_1_measure == '-Y':
                    m_pulse = copy.copy(self.qubit_half_pi['1'])
                    m_pulse.phase = -np.pi/2
                    sequencer.append('charge%s' % '1', m_pulse)

                if qubit_2_measure == 'X':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == 'Y':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = np.pi/2
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == '-X':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = -np.pi
                    sequencer.append('charge%s' % '2', m_pulse)
                elif qubit_2_measure == '-Y':
                    m_pulse = copy.copy(self.qubit_half_pi['2'])
                    m_pulse.phase = -np.pi/2
                    sequencer.append('charge%s' % '2', m_pulse)

                self.readout(sequencer)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def transfer_residue_test(self, sequencer):
        # rabi sequences
        sender_id = self.communication['sender_id']
        receiver_id = self.communication['receiver_id']

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
            sequencer.append('flux%s'%sender_id,self.communication_flux_pi[sender_id])
            sequencer.append('flux%s'%receiver_id,self.communication_flux_pi[receiver_id])

            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
            ef_rabi_pulse = copy.copy(self.qubit_ef_pi[sender_id])
            ef_rabi_pulse.sigma_len = rabi_len
            sequencer.append('charge%s' % sender_id, ef_rabi_pulse)

            self.readout(sequencer)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def two_mode_ghz(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            for measure_id in [0,1]:
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:

                    rabi_pulse = copy.copy(self.qubit_pi[qubit_id])
                    rabi_pulse.sigma_len = rabi_len
                    sequencer.append('charge%s' % qubit_id, rabi_pulse)

                    mm_id = list(map(int, self.expt_cfg['on_mms'][qubit_id]))
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])

                    half_sideband_pulse = copy.copy(self.mm_sideband_pi[qubit_id][mm_id[0]])
                    half_sideband_pulse.flat_len = self.mm_sideband_pi[qubit_id][mm_id[0]].flat_len/2

                    sequencer.append('flux%s'%qubit_id,half_sideband_pulse)

                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('flux%s'%qubit_id,
                                     self.mm_sideband_pi[qubit_id][mm_id[1]])

                    sequencer.append('flux%s'%qubit_id,
                                     self.mm_sideband_pi[qubit_id][mm_id[measure_id]])




                self.readout(sequencer)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def alazar_test(self, sequencer):
        # drag_rabi sequences

        freq_ge = 4.5  # GHz
        alpha = - 0.125  # GHz

        freq_lambda = (freq_ge + alpha) / freq_ge
        optimal_beta = freq_lambda ** 2 / (4 * alpha)

        for rabi_len in np.arange(0, 50, 5):
            sequencer.new_sequence(self)

            self.readout(sequencer)
            # sequencer.append('charge1', Idle(time=100))
            sequencer.append('charge1',
                             Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=self.qubit_freq, phase=0,
                                   plot=False))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def get_experiment_sequences(self, experiment, **kwargs):
        vis = visdom.Visdom()
        vis.close()

        sequencer = Sequencer(self.channels, self.channels_awg, self.awg_info, self.channels_delay)
        self.expt_cfg = self.experiment_cfg[experiment]

        multiple_sequences = eval('self.' + experiment)(sequencer, **kwargs)

        return self.get_sequences(multiple_sequences)

    def get_sequences(self, multiple_sequences):
        seq_num = len(multiple_sequences)

        sequences = {}
        for channel in self.channels:
            channel_waveform = []
            for seq_id in range(seq_num):
                channel_waveform.append(multiple_sequences[seq_id][channel])
            sequences[channel] = np.array(channel_waveform)

        return sequences


    def get_awg_readout_time(self, readout_time_list):
        awg_readout_time_list = {}
        for awg in self.awg_info:
            awg_readout_time_list[awg] = (readout_time_list - self.awg_info[awg]['time_delay'])

        return awg_readout_time_list


if __name__ == "__main__":
    cfg = {
        "qubit": {"freq": 4.5}
    }

    hardware_cfg = {
        "channels": [
            "charge1", "flux1", "charge2", "flux2",
            "hetero1_I", "hetero1_Q", "hetero2_I", "hetero2_Q",
            "m8195a_trig", "readout1_trig", "readout2_trig", "alazar_trig"
        ],
        "channels_awg": {"charge1": "m8195a", "flux1": "m8195a", "charge2": "m8195a", "flux2": "m8195a",
                         "hetero1_I": "tek5014a", "hetero1_Q": "tek5014a", "hetero2_I": "tek5014a",
                         "hetero2_Q": "tek5014a",
                         "m8195a_trig": "tek5014a", "readout1_trig": "tek5014a", "readout2_trig": "tek5014a",
                         "alazar_trig": "tek5014a"},
        "awg_info": {"m8195a": {"dt": 0.0625, "min_increment": 16, "min_samples": 128, "time_delay": 110},
                     "tek5014a": {"dt": 0.83333333333, "min_increment": 16, "min_samples": 128, "time_delay": 0}}
    }

    ps = PulseSequences(cfg, hardware_cfg)

    ps.get_experiment_sequences('rabi')