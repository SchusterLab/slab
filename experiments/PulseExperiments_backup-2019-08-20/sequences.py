try:
    from .sequencer import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a
# from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom
import os
import pickle

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

        self.qubit_ee_freq = self.quantum_device_cfg['qubit']['12']

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
            # Modified by Tanay
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

                    # sequencer.append('charge%s' %self.charge_port[qubit_id], Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                    #             ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=rabi_freq, phase=0))

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
                                        cutoff_sigma=2, freq=flux_freq, phase=self.expt_cfg['phase'][index], plot=False))

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

                # sequencer.append('flux%s' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['flux_amp'], flat_len=rabi_len,
                #                         ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                #                         cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=0))

                for index, flux_line_id in enumerate(self.expt_cfg['flux_line']):
                        sequencer.append('flux%s' %flux_line_id, Square(max_amp=self.expt_cfg['flux_amp'][index], flat_len=rabi_len,
                                                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                                                        cutoff_sigma=2, freq=self.expt_cfg['flux_freq'], phase=self.expt_cfg['phase'][index], plot=False))

                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                        cutoff_sigma=2, freq=self.expt_cfg['qubit_freq'], phase=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def vacuum_rabi(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for iq_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)


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


    def t1(self, sequencer):
        # t1 sequences

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

    def ramsey_while_flux(self, sequencer):
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
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=0.5*np.pi, plot=False))
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
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