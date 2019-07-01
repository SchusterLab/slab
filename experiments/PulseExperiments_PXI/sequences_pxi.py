try:
    from .sequencer_pxi import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone
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



        self.pulse_info = self.quantum_device_cfg['pulse_info']

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']

        # pulse params
        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']}

        self.qubit_sideband_freq = {"1": self.quantum_device_cfg['sideband_prep']['1']['sideband_freq'],
                           "2": self.quantum_device_cfg['sideband_prep']['2']['sideband_freq']}

        self.qubit_ramsey_freq = {"1": self.experiment_cfg['ramsey_sideband']['ramsey_freq'],
                                  "2": self.experiment_cfg['ramsey_sideband']['ramsey_freq']}

        self.qubit_sideband_Iamp = {"1": self.quantum_device_cfg['sideband_prep']['1']['I_amp'],
                                    "2": self.quantum_device_cfg['sideband_prep']['2']['I_amp']}

        self.qubit_sideband_Qamp = {"1": self.quantum_device_cfg['sideband_prep']['1']['Q_amp'],
                                    "2": self.quantum_device_cfg['sideband_prep']['2']['Q_amp']}

        self.qubit_sideband_Iphase = {"1": self.quantum_device_cfg['sideband_prep']['1']['I_phase'],
                                      "2": self.quantum_device_cfg['sideband_prep']['2']['I_phase']}

        self.qubit_sideband_Qphase = {"1": self.quantum_device_cfg['sideband_prep']['1']['Q_phase'],
                                      "2": self.quantum_device_cfg['sideband_prep']['2']['Q_phase']}

        self.qubit_ef_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq']+self.quantum_device_cfg['qubit']['1']['anharmonicity'],
                              "2": self.quantum_device_cfg['qubit']['2']['freq']+self.quantum_device_cfg['qubit']['2']['anharmonicity']}

        self.qubit_pi_I = {
            "1": Square(max_amp=self.pulse_info['1']['pi_amp'], flat_len=self.pulse_info['1']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info['1']['iq_freq'],phase=0),
            "2": Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["2"]['iq_freq'], phase=0)}

        self.qubit_pi_Q = {
            "1": Square(max_amp=self.pulse_info['1']['pi_amp'], flat_len=self.pulse_info['1']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["1"]['iq_freq'], phase=self.pulse_info['1']['Q_phase']),
            "2": Square(max_amp=self.pulse_info['2']['pi_amp'], flat_len=self.pulse_info['2']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["2"]['iq_freq'], phase=self.pulse_info['2']['Q_phase'])}

        self.qubit_sideband_pi_I = {
        "1": Square(max_amp=self.qubit_sideband_Iamp['1'], flat_len=self.pulse_info['1']['pi_len'],
                                ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq["1"],
                                phase=self.qubit_sideband_Iphase["1"]),
        "2": Square(max_amp=self.qubit_sideband_Iamp['2'], flat_len=self.pulse_info['2']['pi_len'],
                                ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq["2"],
                                phase=self.qubit_sideband_Iphase["2"])}

        self.qubit_sideband_pi_Q = {
            "1": Square(max_amp=self.qubit_sideband_Qamp['1'], flat_len=self.pulse_info['1']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq["1"],
                        phase=self.qubit_sideband_Qphase["1"]),
            "2": Square(max_amp=self.qubit_sideband_Qamp['2'], flat_len=self.pulse_info['2']['pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq["2"],
                        phase=self.qubit_sideband_Qphase["2"])}

        self.qubit_half_pi_I = {
            "1": Square(max_amp=self.pulse_info['1']['half_pi_amp'], flat_len=self.pulse_info['1']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["1"]['iq_freq'], phase=0),
            "2": Square(max_amp=self.pulse_info['2']['half_pi_amp'], flat_len=self.pulse_info['2']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["2"]['iq_freq'], phase=0)}

        self.qubit_half_pi_Q = {
            "1": Square(max_amp=self.pulse_info['1']['half_pi_amp'], flat_len=self.pulse_info['1']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["1"]['iq_freq'], phase=self.pulse_info['1']['Q_phase']),
            "2": Square(max_amp=self.pulse_info['2']['half_pi_amp'], flat_len=self.pulse_info['2']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info["2"]['iq_freq'], phase=self.pulse_info['2']['Q_phase'])}

        self.qubit_sideband_half_pi_I = {
            "1": Square(max_amp=self.qubit_sideband_Iamp['1'], flat_len=self.pulse_info['1']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq['1'],
                        phase=self.qubit_sideband_Iphase["1"]),
            "2": Square(max_amp=self.qubit_sideband_Iamp['2'], flat_len=self.pulse_info['2']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq['1'],
                        phase=self.qubit_sideband_Iphase["2"])}

        self.qubit_sideband_half_pi_Q = {
            "1": Square(max_amp=self.qubit_sideband_Qamp['1'], flat_len=self.pulse_info['1']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq['1'],
                        phase=self.qubit_sideband_Qphase["1"]),
            "2": Square(max_amp=self.qubit_sideband_Qamp['2'], flat_len=self.pulse_info['2']['half_pi_len'],
                        ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_sideband_freq['1'],
                        phase=self.qubit_sideband_Qphase["2"])}

        self.qubit_ef_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['pi_ef_amp'], sigma_len=self.pulse_info['1']['pi_ef_len'], cutoff_sigma=2,
                   freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['pi_ef_amp'], sigma_len=self.pulse_info['2']['pi_ef_len'], cutoff_sigma=2,
                   freq=self.qubit_ef_freq["2"], phase=0, plot=False)}

        self.qubit_ef_half_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['half_pi_ef_amp'], sigma_len=self.pulse_info['1']['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['half_pi_ef_amp'], sigma_len=self.pulse_info['2']['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq["2"], phase=0, plot=False)}

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

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)




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

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,plot_visdom=True):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg)
        self.plot_visdom = plot_visdom

    def gen_q(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qubit_id]['iq_freq']+add_freq,
                                    phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,
                         Square(max_amp=amp, flat_len= len,
                                ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qubit_id]['iq_freq']+add_freq,
                                phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.pulse_info[qubit_id]['iq_freq']+add_freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.pulse_info[qubit_id]['iq_freq']+add_freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len=self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len= self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def half_pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], flat_len= self.pulse_info[qubit_id]['half_pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],cutoff_sigma=2,
                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def pi_q_ef(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit']['1']['anharmonicity']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], flat_len= self.pulse_info[qubit_id]['pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['pi_ef_len'],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['pi_ef_len'],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def half_pi_q_ef(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit']['1']['anharmonicity']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], flat_len= self.pulse_info[qubit_id]['half_pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def pi_q_fh(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit']['1']['anharmonicity']+self.quantum_device_cfg['qubit']['1']['anharmonicity_fh']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'], flat_len=self.pulse_info[qubit_id]['pi_fh_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'], flat_len= self.pulse_info[qubit_id]['pi_fh_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'], sigma_len=self.pulse_info[qubit_id]['pi_fh_len'],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'], sigma_len=self.pulse_info[qubit_id]['pi_fh_len'],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def half_pi_q_fh(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit']['1']['anharmonicity']+self.quantum_device_cfg['qubit']['1']['anharmonicity_fh']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_fh_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'], flat_len= self.pulse_info[qubit_id]['half_pi_fh_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_fh_len'],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_fh_len'],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[qubit_id]['Q_phase']))

    def pi_f0g1_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'], phase=phase,
                                plot=False))

    def idle_q(self,sequencer,qubit_id = '1',time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))

    def idle_q_sb(self, sequencer, qubit_id='1', time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))
        sequencer.append('sideband', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_sb(self,sequencer,time=0):
        sequencer.append('sideband', Idle(time=time))

    def pad_start_pxi(self,sequencer,on_qubits=None, time = 500):
        # Need 500 ns of padding for the sequences to work reliably. Not sure exactly why.
        for qubit_id in on_qubits:
            sequencer.append('charge%s_I' % qubit_id,
                             Square(max_amp=0.0, flat_len= time, ramp_sigma_len=0.001, cutoff_sigma=2, freq=0.0,
                                    phase=0))

            sequencer.append('charge%s_Q' % qubit_id,
                         Square(max_amp=0.0, flat_len= time, ramp_sigma_len=0.001, cutoff_sigma=2, freq=0.0,
                                phase=0))

    def pad_start_pxi_tek2(self,sequencer,on_qubits=None, time = 500):
        self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=time)
        sequencer.append('tek2_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

    def readout(self, sequencer, on_qubits=None, sideband = False):
        if on_qubits == None:
            on_qubits = ["1", "2"]

        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('readout_trig') # Earlies was alazar_trig

        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5

        sequencer.append_idle_to_time('readout_trig', readout_time_5ns_multiple)
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

        sequencer.append('readout',
                         Square(max_amp=self.quantum_device_cfg['readout']['amp'],
                                flat_len=self.quantum_device_cfg['readout']['length'],
                                ramp_sigma_len=20, cutoff_sigma=2, freq=self.quantum_device_cfg['readout']['freq'],
                                phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('readout_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def readout_pxi(self, sequencer, on_qubits=None, sideband = False, overlap = False):
        if on_qubits == None:
            on_qubits = ["1", "2"]

        sequencer.sync_channels_time(self.channels)
        readout_time = sequencer.get_time('readout_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('readout_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(self.channels)


        sequencer.append('readout',
                         Square(max_amp=self.quantum_device_cfg['readout']['amp'],
                                flat_len=self.quantum_device_cfg['readout']['length'],
                                ramp_sigma_len=20, cutoff_sigma=2, freq=0,
                                phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('readout_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def parity_measurement(self, sequencer, qubit_id='1'):
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        self.idle_q(sequencer, qubit_id, time=-1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0)
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sequencer.sync_channels_time(self.channels)

    def resonator_spectroscopy(self, sequencer):

        sequencer.new_sequence(self)
        self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s_I' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=0))

                sequencer.append('charge%s_Q' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=self.pulse_info[qubit_id]['Q_phase']))
                self.idle_q(sequencer, time=self.expt_cfg['delay'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                for echo_id in range(self.expt_cfg['echo_times']):
                    self.idle_q(sequencer, time=ramsey_len/(float(2*self.expt_cfg['echo_times'])))
                    if self.expt_cfg['cp']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    elif self.expt_cfg['cpmg']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi/2.0)
                    self.idle_q(sequencer, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=2 * np.pi * ramsey_len*self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def t1rho(self,sequencer):

        for t1rho_len in np.arange(self.expt_cfg['start'],self.expt_cfg['stop'],self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #self.idle_q(sequencer, time= self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['half_pi_len'])
                self.gen_q(sequencer, qubit_id, len=t1rho_len, amp=self.expt_cfg['amp'], phase=np.pi/2,
                           pulse_type=self.expt_cfg['pulse_type'])
                #self.idle_q(sequencer, time=self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['half_pi_len'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=0)


                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_ef_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+self.quantum_device_cfg['qubit']['1']['anharmonicity'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer,qubit_id,len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_t1(self, sequencer):
        # t1 for the e and f level

        for ef_t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:

                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ef_t1_len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                for echo_id in range(self.expt_cfg['echo_times']):
                    self.idle_q(sequencer, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                    if self.expt_cfg['cp']:
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    elif self.expt_cfg['cpmg']:
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                                  phase=np.pi / 2.0)
                    self.idle_q(sequencer, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_fh_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+2*self.quantum_device_cfg['qubit']['1']['anharmonicity'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] + \
                           self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity_fh']
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef_pi']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.gen_q(sequencer,qubit_id,len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = add_freq )
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_ramsey(self, sequencer):


        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


        return sequencer.complete(self, plot=True)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=qubit_id, time=500)
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()
                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()
                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def sideband_histogram(self, sequencer):
        # vacuum rabi sequences


        center_freq = self.expt_cfg['center_freq']
        offset_freq = self.expt_cfg['offset_freq']
        sideband_freq = self.expt_cfg['single_freq']
        detuning = self.expt_cfg['detuning']
        freq1, freq2 = center_freq + offset_freq - detuning / 2.0, center_freq + offset_freq + detuning / 2.0
        rabi_len = self.expt_cfg['length']

        for ii in range(self.expt_cfg['num_seq_sets']):
            for qubit_id in self.expt_cfg['on_qubits']:

                sequencer.new_sequence(self)
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['single_tone']:
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                            plot=False))
                    self.gen_q(sequencer, qubit_id, len=rabi_len, phase=0,
                               pulse_type="square",
                               add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                else:
                    sequencer.append('sideband',
                                 Square_two_tone(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                                 ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                     'ramp_sigma_len'], cutoff_sigma=2, freq1=freq1, freq2=freq2,
                                                 phase1=0, phase2=0, plot=False))
                for qubit_id in self.expt_cfg['on_qubits']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()

                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['single_tone']:
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                            plot=False))
                    self.gen_q(sequencer, qubit_id, len=rabi_len, phase=0,
                               pulse_type="square",
                               add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                else:
                    sequencer.append('sideband',
                                 Square_two_tone(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                                 ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                     'ramp_sigma_len'], cutoff_sigma=2, freq1=freq1, freq2=freq2,
                                                 phase1=0, phase2=0, plot=False))

                for qubit_id in self.expt_cfg['on_qubits']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()

                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['single_tone']:
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                            plot=False))
                    self.gen_q(sequencer, qubit_id, len=rabi_len, phase=0,
                               pulse_type="square",
                               add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                else:
                    sequencer.append('sideband',
                                 Square_two_tone(max_amp=self.expt_cfg['amp'], flat_len=rabi_len,
                                                 ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                     'ramp_sigma_len'], cutoff_sigma=2, freq1=freq1, freq2=freq2,
                                                 phase1=0, phase2=0, plot=False))

                for qubit_id in self.expt_cfg['on_qubits']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def sideband_rabi(self, sequencer):
        # sideband rabi time domain
        sideband_freq = self.expt_cfg['freq']
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['f0g1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                elif self.expt_cfg['h0e1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        plot=False))
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                if self.expt_cfg['h0e1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_rabi_freq_scan(self, sequencer):

        sideband_freq = self.expt_cfg['freq']
        length = self.expt_cfg['length']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer,on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['f0g1']:self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                elif self.expt_cfg['h0e1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                             Square(max_amp=self.expt_cfg['amp'], flat_len=length, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                    plot=False))
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                if self.expt_cfg['h0e1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_rabi_two_tone(self, sequencer):
        # sideband rabi time domain
        center_freq = self.expt_cfg['center_freq']
        offset_freq = self.expt_cfg['offset_freq']
        detuning = self.expt_cfg['detuning']
        freq1, freq2 = center_freq + offset_freq - detuning / 2.0, center_freq + offset_freq + detuning / 2.0
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer,on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                # self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                             Square_two_tone(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq1=freq1,freq2 =freq2,
                                             phase1=0,phase2 = np.pi,plot=False))
                if self.expt_cfg['ef']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_rabi_two_tone_freq_scan(self, sequencer):
        # sideband rabi time domain
        center_freq = self.expt_cfg['center_freq']
        offset_freq = self.expt_cfg['offset_freq']
        rabi_len = self.expt_cfg['length']
        for detuning in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            freq1, freq2 = center_freq+offset_freq - detuning / 2.0, center_freq+offset_freq + detuning / 2.0
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer,on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                             Square_two_tone(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq1=freq1,freq2 =freq2,
                                             phase1=0,phase2 = 0,plot=False))
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=0000.0)
                if self.expt_cfg['ef']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['f0g1']:
                    # self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square')
                    sequencer.append('sideband',
                                     Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'],
                                            flat_len=5000,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                            cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'],
                                            phase=0,plot=False))
                    sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=40.0)
                sequencer.append('charge%s_I' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2,
                                        freq=self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=0))

                sequencer.append('charge%s_Q' % qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2,
                                        freq=self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=self.pulse_info[qubit_id]['Q_phase']))
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_t1(self, sequencer):
        # sideband rabi time domain
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer,on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square')
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer,qubit_id,time=length + 40.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square')
                if self.expt_cfg['pi_calibration']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=40)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_ramsey(self, sequencer):
        # sideband rabi time domain

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'] + self.expt_cfg['ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=ramsey_len+50.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = 2*np.pi*ramsey_len*phase_freq-offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_pi_pi_offset(self, sequencer):
        # sideband rabi time domain
        offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
        for offset_phase2 in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'] + self.expt_cfg[
                    'ramsey_freq']

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=self.expt_cfg['delay'])
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = -offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type']+offset_phase2)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ge_calibration(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'] + self.expt_cfg['ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, qubit_id, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = 2*np.pi*ramsey_len*phase_freq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ef_calibration(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef'] + self.expt_cfg[
                    'ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                sequencer.sync_channels_time(self.channels)

                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                                  phase=2 * np.pi * ramsey_len * phase_freq)
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_gf_calibration(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_f'] + self.expt_cfg[
                    'ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                sequencer.sync_channels_time(self.channels)

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                  phase=2 * np.pi * ramsey_len * phase_freq)


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_transmon_reset(self, sequencer):

        sideband_freq = self.expt_cfg['sideband_freq']
        sideband_pulse_length =  self.expt_cfg['sideband_pulse_length']
        wait_after_reset = self.expt_cfg['wait_after_reset']

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                # Reset pulse
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=sideband_pulse_length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=wait_after_reset)
                # Temperature measurement
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer, qubit_id, len=rabi_len, phase=0,
                           pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                           add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_parity_measurement(self, sequencer):
        # sideband parity measurement

        sideband_freq = self.expt_cfg['sideband_freq']
        sideband_pulse_length =  self.expt_cfg['sideband_pulse_length']
        wait_after_reset = self.expt_cfg['wait_after_reset']

        for ii in np.arange(self.expt_cfg['num_expts']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                if self.expt_cfg['add_photon']:
                    if ii > self.expt_cfg['num_expts'] / 2.0:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        sequencer.sync_channels_time(self.channels)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                        sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['reset']:
                    self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=sideband_pulse_length,
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                            plot=False))
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=wait_after_reset)



                self.idle_q_sb(sequencer, qubit_id, time=self.expt_cfg['wait_before_parity'])
                self.parity_measurement(sequencer,qubit_id)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_cavity_photon_number(self, sequencer):
        # sideband parity measurement

        cavity_freq = self.expt_cfg['cavity_freq']
        cavity_pulse_length =  self.expt_cfg['cavity_pulse_length']
        wait_after_cavity_drive = self.expt_cfg['wait_after_cavity_drive']

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.append('sideband',
                                 Square(max_amp= self.expt_cfg['cavity_drive_amp'], flat_len=cavity_pulse_length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=cavity_freq, phase=0,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=wait_after_cavity_drive)
                self.gen_q(sequencer, qubit_id, amp = self.expt_cfg['qubit_drive_amp'],len=self.expt_cfg['pulse_length'], phase=0, pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def qp_pumping_t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            N_pump = self.expt_cfg['N_pump']
            pump_wait = self.expt_cfg['pump_wait']

            for pump in range(N_pump):
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q(sequencer, time=pump_wait)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
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