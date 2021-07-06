try:
    from .sequencer_pxi import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone,Double_Square
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

        try:self.cavity_pulse_info = self.quantum_device_cfg['cavity_pulse_info']
        except:print("No cavity pulses")

        try:self.transmon_pulse_info_tek2 = self.quantum_device_cfg['transmon_pulse_info_tek2']
        except:print("No tek2 transmon drive pulse information")

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']

        # pulse params
        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']}

        try:self.cavity_freq = {"1": self.quantum_device_cfg['cavity']['1']['freq']}
        except:pass

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
#    def pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square'):
#        if pulse_type.lower() == 'square':
#            sequencer.append('charge%s_I' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len=self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
#                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
#            sequencer.append('charge%s_Q' % qubit_id,Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len= self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
#                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))
#        elif pulse_type.lower() == 'gauss':
#            sequencer.append('charge%s_I' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
#                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase))
#            sequencer.append('charge%s_Q' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
#                            freq=self.pulse_info[qubit_id]['iq_freq'],phase=phase+self.pulse_info[qubit_id]['Q_phase']))

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
            # ALL THIS BELOW WAS ADDED BY MEG, OCT. 4 2019 IN COMPARISON TO PI_Q ORIGINAL. ORIGINAL IS COMMENTED OUT ABOVE. pi_q seems to be called implicitly if pi_calibration is true. sorry!
        #sequencer.append_idle_to_time('cavity1_I', 2150)
        #sequencer.append_idle_to_time('cavity1_Q', 2150)
        #self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,len=self.pulse_info[qubit_id]['pi_len']*4)

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
            # ALL THIS ADDED BY MEG ON OCT 7, 2019 AS EF PI CALIBRATION IS APPARENTLY SOMETIMES SECRETLY CALLED AND ITSELF CALLS THIS.
        # sequencer.append_idle_to_time('cavity1_I', 2150)
        # sequencer.append_idle_to_time('cavity1_Q', 2150)
        # self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001, len=self.pulse_info[qubit_id]['pi_ef_len']*4)

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

    def pi_f0g1_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',add_freq = 0,mode_index = 0):
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'][mode_index],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_len'][mode_index],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][mode_index]+add_freq, phase=phase,fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][mode_index],plot=False))

    def pi_fnm1gn_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',n = 0):
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_fnm1gn_amps'][mode_index][int(n-1)],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_fnm1gn_lens'][mode_index][int(n-1)],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fnm1gn_freqs'][mode_index][int(n-1)], phase=phase,fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][mode_index],plot=False))

    def prep_fock_n(self,sequencer,qubit_id = '1',fock_state=1,nshift =True):
        for nn in range(fock_state):
            if nshift:
                add_freq_ge = (nn) * 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e']
                add_freq_ef = (nn) * 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_ef']
            else:
                add_freq_ge, add_freq_ef = 0.0, 0.0
            self.sideband_pi_q(sequencer, qubit_id,
                               pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],
                               add_freq=add_freq_ge)
            self.sideband_pi_q_ef(sequencer, qubit_id,
                                  pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'],
                                  add_freq=add_freq_ef)
            self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0, n=nn + 1)

    def sideband_gen_q(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,
                                    phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,phase=phase))

    def sideband_pi_q(self, sequencer, qubit_id='1',phase=0, pulse_type='square',add_freq = 0):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_amp'], flat_len=self.transmon_pulse_info_tek2[qubit_id]['pi_len'],
                                                ramp_sigma_len=0.001, cutoff_sigma=2,
                                                freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,
                                                phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_amp'], sigma_len=self.transmon_pulse_info_tek2[qubit_id]['pi_len'],
                                               cutoff_sigma=2,
                                               freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,
                                               phase=phase))

    def sideband_pi_q_resolved(self, sequencer, qubit_id='1',phase=0, pulse_type='square',add_freq = 0):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_amp_resolved'], flat_len=self.transmon_pulse_info_tek2[qubit_id]['pi_len_resolved'],
                                                ramp_sigma_len=0.001, cutoff_sigma=2,
                                                freq=self.quantum_device_cfg['qubit'][qubit_id]['freq'] +add_freq,
                                                phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_amp_resolved'], sigma_len=self.transmon_pulse_info_tek2[qubit_id]['pi_len_resolved'],
                                               cutoff_sigma=2,
                                               freq=self.quantum_device_cfg['qubit'][qubit_id]['freq'] + add_freq,
                                               phase=phase))

    def sideband_half_pi_q(self, sequencer, qubit_id='1',phase=0, pulse_type='square',add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=self.transmon_pulse_info_tek2[qubit_id]['half_pi_amp'], flat_len=self.transmon_pulse_info_tek2[qubit_id]['half_pi_len'],
                                                ramp_sigma_len=0.001, cutoff_sigma=2,
                                                freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,
                                                phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=self.transmon_pulse_info_tek2[qubit_id]['half_pi_amp'], sigma_len=self.transmon_pulse_info_tek2[qubit_id]['half_pi_len'],
                                               cutoff_sigma=2,
                                               freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+add_freq,
                                               phase=phase))

    def sideband_pi_q_ef(self, sequencer, qubit_id='1',phase=0, pulse_type='square',add_freq = 0):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_ef_amp'], flat_len=self.transmon_pulse_info_tek2[qubit_id]['pi_ef_len'],
                                                ramp_sigma_len=0.001, cutoff_sigma=2,
                                                freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+add_freq,
                                                phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=self.transmon_pulse_info_tek2[qubit_id]['pi_ef_amp'], sigma_len=self.transmon_pulse_info_tek2[qubit_id]['pi_ef_len'],
                                               cutoff_sigma=2,
                                               freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+add_freq,
                                               phase=phase))

    def sideband_half_pi_q_ef(self, sequencer, qubit_id='1',phase=0, pulse_type='square',add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband', Square(max_amp=self.transmon_pulse_info_tek2[qubit_id]['half_pi_ef_amp'], flat_len=self.transmon_pulse_info_tek2[qubit_id]['half_pi_ef_len'],
                                                ramp_sigma_len=0.001, cutoff_sigma=2,
                                                freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+add_freq,
                                                phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband', Gauss(max_amp=self.transmon_pulse_info_tek2[qubit_id]['half_pi_ef_amp'], sigma_len=self.transmon_pulse_info_tek2[qubit_id]['half_pi_ef_len'],
                                               cutoff_sigma=2,
                                               freq=self.quantum_device_cfg['qubit'][qubit_id]['freq']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+add_freq,
                                               phase=phase))

    def idle_q(self,sequencer,qubit_id = '1',time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))

    def idle_q_sb(self, sequencer, qubit_id='1', time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))
        sequencer.append('sideband', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_sb(self,sequencer,time=0.0):
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
        # It is unclear if readout_trig appends to card delay or not -- does it delay digitizer?? Sep 2020 MGP
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
        #sequencer.append('readout_trig', Idle(time=400))
        sequencer.append('readout_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def readout_pxi_onqubittone(self, sequencer, on_qubits=None, sideband = False, overlap = False,len=10):
        if on_qubits == None:
            on_qubits = ["1", "2"]
        sequencer.append_idle_to_time('readout_trig',2020)
        sequencer.append_idle_to_time('readout', 2020)
        readout_time = sequencer.get_time('readout_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('readout_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(self.channels)
        sequencer.append('readout',
                         Square(max_amp=0.00001,
                                flat_len=len,
                                ramp_sigma_len=20, cutoff_sigma=2, freq=0,
                                phase=0, phase_t0=readout_time_5ns_multiple))
        #self.idle_q(sequencer, time=400)
        sequencer.append('readout_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time


    def parity_measurement(self, sequencer, qubit_id='1'):
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        # self.idle_q(sequencer, qubit_id, time=np.abs(1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0))
        self.idle_q(sequencer, qubit_id,time= self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'])
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sequencer.sync_channels_time(self.channels)

    def parity_measurement_alltek2(self, sequencer, qubit_id='1'):
        self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
        self.idle_sb(sequencer,time= self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'])
        self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],phase=np.pi)

    def gen_c(self, sequencer, cavity_id='1', len=10, amp=1, add_freq=0, phase=0, phase2=0, pulse_type='square'):
        if pulse_type.lower() == 'square':
            sequencer.append('cavity%s_I' % cavity_id, Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq,
                                                              phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id,
                             Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2,
                                    freq=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq,
                                    phase=phase + self.cavity_pulse_info[cavity_id]['Q_phase']))
        if pulse_type.lower() == 'square_two_tone':
            sequencer.append('cavity%s_I' % cavity_id, Square_two_tone(max_amp=amp, flat_len= len, ramp_sigma_len=0.001,
                                                             cutoff_sigma=2, freq1=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq, freq2 = self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"],
                                                             phase1=phase, phase2 = phase2))
            sequencer.append('cavity%s_Q' % cavity_id, Square_two_tone(max_amp=amp, flat_len= len, ramp_sigma_len=0.001, cutoff_sigma=2, freq1=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq,
                                                             freq2=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"], phase1=phase + self.cavity_pulse_info[cavity_id]['Q_phase'], phase2= phase2+ self.cavity_pulse_info[cavity_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity%s_I' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id][
                                                                                      'iq_freq'] + add_freq,
                                                             phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id][
                                                                                      'iq_freq'] + add_freq,
                                                             phase=phase + self.cavity_pulse_info[cavity_id][
                                                                 'Q_phase']))

    def gen_c2(self, sequencer, cavity_id='1', len=10, amp=1, add_freq=0, phase=0, phase2=0, pulse_type='square'):
        if pulse_type.lower() == 'square':
            sequencer.append('cavity%s_I' % cavity_id, Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"] + add_freq,
                                                              phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id,
                             Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2,
                                    freq=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"] + add_freq,
                                    phase=phase + self.cavity_pulse_info[cavity_id]['Q_phase']))
        # if pulse_type.lower() == 'square_two_tone':
        #     sequencer.append('cavity%s_I' % cavity_id, Square_two_tone(max_amp=amp, flat_len= len, ramp_sigma_len=0.001,
        #                                                      cutoff_sigma=2, freq1=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq, freq2 = self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"],
        #                                                      phase1=phase, phase2 = phase2))
        #     sequencer.append('cavity%s_Q' % cavity_id, Square_two_tone(max_amp=amp, flat_len= len, ramp_sigma_len=0.001, cutoff_sigma=2, freq1=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq,
        #                                                      freq2=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"], phase1=phase + self.cavity_pulse_info[cavity_id]['Q_phase'], phase2= phase2+ self.cavity_pulse_info[cavity_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity%s_I' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"] + add_freq,
                                                             phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id]['iq_freq'] - self.quantum_device_cfg["cavity"]["1"]["freq"] + self.quantum_device_cfg["cavity"]["2"]["freq"] + add_freq,
                                                             phase=phase + self.cavity_pulse_info[cavity_id][
                                                                 'Q_phase']))

    def gen_f0g1(self, sequencer, cavity_id='1', len=10, amp=1, add_freq=0, phase=0, pulse_type='square'):
        if pulse_type.lower() == 'square':
            sequencer.append('cavity%s_I' % cavity_id, Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_pulse_info[cavity_id][
                                                                       'iq_freq'] + add_freq,
                                                              phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id,
                             Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2,
                                    freq=self.cavity_pulse_info[cavity_id]['iq_freq'] + add_freq,
                                    phase=phase + self.cavity_pulse_info[cavity_id]['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity%s_I' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id][
                                                                                      'iq_freq'] + add_freq,
                                                             phase=phase))
            sequencer.append('cavity%s_Q' % cavity_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_pulse_info[cavity_id][
                                                                                      'iq_freq'] + add_freq,
                                                             phase=phase + self.cavity_pulse_info[cavity_id][
                                                                 'Q_phase']))

    def wigner_tomography(self, sequencer, qubit_id='1',cavity_id = '1',amp = 0, phase=0,len = 0):
        self.gen_c(sequencer, cavity_id=cavity_id, amp=amp,len=len,phase=phase)
        sequencer.sync_channels_time(self.channels)
        self.parity_measurement(sequencer,qubit_id)

    def wigner_tomography_sideband_only(self, sequencer, qubit_id='1',cavity_id = '1',amp = 0, phase=0,len = 0):
        sequencer.append('sideband',
                         Square(max_amp=amp, flat_len=len,
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[cavity_id], phase=phase,
                                plot=False))
        sequencer.sync_channels_time(self.channels)
        self.idle_q_sb(sequencer, qubit_id, time=50)
        self.parity_measurement_alltek2(sequencer,qubit_id)

    def wigner_tomography_alltek2(self, sequencer, qubit_id='1',cavity_id = '1',amp = 0, phase=0,len = 0):
        sequencer.append('sideband',
                         Square(max_amp=amp, flat_len=len,
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[cavity_id], phase=phase,
                                plot=False))
        self.idle_sb(sequencer, time=50)
        self.parity_measurement_alltek2(sequencer,qubit_id)

    def wigner_tomography_swap_and_cavity_drive(self, sequencer, qubit_id='1',cavity_id = '1',amp2 = 0, phase1=0, phase2=0,len2 = 0,delay = 0,doubling_trick = False,fix_phase=False):
        sequencer.append('sideband',
                         Double_Square(max_amp1=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'pi_f0g1_amp'],max_amp2=amp2, flat_len1=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'pi_f0g1_len'], flat_len2=len2, delay = delay,
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'ramp_sigma_len'], cutoff_sigma=2, freq1 =self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'f0g1_freq'], freq2=self.cavity_freq[cavity_id], phase1=phase1,phase2=phase2,plot=False,doubling_trick=doubling_trick,fix_phase=fix_phase))
        sequencer.sync_channels_time(self.channels)
        self.idle_q_sb(sequencer, qubit_id
                       , time=50)
        self.parity_measurement(sequencer,qubit_id)

    def cavity_drive_chi_dressing_calibration(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                add_phase = 0
                if self.expt_cfg['add_photon']:
                    phase_freq =  self.expt_cfg['ramsey_freq']
                else:
                    phase_freq = self.expt_cfg['ramsey_freq']
                if self.expt_cfg["h0e1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][
                                        self.expt_cfg['mode_index']] + self.expt_cfg['detuning']
                elif self.expt_cfg["f0g1"]:
                    sideband_freq = self.quantum_device_cfg['cavity']['1']['freq'] + self.expt_cfg['detuning']
                else:
                    sideband_freq = 0.0
                    print("what transition do you want to use for dressing good sir/madam")
                #offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][
                #    self.expt_cfg['mode_index']]
                if self.expt_cfg['add_photon']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'],
                          len=self.expt_cfg['pi_len'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q_sb(sequencer, qubit_id, time=2)
                dc_offset = 0
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['amp'],
                           add_freq=self.expt_cfg['detuning'],
                           len=ramsey_len,
                           pulse_type=self.expt_cfg['sideband_pulse_type'])
                self.idle_q_sb(sequencer, qubit_id, time=2)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=add_phase + 2 * np.pi * ramsey_len * phase_freq)
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sequential_chi_dressing_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_chi_dressing_calibration']
        varlist = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        experiment_name = 'cavity_drive_chi_dressing_calibration'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_chi_dressing_calibration',
                                                       suffix='.h5'))
        for ii, x in enumerate(varlist):
            if swp_cfg['sweep_detuning']:
                experiment_cfg[experiment_name]['detuning'] = x
            elif swp_cfg['sweep_amp']:
                experiment_cfg[experiment_name]['amp'] = x
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def resonator_spectroscopy(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

        if self.expt_cfg['pi_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

        if self.expt_cfg['pi_ef_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

        self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def resonator_spectroscopy_mixedtones(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

        if self.expt_cfg['pi_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001, len=self.pulse_info[qubit_id]['pi_len'])


        if self.expt_cfg['pi_ef_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001, len=self.pulse_info[qubit_id]['pi_ef_len'])

        self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)

    def cavity_drive_resonator_spectroscopy_mixedtones(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

        if self.expt_cfg['pi_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.pulse_info[qubit_id]['pi_len'])
                #removed *4

        if self.expt_cfg['pi_ef_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.idle_q(sequencer, time=self.pulse_info[qubit_id]['pi_len']+000)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.pulse_info[qubit_id]['pi_ef_len'])

        self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)

    def pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                ## Meg commented this out 9/16/2020 because it wasn't working when RF sources got switched
                ## Seems like newer versions use a different pulse structure, including gen_q?
                ## pulled in gen_q from cavity_drive_pulse_probe_iq which definitely worked in the spring
                # sequencer.append('charge%s_I' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                #                         ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                #                         phase=0))
                #
                # sequencer.append('charge%s_Q' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                #                         ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                #                         phase=self.pulse_info[qubit_id]['Q_phase']))
                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=self.expt_cfg['pulse_length'],
                           phase=0, pulse_type="square", add_freq=dfreq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def cavity_drive_pulse_probe_iq_mixedtones(self, sequencer):

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
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.expt_cfg['pulse_length'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_pulse_probe_ef_iq_mixedtones(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #pi_q already has the idle and gen_c embedded so no need to update this until later.... if you change the code add idles and gen_c
                self.gen_q(sequencer, qubit_id, amp = self.expt_cfg['amp'],len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+self.quantum_device_cfg['qubit']['1']['anharmonicity'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.expt_cfg['pulse_length'] * 4)
            if self.expt_cfg['pi_calibration']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef_pi_for_cal']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_t1_mixedtones(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['pi_len']) + 70, pulse_type ='square')
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_ramsey_mixedtones(self, sequencer):
# CHANGED ALL PULSES SQUARE
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['half_pi_len']), pulse_type ='square')
                self.idle_q(sequencer, time=ramsey_len)
                sequencer.append_idle_to_time('cavity1_I', ramsey_len+2150+20)
                sequencer.append_idle_to_time('cavity1_Q', ramsey_len+2150+20)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                #sequencer.append_idle_to_time('cavity1_I', ramsey_len + 2150 + 70+5)
                #sequencer.append_idle_to_time('cavity1_Q', ramsey_len + 2150 + 70 + 5)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.pulse_info[qubit_id]['half_pi_len'], pulse_type ='square')
                # again times 4 (or 2) on gen c pulse or make it gaussian??? unclear whether we need to multiply the 70. use oscilloscope.
               #if self.expt_cfg['pi_calibration']:
                 #   self.idle_q(sequencer, time=40)
                  #  self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)


    def cavity_drive_ef_ramsey_mixedtones(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['pi_len']) , pulse_type='square')
                # this idle appears to be optional
                self.idle_q(sequencer, time=ramsey_len + 000)
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['half_pi_ef_len']+ramsey_len+self.pulse_info[qubit_id]['half_pi_ef_len']), pulse_type='square')
                self.idle_q(sequencer, time=ramsey_len)
                # sequencer.append_idle_to_time('cavity1_I', ramsey_len)
                # sequencer.append_idle_to_time('cavity1_Q', ramsey_len)
                # sequencer.append_idle_to_time('cavity1_I', ramsey_len + 2150 + 70 + 5)
                # sequencer.append_idle_to_time('cavity1_Q', ramsey_len + 2150 + 70 + 5)
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                # Check if need the two sequencer.append bits below
                sequencer.append_idle_to_time('cavity1_I', ramsey_len+2150)
                sequencer.append_idle_to_time('cavity1_Q', ramsey_len+2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['half_pi_ef_len']), pulse_type='square')
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                               len=(self.pulse_info[qubit_id]['pi_len']), pulse_type='square')
                    # self.idle_q(sequencer, time=(self.pulse_info[qubit_id]['pi_len'] + 70))
                if self.expt_cfg['ef_pi_for_cal']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                               len=(self.pulse_info[qubit_id]['pi_ef_len']), pulse_type='square')
                    # self.idle_q(sequencer, time=(self.pulse_info[qubit_id]['pi_ef_len'] + 70))

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)

    def cavity_drive_echo_mixedtones(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['half_pi_len']) * 4 + 70, pulse_type='gauss')
                for echo_id in range(self.expt_cfg['echo_times']):
                    self.idle_q(sequencer, time=ramsey_len/(float(2*self.expt_cfg['echo_times'])))
                    sequencer.append_idle_to_time('cavity1_I', ramsey_len/(float(2*self.expt_cfg['echo_times'])) + 2150 + 70 + 5)
                    sequencer.append_idle_to_time('cavity1_Q', ramsey_len/(float(2*self.expt_cfg['echo_times'])) + 2150 + 70 + 5)
                    if self.expt_cfg['cp']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    elif self.expt_cfg['cpmg']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi/2.0)
                    self.idle_q(sequencer, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                    sequencer.append_idle_to_time('cavity1_I',
                                                  ramsey_len / (float(2 * self.expt_cfg['echo_times'])) + 2150 + 70 + 5)
                    sequencer.append_idle_to_time('cavity1_Q',
                                                  ramsey_len / (float(2 * self.expt_cfg['echo_times'])) + 2150 + 70 + 5)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=2 * np.pi * ramsey_len*self.expt_cfg['ramsey_freq'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=(self.pulse_info[qubit_id]['half_pi_len']) * 4 + 70, pulse_type='gauss')

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
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


    def amplitude_rabi(self, sequencer):

        for rabi_amp in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                #self.gen_q(sequencer,qubit_id,len=self.pulse_info['1']['pi_len'],amp = rabi_amp,phase=0,pulse_type=self.expt_cfg['pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['qubit_pulse_len'], amp=rabi_amp, phase=0,
                           pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

        # for rabi_amp in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
        #
        #     for rabi_len in np.arange(self.expt_cfg['starttime'], self.expt_cfg['stoptime'], self.expt_cfg['steptime']):
        #         sequencer.new_sequence(self)
        #         self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)
        #
        #         for qubit_id in self.expt_cfg['on_qubits']:
        #             self.gen_q(sequencer, qubit_id, len=rabi_len, amp=rabi_amp, phase=0,
        #                pulse_type=self.expt_cfg['pulse_type'])
        #         self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        #         sequencer.end_sequence()
        #
        # return sequencer.complete(self, plot=True)

    def cavity_drive_rabi_twotone(self, sequencer):
        # This should have been named _mixedtones but that didn't happen and now something else is named that.
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001, len=rabi_len)
                # Times 4 accommodates the fact that gen_c (defaulted to square pulse type, ok if gauss) is too short and cuts off the Rabi pulse
                # Explicit pi calibration below:
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #     sequencer.append_idle_to_time('cavity1_I', 2150)
                #     sequencer.append_idle_to_time('cavity1_Q', 2150)
                #     #changed idles from 2150
                #     self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,len=self.pulse_info[qubit_id]['pi_len']*4)
            sequencer.sync_channels_time(self.channels)
            sequencer.append_idle_to_time('readout_trig', rabi_len + 2230)
            sequencer.append_idle_to_time('readout', rabi_len + 2230)
            # added idles while troubleshooting overlap of readout pulse with rabi, dec 12 2019

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
# change this back to .complete
        return sequencer.complete_mixedtones(self, plot=True)

    def cavity_drive_ef_rabi_mixedtones(self, sequencer):
        # JUST A TEST FUNCTION DESPITE THE MISLEADING NAME
        # cavity_drive_ef_rabi_testing_mixedtones is the good one
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=rabi_len, phase=0,
                           pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                           add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=self.expt_cfg['stop'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)

#The below currently doesn't work. Must make updates to post-experiment analysis
    def cavity_drive_ef_rabi_testing_mixedtones(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                              len=self.pulse_info[qubit_id]['pi_len'])
                    # removed *4
                # sequencer.append_idle_to_time('cavity1_I', 2150)
                # sequencer.append_idle_to_time('cavity1_Q', 2150)
                #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['pi_len']+000)
                # plus 220 on the length seems to make it do ok, in terms of pulse separation
                # rabi osc are taller when no additional lag, plus this matches OG ef_rabi, so leaving it
                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=rabi_len, phase=0,
                           pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                           add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                           len=rabi_len)
                # could make this expt cfg stop and did indeed for a while
                #gen_c and gen_q both default to square so you don't have to stretch out the gen_c as much

                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                                    len=self.pulse_info[qubit_id]['pi_len'])
                    #remvoved *4
                if self.expt_cfg['ef_pi_for_cal']:
                    #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['pi_len'] + 000)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                                    len=self.pulse_info[qubit_id]['pi_ef_len'])
            sequencer.sync_channels_time(self.channels)
            sequencer.append_idle_to_time('readout_trig', rabi_len + 2360)
            sequencer.append_idle_to_time('readout', rabi_len + 2360)
            # added these three lines above to test stuff dec 16
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)

 #   if self.expt_cfg['pi_calibration']:
 #       sequencer.sync_channels_time(self.channels)
 #       self.idle_q_sb(sequencer, qubit_id, time=500)
 #       self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
 #       sequencer.append_idle_to_time('cavity1_I', 2150)
 #       sequencer.append_idle_to_time('cavity1_Q', 2150)
 #       self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
  #                 len=self.pulse_info[qubit_id]['pi_len'] * 4)

#    def cavity_drive_rabi_twotone(self, sequencer):

#        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
#            sequencer.new_sequence(self)
#            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

#            for qubit_id in self.expt_cfg['on_qubits']:

                #self.idle_q(sequencer, time=rabi_len)
                #self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
                #
#                self.gen_q(sequencer, qubit_id, len=rabi_len, amp=self.expt_cfg['amp'], phase=0, pulse_type=self.expt_cfg['pulse_type'])
 #               sequencer.append_idle_to_time('cavity1_I', 2150)
 #               sequencer.append_idle_to_time('cavity1_Q', 2150)
 #               self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
#                                     len=rabi_len)
                #self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                 #          len=rabi_len)
               # self.readout_pxi_onqubittone(sequencer, self.expt_cfg['on_qubits'], overlap=True, len=rabi_len)
                #self.idle_q(sequencer, time=400)


 #           self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)
 #           sequencer.end_sequence()

 #       return sequencer.complete(self, plot=True)


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

    def pulse_probe_ef_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer, qubit_id, amp = self.expt_cfg['amp'],len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+self.quantum_device_cfg['qubit']['1']['anharmonicity'])
            if self.expt_cfg['pi_calibration']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef_pi_for_cal']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

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
                self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef_pi_for_cal']:
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
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['ef_pi_for_cal']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])


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
                if self.expt_cfg['ef_pi_for_cal']:
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
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def gf_ramsey(self, sequencer):
        # gf ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

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

    # Written by Meg who thinks she deleted it to write cavity_drive_histogram_mixedtones for some unknown, mistaken reason
    def histogram(self, sequencer):
        for ii in range(self.expt_cfg['num_seq_sets']):

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=qubit_id, time=500)
                self.readout_pxi(sequencer, qubit_id)
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

    def cavity_drive_histogram_mixedtones(self, sequencer):
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
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                               len=self.pulse_info[qubit_id]['pi_len']*2)
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()
                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                               len=self.pulse_info[qubit_id]['pi_len']*2)
                    # Added this idle_q to mirror the ef rabi that seems to be working okay but needs the idle
                    self.idle_q(sequencer, time=self.pulse_info[qubit_id]['pi_len'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.append_idle_to_time('cavity1_I', 2150)
                    sequencer.append_idle_to_time('cavity1_Q', 2150)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.0001,
                               len=self.pulse_info[qubit_id]['pi_ef_len']*2)
                    # could remove the *4s here if pulse types are all square
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()
        return sequencer.complete_mixedtones(self, plot=False)

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
                    dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                elif self.expt_cfg['h0e1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_dc_offset']
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
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
                if self.expt_cfg['f0g1']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                elif self.expt_cfg['h0e1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_dc_offset']
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                             Square(max_amp=self.expt_cfg['amp'], flat_len=length, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                    fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                    dc_offset=dc_offset,plot=False))
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
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer,qubit_id,time=length + 40.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',mode_index=self.expt_cfg['mode_index'])
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

                if self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase']:phase_freq = self.expt_cfg['ramsey_freq']
                else:phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][self.expt_cfg['mode_index']] + self.expt_cfg['ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=ramsey_len+50.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = 2*np.pi*ramsey_len*phase_freq-offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_pi_pi_offset(self, sequencer):
        # sideband rabi time domain

        for offset_phase2 in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase2/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = -offset_phase2/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ge_calibration(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['test_parity']:
                    add_phase = np.pi
                    phase_freq = 0
                else:
                    add_phase = 0
                    phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                # self.idle_q_sb(sequencer, qubit_id, time=10.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, qubit_id, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = add_phase+2*np.pi*ramsey_len*phase_freq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ef_calibration(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

            for qubit_id in self.expt_cfg['on_qubits']:

                phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
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

                phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_f'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                # self.idle_q_sb(sequencer, qubit_id, time=10.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
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
            # self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
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
        # Hack for using tek2 to do the number splitting experiment.

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

    def cavity_drive_direct_spectroscopy(self, sequencer):
        # Direct cavity spectroscopy by monitoring the readout resonator
        # Completely rewritten by MGP on Jan. 22, 2019 because the old one wasn't working, sorry :(
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            #for qubit_id in self.expt_cfg['on_qubits']:
            self.pad_start_pxi_tek2(sequencer, on_qubits = self.expt_cfg['on_qubits'])
            freqvar = self.cavity_pulse_info['1']['iq_freq'] + dfreq
            #self.gen_c(sequencer, cavity_id= '1', len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'], pulse_type = self.expt_cfg['cavity_pulse_type'], add_freq= dfreq)
                # code doesn't like the add_freq in gen_c for some reason -- ohhh, it doesn't like the adding

                #cavity_id = self.expt_cfg['on_cavities']
            sequencer.append('cavity%s_I' % '1', Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=freqvar, phase=0))
            sequencer.append('cavity%s_Q' % '1', Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=freqvar, phase=0 + self.cavity_pulse_info['1']['Q_phase']))

            # sequencer.append('cavity%s_I' % '1', Gauss(max_amp=self.expt_cfg['amp'], sigma_len=self.expt_cfg['pulse_length'],
            #                                                  cutoff_sigma=2, freq=freqvar,
            #                                                  phase=0))
            # sequencer.append('cavity%s_Q' % '1', Gauss(max_amp=self.expt_cfg['amp'], sigma_len=self.expt_cfg['pulse_length'],
            #                                                  cutoff_sigma=2, freq=freqvar,
            #                                                  phase=0 + self.cavity_pulse_info['1'][
            #                                                      'Q_phase']))


                # maybe drop a delay in here so that you pulse the cavity before you have the qubit do anything
            #self.idle_q(sequencer, time=self.expt_cfg['idle_time'])
                # hopefully time units are in ns

                # watch this to make sure that it is actually working
            self.idle_q(sequencer, time=self.expt_cfg['idle_time'])
            if not self.expt_cfg['overlap']:
                sequencer.sync_channels_time(self.channels)
                #self.idle_q(sequencer, time=self.expt_cfg['pulse_length'])
            qubitpulselength = self.expt_cfg['qubit_pulse']
            self.gen_q(sequencer, len = qubitpulselength, amp = self.expt_cfg['qubit_amp'], pulse_type = self.expt_cfg['qubit_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_pulse_probe_iq(self, sequencer):
        # Cavity spectroscopy by monitoring the qubit peak
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

                #self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['cavity_pulse_len'])
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['cavity_pulse_len'], pulse_type = 'gauss')
                self.idle_q(sequencer, time=self.expt_cfg['idle_time'])
                if not self.expt_cfg['overlap']:
                    sequencer.sync_channels_time(self.channels)

                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['qubit_amp'], len=self.expt_cfg['qubit_pulse_len'],
                           phase=0, pulse_type=self.expt_cfg['pulse_type'], add_freq=dfreq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_pulse_probe_iq_signalcore(self, sequencer):
        # Cavity spectroscopy by monitoring the qubit peak
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

                #self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['cavity_pulse_len'])
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['cavity_pulse_len'], pulse_type = 'gauss')
                self.idle_q(sequencer, time=self.expt_cfg['idle_time'])
                if not self.expt_cfg['overlap']:
                    sequencer.sync_channels_time(self.channels)

                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['qubit_amp'], len=self.expt_cfg['qubit_pulse_len'],
                           phase=0, pulse_type=self.expt_cfg['pulse_type'], add_freq=dfreq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_pulse_probe_iq_wf0g1(self, sequencer):
        # Cavity spectroscopy by monitoring the qubit peak
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])

                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['cavity_pulse_len'])
                if not self.expt_cfg['overlap']:
                    sequencer.sync_channels_time(self.channels)

                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['qubit_amp'], len=self.expt_cfg['qubit_pulse_len'],
                           phase=0, pulse_type=self.expt_cfg['pulse_type'], add_freq=dfreq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_sideband_rabi_freq_scan(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)


            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['pulse_length'], pulse_type=self.expt_cfg['cavity_pulse_type'],add_freq=dfreq)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=40)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

    def cavity_drive_lattice_swap(self, sequencer):

        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_cavity']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['pi_len1'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])


                self.idle_q(sequencer, time=len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def cavity_drive_lattice_pulse(self, sequencer):

        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_cavity']:
                self.gen_c(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=self.expt_cfg['pulse_length'], phase=0,
                           pulse_type=self.expt_cfg['pulse_type']
                           )



                self.idle_q(sequencer, time=len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_lattice_swap(self, sequencer):

        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['pi_len1'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)





                self.idle_q(sequencer, time=len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_rabi_freq_scan(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'],
                           amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['pulse_length'],
                           pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=dfreq)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



                #self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_rabi_freq_scan_mixedtones(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.idle_q_sb(sequencer, qubit_id, time=30)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_len'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_ef_len'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'],
                           amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['pulse_length'],
                           pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=dfreq+self.quantum_device_cfg['cavity']['1']['f0g1Delta'])
                self.gen_q(sequencer, qubit_id=self.expt_cfg['on_qubits'],
                           amp=0.0000001, len=self.expt_cfg['pulse_length'],
                           pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=dfreq)
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=100)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                #sequencer.append_idle_to_time('cavity1_I', 2155)
               # sequencer.append_idle_to_time('cavity1_Q', 2155)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_ef_len'])



                #self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)
        # there is never any calibration here so the _mixedtones is kind of overkill. Implemented for consistency.


    def cavity_sideband_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.expt_cfg['on_qubits'],time=500)


            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'], len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=40)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



                #self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'],
                           len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_rabi_mixedtones_withf0g1_fromfreqscan(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                # Check this idle
                #self.idle_q_sb(sequencer, qubit_id, time=30)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_len'])
                # altered this gen_c to include an additional add_freq. see if it works??
                # there's not a dfreq in this unlike in freq scan so we took that bit out

                # THE BELOW ARE TAKEN FROM FREQ SCAN
                # sequencer.sync_channels_time(self.channels)
                # self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'],
                #            amp=self.expt_cfg['cavity_amp'], len=self.expt_cfg['pulse_length'],
                #            pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=dfreq+self.quantum_device_cfg['cavity']['1']['f0g1Delta'])
                # self.gen_q(sequencer, qubit_id=self.expt_cfg['on_qubits'],
                #            amp=0.0000001, len=self.expt_cfg['pulse_length'],
                #            pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=dfreq)
                # sequencer.sync_channels_time(self.channels)
                # self.idle_q_sb(sequencer, qubit_id, time=100)
                # self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                # sync_channels has been added up here, whereas it's below the gen_c in cavity_drive_rabi_freq_scan_mixedtones -- ok?
                sequencer.sync_channels_time(self.channels)
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_ef_len'])

                # self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'],
                #            len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])
                # self.gen_q(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=.000001,
                #            len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])
                # copy pasted stuff below from the other code that works
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], len=rabi_len,
                           amp=self.expt_cfg['cavity_amp'], add_freq=self.quantum_device_cfg['cavity']['1']['f0g1Delta'] +self.quantum_device_cfg['cavity']['1']['f0g1AddFreq'] ,
                           pulse_type=self.expt_cfg['cavity_pulse_type'])
                self.gen_q(sequencer, qubit_id=self.expt_cfg['on_qubits'],
                           amp=0.000000001, len=rabi_len,
                           pulse_type=self.expt_cfg['cavity_pulse_type'])

                sequencer.sync_channels_time(self.channels)
                # Check this idle too
                self.idle_q_sb(sequencer, qubit_id, time=100)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.append_idle_to_time('cavity1_I', 2150)
                sequencer.append_idle_to_time('cavity1_Q', 2150)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=0.00000000001,
                           len=self.pulse_info[qubit_id]['pi_ef_len'])
                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete_mixedtones(self, plot=True)


    def cavity_drive_rabi_geramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'],
                           len=self.expt_cfg['pi_len'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])


                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_sideband_geramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['swap']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp'],
                          len=self.expt_cfg['pi_len'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)

                self.idle_q_sb(sequencer, qubit_id, time=5)

                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['sideband_amp'],add_freq=self.expt_cfg['detuning'],
                           len=self.quantum_device_cfg['pulse_info']['1']['pi_len']+ramsey_len+100, pulse_type=self.expt_cfg['sideband_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])


                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_twotonerabi_geramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['pi_len1'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])


                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_twotonerabi_correlation1(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                            len=self.expt_cfg['pi_len3'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=2)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])


                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_twotonerabi_bellstate_ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['pi_len1'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])




                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.idle_q_sb(sequencer, qubit_id, time=5)

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])




                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_twotonerabi_bellstate_pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['pi_len1']+200, pulse_type=self.expt_cfg['cavity_pulse_type'])
                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=5)

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





                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_f0g1_pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                            len=self.expt_cfg['pi_len2'], pulse_type=self.expt_cfg['cavity_pulse_type'])

                self.idle_q_sb(sequencer, qubit_id, time=5)


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





                #self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_drive_twotones(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.expt_cfg['on_qubits'], time=500)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=self.expt_cfg['cavity1pitime'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c2(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp2'],
                           len=self.expt_cfg['cavity2pitime'], pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)

                self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['cavity_amp1'],
                           len=rabi_len, pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)

                self.idle_q_sb(sequencer, qubit_id, time=5)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                # self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                # if self.expt_cfg['pi_calibration']:
                #     self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # if self.expt_cfg['ef_pi_for_cal']:
                #     self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def wigner_tomography_test_cavity_drive_sideband(self, sequencer):
        # wigner tomography with both cavity drive from PXI  and sideband drive from tek2
        for amp in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)

                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)

                elif self.expt_cfg['state'] == 'coherent_state':
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['prep_cav_amp'],
                          len=self.expt_cfg['prep_cav_len'])
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0':
                    pass

                self.wigner_tomography(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = amp,phase=self.expt_cfg['phase'],len = self.expt_cfg['cavity_pulse_len'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def direct_cavity_sideband_spectroscopy(self, sequencer):
        # Hack for using tek2 to do the number splitting experiment.

        cavity_freq = self.expt_cfg['cavity_freq']
        cavity_pulse_length =  self.expt_cfg['cavity_pulse_length']
        wait_after_cavity_drive = self.expt_cfg['wait_after_cavity_drive']

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:

                sequencer.append('sideband',
                                 Square(max_amp= self.expt_cfg['cavity_drive_amp'], flat_len=cavity_pulse_length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=cavity_freq + dfreq, phase=0,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=wait_after_cavity_drive)
                self.gen_q(sequencer, qubit_id, amp = self.expt_cfg['qubit_drive_amp'],len=self.expt_cfg['pulse_length'], phase=0, pulse_type=self.expt_cfg['pulse_type'],add_freq=0)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def wigner_tomography_test_sideband_only(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == 'coherent_state':
                    self.gen_c(sequencer, cavity_id=self.expt_cfg['on_cavities'], amp=self.expt_cfg['prep_cav_amp'],
                          len=self.expt_cfg['prep_cav_len'])
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0':
                    pass
                self.idle_q_sb(sequencer, qubit_id, time=50)
                if self.expt_cfg['sweep_phase']:self.wigner_tomography_sideband_only(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'])
                else:self.wigner_tomography_sideband_only(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = x,phase=self.expt_cfg['phase'],len = self.expt_cfg['cavity_pulse_len'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def wigner_tomography_test_sideband_one_pulse(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2, and generated using a single double square pulse
        for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    # self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    # sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    # self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    # sequencer.sync_channels_time(self.channels)

                elif self.expt_cfg['state'] == '0':
                    pass

                if self.expt_cfg['sweep_phase']:
                    self.wigner_tomography_swap_and_cavity_drive(sequencer, qubit_id, cavity_id=self.expt_cfg['on_cavities'],
                                                         amp2=self.expt_cfg['amp'], phase1=0,phase2=x,
                                                         len2=self.expt_cfg['cavity_pulse_len'],delay = self.expt_cfg['delay'],doubling_trick=self.expt_cfg['doubling_trick'],fix_phase=self.expt_cfg['fix_phase'])
                else:
                    self.wigner_tomography_swap_and_cavity_drive(sequencer, qubit_id, cavity_id=self.expt_cfg['on_cavities'],
                                                         amp2=x,phase1=0, phase2=self.expt_cfg['phase'],
                                                         len2=self.expt_cfg['cavity_pulse_len'],delay = self.expt_cfg['delay'],doubling_trick=self.expt_cfg['doubling_trick'],fix_phase=self.expt_cfg['fix_phase'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_transmon_ge_rabi(self, sequencer):
        # transmon ge rabi with tek2
        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.sideband_gen_q(sequencer, qubit_id, len=len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=0)
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_transmon_ge_ramsey(self, sequencer):
        # transmon ge rabi with tek2
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.sideband_half_pi_q(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.idle_sb(sequencer, time=ramsey_len)
                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_transmon_ef_rabi(self, sequencer):
        # transmon ef rabi with tek2
        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.idle_sb(sequencer, time  = 5.0)
                self.sideband_gen_q(sequencer, qubit_id, len=len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                    add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                # self.idle_sb(sequencer,time = 10.0)
                if self.expt_cfg['pi_calibration']:
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_transmon_pulse_probe_ef(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.idle_sb(sequencer, time  = 5.0)
                self.sideband_gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                    add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] + df)
                if self.expt_cfg['pi_calibration']:
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_transmon_pulse_probe_ge(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['add_photon']:
                    offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                    self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                self.sideband_gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                    add_freq= df)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_f0g1rabi_freq_scan(self,sequencer):
        sideband_freq = self.expt_cfg['freq']
        length = self.expt_cfg['length']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                self.idle_sb(sequencer, time=5.0)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset, plot=False))


                self.idle_sb(sequencer,time=40)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_fnm1gnrabi_freq_scan(self,sequencer):
        sideband_freq = self.expt_cfg['freq']
        length = self.expt_cfg['length']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                self.sideband_pi_q(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                for nn in range(self.expt_cfg['n']-1):
                    self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0, n=nn)
                    if self.expt_cfg['numbershift_qdrive_freqs']:
                        add_freq_ge = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                        add_freq_ef = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef']
                    else:add_freq_ge,add_freq_ef=0.0,0.0
                    self.sideband_pi_q(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],add_freq=add_freq_ge)
                    self.sideband_pi_q_ef(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'],add_freq=add_freq_ef)
                self.idle_sb(sequencer, time=5.0)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset, plot=False))
                self.idle_sb(sequencer,time=40)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_f0g1rabi(self, sequencer):
        # sideband rabi time domain
        sideband_freq = self.expt_cfg['freq']
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                self.idle_sb(sequencer, time=5.0)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))

                self.idle_sb(sequencer,time=5.0)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])



            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_fnm1gnrabi(self, sequencer):
        # sideband rabi time domain
        sideband_freq = self.expt_cfg['freq']
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                for nn in range(self.expt_cfg['n']-1):
                    self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0, n=nn+1)
                    if self.expt_cfg['numbershift_qdrive_freqs']:
                        add_freq_ge = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                        add_freq_ef = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef']
                    else:add_freq_ge,add_freq_ef=0.0,0.0
                    self.sideband_pi_q(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],add_freq=add_freq_ge)
                    self.sideband_pi_q_ef(sequencer, qubit_id,pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'],add_freq=add_freq_ef)
                self.idle_sb(sequencer, time=5.0)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                self.idle_sb(sequencer,time=5.0)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_f0g1ramsey(self, sequencer):
        # sideband rabi time domain

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase']:phase_freq = self.expt_cfg['ramsey_freq']
                else:phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'] + self.expt_cfg['ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_sb(sequencer,time=ramsey_len+50.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = 2*np.pi*ramsey_len*phase_freq-offset_phase/2.0)
                self.idle_q_sb(sequencer, qubit_id, time=10.0)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_f0g1_pi_pi_offset(self, sequencer):
        # sideband rabi time domain

        for ph in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = ph/2.0)
                sequencer.sync_channels_time(self.channels)
                self.idle_sb(sequencer,time=50.0)
                self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = -ph/2.0)
                self.idle_q_sb(sequencer, qubit_id, time=10.0)
                self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ge_calibration_alltek2(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['test_parity']:
                    add_phase = np.pi
                    phase_freq = 0
                else:
                    add_phase = 0
                    phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'] + self.expt_cfg[
                        'ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']
                if self.expt_cfg['add_photon']:
                    self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer,time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0)
                self.idle_sb(sequencer,time=15.0)
                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.idle_sb(sequencer, time=ramsey_len)
                self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],phase = add_phase+2*np.pi*ramsey_len*phase_freq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_cavity_spectroscopy_alltek2(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['excite_qubit']:
                    self.sideband_pi_q(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])

                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['prep_cav_amp'] / 2.0,
                                        flat_len=self.expt_cfg['prep_cav_len'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2,
                                        freq=self.cavity_freq[self.expt_cfg['on_cavities']] + df,
                                        phase=self.expt_cfg['prep_cav_phase'],
                                        plot=False))
                if self.expt_cfg['excite_qubit']:
                    self.sideband_pi_q(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                self.sideband_pi_q_resolved(sequencer, qubit_id,
                                        pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_number_splitting_pulse_probe_ge_tek2(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['state'] == '1':
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id,
                                          pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0+1':
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id,
                                          pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                elif self.expt_cfg['state'] == 'alpha':
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == '0+alpha':
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'] / 2.0,
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'] + np.pi,
                                            plot=False))


                elif self.expt_cfg['state'] == '0':
                    pass


                self.sideband_gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                    add_freq= df)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_photon_number_distribution_tek2(self, sequencer):
        # transmon ef rabi with tek2
        for n in np.arange(self.expt_cfg['N_max']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['state'] == '1':
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id,
                                          pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0+1':
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id,
                                          pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                elif self.expt_cfg['state'] == 'alpha':
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == '0+alpha':
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.sideband_half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'],
                                            plot=False))
                    self.sideband_pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'] / 2.0,
                                            flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2,
                                            freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                            phase=self.expt_cfg['prep_cav_phase'] + np.pi,
                                            plot=False))


                elif self.expt_cfg['state'] == '0':
                    pass

                self.sideband_pi_q_resolved(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'],add_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']*n)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def wigner_tomography_test_sideband_alltek2(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                if self.expt_cfg['state'] == '1':
                    self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['state'] == '0+1':
                    self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                    self.idle_sb(sequencer, time=5.0)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                elif self.expt_cfg['state'] == 'alpha':
                    sequencer.append('sideband',
                                     Square(max_amp=self.expt_cfg['prep_cav_amp'], flat_len=self.expt_cfg['prep_cav_len'],
                                            ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[cavity_id],
                                            phase=phase,
                                            plot=False))
                elif self.expt_cfg['state'] == '0':
                    pass
                self.idle_sb(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:self.wigner_tomography_alltek2(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'])
                else:self.wigner_tomography_alltek2(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = x,phase=self.expt_cfg['phase'],len = self.expt_cfg['cavity_pulse_len'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def wigner_tomography_2d_sideband_alltek2(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']
        for y in (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0):
            for x in (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0):

                tom_amp = np.sqrt(x**2+y**2)
                tom_phase = np.arctan2(y,x)
                sequencer.new_sequence(self)
                self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.sync_channels_time(self.channels)
                for qubit_id in self.expt_cfg['on_qubits']:
                    offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset']

                    if self.expt_cfg['state'] == "1":
                        self.sideband_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                        self.idle_sb(sequencer, time=5.0)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['state'] == "0+1":
                        self.sideband_half_pi_q(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        self.sideband_pi_q_ef(sequencer, qubit_id, pulse_type=self.transmon_pulse_info_tek2[qubit_id]['ef_pulse_type'])
                        self.idle_sb(sequencer, time=5.0)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                    elif self.expt_cfg['state'] == "alpha":
                        print ("goes_here")
                        sequencer.append('sideband',
                                         Square(max_amp=self.expt_cfg['prep_cav_amp'], flat_len=self.expt_cfg['prep_cav_len'],
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                                phase=self.expt_cfg['prep_cav_phase'],
                                                plot=False))
                    elif self.expt_cfg['state'] == "cavity_drive_res_check":
                        self.sideband_pi_q(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        sequencer.append('sideband',
                                         Square(max_amp=self.expt_cfg['prep_cav_amp'], flat_len=self.expt_cfg['prep_cav_len'],
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                                phase=self.expt_cfg['prep_cav_phase'],
                                                plot=False))
                        self.sideband_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    elif self.expt_cfg['state'] == "0+alpha":
                        self.sideband_half_pi_q(sequencer, qubit_id,
                                                    pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        sequencer.append('sideband',
                                         Square(max_amp=self.expt_cfg['prep_cav_amp'], flat_len=self.expt_cfg['prep_cav_len'],
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                                phase=self.expt_cfg['prep_cav_phase'],
                                                plot=False))
                        self.sideband_pi_q_resolved(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    elif self.expt_cfg['state'] == "alpha+-alpha":
                        self.sideband_half_pi_q(sequencer, qubit_id,
                                                pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        sequencer.append('sideband',
                                         Square(max_amp=self.expt_cfg['prep_cav_amp'],
                                                flat_len=self.expt_cfg['prep_cav_len'],
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                    'ramp_sigma_len'], cutoff_sigma=2,
                                                freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                                phase=self.expt_cfg['prep_cav_phase'],
                                                plot=False))
                        self.sideband_pi_q_resolved(sequencer, qubit_id,
                                                    pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        sequencer.append('sideband',
                                         Square(max_amp=self.expt_cfg['prep_cav_amp']/2.0, flat_len=self.expt_cfg['prep_cav_len'],
                                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                                    'ramp_sigma_len'], cutoff_sigma=2, freq=self.cavity_freq[self.expt_cfg['on_cavities']],
                                                phase=self.expt_cfg['prep_cav_phase']+np.pi,
                                                plot=False))
                    elif self.expt_cfg['state'] == 'fock_n':
                        self.prep_fock_n(sequencer, qubit_id=qubit_id, fock_state=self.expt_cfg['n'], nshift=self.expt_cfg['numbershifr_qdrive_freqs'])
                    elif self.expt_cfg['state'] == '0':
                        pass
                    self.idle_sb(sequencer, time=5.0)
                    if self.expt_cfg['sweep_phase']:self.wigner_tomography_alltek2(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'])
                    else:self.wigner_tomography_alltek2(sequencer, qubit_id,cavity_id=self.expt_cfg['on_cavities'],amp = tom_amp,phase=tom_phase,len = self.expt_cfg['cavity_pulse_len'])

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

        multiple_sequences = eval('self.' + experiment)(sequencer)

        try:
            if kwargs['save']: self.save_sequences(multiple_sequences,kwargs['filename'])
        except:pass

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

    def save_sequences(self,multiple_sequences,filename = 'test'):
        seq_num = len(multiple_sequences)

        channel_waveforms = []
        for channel in self.channels:
            channel_waveform = []
            for seq_id in range(seq_num):
                channel_waveform.append(multiple_sequences[seq_id][channel])
            channel_waveforms.append(channel_waveform)

        np.save(filename,np.array(channel_waveforms))

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