try:
    from .sequencer_pxi_StimEm import Sequencer
    from .pulse_classes_StimEm import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone,Double_Square, ARB, ARB_Sum,\
        Square_multitone,Gauss_multitone,Square_with_blockade,Gauss_with_blockade, ARB_with_blockade, ARB_Sum_with_blockade,\
        Square_multitone_with_blockade, Gauss_multitone_with_blockade
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a, ARB, ARB_Sum

import numpy as np
import visdom
import os
import pickle

import copy

from h5py import File
from scipy import interpolate
from itertools import combinations

class PulseSequences:
    # channels and awgs

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.pulse_info = self.quantum_device_cfg['pulse_info']
        self.sideband_pulse_info = self.quantum_device_cfg['sideband_iq_pulse_info']['1']
        self.multimodes = self.quantum_device_cfg['flux_pulse_info']

        try:self.cavity_pulse_info = self.quantum_device_cfg['cavity_pulse_info']
        except:print("No cavity pulses")

        try:self.cavity_freq = self.multimodes['1']['cavity_freqs']
        except:pass

        try:self.transmon_pulse_info_tek2 = self.quantum_device_cfg['transmon_pulse_info_tek2']
        except:print("No tek2 transmon drive pulse information")

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']

        # pulse params
        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']}

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=True):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg)
        self.plot_visdom = plot_visdom

    def gen_q(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_freq[qubit_id]+add_freq,
                                    phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            
    def gen_q_multitone(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freqs = np.array([0]),phases = np.array([0]),pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freqs=self.qubit_freq[qubit_id]+np.array(add_freqs),
                                    phases=phases))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freqs=self.qubit_freq[qubit_id]+np.array(add_freqs)
                                                                    ,phase=phases))

    def gen_q_weak(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_freq[qubit_id]+add_freq,
                                    phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def gen_q_weak_multitone(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freqs = np.array([0]),phases = np.array([0]),pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square_multitone(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freqs=self.qubit_freq[qubit_id]+np.array(add_freqs),
                                    phases=phases))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss_multitone(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freqs=self.qubit_freq[qubit_id]+np.array(add_freqs),phases=phases))

    def gen_q_and_blockade(self, sequencer, qubit_id='1', len=10, amp=1, add_freq=0, phase=0, pulse_type="square",
                                blockade_amp=1.0, blockade_freqs=None, blockade_pulse_type="square",
                                use_transfer=False):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + add_freq
        if pulse_type.lower() == "square":
            sequencer.append("charge" + qubit_id, Square_with_blockade(max_amp=amp, flat_len=len, ramp_sigma_len=0.001,
                                                               cutoff_sigma=2, freq=freq, phase=phase,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == "gauss":
            sequencer.append("charge" + qubit_id, Gauss_with_blockade(max_amp=amp, sigma_len=len,
                                                              cutoff_sigma=2, freq=freq, phase=phase,
                                                              blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                              blockade_pulse_type=blockade_pulse_type))

    def gen_q_and_blockade_weak(self, sequencer, qubit_id = '1', len = 10, amp=1, add_freq=0, phase=0, pulse_type="square",
                                blockade_amp=1.0, blockade_freqs=None, blockade_pulse_type="square", use_transfer=False):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + add_freq
        if pulse_type.lower() == "square":
            sequencer.append("qubitweak", Square_with_blockade(max_amp=amp, flat_len=len, ramp_sigma_len=0.001,
                                                               cutoff_sigma=2, freq=freq, phase=phase,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == "gauss":
            sequencer.append("qubitweak", Gauss_with_blockade(max_amp=amp, sigma_len=len,
                                                               cutoff_sigma=2, freq=freq, phase=phase,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))

    def gen_q_and_blockade_multitone(self, sequencer, qubit_id='1', len=10, amp=1, add_freqs=0, phases=0, pulse_type="square",
                                blockade_amp=1.0, blockade_freqs=None, blockade_pulse_type="square",
                                use_transfer=False):
        freqs = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + add_freqs
        if pulse_type.lower() == "square":
            sequencer.append("charge" + qubit_id, Square_multitone_with_blockade(max_amp=amp, flat_len=len, ramp_sigma_len=0.001,
                                                               cutoff_sigma=2, freq=freqs, phases=phases,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == "gauss":
            sequencer.append("charge" + qubit_id, Gauss_multitone_with_blockade(max_amp=amp, sigma_len=len,
                                                              cutoff_sigma=2, freq=freqs, phases=phases,
                                                              blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                              blockade_pulse_type=blockade_pulse_type))

    def gen_q_and_blockade_weak_multitone(self, sequencer, qubit_id = '1', len = 10, amp=1, add_freqs=0, phases=0, pulse_type="square",
                                blockade_amp=1.0, blockade_freqs=None, blockade_pulse_type="square", use_transfer=False):
        freqs = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + np.array(add_freqs)
        if pulse_type.lower() == "square":
            sequencer.append("qubitweak", Square_multitone_with_blockade(max_amp=amp, flat_len=len, ramp_sigma_len=0.001,
                                                               cutoff_sigma=2, freqs=freqs, phases=phases,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == "gauss":
            sequencer.append("qubitweak", Gauss_multitone_with_blockade(max_amp=amp, sigma_len=len,
                                                               cutoff_sigma=2, freqs=freqs, phases=phases,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))

    def gen_sideband(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,iq_freq = 0.2,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('sideband_I', Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq+add_freq,
                                    phase=phase))
            sequencer.append('sideband_Q',
                         Square(max_amp=amp, flat_len= len,
                                ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq+add_freq,
                                phase=phase+self.sideband_pulse_info['Q_phase']))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('sideband_I', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=iq_freq+add_freq,phase=phase))
            sequencer.append('sideband_Q', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=iq_freq+add_freq,phase=phase+self.sideband_pulse_info['Q_phase']))


    def pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len=self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            if self.pulse_info[qubit_id]['use_drag_pi']:
                sequencer.append('charge%s' % qubit_id, DRAG(A=self.pulse_info[qubit_id]['pi_amp'], beta=self.pulse_info[qubit_id]['drag_beta_pi'],
                                                             sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                                                             freq=self.qubit_freq[qubit_id]+add_freq,
                                                             phase=phase))
            else:
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def half_pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            if self.pulse_info[qubit_id]['use_drag_half_pi']:
                sequencer.append('charge%s' % qubit_id, DRAG(A=self.pulse_info[qubit_id]['half_pi_amp'], beta=self.pulse_info[qubit_id]['drag_beta_half_pi'],
                                                             sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                                                             freq=self.qubit_freq[qubit_id]+add_freq,
                                                             phase=phase))
            else:
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def pi_q_ef(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['pi_ef_len'],cutoff_sigma=2,
                            freq=freq+add_freq,phase=phase))

    def half_pi_q_ef(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_ef_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],cutoff_sigma=2,
                            freq=freq+add_freq,phase=phase))

    def pi_q_fh(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity_fh']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'],
                                                           flat_len=self.pulse_info[qubit_id]['pi_fh_len'],
                                                           ramp_sigma_len=0.001, cutoff_sigma=2,
                                                           freq=freq+addfreq, phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_fh_amp'],
                                                          sigma_len=self.pulse_info[qubit_id]['pi_fh_len'],
                                                          cutoff_sigma=2,
                                                          freq=freq+add_freq, phase=phase))

    def half_pi_q_fh(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        freq = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + self.quantum_device_cfg['qubit'][qubit_id ]['anharmonicity']+self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity_fh']
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'],
                                                           flat_len=self.pulse_info[qubit_id]['half_pi_fh_len'],
                                                           ramp_sigma_len=0.001, cutoff_sigma=2,
                                                           freq=freq+add_freq, phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_fh_amp'],
                                                          sigma_len=self.pulse_info[qubit_id]['half_pi_fh_len'],
                                                          cutoff_sigma=2,
                                                          freq=freq+add_freq, phase=phase))

    def pi_q_resolved(self, sequencer, qubit_id='1', phase=0, add_freq = 0, use_weak_drive=False):
        if not use_weak_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'], flat_len=self.pulse_info[qubit_id]['pi_len_resolved'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'], sigma_len=self.pulse_info[qubit_id]['pi_len_resolved'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'], flat_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'], sigma_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def pi_q_resolved_ef(self, sequencer, qubit_id='1',phase=0,add_freq = 0,use_weak_drive=False):
        alpha = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']
        if not use_weak_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_ef']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_ef'],
                                                               flat_len=self.pulse_info[qubit_id]['pi_len_resolved_ef'],
                                                               ramp_sigma_len=0.001, cutoff_sigma=2,
                                                               freq=self.qubit_freq[qubit_id]+add_freq+alpha,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_ef'],
                                                              sigma_len=self.pulse_info[qubit_id]['pi_len_resolved_ef'],cutoff_sigma=2,
                                                              freq=self.qubit_freq[qubit_id]+add_freq+alpha,phase=phase))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak_ef']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak_ef'],
                                                     flat_len=self.pulse_info[qubit_id]['pi_len_resolved_weak_ef'],ramp_sigma_len=0.001,
                                                     cutoff_sigma=2,
                                                     freq=self.qubit_freq[qubit_id]+add_freq+alpha,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak_ef'],
                                                    sigma_len=self.pulse_info[qubit_id]['pi_len_resolved_weak_ef'],cutoff_sigma=2,
                                                    freq=self.qubit_freq[qubit_id]+add_freq+alpha,phase=phase))
                
    def half_pi_q_resolved(self, sequencer, qubit_id='1',phase=0,add_freq = 0,use_weak_drive=False):
        if not use_weak_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp_resolved'], flat_len=self.pulse_info[qubit_id]['half_pi_len_resolved'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp_resolved'], sigma_len=self.pulse_info[qubit_id]['half_pi_len_resolved'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp_resolved_weak'], flat_len=self.pulse_info[qubit_id]['half_pi_len_resolved_weak'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp_resolved_weak'], sigma_len=self.pulse_info[qubit_id]['half_pi_len_resolved_weak'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
                

    def two_pi_q_resolved(self, sequencer, qubit_id='1',phase=0,add_freq = 0,use_weak_drive=False):
        if not use_weak_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['2pi_amp_resolved'], flat_len=self.pulse_info[qubit_id]['2pi_len_resolved'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['2pi_amp_resolved'], sigma_len=self.pulse_info[qubit_id]['2pi_len_resolved'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square(max_amp=self.pulse_info[qubit_id]['2pi_amp_resolved_weak'], flat_len=self.pulse_info[qubit_id]['2pi_len_resolved_weak'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss(max_amp=self.pulse_info[qubit_id]['2pi_amp_resolved_weak'], sigma_len=self.pulse_info[qubit_id]['2pi_len_resolved_weak'],cutoff_sigma=2,
                                freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def pi_f0g1_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',add_freq = 0,mode_index = 0):
        self.gen_sideband(sequencer,qubit_id = qubit_id,len = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_len'][mode_index],amp = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'][mode_index],iq_freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_iq_freq'][mode_index],add_freq = add_freq,phase = phase,pulse_type = pulse_type)

    def pi_h0e1_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',add_freq = 0,mode_index = 0):
        self.gen_sideband(sequencer,qubit_id = qubit_id,len = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_h0e1_len'][mode_index],amp = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_h0e1_amp'][mode_index],iq_freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_iq_freq'][mode_index],add_freq = add_freq,phase = phase,pulse_type = pulse_type)

    def pi_fnm1gn_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',n=0, mode_index=0):
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_fnm1gn_amps'][mode_index][int(n-1)],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_fnm1gn_lens'][mode_index][int(n - 1)],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fnm1gn_freqs'][mode_index][int(n-1)], phase=phase,fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][mode_index],plot=False))

    def prep_fock_n(self,sequencer,qubit_id = '1',fock_state=1,nshift=True, offset_phase=0,mode_index=0):
        for nn in range(fock_state):
            if nshift:
                add_freq_ge = (nn) * 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e'][mode_index]
                add_freq_ef = (nn) * 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_ef'][mode_index]
            else:
                add_freq_ge, add_freq_ef = 0.0, 0.0
            self.pi_q(sequencer, qubit_id,
                               pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               add_freq=add_freq_ge)
            self.pi_q_ef(sequencer, qubit_id,
                                  pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                                  add_freq=add_freq_ef)
            sequencer.sync_channels_time(self.channels)
            self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase/2.0, n=nn+1, mode_index=0)
            sequencer.sync_channels_time(self.channels)

    def prep_fock_superposition(self,sequencer,qubit_id = '1',fock_state=1,nshift=True, offset_phase=0,mode_index=0):
        for nn in range(fock_state):
            if nshift:
                add_freq_ge = (nn) * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e'][mode_index]
                add_freq_ef = (nn) *self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_ef'][mode_index]
            else:
                add_freq_ge, add_freq_ef = 0.0, 0.0

            if nn == 0:self.half_pi_q(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'],add_freq=add_freq_ge)
            else:self.pi_q(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'],add_freq=add_freq_ge)
            self.pi_q_ef(sequencer, qubit_id,
                                  pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                                  add_freq=add_freq_ef)

            if nn != fock_state:self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'], add_freq=add_freq_ge)
            sequencer.sync_channels_time(self.channels)
            self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase/2.0, n=nn+1, mode_index=0)
            sequencer.sync_channels_time(self.channels)

    def idle_q(self,sequencer,qubit_id = '1',time=0):
        sequencer.append('charge%s' % qubit_id, Idle(time=time))

    def idle_q_sb(self, sequencer, qubit_id='1', time=0):
        sequencer.append('charge%s' % qubit_id, Idle(time=time))
        sequencer.append('sideband_I', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_all(self, sequencer, qubit_id='1', time=0):
        sequencer.sync_channels_time(self.channels)
        sequencer.append('charge%s' % qubit_id, Idle(time=time))
        sequencer.append('sideband_I', Idle(time=time))
        sequencer.append('sideband_Q', Idle(time=time))
        sequencer.append('cavity', Idle(time=time))
        sequencer.append('qubitweak', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_sb(self,sequencer,time=0.0):
        sequencer.append('sideband_I', Idle(time=time))
        sequencer.append('sideband_Q', Idle(time=time))

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
        readout_time = sequencer.get_time('readout_trig') # Earlier was alazar_tri
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
        # self.idle_q(sequencer, qubit_id, time=np.abs(1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0))
        self.idle_q(sequencer, qubit_id,time= self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'][self.expt_cfg['mode_index']])
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sequencer.sync_channels_time(self.channels)

    def parity_measurement_with_sideband(self, sequencer, qubit_id='1'):
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        # self.idle_q(sequencer, qubit_id, time=np.abs(1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0))
        self.idle_q(sequencer, qubit_id,time= self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'][self.expt_cfg['mode_index']])
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sideband_len = 2 * self.pulse_info[qubit_id]["half_pi_len"] + self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'][self.expt_cfg['mode_index']]
        self.gen_sideband(sequencer, qubit_id, len=sideband_len, amp=self.expt_cfg['chi_dressing_amp'], add_freq=0.0,
                          iq_freq=self.sideband_pulse_info['iq_freq'])
        sequencer.sync_channels_time(self.channels)

    def generalized_parity_measurement(self, sequencer, qubit_id='1', ramsey_time=0.0,add_phase=0.0):
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        # self.idle_q(sequencer, qubit_id, time=np.abs(1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0))
        self.idle_q(sequencer, qubit_id, time=ramsey_time)
        # print ("Ramsey time = ",ramsey_time)
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'], phase=np.pi+add_phase)
        sequencer.sync_channels_time(self.channels)

    def pi_resolved_joint_parity_measurement(self, sequencer, mode1=1, mode2=2, qubit_id='1', n_max=2, use_weak_drive=True, phases=[0],
                                             only_mode1 = False):
        nlist = np.arange(n_max + 1)
        add_freqs = []
        chi_ges = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
        if not only_mode1:
            for n1 in nlist:
                for n2 in nlist:
                    if (n1+n2)%2 == 1:
                        add_freqs.append(2 * chi_ges[mode1] * n1 + 2 * chi_ges[mode2] * n2)
        else:
            for n1 in nlist:
                if n1 % 2 == 1:
                    add_freqs.append(2 * chi_ges[mode1] * n1)
        add_freqs = np.array(add_freqs)
        phases = phases * len(add_freqs)

        if not use_weak_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                               flat_len=self.pulse_info[qubit_id]['pi_len_resolved'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                                               freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases,
                                                                         subtract_t0=True))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                                        sigma_len=self.pulse_info[qubit_id]['pi_len_resolved'],cutoff_sigma=2,
                                freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases, subtract_t0=True))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                     flat_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                                     freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases, subtract_t0=True))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                              sigma_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],cutoff_sigma=2,
                                freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases, subtract_t0=True))


    def gen_c(self, sequencer, mode_index = 0, len=10, amp=1, add_freq=0, phase=0, pulse_type='square', use_transfer=False):
        if use_transfer:
            amp = self.transfer_function_blockade(amp, channel='cavity_amp_vs_amp')
        if pulse_type.lower() == 'square':
            sequencer.append('cavity', Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_freq[mode_index] + add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_freq[mode_index] + add_freq,
                                                             phase=phase))

    def gen_c_weak(self, sequencer, mode_index = 0, len=10, amp=1, add_freq=0, phase=0, pulse_type='square', use_transfer=False):
        if use_transfer:
            amp = self.transfer_function_blockade(amp, channel='cavity_amp_vs_amp')
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_freq[mode_index] + add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_freq[mode_index] + add_freq,
                                                             phase=phase))

    def gen_c_multitone(self, sequencer, mode_indices = [0], len=10, amps=[1], add_freqs=[0], phases=[0], pulse_type='square'):
        freqs = []
        for mode_index in mode_indices:
            freqs.append(self.cavity_freq[mode_index] + add_freqs[mode_index])
        if pulse_type.lower() == 'square':
            sequencer.append('cavity', Square_multitone(max_amp=amps, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freqs=freqs,
                                    phases=phases))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity', Gauss_multitone(max_amp=amps, sigma_len=len,
                                                             cutoff_sigma=2,freqs=freqs,phases=phases))

    def gen_c_weak_multitone(self, sequencer, mode_indices = [0], len=10, amps=[1], add_freqs=[0], phases=[0], pulse_type='square'):
        freqs = []
        for mode_index in mode_indices:
            freqs.append(self.cavity_freq[mode_index] + add_freqs[mode_index])
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square_multitone(max_amp=amps, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freqs=freqs,
                                    phases=phases))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss_multitone(max_amp=amps, sigma_len=len,
                                                             cutoff_sigma=2,freqs=freqs,phases=phases))

    def gen_c_and_blockade_weak(self, sequencer, mode_index = 0, len=10, amp=1, add_freq=0, phase=0, pulse_type='square',
                                blockade_amp = 1.0,blockade_freqs = None,blockade_pulse_type='square',use_transfer=False):
        if use_transfer:
            amp = self.transfer_function_blockade(amp, channel='cavity_amp_vs_amp')
        if isinstance(mode_index, (int, float)):
            freqs = self.cavity_freq[mode_index] + add_freq
        else:
            freqs = []
            for i, mode in enumerate(mode_index):
                freqs.append(self.cavity_freq[index] + add_freq[i])
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square_with_blockade(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=freqs,phase=phase,
                                                               blockade_amp=blockade_amp,blockade_freqs=blockade_freqs, blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss_with_blockade(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=freqs,
                                                             phase=phase,blockade_amp=blockade_amp,blockade_freqs=blockade_freqs, blockade_pulse_type=blockade_pulse_type))

    def gen_c_multitone_and_blockade_weak(self, sequencer, mode_indices=[0], len=10, amps=[1], add_freqs=[0], phases=[0], pulse_type='square',
                                blockade_amp=1.0, blockade_freqs=None, blockade_pulse_type='square',
                                use_transfer=False):
        if use_transfer:
            amp = self.transfer_function_blockade(amp, channel='cavity_amp_vs_amp')
        freqs = []
        for mode_index in mode_indices:
            freqs.append(self.cavity_freq[mode_index] + add_freqs[mode_index])
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square_multitone_with_blockade(max_amp=amps, flat_len=len,
                                                               ramp_sigma_len=0.001, cutoff_sigma=2,
                                                               freqs=freqs, phases=phases,
                                                               blockade_amp=blockade_amp, blockade_freqs=blockade_freqs,
                                                               blockade_pulse_type=blockade_pulse_type))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss_multitone_with_blockade(max_amp=amps, sigma_len=len,
                                                              cutoff_sigma=2, freqs=freqs,
                                                              phases=phases, blockade_amp=blockade_amp,
                                                              blockade_freqs=blockade_freqs,
                                                              blockade_pulse_type=blockade_pulse_type))

    def half_pi_blockade_rotation(self, sequencer, qubit_id="1", mode_index=0, phase=0):
        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                    len=blockade_pulse_info['blockade_half_pi_length'][mode_index],
                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                        mode_index],
                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                        mode_index],
                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                    blockade_levels=[2], dressing_pulse_type="square",
                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                    phase=phase,
                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                        mode_index],
                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                        'use_weak_for_cavity'])
        return blockade_pulse_info['blockade_half_pi_length'][mode_index]

    def half_pi_multimode_blockade_rotation(self, sequencer, qubit_id="1", mode_indices=[0], phases=[0]):
        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
        # sort mode indices in increasing order of necessary blockade pulse length
        sort_list = np.argsort(np.array(blockade_pulse_info['blockade_half_pi_length'])[mode_indices])
        mode_indices = np.array(mode_indices)[sort_list]
        phases = np.array(phases)[sort_list]
        # input all blockades for their correct lengths
        current_len = 0
        orig_phases = phases
        for j in range(len(mode_indices)):
            self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                  blockade_mode_indices=mode_indices,
                                                  cavity_drive_mode_indices=mode_indices,
                                                  length=blockade_pulse_info['blockade_half_pi_length'][mode_indices[0]] - current_len,
                                                  cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_indices[0]],
                                                  use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_indices[0]],
                                                  dressing_amps=np.array(blockade_pulse_info['blockade_pi_amp_qubit'])[mode_indices],
                                                  blockade_levels=[[2] for i in range(15)],
                                                  dressing_pulse_type="square",
                                                  cavity_amps=np.array(blockade_pulse_info['blockade_pi_amp_cavity'])[mode_indices],
                                                  phases=phases,
                                                  add_freqs=np.array(blockade_pulse_info['blockade_cavity_offset_freq']),
                                                  add_dressing_drive_offsets=[[0] for i in range(15)],
                                                  weak_cavity=blockade_pulse_info['use_weak_for_cavity'][mode_indices[0]])
            current_len = blockade_pulse_info['blockade_half_pi_length'][mode_indices[0]]
            mode_indices = mode_indices[1:]
            # probably need to add some correction to phases to keep them consistent
            phases = orig_phases[j+1:] + 2*np.pi*current_len*(np.array(blockade_pulse_info['blockade_cavity_offset_freq'])[mode_indices] + \
                     np.array(self.quantum_device_cfg['flux_pulse_info'][qubit_id]['cavity_freqs'])[mode_indices])

    def gen_two_mode_correlator(self, sequencer, qubit_id="1", mode1=1, mode2=2, op1='Z', op2='I', use_weak_resolved_pulse_drive=True,
                                simultaneous_rotations=False, qubit_phase=0):
        # default added frequencies to probe each of the 2 modes
        add_freqs = [2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode1],
                     2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode2]]
        add_freqs = np.array(add_freqs)
        if op1.lower() == 'i':  # only really measuring second mode, so remove probe frequency that corresponds to mode1
            add_freqs = add_freqs[1:]
        if op2.lower() == 'i':  # only really measuring first mode, so remove probe frequency that corresponds to mode2
            add_freqs = add_freqs[:1]

        # not sure if for nonsimultaneous blockades we need to add phase corrections to the second blockade. Maybe..
        # since they are sequential; currently have the half_pi_blockade_rotation function returning the blockade length
        # for that purpose if desired
        if op1.lower() == 'x':
            if op2.lower() == 'x':
                if simultaneous_rotations:
                    self.half_pi_multimode_blockade_rotation(sequencer, qubit_id=qubit_id, mode_indices=[mode1, mode2], phases=[0, 0])
                else:
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=0)
                    sequencer.sync_channels_time(self.channels)
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=0)
            elif op2.lower() == 'y':
                if simultaneous_rotations:
                    self.half_pi_multimode_blockade_rotation(sequencer, qubit_id=qubit_id, mode_indices=[mode1, mode2], phases=[0, -np.pi/2])
                else:
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=0)
                    sequencer.sync_channels_time(self.channels)
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=-np.pi/2)
            else:  # no rotation needed for operator 2, only operator 1, x rotation
                self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=0)

        elif op1.lower() == 'y':
            if op2.lower() == 'x':
                if simultaneous_rotations:
                    self.half_pi_multimode_blockade_rotation(sequencer, qubit_id=qubit_id, mode_indices=[mode1, mode2], phases=[-np.pi/2, 0])
                else:
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=-np.pi/2)
                    sequencer.sync_channels_time(self.channels)
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=0)
            elif op2.lower() == 'y':
                if simultaneous_rotations:
                    self.half_pi_multimode_blockade_rotation(sequencer, qubit_id=qubit_id, mode_indices=[mode1, mode2], phases=[-np.pi/2, -np.pi/2])
                else:
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=-np.pi/2)
                    sequencer.sync_channels_time(self.channels)
                    self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=-np.pi/2)
            else:  # no rotation needed for operator 2, only operator 1, y rotation
                self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode1, phase=-np.pi/2)

        else:  # op1 is Z or I, only possibly need to rotate op2
            if op2.lower() == 'x':
                self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=0)
            elif op2.lower() == 'y':
                self.half_pi_blockade_rotation(sequencer, qubit_id=qubit_id, mode_index=mode2, phase=-np.pi/2)

        sequencer.sync_channels_time(self.channels)
        phases = [qubit_phase] * len(add_freqs)
        if not use_weak_resolved_pulse_drive:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
            if pulse_type.lower() == 'square':
                sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                               flat_len=self.pulse_info[qubit_id]['pi_len_resolved'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                                               freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                                        sigma_len=self.pulse_info[qubit_id]['pi_len_resolved'],cutoff_sigma=2,
                                freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases))
        else:
            pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
            if pulse_type.lower() == 'square':
                sequencer.append('qubitweak', Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                     flat_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],ramp_sigma_len=0.001, cutoff_sigma=2,
                                                     freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases))
            elif pulse_type.lower() == 'gauss':
                sequencer.append('qubitweak', Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                              sigma_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],cutoff_sigma=2,
                                freqs=self.qubit_freq[qubit_id]+add_freqs,phases=phases))


    def cavity_blockade_tomography(self, sequencer, qubit_id='1', mode_index=0, use_weak_resolved_pulse_drive=True,
                                   qubit_phase=0):
        phase = qubit_phase
        add_freq = 2 * self.quantum_device_cfg['flux_pulse_info']['chiby2pi_e'][mode_index]
        for i in range(3):  # measure x, measure y, measure z
            if i == 0:  # measure x
                self.half_pi_blockade_rotation(qubit_id=qubit_id, mode_index=mode_index, phase=0)
            elif i == 1:  # measure y
                self.half_pi_blockade_rotation(qubit_id=qubit_id, mode_index=mode_index, phase=-np.pi/2)
            # i == 2 case is measure z, which does not require rotation
            if not use_weak_resolved_pulse_drive:
                pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
                if pulse_type.lower() == 'square':
                    sequencer.append('charge%s' % qubit_id,
                                     Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                      flat_len=self.pulse_info[qubit_id]['pi_len_resolved'],
                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                      freq=self.qubit_freq[qubit_id] + add_freq, phase=phase))
                elif pulse_type.lower() == 'gauss':
                    sequencer.append('charge%s' % qubit_id,
                                     Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved'],
                                                     sigma_len=self.pulse_info[qubit_id]['pi_len_resolved'], cutoff_sigma=2,
                                                     freq=self.qubit_freq[qubit_id] + add_freq, phase=phase))
            else:
                pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
                if pulse_type.lower() == 'square':
                    sequencer.append('qubitweak',
                                     Square_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                      flat_len=self.pulse_info[qubit_id]['pi_len_resolved_weak'],
                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                      freq=self.qubit_freq[qubit_id] + add_freq, phase=phase))
                elif pulse_type.lower() == 'gauss':
                    sequencer.append('qubitweak', Gauss_multitone(max_amp=self.pulse_info[qubit_id]['pi_amp_resolved_weak'],
                                                                  sigma_len=self.pulse_info[qubit_id][
                                                                      'pi_len_resolved_weak'], cutoff_sigma=2,
                                                                  freq=self.qubit_freq[qubit_id] + add_freq, phase=phase))


    def wigner_tomography(self, sequencer, qubit_id='1',mode_index = 0,amp = 0, phase=0,len = 0,pulse_type = 'square',
                          use_transfer=False):
        self.gen_c(sequencer,mode_index=mode_index, len=len, amp = amp, add_freq=0, phase=phase + np.pi, pulse_type=pulse_type,
                   use_transfer=use_transfer)
        sequencer.sync_channels_time(self.channels)
        # self.idle_all(sequencer, time=10)
        self.parity_measurement(sequencer,qubit_id)


    def generalized_wigner_tomography(self, sequencer, qubit_id='1', mode_index=0, amp=0, phase=0, len=0,
                                      pulse_type='square', ramsey_time=0.0):
        self.gen_c(sequencer, mode_index=mode_index, len=len, amp=amp, add_freq=0, phase=phase + np.pi,
                   pulse_type=pulse_type)
        sequencer.sync_channels_time(self.channels)
        # self.idle_all(sequencer, time=10)
        self.generalized_parity_measurement(sequencer, qubit_id, ramsey_time=ramsey_time)


    def wigner_tomography_with_sideband(self, sequencer, qubit_id='1',mode_index = 0,amp = 0, phase=0,len = 0,pulse_type = 'square',
                          use_transfer=False):
        self.gen_c(sequencer, mode_index=mode_index, len=len, amp=amp, add_freq=0, phase=phase + np.pi,
                   pulse_type=pulse_type,
                   use_transfer=use_transfer)
        sideband_len = len
        if pulse_type == "gauss":
            sideband_len *= 4
        self.gen_sideband(sequencer, qubit_id, len=sideband_len, amp=self.expt_cfg['chi_dressing_amp'], add_freq=0.0,
                          iq_freq=self.sideband_pulse_info['iq_freq'])
        sequencer.sync_channels_time(self.channels)
        self.idle_all(sequencer, time=10)
        self.parity_measurement_with_sideband(sequencer, qubit_id)


    def joint_wigner_tomography(self, sequencer, qubit_id='1', modes=[1, 2], amps=[0, 0], phases=[0, 0], lengths=[0, 0],
                                pulse_type='gauss', sequential=True, qubit_phases=[0, 0], use_weak_qubit_drive=True,
                                n_max=2, pi_resolved_parity=True, ramsey_parity=False, displace_cav=True,add_final_piby2_phase = 0.0):
        cav_freqs = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['cavity_freqs']
        if sequential:
            if displace_cav:
                for j, mode in enumerate(modes):
                    # print("tomography mode = ",mode,"len = ",lengths[j],"amp = ",amps[j])
                    self.gen_c(sequencer, mode_index=mode, len=lengths[j], amp=amps[j],
                               phase=phases[j] + np.pi, pulse_type=pulse_type)
        elif displace_cav:
            sort_list = np.argsort(lengths)
            modes = np.array(modes)[sort_list]
            amps = np.array(amps)[sort_list]
            phases = np.array(phases)[sort_list] + np.pi
            lengths = np.array(lengths)[sort_list]
            # input all cavity drives for their correct lengths
            orig_phases = phases
            mode_indices = modes
            for j in range(len(mode_indices)):
                if lengths[0]-current_len > 0:
                    self.gen_c_multitone(sequencer, mode_indices=mode_indices, len=lengths[0]-current_len, phases=phases,
                                         pulse_type=pulse_type, amps=amps, add_freqs=[0.0 for j in range(15)])
                    current_len = lengths[0]
                    mode_indices = mode_indices[1:]
                    amps = amps[1:]
                    lengths = lengths[1:]
                    # probably need to add some correction to phases to keep them consistent
                    phases = orig_phases[j + 1:] - 2 * np.pi * current_len * pulse_scale * \
                                np.array(cav_freqs)[mode_indices]
                else:
                    break
        sequencer.sync_channels_time(self.channels)
        if pi_resolved_parity:
            self.pi_resolved_joint_parity_measurement(sequencer, mode1=modes[0], mode2=modes[1], qubit_id=qubit_id, n_max=n_max,
                                                      use_weak_drive=use_weak_qubit_drive, phases=qubit_phases,
                                                      only_mode1=self.expt_cfg['pi_resolved_on_first_mode_only'])
        elif ramsey_parity:
            self.generalized_parity_measurement(sequencer, qubit_id, ramsey_time=self.expt_cfg['ramsey_parity_time'],add_phase=add_final_piby2_phase)
        else:
            print("!!!NOT PERFORMING PARITY MEASUREMENT!!!")


    # def joint_wigner_tomography_phase_advancing(self, sequencer, qubit_id='1', modes=[1,2], amps=[0,0], phases=[0,0], lengths=[0,0],
    #                             pulse_type='gauss', sequential=True, qubit_phases=[0,0], use_weak_qubit_drive=True,
    #                             n_max=2, joint_parity=True, parity_meas=True, displace_cav=True):
    #     pulse_scale = 1
    #     if pulse_type == "gauss":
    #         pulse_scale = 4
    #     current_len = 0
    #     cav_freqs = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['cavity_freqs']
    #     if sequential:
    #         if displace_cav:
    #             for j, mode in enumerate(modes):
    #                 self.gen_c(sequencer, mode_index=mode, len=lengths[j], amp=amps[j],
    #                            phase=phases[j]+np.pi-current_len*2*np.pi*cav_freqs[modes[j]],
    #                            pulse_type=pulse_type)
    #                 current_len += lengths[j] * pulse_scale
    #     elif displace_cav:
    #         sort_list = np.argsort(lengths)
    #         modes = np.array(modes)[sort_list]
    #         amps = np.array(amps)[sort_list]
    #         phases = np.array(phases)[sort_list] + np.pi
    #         lengths = np.array(lengths)[sort_list]
    #         # input all cavity drives for their correct lengths
    #         orig_phases = phases
    #         mode_indices = modes
    #         for j in range(len(mode_indices)):
    #             if lengths[0]-current_len > 0:
    #                 self.gen_c_multitone(sequencer, mode_indices=mode_indices, len=lengths[0]-current_len, phases=phases,
    #                                      pulse_type=pulse_type, amps=amps, add_freqs=[0.0 for j in range(15)])
    #                 current_len = lengths[0]
    #                 mode_indices = mode_indices[1:]
    #                 amps = amps[1:]
    #                 lengths = lengths[1:]
    #                 # probably need to add some correction to phases to keep them consistent
    #                 phases = orig_phases[j + 1:] - 2 * np.pi * current_len * pulse_scale * \
    #                             np.array(cav_freqs)[mode_indices]
    #             else:
    #                 break
    #     sequencer.sync_channels_time(self.channels)
    #     if parity_meas:
    #         if joint_parity:
    #             self.pi_resolved_joint_parity_measurement(sequencer, mode1=modes[0], mode2=modes[1], qubit_id=qubit_id, n_max=n_max,
    #                                                       use_weak_drive=use_weak_qubit_drive, phases=qubit_phases,
    #                                                       only_mode1=self.expt_cfg['pi_resolved_on_first_mode_only'])
    #         else:
    #             if self.expt_cfg['use_parity_time_from_config']:
    #                 # print("Using parity time from expt cfg")
    #                 self.generalized_parity_measurement(sequencer, qubit_id, ramsey_time=self.expt_cfg['parity_time'])
    #             else:
    #                 self.parity_measurement(sequencer, qubit_id)


    def resonator_spectroscopy(self, sequencer):
        for i in range(3):
            sequencer.new_sequence(self)
            if self.expt_cfg['pi_qubit']:
                for qubit_id in self.expt_cfg['on_qubits']:
                    if self.expt_cfg['resolved_pi']:self.pi_q_resolved(sequencer, qubit_id, use_weak_drive=self.expt_cfg['use_weak_drive'])
                    else: self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    if self.expt_cfg['pi_qubit_ef']:
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
    
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cross_kerr_with_readout_spectroscopy(self, sequencer):
        for i in range(3):
            sequencer.new_sequence(self)
            if self.expt_cfg['excite_mode']:
                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                # print (blockade_pulse_info)
                mode_index = self.expt_cfg['prep_mode_index']
                qubit_id = "1"
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                            len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                            cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                mode_index],
                                            use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                mode_index],
                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                            blockade_levels=[2], dressing_pulse_type="square",
                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                            phase=0,
                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                            weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                'use_weak_for_cavity'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def resonator_spectroscopy_weak_qubit_drive(self, sequencer):
        sequencer.new_sequence(self)
        if self.expt_cfg['pi_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q_resolved(sequencer, qubit_id,use_weak_drive=True,add_freq=self.expt_cfg['add_freq'])
                

        self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                                               ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_freq[qubit_id] + dfreq,
                                                               phase=0))

                # sequencer.append('charge%s_Q' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                #                         ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                #                         phase=self.pulse_info[qubit_id]['Q_phase']))
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi_drag(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, DRAG(A=self.expt_cfg['amp'], beta=self.expt_cfg['drag_beta'],
                                                             sigma_len=rabi_len, cutoff_sigma=2,
                                                             freq=self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                                                             phase=0))
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                #self.gen_q(sequencer, qubit_id, len=ramsey_len, amp=0.0, phase=0,pulse_type='square',add_freq=0.5)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey_after_readout(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            readout_time = sequencer.get_time('readout_trig')  # Earlies was alazar_tri
            readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
            # sequencer.append_idle_to_time('readout_trig', readout_time_5ns_multiple)
            sequencer.append('readout',
                             Square(max_amp=self.quantum_device_cfg['readout']['amp'],
                                    flat_len=self.quantum_device_cfg['readout']['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=0,
                                    phase=0, phase_t0=readout_time_5ns_multiple))
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_time'])
            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                #self.gen_q(sequencer, qubit_id, len=ramsey_len, amp=0.0, phase=0,pulse_type='square',add_freq=0.5)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase= 2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
    
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
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer, qubit_id, amp = self.expt_cfg['amp'],len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],
                           add_freq=dfreq+self.quantum_device_cfg['qubit']['1']['anharmonicity'])
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
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['use_weak_drive']:
                    sequencer.sync_channels_time(self.channels)
                    self.gen_q_weak(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=rabi_len, phase=0,
                               pulse_type=self.expt_cfg['ef_pulse_type'],
                               add_freq=self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
                    sequencer.sync_channels_time(self.channels)
                else:
                    self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.expt_cfg['ef_pulse_type'],
                           add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'])
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
            for qubit_id in self.expt_cfg['on_qubits']:

                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ef_t1_len)

                if self.expt_cfg['ef_pi_for_cal']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

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

    def gf_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                if self.expt_cfg['echo']:
                    n  = self.expt_cfg['number']
                    for i in range(n):
                        self.idle_q(sequencer, time=ramsey_len/(n+1))
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.idle_q(sequencer, time=ramsey_len / (n+1))

                else:self.idle_q(sequencer, time=ramsey_len)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'], phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq']+self.expt_cfg['final_phase'])


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

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

    def pulse_probe_fh_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
           
            for qubit_id in self.expt_cfg['on_qubits']:
                add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] + \
                           self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity_fh']

                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.sync_channels_time(self.channels)

                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.gen_q(sequencer, qubit_id, amp=self.expt_cfg['amp'], len=self.expt_cfg['pulse_length'], phase=0,
                           pulse_type=self.expt_cfg['pulse_type'], add_freq=add_freq +dfreq)
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['pi_calibration']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
           

            for qubit_id in self.expt_cfg['on_qubits']:
                add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity_fh']
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['ef_pi']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq = add_freq )
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['pi_calibration']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_ramsey(self, sequencer):


        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
           
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
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()
                # with pi pulse (e state)
                sequencer.new_sequence(self)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()
                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.readout_pxi(sequencer, qubit_id)
                    sequencer.end_sequence()
                if self.expt_cfg['include_h']:
                    sequencer.new_sequence(self)
                    for qubit_id in self.expt_cfg['on_qubits']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                        self.readout_pxi(sequencer, qubit_id)
                        sequencer.end_sequence()
        return sequencer.complete(self, plot=False)

    def oct_histogram(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):
            for qubit_id in self.expt_cfg['on_qubits']:
                """Readout at |g, 0>"""
                sequencer.new_sequence(self)
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()
                """Readout at |e, 0>"""
                sequencer.new_sequence(self)
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()
                """Prepare all the Fock states sequentially measure one at a time and then repeat it.
                    Efficient than measuring one Fock state at a time.
                """
                for n, filename in enumerate(self.expt_cfg['filenames']):
                    """Readout at |g, n>"""
                    sequencer.new_sequence(self)
                    self.prep_optimal_control_pulse_1step(sequencer, filename=filename)
                    self.readout_pxi(sequencer, qubit_id)
                    sequencer.end_sequence()
                    """Readout at |e, n>"""
                    sequencer.new_sequence(self)
                    self.prep_optimal_control_pulse_1step(sequencer, filename=filename)
                    add_freq = 2 * (n+1) * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                        'chiby2pi_e'][self.expt_cfg['mode_index']]
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'], add_freq=add_freq)
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
        sideband_freq_offset = self.expt_cfg['freq_offset']
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['f0g1']:
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    dc_offset=0.0
                elif self.expt_cfg['h0e1']:
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    dc_offset = 0.0
                elif self.expt_cfg['e0g2']:
                    dc_offset = 0.0
                sequencer.sync_channels_time(self.channels)
                self.gen_sideband(sequencer,qubit_id,len= length,amp = self.expt_cfg['amp'],add_freq = sideband_freq_offset,iq_freq = self.sideband_pulse_info['iq_freq'])

                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # # COMMENT OUT IF SIDEBAND FREQUENCY IS NOT QUBIT
                    # self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # ge pulse for testing
                    # self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    # self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # # --------------------------------------------------------------------------------------#
                if self.expt_cfg['h0e1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_rabi_phase_test(self, sequencer):
        # sideband rabi time domain
        sideband_freq = self.expt_cfg['freq']
        for phase in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['f0g1']:
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                elif self.expt_cfg['h0e1']:
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_dc_offset']
                elif self.expt_cfg['e0g2']:
                    dc_offset = 0.0
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['length'],
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'], phase=phase)
                    # self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # ge pulse for testing
                if self.expt_cfg['h0e1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])


            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_rabi_freq_scan(self, sequencer):
        length = self.expt_cfg['length']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['f0g1']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                elif self.expt_cfg['h0e1']:
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_dc_offset']
                elif self.expt_cfg['e0g2']:
                    dc_offset = 0.0
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=20)
                self.gen_sideband(sequencer, qubit_id, len=length, amp=self.expt_cfg['amp'],
                                  add_freq=dfreq, iq_freq=self.sideband_pulse_info['iq_freq'])
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=20)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # # COMMENT OUT IF "SIDEBAND" FREQUENCY IS NOT QUBIT FREQUENCY
                    # self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # ge pulse for testing
                    # self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    # self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # --------------------------------------------------------------------------------------#
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

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['prep_using_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    # print (blockade_pulse_info)
                    mode_index = self.expt_cfg['mode_index']
                    sequencer.new_sequence(self)
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                else:
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
            if self.expt_cfg['f0g1']:
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
            elif self.expt_cfg['e0g2']:
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase2 / 2.0,
                                    mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=50.0)
                    self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square', phase=-offset_phase2 / 2.0,
                                    mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=50.0)
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            elif self.expt_cfg['h0e1']:
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # g-e
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])  # g-f
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # e-f
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])  # e-h
                    sequencer.sync_channels_time(self.channels)
                    self.pi_h0e1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase2 / 2.0,
                                    mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=50.0)
                    self.pi_h0e1_sb(sequencer, qubit_id, pulse_type='square', phase=-offset_phase2 / 2.0,
                                    mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=50.0)
                    self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ge_calibration(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            mode_index = self.expt_cfg['mode_index']
            for qubit_id in self.expt_cfg['on_qubits']:

                if self.expt_cfg['test_parity']:
                    add_phase = np.pi
                    phase_freq = 0
                else:
                    add_phase = 0
                    phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                if self.expt_cfg['add_photon_with_sideband']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0,
                                    mode_index=self.expt_cfg['mode_index'])
                elif self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['delay_before_ramsey']: self.idle_q(sequencer, qubit_id, time=self.expt_cfg['delay_time'])

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, qubit_id, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = add_phase+2*np.pi*ramsey_len*phase_freq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_ef_calibration(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                mode_index = self.expt_cfg['mode_index']
                if self.expt_cfg['add_photon_with_sideband']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                elif self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
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
            for qubit_id in self.expt_cfg['on_qubits']:

                phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_f'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']
                mode_index = self.expt_cfg['mode_index']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                if self.expt_cfg['add_photon_with_sideband']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                elif self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
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

    def sideband_chi_dressing_calibration(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            mode_index = self.expt_cfg['mode_index']
            for qubit_id in self.expt_cfg['on_qubits']:
                add_phase = 0
                if self.expt_cfg['add_photon_with_sideband'] or self.expt_cfg['add_photon_with_blockade']:
                    phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['ramsey_freq']
                else:
                    phase_freq =self.expt_cfg['ramsey_freq']
                if self.expt_cfg['use_freq_from_expt_cfg']:
                    sideband_freq = self.expt_cfg['sideband_freq']
                elif self.expt_cfg["h0e1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                elif self.expt_cfg["f0g1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                else:
                    sideband_freq = 0.0
                    print ("what transition do you want to use for dressing good sir/madam")

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

                if self.expt_cfg['add_photon_with_sideband']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_all(sequencer, qubit_id, time=2)
                dc_offset = 0
                self.gen_sideband(sequencer, qubit_id, len=ramsey_len, amp=self.expt_cfg['amp'],
                                  add_freq=0.0, iq_freq=self.sideband_pulse_info['iq_freq'])
                # sequencer.append('sideband',
                #                  Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len,
                #                         ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                #                             'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                #                         fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                #                         dc_offset=dc_offset,
                #                         plot=False))
                self.idle_all(sequencer, qubit_id, time=2)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = add_phase+2*np.pi*ramsey_len*phase_freq)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_dressing_t1test(self, sequencer):
        for len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            mode_index = self.expt_cfg['mode_index']
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['use_freq_from_expt_cfg']:
                    sideband_freq = self.expt_cfg['sideband_freq']
                elif self.expt_cfg["h0e1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                elif self.expt_cfg["f0g1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                else:
                    sideband_freq = 0.0
                    print ("what transition do you want to use for dressing good sir/madam")

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

                if self.expt_cfg['add_photon_with_sideband']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                elif self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_all(sequencer, qubit_id, time=2)
                dc_offset = 0
                self.gen_sideband(sequencer, qubit_id, len=len, amp=self.expt_cfg['amp'],
                                  add_freq=0.0, iq_freq=self.sideband_pulse_info['iq_freq'])
    
                self.idle_all(sequencer, qubit_id, time=2)
            

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def chi_dressing_pnrqs(self, sequencer):
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            mode_index = self.expt_cfg['mode_index']
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg["h0e1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][self.expt_cfg['mode_index']] + self.expt_cfg['detuning']
                elif self.expt_cfg["f0g1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
                                        self.expt_cfg['mode_index']] + self.expt_cfg['detuning']
                else:
                    sideband_freq = 0.0
                    print("what transition do you want to use for dressing good sir/madam 8)")
                if self.expt_cfg['prep_use_weak_cavity']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                              amp=self.expt_cfg['prep_cav_amp'], phase=0,
                              pulse_type=self.expt_cfg['cavity_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['use_weak_drive_for_probe']:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                    if self.pulse_info[qubit_id]['resolved_pulse_type_weak'] == "gauss":
                        dressing_len *= 4
                else:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved']
                    if self.pulse_info[qubit_id]['resolved_pulse_type'] == "gauss":
                        dressing_len *= 4
                dc_offset = 0
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['dressing_amp'], flat_len=dressing_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=df, use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_chi_dressing_pnrqs(self, sequencer):
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            mode_index = self.expt_cfg['mode_index']
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['prep_use_weak_cavity']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                              amp=self.expt_cfg['prep_cav_amp'], phase=0,
                              pulse_type=self.expt_cfg['cavity_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['use_weak_drive_for_probe']:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                    if self.pulse_info[qubit_id]['resolved_pulse_type_weak'] == "gauss":
                        dressing_len *= 4
                else:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved']
                    if self.pulse_info[qubit_id]['resolved_pulse_type'] == "gauss":
                        dressing_len *= 4
                dc_offset = 0
                self.gen_sideband(sequencer, qubit_id=self.expt_cfg['on_qubits'], len=dressing_len, amp=self.expt_cfg['dressing_amp'],
                                  add_freq=0.0, iq_freq = self.sideband_pulse_info['iq_freq'])
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=df, use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def chi_dressing_blockade_cavity_spectroscopy(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg["h0e1"]:
                sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][
                                    self.expt_cfg['chi_dressing_mode_index']] + self.expt_cfg['detuning']
            elif self.expt_cfg["f0g1"]:
                sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
                                    self.expt_cfg['chi_dressing_mode_index']] + self.expt_cfg['detuning']
            else:
                sideband_freq = 0.0
                print("what transition do you want to use for dressing good sir/madam 8)")
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['blockade_mode_index'],
                                            len=self.expt_cfg['cavity_pulse_len'],
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq']+df,
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=True)
                dc_offset = 0
                dressing_len = self.expt_cfg['cavity_pulse_len']
                if self.expt_cfg['cavity_pulse_type'] == "gauss":
                    dressing_len *= 4
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['chi_dressing_amp'], flat_len=dressing_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['use_weak_drive_for_probe']:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                    if self.pulse_info[qubit_id]['resolved_pulse_type_weak'] == "gauss":
                        dressing_len *= 4
                else:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved']
                    if self.pulse_info[qubit_id]['resolved_pulse_type'] == "gauss":
                        dressing_len *= 4
                dc_offset = 0
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['chi_dressing_amp'], flat_len=dressing_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['probe_mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def chi_dressing_blockade_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg["h0e1"]:
                sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][
                                    self.expt_cfg['chi_dressing_mode_index']] + self.expt_cfg['detuning']
            elif self.expt_cfg["f0g1"]:
                sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
                                    self.expt_cfg['chi_dressing_mode_index']] + self.expt_cfg['detuning']
            else:
                sideband_freq = 0.0
                print("what transition do you want to use for dressing good sir/madam 8)")
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['blockade_mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=True)
                dc_offset = 0
                dressing_len = rabi_len
                if self.expt_cfg['cavity_pulse_type'] == "gauss":
                    dressing_len *= 4
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['chi_dressing_amp'], flat_len=dressing_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['use_weak_drive_for_probe']:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                    if self.pulse_info[qubit_id]['resolved_pulse_type_weak'] == "gauss":
                        dressing_len *= 4
                else:
                    dressing_len = self.pulse_info[qubit_id]['pi_len_resolved']
                    if self.pulse_info[qubit_id]['resolved_pulse_type'] == "gauss":
                        dressing_len *= 4
                dc_offset = 0
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['chi_dressing_amp'], flat_len=dressing_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['probe_mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def sideband_transmon_reset(self, sequencer):
        sideband_freq = self.expt_cfg['sideband_freq']
        sideband_pulse_length =  self.expt_cfg['sideband_pulse_length']
        wait_after_reset = self.expt_cfg['wait_after_reset']

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
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

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
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

    def sideband_repetitive_parity_measurement(self, sequencer):
        # sideband repetitive parity measurement

        for ii in np.arange(self.expt_cfg['num_expts']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if ii == 40:
                    offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                    if self.expt_cfg['add_photon']:
                        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        sequencer.sync_channels_time(self.channels)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)
                        sequencer.sync_channels_time(self.channels)

                else:
                    self.idle_q_sb(sequencer, qubit_id, time=self.expt_cfg['wait_before_parity'])
                    self.parity_measurement(sequencer, qubit_id)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def repetitive_parity_measurement(self, sequencer):
        # cavity drive + repetitive parity measurement

        for ii in np.arange(self.expt_cfg['num_expts']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if ii == 10:
                    if self.expt_cfg['add_photon']:
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'],
                                   len=self.expt_cfg['alpha_len'], amp=self.expt_cfg['alpha_amp'],
                                   phase=0, pulse_type='gauss')

                        sequencer.sync_channels_time(self.channels)

                else:
                    self.idle_q_sb(sequencer, qubit_id, time=self.expt_cfg['wait_before_parity'])
                    self.parity_measurement(sequencer, qubit_id)

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_e0g2_t1(self, sequencer):
        # sideband rabi time domain
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square',mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer,qubit_id,time=length + 40.0)
                self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square',mode_index=self.expt_cfg['mode_index'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_e0g2_ramsey(self, sequencer):
        # sideband rabi time domain

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:

                if self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase']:phase_freq = self.expt_cfg['ramsey_freq']
                else:phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['e0g2_dc_offset'][self.expt_cfg['mode_index']] + self.expt_cfg['ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['e0g2_pi_pi_offset'][self.expt_cfg['mode_index']]

                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=ramsey_len+50.0)
                self.pi_e0g2_sb(sequencer, qubit_id, pulse_type='square',phase = 2*np.pi*ramsey_len*phase_freq-offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_h0e1_ramsey(self, sequencer):
        # sideband rabi time domain
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase']:phase_freq = self.expt_cfg['ramsey_freq']
                else:phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_dc_offset'][self.expt_cfg['mode_index']] + self.expt_cfg['ramsey_freq']
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_pi_pi_offset'][self.expt_cfg['mode_index']]
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # g-e
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])  # g-f
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # e-f
                self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])  # e-h
                sequencer.sync_channels_time(self.channels)
                self.pi_h0e1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=ramsey_len+50.0)
                self.pi_h0e1_sb(sequencer, qubit_id, pulse_type='square',phase = 2*np.pi*ramsey_len*phase_freq-offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q_sb(sequencer, qubit_id, time=50.0)
                self.pi_q_fh(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['fh_pulse_type'])  # e-f
                self.pi_q_(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # g-f
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])  # g-e
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
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

    def direct_cavity_spectroscopy(self, sequencer):
        # Direct Cavity spectroscopy by monitoring the readout resonator
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sequencer.append('cavity',
                             Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.cavity_freq[self.expt_cfg['mode_index']] + dfreq,
                                    phase=0))

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def sideband_fnm1gnrabi_freq_scan(self,sequencer):
        sideband_freq = self.expt_cfg['freq']
        length = self.expt_cfg['length']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                dc_offset = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset']
                self.pi_q(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                for nn in range(self.expt_cfg['n']-1):
                    sequencer.sync_channels_time(self.channels)
                    self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase/2.0, n=nn+1,mode_index=self.expt_cfg['mode_index'])
                    if self.expt_cfg['numbershift_qdrive_freqs']:
                        add_freq_ge = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']]
                        add_freq_ef = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef'][self.expt_cfg['mode_index']]
                    else:add_freq_ge,add_freq_ef=0.0,0.0
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'],add_freq=add_freq_ge)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq=add_freq_ef)
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                        fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset, plot=False))
                sequencer.sync_channels_time(self.channels)
                self.idle_sb(sequencer,time=40)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def sideband_fnm1gnrabi(self, sequencer):
        # sideband rabi time domain
        sideband_freq = self.expt_cfg['freq']
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][self.expt_cfg['mode_index']]
                for nn in range(self.expt_cfg['n']-1):
                    sequencer.sync_channels_time(self.channels)
                    self.pi_fnm1gn_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0, n=nn+1)
                    if self.expt_cfg['numbershift_qdrive_freqs']:
                        add_freq_ge = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']]
                        add_freq_ef = (nn+1)*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_ef'][self.expt_cfg['mode_index']]
                    else:add_freq_ge,add_freq_ef=0.0,0.0
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'],add_freq=add_freq_ge)
                    self.pi_q_ef(sequencer, qubit_id,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq=add_freq_ef)
                sequencer.sync_channels_time(self.channels)
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                sequencer.sync_channels_time(self.channels)
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def qubit_rabi_with_sideband(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'], phase=0,
                           pulse_type=self.expt_cfg['pulse_type'], add_freq=self.expt_cfg['add_qubit_freq'])
                sideband_len = rabi_len
                if self.expt_cfg['pulse_type'] == "gauss":
                    sideband_len *= 4
                self.gen_sideband(sequencer, qubit_id=qubit_id, len=sideband_len,
                                amp=self.expt_cfg['sideband_amp'],
                                add_freq=0.0, iq_freq=self.sideband_pulse_info['iq_freq'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def cavity_spectroscopy_resolved_qubit_pulse(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['excite_qubit']:
                    self.pi_q(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index],
                                                    phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['use_weak_cavity']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                               amp=self.expt_cfg['cavity_amp'], add_freq=df, phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                else:
                    self.gen_c(sequencer,mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'], 
                            amp=self.expt_cfg['cavity_amp'], add_freq=df, phase=0, pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id, use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def photon_number_resolved_qubit_spectroscopy(self, sequencer):
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == 'alpha':
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'],add_freq=self.expt_cfg['add_cavity_drive_detuning'])

                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                elif self.expt_cfg['state'] == '0+alpha':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase']+np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] in ['2', '3', '4','5']:
                    self.prep_fock_n(sequencer, qubit_id, fock_state=eval(self.expt_cfg['state']),
                                     offset_phase=offset_phase, mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['0+2', '0+3', '0+4']:
                    self.prep_fock_superposition(sequencer, qubit_id, fock_state=eval(self.expt_cfg['state']),
                                     offset_phase=offset_phase, mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['SNAP_g1']:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][0],
                               amp=self.expt_cfg['snap_cav_amps'][0], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.two_pi_q_resolved(sequencer, qubit_id, add_freq=0.0, phase=self.expt_cfg['snap_phase'],
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_SNAPg1'])
                    # self.pi_q_resolved(sequencer, qubit_id,
                    #                        pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                    #                        add_freq=0.0, phase=self.expt_cfg['snap_phase'])
                    # self.pi_q_resolved(sequencer, qubit_id,
                    #                        pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                    #                        add_freq=0.0, phase=self.expt_cfg['snap_phase']+np.pi)
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][1],
                               amp=self.expt_cfg['snap_cav_amps'][1], phase=np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == '0':
                    pass

                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df+self.expt_cfg['add_freq'], phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive'])

                else:
                    if self.expt_cfg['use_weak_drive']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df+self.expt_cfg['add_freq'])
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                            add_freq= df+self.expt_cfg['add_freq'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def photon_number_resolved_qubit_spectroscopy_with_sideband_on(self, sequencer):
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == 'alpha':
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                elif self.expt_cfg['state'] == '0+alpha':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase']+np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] in ['2', '3', '4','5']:
                    self.prep_fock_n(sequencer, qubit_id, fock_state=eval(self.expt_cfg['state']),
                                     offset_phase=offset_phase, mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['0+2', '0+3', '0+4']:
                    self.prep_fock_superposition(sequencer, qubit_id, fock_state=eval(self.expt_cfg['state']),
                                     offset_phase=offset_phase, mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['SNAP_g1']:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][0],
                               amp=self.expt_cfg['snap_cav_amps'][0], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.two_pi_q_resolved(sequencer, qubit_id, add_freq=0.0, phase=self.expt_cfg['snap_phase'],
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_SNAPg1'])
                    # self.pi_q_resolved(sequencer, qubit_id,
                    #                        pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                    #                        add_freq=0.0, phase=self.expt_cfg['snap_phase'])
                    # self.pi_q_resolved(sequencer, qubit_id,
                    #                        pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                    #                        add_freq=0.0, phase=self.expt_cfg['snap_phase']+np.pi)
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][1],
                               amp=self.expt_cfg['snap_cav_amps'][1], phase=np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == '0':
                    pass

                sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df+self.expt_cfg['add_freq'], phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive'])
                    if self.expt_cfg['use_weak_drive']:
                        sideband_len = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_len_resolved_weak']
                        if self.quantum_device_cfg['pulse_info'][qubit_id]['resolved_pulse_type_weak'] == "gauss":
                            sideband_len *= 4
                    else:
                        sideband_len = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_len_resolved']
                        if self.quantum_device_cfg['pulse_info'][qubit_id]['resolved_pulse_type'] == "gauss":
                            sideband_len *= 4
                else:
                    if self.expt_cfg['use_weak_drive']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df+self.expt_cfg['add_freq'])
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],
                                   phase=0,pulse_type=self.expt_cfg['pulse_type'], add_freq= df+self.expt_cfg['add_freq'])
                self.gen_sideband(sequencer, qubit_id=self.expt_cfg['on_qubits'], len=sideband_len, amp=self.expt_cfg['sideband_amp'],
                                  add_freq=0.0, iq_freq = self.sideband_pulse_info['iq_freq'])
    
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def photon_number_distribution_measurement(self, sequencer):
        # transmon ef rabi with tek2
        for n in np.arange(self.expt_cfg['N_max']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][
                    self.expt_cfg['mode_index']]
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == 'alpha':
                    if self.expt_cfg['weak_cavity_prep_alpha_only']:
                        self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                    else:
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                elif self.expt_cfg['state'] == '0+alpha':
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.half_pi_q(sequencer, qubit_id,
                                            pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase']+np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] in ['2', '3', '4']:
                    self.prep_fock_n(sequencer, qubit_id, fock_state=eval(self.expt_cfg['state']),
                                     offset_phase=offset_phase, mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['SNAP_g1']:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][0],
                               amp=self.expt_cfg['snap_cav_amps'][0], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.two_pi_q_resolved(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                       add_freq=0.0,phase = self.expt_cfg['snap_phase'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][1],
                               amp=self.expt_cfg['snap_cav_amps'][1], phase=np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == '0':
                    pass
                sequencer.sync_channels_time(self.channels)

                self.pi_q_resolved(sequencer, qubit_id,add_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']]*n,
                                   use_weak_drive=self.expt_cfg['use_weak_drive'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def photon_number_expt_fit_wigner_alphas(self, sequencer):
        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])
        x1 = xs[self.expt_cfg['tom_index_pt']]
        y1 = ys[self.expt_cfg['tom_index_pt']]
        print(x1 * self.expt_cfg['tomography_pulse_len'], y1 * self.expt_cfg['tomography_pulse_len'])
        if self.expt_cfg['transfer_fn_wt']:
            tom_amp = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                       channel='cavity_amp_vs_freq_list')
        else:
            tom_amp = np.sqrt(x1 ** 2 + y1 ** 2)
        tom_phase = np.arctan2(y1, x1)

        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['pnrqs']:
                loop_freqs = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
            elif self.expt_cfg['photon_number_distribution_measurement']:
                loop_freqs = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] * np.arange(self.expt_cfg['N_max'])
            for df in loop_freqs:
                sequencer.new_sequence(self)
                for qubit_id in self.expt_cfg['on_qubits']:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['tomography_pulse_len'],
                               amp=tom_amp, phase=tom_phase,
                               pulse_type=self.expt_cfg['tomography_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def cavity_drive_ramsey(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                phase_freq = self.expt_cfg['ramsey_freq']
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_drive_len'],
                           amp=self.expt_cfg['cavity_drive_amp'], phase=0.0,
                           pulse_type=self.expt_cfg['cavity_pulse_type'])
                self.idle_all(sequencer, qubit_id, time=ramsey_len)
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_drive_len'],
                           amp=self.expt_cfg['cavity_drive_amp'], phase=np.pi + 2*np.pi*phase_freq*ramsey_len,
                           pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,add_freq = 0, use_weak_drive=self.expt_cfg['use_weak_probe'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def wigner_tomography_1d_sweep(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            # self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.expt_cfg['on_qubits']:
                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]
                if self.expt_cfg['state'] == '1':
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)



                elif self.expt_cfg['state'] == '0+1':
                    self.half_pi_q(sequencer, qubit_id,
                                   pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id,
                                 pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                elif self.expt_cfg['state'] == 'alpha':
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q(sequencer, qubit_id,
                              pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == '0+alpha':
                    self.half_pi_q(sequencer, qubit_id,
                                   pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q_resolved(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif self.expt_cfg['state'] == "alpha+-alpha":
                    self.half_pi_q(sequencer, qubit_id,
                                   pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q_resolved(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                               amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'] + np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])

                elif self.expt_cfg['state'] in ['2','3','4']:
                    self.prep_fock_n(sequencer,qubit_id,fock_state=eval(self.expt_cfg['state']),offset_phase=offset_phase,mode_index=self.expt_cfg['mode_index'])

                elif self.expt_cfg['state'] in ['SNAP_g1']:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][0],
                               amp=self.expt_cfg['snap_cav_amps'][0], phase=0,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.two_pi_q_resolved(sequencer, qubit_id,add_freq=0.0,phase=self.expt_cfg['snap_phase'],use_weak_drive=self.expt_cfg['use_weak_drive'])
                    sequencer.sync_channels_time(self.channels)
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['snap_cav_lens'][1],
                               amp=self.expt_cfg['snap_cav_amps'][1], phase=np.pi,
                               pulse_type=self.expt_cfg['cavity_pulse_type'])


                elif self.expt_cfg['state'] == '0':
                    pass

                self.idle_all(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:self.wigner_tomography(sequencer, qubit_id,mode_index = self.expt_cfg['mode_index'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])
                else:self.wigner_tomography(sequencer, qubit_id,mode_index = self.expt_cfg['mode_index'],amp = x,phase=self.expt_cfg['phase'],len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def wigner_tomography_2d_sweep(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']
        for y in (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0):
            for x in (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0):

                tom_amp = np.sqrt(x**2+y**2)
                tom_phase = np.arctan2(y,x)
                sequencer.new_sequence(self)
                # self.pad_start_pxi_tek2(sequencer, on_qubits=self.expt_cfg['on_qubits'])
                sequencer.sync_channels_time(self.channels)
                for qubit_id in self.expt_cfg['on_qubits']:
                    offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

                    if self.expt_cfg['state'] == '1':
                        self.pi_q(sequencer, qubit_id,
                                  pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id,
                                     pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        self.idle_sb(sequencer, time=5.0)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                    elif self.expt_cfg['state'] == '0+1':
                        self.half_pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.pi_q_ef(sequencer, qubit_id,
                                     pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                        self.idle_sb(sequencer, time=5.0)
                        self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square', phase=offset_phase / 2.0)

                    elif self.expt_cfg['state'] == 'alpha':
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])

                        sequencer.sync_channels_time(self.channels)

                    elif self.expt_cfg['state'] == 'cavity_drive_res_check':
                        self.pi_q(sequencer, qubit_id,
                                  pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                        self.pi_q(sequencer, qubit_id,
                                  pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                    elif self.expt_cfg['state'] == '0+alpha':
                        self.half_pi_q(sequencer, qubit_id,
                                       pulse_type=self.transmon_pulse_info_tek2[qubit_id]['pulse_type'])
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                        self.pi_q_resolved(sequencer, qubit_id,
                                           pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                    elif self.expt_cfg['state'] == "alpha+-alpha":
                        self.half_pi_q(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                        self.pi_q_resolved(sequencer, qubit_id,
                                           pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                                   amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'] + np.pi,
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])

                    elif self.expt_cfg['state'] in ['SNAP_g1']:
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'],
                                   len=self.expt_cfg['snap_cav_lens'][0],
                                   amp=self.expt_cfg['snap_cav_amps'][0], phase=0,
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                        sequencer.sync_channels_time(self.channels)
                        self.two_pi_q_resolved(sequencer, qubit_id,
                                               pulse_type=self.pulse_info[qubit_id]['resolved_pulse_type'],
                                               add_freq=0.0, phase=self.expt_cfg['snap_phase'],use_weak_drive=self.expt_cfg['use_weak_drive'])
                        sequencer.sync_channels_time(self.channels)
                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'],
                                   len=self.expt_cfg['snap_cav_lens'][1],
                                   amp=self.expt_cfg['snap_cav_amps'][1], phase=np.pi,
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])

                    elif self.expt_cfg['state'] == '0':
                        pass

                    self.idle_sb(sequencer, time=5.0)
                    if self.expt_cfg['sweep_phase']:self.wigner_tomography(sequencer, qubit_id,mode_index=self.expt_cfg['mode_index'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])
                    else:self.wigner_tomography(sequencer, qubit_id,mode_index=self.expt_cfg['mode_index'],amp = tom_amp,phase=tom_phase,len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])

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
            if kwargs['save']: self.save_sequences(multiple_sequences, kwargs['filename'])
        except:
            pass

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

    def save_sequences(self, multiple_sequences, filename = 'Z:\\2020-08-28-StimEm\\test'):
        seq_num = len(multiple_sequences)
        channel_waveforms = []
        for channel in self.channels:
            channel_waveform = []
            for seq_id in range(seq_num):
                channel_waveform.append(multiple_sequences[seq_id][channel])
            channel_waveforms.append(channel_waveform)
        np.save(filename, np.array(channel_waveforms))

    def get_awg_readout_time(self, readout_time_list):
        awg_readout_time_list = {}
        for awg in self.awg_info:
            awg_readout_time_list[awg] = (readout_time_list - self.awg_info[awg]['time_delay'])

        return awg_readout_time_list

    def ge_to_01(self, sequencer):  # pulses from file known to be qubit ge, qubit ef, cavity
        with File("S:/_Data/190826 - optimal_control_tests/sequences/00004_ge_to_01.h5", 'r') as f:
            pulses = f['uks'][-1]
            carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                             "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
                                      self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
                             "cavity": self.quantum_device_cfg['cavity']['1']['freq']}
            total_time = f['total_time'][()]
            counter = 0
            sequencer.append('charge1', ARB_Sum(A_list_list=[pulses[2 * counter], pulses[2 * counter + 2]],
                                                B_list_list=[pulses[2 * counter + 1], pulses[2 * counter + 3]],
                                                length=total_time,
                                                freq_list=[carrier_freqs["pi_ge"], carrier_freqs["pi_ef"]],
                                                phase_list=[0, 0], scale=1.0))
            counter += 2
            sequencer.append('cavity', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                           len=total_time, freq=carrier_freqs["cavity"], phase=0, scale=1.0))

    def ge_to_01_inverse(self, sequencer):  # pulses from file known to be qubit ge, qubit ef, cavity
        with File("S:/_Data/190826 - optimal_control_tests/sequences/00004_01_to_ge.h5", 'r') as f:
            pulses = f['uks'][-1]
            carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                             "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
                                      self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
                             "cavity": self.quantum_device_cfg['cavity']['1']['freq'],
                             "f0g1": self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][0]}
            total_time = f['total_time'][()]
            counter = 0
            sequencer.append('charge1', ARB_Sum(A_list_list=[pulses[2 * counter], pulses[2 * counter + 2]],
                                                B_list_list=[pulses[2 * counter + 1], pulses[2 * counter + 3]],
                                                length=total_time,
                                                freq_list=[carrier_freqs["pi_ge"], carrier_freqs["pi_ef"]],
                                                phase_list=[0, 0], scale=1.0))
            counter += 2
            sequencer.append('cavity', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                           len=total_time, freq=carrier_freqs["cavity"], phase=0, scale=1.0))

    def g0_to_g2_SNAP(self, sequencer):  # pulses from file known to be qubit ge, qubit ef, cavity
        with File("S:/_Data/190826 - optimal_control_tests/sequences/00000_g2_from_g0_SNAP.h5", 'r') as f:
            pulses = f['uks'][-1]
            carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                             "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
                                      self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
                             "cavity": self.quantum_device_cfg['cavity']['1']['freq'],
                             "f0g1": self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][0]}
            total_time = f['total_time'][()]
            counter = 0
            sequencer.append('charge1', ARB_Sum(A_list_list=[pulses[2 * counter], pulses[2 * counter + 2]],
                                                B_list_list=[pulses[2 * counter + 1], pulses[2 * counter + 3]],
                                                length=total_time,
                                                freq_list=[carrier_freqs["pi_ge"], carrier_freqs["pi_ef"]],
                                                phase_list=[0, 0], scale=1.0))
            counter += 2
            sequencer.append('cavity', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                           len=total_time, freq=carrier_freqs["cavity"], phase=0, scale=1.0))

    def prep_optimal_control_pulse_1step(self, sequencer, pulse_frac = 1.0, print_times=False, new=True, filename=None):
        if filename is None:
            filename = self.expt_cfg['filename']
        if filename.split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(filename, 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:
                    print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(filename)
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file
        num_pulses = len(pulses)
        if new:
            sequencer.new_sequence(self)
        # write pulses to their appropriate channels
        counter = 0
        # currently only works for 1 qubit, kept for loop for convention / similarity to other experiments
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['use_weak_drive']:
                qub_channel = 'qubitweak'
                qub_channel_transfer = 'qubitweak'
                print ("Using weak drive port for qubit")
                qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge_weak']
                qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef_weak']
            else:
                qub_channel = 'charge%s' % qubit_id
                qub_channel_transfer = "qubit"
                qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge']
                qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef']
            if self.expt_cfg['use_weak_cavity_drive']:
                cav_channel = 'qubitweak'
                cav_channel_transfer = 'cavity_weak'
                print("Using weak drive port for cavity")
                cav_scale = self.expt_cfg['calibrations']['cavity_weak']
            else:
                cav_channel = 'cavity'
                cav_channel_transfer = 'cavity'
                cav_scale = self.expt_cfg['calibrations']['cavity']

            # if carrier freqs not specified (empty list), assume qubit ge, qubit ef, cavity, f0g1 sideband
            # pulse h5 file also assumed to be in that order
            if not self.expt_cfg['carrier_freqs']:
                carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                                 "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
                                          self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
                                 "cavity": self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][
                                     self.expt_cfg['mode_index']],
                                 "f0g1": self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
                                     self.expt_cfg['mode_index']]}
            else:
                carrier_freqs = self.expt_cfg['carrier_freqs']
            if pulse_frac != 0:
                if qub_channel != cav_channel or not (self.expt_cfg['cavity_on'] and
                                                      (self.expt_cfg['ge_on'] or self.expt_cfg['ef_on'])):
                # combine qubit ge and ef pulses
                    if self.expt_cfg['ge_on'] and self.expt_cfg['ef_on']:
                        sequencer.append(qub_channel, ARB_Sum(A_list_list=[pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
                                                                                     pulses[self.expt_cfg['pulse_number_map']['ge'][1]]],
                                                                        B_list_list=[pulses[self.expt_cfg['pulse_number_map']['ef'][0]],
                                                                                     pulses[self.expt_cfg['pulse_number_map']['ef'][1]]],
                                                                        length=total_time * pulse_frac,
                                                                        freq_list=[carrier_freqs["pi_ge"],
                                                                                   carrier_freqs["pi_ef"]],
                                                                        phase_list=[0, 0],
                                                                        scale_list=[qub_scale_ge,qub_scale_ef]))
                    elif self.expt_cfg['ge_on']:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append(qub_channel,
                                             ARB(A_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
                                                                               channel=qub_channel_transfer),
                                                 B_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
                                                                               channel=qub_channel_transfer),
                                                 len=total_time * pulse_frac, freq=carrier_freqs["pi_ge"], phase=0))
                        else:
                            sequencer.append(qub_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['ge'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
                                                           len=total_time * pulse_frac,
                                                           freq=carrier_freqs["pi_ge"], phase=0,
                                                           scale=qub_scale_ge))

                    elif self.expt_cfg['ef_on']:
                        sequencer.append(qub_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['ef'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['ef'][0]],
                                                       len=total_time * pulse_frac,
                                                       freq=carrier_freqs["pi_ef"], phase=0,
                                                       scale=qub_scale_ef))

                    if self.expt_cfg['cavity_on']:  # cavity drive
                        if self.expt_cfg['try_transfer_function']:
                            # hard coded values of /3 since actually putting qubit drive in
                            sequencer.append(cav_channel,
                                             ARB(A_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], channel=cav_channel_transfer),
                                                 B_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][1]], channel=cav_channel_transfer),
                                                 len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0))
                        else:
                            sequencer.append(cav_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                                           len=total_time * pulse_frac,
                                                           freq=carrier_freqs["cavity"], phase=0,
                                                           scale=cav_scale))
                else:  # any overlapping can only happen through the weak channel
                    if self.expt_cfg['try_transfer_function']:
                        if self.expt_cfg['ge_on'] and not self.expt_cfg['ef_on']:
                            sequencer.append(qub_channel,
                                             ARB_Sum(A_list_list=[self.transfer_function(
                                                 pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
                                                 channel='qubitweak'), self.transfer_function(
                                                 pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
                                                 channel='cavity_weak')],
                                                 B_list_list=[self.transfer_function(
                                                     pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
                                                     channel='qubitweak'), self.transfer_function(
                                                     pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                                     channel='cavity_weak')],
                                                 length=total_time * pulse_frac, freq_list=[carrier_freqs["pi_ge"], carrier_freqs['cavity']],
                                                 phase_list=[0,0], scale_list=[1.0, 1.0]))
                        else:
                            print("code not written to handle this case yet")
                    else:
                        if self.expt_cfg['ge_on'] and not self.expt_cfg['ef_on']:
                            sequencer.append(qub_channel, ARB_Sum(
                                A_list_list=[pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
                                             pulses[self.expt_cfg['pulse_number_map']['ge'][1]]],
                                B_list_list=[pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
                                             pulses[self.expt_cfg['pulse_number_map']['cavity'][1]]],
                                length=total_time * pulse_frac,
                                freq_list=[carrier_freqs["pi_ge"],
                                           carrier_freqs["cavity"]],
                                phase_list=[0, 0],
                                scale_list=[qub_scale_ge, cav_scale]))
                        else:
                            print("code not written to handle this case yet")

                if self.expt_cfg['sideband_on']:  # sideband drive
                    sequencer.append('sideband', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                                     len=total_time * pulse_frac,
                                                     freq=carrier_freqs["f0g1"], phase=0,
                                                     scale=2*self.expt_cfg['calibrations']['sideband']))
                    counter += 1
        sequencer.sync_channels_time(self.channels)
    # def prep_optimal_control_pulse_1step(self, sequencer, pulse_frac = 1.0, print_times=False, new=True, filename=None):
    #     if filename is None:
    #         filename = self.expt_cfg['filename']
    #     if filename.split(".")[-1] == 'h5':  # detect if file is an h5 file
    #         with File(filename, 'r') as f:
    #             pulses = f['uks'][-1]
    #             total_time = f['total_time'][()] + 0.0
    #             dt = total_time / f['steps'][()]
    #             if print_times:
    #                 print(total_time, dt)
    #     else:  # if not h5, read from a text file
    #         pulses = np.genfromtxt(filename)
    #         total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file
    #     num_pulses = len(pulses)
    #     if new:
    #         sequencer.new_sequence(self)
    #     # write pulses to their appropriate channels
    #     counter = 0
    #     # currently only works for 1 qubit, kept for loop for convention / similarity to other experiments
    #     for qubit_id in self.expt_cfg['on_qubits']:
    #         if self.expt_cfg['use_weak_drive']:
    #             qub_channel = 'qubitweak'
    #             qub_channel_transfer = 'qubitweak'
    #             print ("Using weak drive port for qubit")
    #             qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge_weak']
    #             qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef_weak']
    #         else:
    #             qub_channel = 'charge%s' % qubit_id
    #             qub_channel_transfer = "qubit"
    #             qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge']
    #             qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef']
    #         if self.expt_cfg['use_weak_cavity_drive']:
    #             cav_channel = 'qubitweak'
    #             cav_channel_transfer = 'cavity_list_weak'
    #             print("Using weak drive port for cavity")
    #             cav_scale = self.expt_cfg['calibrations']['cavity_weak']
    #         else:
    #             cav_channel = 'cavity'
    #             cav_channel_transfer = 'cavity_list'
    #             cav_scale = self.expt_cfg['calibrations']['cavity']
    #
    #         # if carrier freqs not specified (empty list), assume qubit ge, qubit ef, cavity, f0g1 sideband
    #         # pulse h5 file also assumed to be in that order
    #         if not self.expt_cfg['carrier_freqs']:
    #             carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
    #                              "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
    #                                       self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
    #                              "cavity": self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][
    #                                  self.expt_cfg['mode_index']],
    #                              "f0g1": self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
    #                                  self.expt_cfg['mode_index']]}
    #         else:
    #             carrier_freqs = self.expt_cfg['carrier_freqs']
    #         if pulse_frac != 0:
    #             if qub_channel != cav_channel or not (self.expt_cfg['cavity_on'] and
    #                                                   (self.expt_cfg['ge_on'] or self.expt_cfg['ef_on'])):
    #             # combine qubit ge and ef pulses
    #                 if self.expt_cfg['ge_on'] and self.expt_cfg['ef_on']:
    #                     sequencer.append(qub_channel, ARB_Sum(A_list_list=[pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
    #                                                                                  pulses[self.expt_cfg['pulse_number_map']['ge'][1]]],
    #                                                                     B_list_list=[pulses[self.expt_cfg['pulse_number_map']['ef'][0]],
    #                                                                                  pulses[self.expt_cfg['pulse_number_map']['ef'][1]]],
    #                                                                     length=total_time * pulse_frac,
    #                                                                     freq_list=[carrier_freqs["pi_ge"],
    #                                                                                carrier_freqs["pi_ef"]],
    #                                                                     phase_list=[0, 0],
    #                                                                     scale_list=[qub_scale_ge,qub_scale_ef]))
    #                 elif self.expt_cfg['ge_on']:
    #                     if self.expt_cfg['try_transfer_function']:
    #                         sequencer.append(qub_channel,
    #                                          ARB(A_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
    #                                                                            channel=qub_channel_transfer),
    #                                              B_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
    #                                                                            channel=qub_channel_transfer),
    #                                              len=total_time * pulse_frac, freq=carrier_freqs["pi_ge"], phase=0))
    #                     else:
    #                         sequencer.append(qub_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['ge'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
    #                                                        len=total_time * pulse_frac,
    #                                                        freq=carrier_freqs["pi_ge"], phase=0,
    #                                                        scale=qub_scale_ge))
    #
    #                 elif self.expt_cfg['ef_on']:
    #                     sequencer.append(qub_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['ef'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['ef'][0]],
    #                                                    len=total_time * pulse_frac,
    #                                                    freq=carrier_freqs["pi_ef"], phase=0,
    #                                                    scale=qub_scale_ef))
    #
    #                 if self.expt_cfg['cavity_on']:  # cavity drive
    #                     if self.expt_cfg['try_transfer_function']:
    #                         # hard coded values of /3 since actually putting qubit drive in
    #                         sequencer.append(cav_channel,
    #                                          ARB(A_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
    #                                                                            channel=cav_channel_transfer, mode_index=self.expt_cfg['mode_index']),
    #                                              B_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
    #                                                                            channel=cav_channel_transfer, mode_index=self.expt_cfg['mode_index']),
    #                                              len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0))
    #                     else:
    #                         sequencer.append(cav_channel, ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
    #                                                        len=total_time * pulse_frac,
    #                                                        freq=carrier_freqs["cavity"], phase=0,
    #                                                        scale=cav_scale))
    #             else:  # any overlapping can only happen through the weak channel
    #                 if self.expt_cfg['try_transfer_function']:
    #                     if self.expt_cfg['ge_on'] and not self.expt_cfg['ef_on']:
    #                         sequencer.append(qub_channel,
    #                                          ARB_Sum(A_list_list=[self.transfer_function(
    #                                              pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
    #                                              channel=qub_channel_transfer), self.transfer_function(
    #                                              pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
    #                                              channel=cav_channel_transfer, mode_index=self.expt_cfg['mode_index'])],
    #                                              B_list_list=[self.transfer_function(
    #                                                  pulses[self.expt_cfg['pulse_number_map']['ge'][1]],
    #                                                  channel=qub_channel_transfer), self.transfer_function(
    #                                                  pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
    #                                                  channel=cav_channel_transfer, mode_index=self.expt_cfg['mode_index'])],
    #                                              length=total_time * pulse_frac, freq_list=[carrier_freqs["pi_ge"], carrier_freqs['cavity']],
    #                                              phase_list=[0,0], scale_list=[1.0, 1.0]))
    #                     else:
    #                         print("code not written to handle this case yet")
    #                 else:
    #                     if self.expt_cfg['ge_on'] and not self.expt_cfg['ef_on']:
    #                         sequencer.append(qub_channel, ARB_Sum(
    #                             A_list_list=[pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
    #                                          pulses[self.expt_cfg['pulse_number_map']['ge'][1]]],
    #                             B_list_list=[pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
    #                                          pulses[self.expt_cfg['pulse_number_map']['cavity'][1]]],
    #                             length=total_time * pulse_frac,
    #                             freq_list=[carrier_freqs["pi_ge"],
    #                                        carrier_freqs["cavity"]],
    #                             phase_list=[0, 0],
    #                             scale_list=[qub_scale_ge, cav_scale]))
    #                     else:
    #                         print("code not written to handle this case yet")
    #
    #             if self.expt_cfg['sideband_on']:  # sideband drive
    #                 sequencer.append('sideband', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
    #                                                  len=total_time * pulse_frac,
    #                                                  freq=carrier_freqs["f0g1"], phase=0,
    #                                                  scale=2*self.expt_cfg['calibrations']['sideband']))
    #                 counter += 1
    #     sequencer.sync_channels_time(self.channels)

    def prep_optimal_control_pulse_before_blockade(self, sequencer, pulse_frac=1.0, print_times=False):
        if self.expt_cfg['prep_state_filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['prep_state_filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:
                    print(total_time, dt)
        else:  # if not h5, read from a text file
            print("not an h5 file, will not handle properly!!!")
        num_pulses = len(pulses)
        sequencer.new_sequence(self)
        # write pulses to their appropriate channels
        counter = 0
        # currently only works for 1 qubit, kept for loop for convention / similarity to other experiments
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['prep_use_weak_drive']:
                qub_channel = 'qubitweak'
                print ("Using weak drive port for qubit")
                qub_scale_ge = self.expt_cfg['prep_calibrations']['qubit_ge_weak']
                qub_channel_transfer = "qubitweak"
            else:
                qub_channel = 'charge%s' % qubit_id
                qub_channel_transfer = "qubit"
                qub_scale_ge = self.expt_cfg['prep_calibrations']['qubit_ge']
            if self.expt_cfg['prep_use_weak_cavity_drive']:
                cav_channel = 'qubitweak'
                cav_channel_transfer = 'cavity_weak'
                print("Using weak drive port for cavity")
                cav_scale = self.expt_cfg['prep_calibrations']['cavity_weak']
            else:
                cav_channel = 'cavity'
                cav_channel_transfer = 'cavity'
                cav_scale = self.expt_cfg['prep_calibrations']['cavity']

            # if carrier freqs not specified (empty list), assume qubit ge, qubit ef, cavity, f0g1 sideband
            # pulse h5 file also assumed to be in that order
            if not self.expt_cfg['carrier_freqs']:
                carrier_freqs = {"pi_ge": self.quantum_device_cfg['qubit'][qubit_id]['freq'],
                                 "pi_ef": self.quantum_device_cfg['qubit'][qubit_id]['freq'] + \
                                          self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'],
                                 "cavity": self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][
                                     self.expt_cfg['mode_index']],
                                 "f0g1": self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][
                                     self.expt_cfg['mode_index']]}
            else:
                carrier_freqs = self.expt_cfg['carrier_freqs']
            if self.expt_cfg['prep_using_blockade_oc']:
                print("preparing state using blockaded optimal control")
                if pulse_frac != 0:
                    # write pulses to their appropriate channels
                    for qubit_id in self.expt_cfg['on_qubits']:
                        freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + 2 * \
                                   self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['prep_mode_index']] * \
                                   np.array(self.expt_cfg['prep_blockade_params']['levels'])
                        if self.expt_cfg['prep_use_weak_cavity_drive'] and self.expt_cfg['prep_cavity_on']:
                            print("Using weak drive port for cavity")
                            cav_scale = self.expt_cfg['prep_calibrations']['cavity_weak']
                            if self.expt_cfg['prep_use_weak_drive_for_dressing']:
                                if self.expt_cfg['try_transfer_function']:
                                    sequencer.append('qubitweak',
                                                     ARB_with_blockade(A_list=self.transfer_function(
                                                         pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                         channel='cavity_weak'),
                                                                       B_list=self.transfer_function(pulses[
                                                                                                         self.expt_cfg[
                                                                                                             'prep_pulse_number_map'][
                                                                                                             'cavity'][
                                                                                                             1]],
                                                                                                     channel='cavity_weak'),
                                                                       len=total_time * pulse_frac,
                                                                       freq=carrier_freqs["cavity"], phase=0,
                                                                       blockade_amp=self.expt_cfg['prep_blockade_params'][
                                                                           'amp'],
                                                                       blockade_freqs=freqlist, blockade_pulse_type=
                                                                       self.expt_cfg['prep_blockade_params']['pulse_type']))
                                else:
                                    sequencer.append('qubitweak', ARB_with_blockade(
                                        A_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                        B_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                        len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                        scale=cav_scale, blockade_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                        blockade_freqs=freqlist,
                                        blockade_pulse_type=self.expt_cfg['prep_blockade_params']['pulse_type']))
                            else:
                                if self.expt_cfg['try_transfer_function']:
                                    sequencer.append('qubitweak',
                                                     ARB(A_list=self.transfer_function(
                                                         pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                         channel='cavity_weak'),
                                                         B_list=self.transfer_function(
                                                             pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                             channel='cavity_weak'),
                                                         len=total_time * pulse_frac, freq=carrier_freqs["cavity"],
                                                         phase=0))
                                else:
                                    sequencer.append('qubitweak',
                                                     ARB(A_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                         B_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                         len=total_time * pulse_frac, freq=carrier_freqs["cavity"],
                                                         phase=0,
                                                         scale=cav_scale))
                                if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                                    sequencer.append('charge%s' % qubit_id,
                                                     Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                      flat_len=total_time * pulse_frac,
                                                                      ramp_sigma_len=0.001,
                                                                      cutoff_sigma=2, freqs=freqlist,
                                                                      phases=np.zeros(len(freqlist))))
                                elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                                    sequencer.append('charge%s' % qubit_id,
                                                     Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                     sigma_len=total_time * pulse_frac / (2 * 2.0),
                                                                     cutoff_sigma=2,
                                                                     freqs=freqlist, phases=np.zeros(len(freqlist))))
                                else:
                                    print("blockade pulse type not recognized, not blockading")
                        elif self.expt_cfg['prep_cavity_on']:
                            cav_scale = self.expt_cfg['prep_calibrations']['cavity']
                            if self.expt_cfg['try_transfer_function']:
                                sequencer.append('cavity',
                                                 ARB(A_list=self.transfer_function(
                                                     pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                     channel='cavity'),
                                                     B_list=self.transfer_function(
                                                         pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                         channel='cavity'),
                                                     len=total_time * pulse_frac, freq=carrier_freqs["cavity"],
                                                     phase=0))
                            else:
                                sequencer.append('cavity',
                                                 ARB(A_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                     B_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                     len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                                     scale=cav_scale,
                                                     blockade_amp=self.expt_cfg['prep_blockade_params']['amp']))
                            if self.expt_cfg['use_weak_drive_for_dressing']:
                                if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                                    sequencer.append('qubitweak',
                                                     Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                      flat_len=total_time * pulse_frac,
                                                                      ramp_sigma_len=0.001,
                                                                      cutoff_sigma=2, freqs=freqlist,
                                                                      phases=np.zeros(len(freqlist))))
                                elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                                    sequencer.append('qubitweak',
                                                     Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                     sigma_len=total_time * pulse_frac / 4.0,
                                                                     cutoff_sigma=2,
                                                                     freqs=freqlist, phases=np.zeros(len(freqlist))))
                                else:
                                    print("blockade pulse type not recognized, not blockading")
                            else:
                                if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                                    sequencer.append('charge%s' % qubit_id,
                                                     Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                      flat_len=total_time * pulse_frac,
                                                                      ramp_sigma_len=0.001,
                                                                      cutoff_sigma=2, freqs=freqlist,
                                                                      phases=np.zeros(len(freqlist))))
                                elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                                    sequencer.append('charge%s' % qubit_id,
                                                     Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                     sigma_len=total_time * pulse_frac / 4.0,
                                                                     cutoff_sigma=2,
                                                                     freqs=freqlist, phases=np.zeros(len(freqlist))))
                                else:
                                    print("blockade pulse type not recognized, not blockading")
                        else:
                            print("cavity drive not on during prep before blockade experiment")
            elif self.expt_cfg['prep_using_blockade_rabi']:
                print("preparing state with blockade rabi")
                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                # print (blockade_pulse_info)
                mode_index = self.expt_cfg['prep_mode_index']
                sequencer.new_sequence(self)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                            len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                            cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                mode_index],
                                            use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                mode_index],
                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                            blockade_levels=[2], dressing_pulse_type="square",
                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                            phase=0,
                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                            weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                'use_weak_for_cavity'])
            else:
                if pulse_frac != 0:
                    if qub_channel != cav_channel or not (self.expt_cfg['cavity_on'] and self.expt_cfg['ge_on']):
                        if self.expt_cfg['prep_ge_on']:
                            if self.expt_cfg['try_transfer_function']:
                                sequencer.append(qub_channel,
                                                 ARB(A_list=self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map']['ge'][0]],
                                                                                   channel=qub_channel_transfer),
                                                     B_list=self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map']['ge'][1]],
                                                                                   channel=qub_channel_transfer),
                                                     len=total_time * pulse_frac, freq=carrier_freqs["pi_ge"], phase=0))
                            else:
                                sequencer.append(qub_channel, ARB(A_list=pulses[self.expt_cfg['prep_pulse_number_map']['ge'][0]],
                                                                  B_list=pulses[self.expt_cfg['prep_pulse_number_map']['ge'][1]],
                                                               len=total_time * pulse_frac,
                                                               freq=carrier_freqs["pi_ge"], phase=0,
                                                               scale=qub_scale_ge))
                        if self.expt_cfg['prep_cavity_on']:  # cavity drive
                            if self.expt_cfg['try_transfer_function']:
                                sequencer.append(cav_channel,
                                                 ARB(A_list=self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]], channel=cav_channel_transfer),
                                                     B_list=self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]], channel=cav_channel_transfer),
                                                     len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0))
                            else:
                                sequencer.append(cav_channel, ARB(A_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                                  B_list=pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                               len=total_time * pulse_frac,
                                                               freq=carrier_freqs["cavity"], phase=0,
                                                               scale=cav_scale))
                    else:  # any overlapping can only happen through the weak channel
                        if self.expt_cfg['try_transfer_function']:
                            if self.expt_cfg['prep_ge_on']:
                                sequencer.append(qub_channel,
                                                 ARB_Sum(A_list_list=[self.transfer_function(
                                                     pulses[self.expt_cfg['prep_pulse_number_map']['ge'][0]],
                                                     channel='qubitweak'), self.transfer_function(
                                                     pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                     channel='cavity_weak')],
                                                     B_list_list=[self.transfer_function(
                                                         pulses[self.expt_cfg['prep_pulse_number_map']['ge'][1]],
                                                         channel='qubitweak'), self.transfer_function(
                                                         pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]],
                                                         channel='cavity_weak')],
                                                     length=total_time * pulse_frac, freq_list=[carrier_freqs["pi_ge"], carrier_freqs['cavity']],
                                                     phase_list=[0,0], scale_list=[1.0, 1.0]))
                            else:
                                print("code not written to handle this case yet")
                        else:
                            if self.expt_cfg['prep_ge_on']:
                                sequencer.append(qub_channel, ARB_Sum(
                                    A_list_list=[pulses[self.expt_cfg['prep_pulse_number_map']['ge'][0]],
                                                 pulses[self.expt_cfg['prep_pulse_number_map']['ge'][1]]],
                                    B_list_list=[pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][0]],
                                                 pulses[self.expt_cfg['prep_pulse_number_map']['cavity'][1]]],
                                    length=total_time * pulse_frac,
                                    freq_list=[carrier_freqs["pi_ge"],
                                               carrier_freqs["cavity"]],
                                    phase_list=[0, 0],
                                    scale_list=[qub_scale_ge, cav_scale]))
                            else:
                                print("code not written to handle this case yet")
        sequencer.sync_channels_time(self.channels)

    def optimal_control_test(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        for ii in range(self.expt_cfg['steps']):
            if self.expt_cfg['measurement'] == 'wigner_tomography_1d':
                for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                    self.prep_optimal_control_pulse(sequencer,ii=ii, print_times=print_times)
                    print_times = False
                    self.idle_all(sequencer, time=5.0)
                    if self.expt_cfg['sweep_phase']:
                        self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'],
                                               amp=self.expt_cfg['amp'], phase=x, len=self.expt_cfg['cavity_pulse_len'],
                                               pulse_type=self.expt_cfg['tomography_pulse_type'])
                    else:
                        self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=x,
                                               phase=self.expt_cfg['phase'], len=self.expt_cfg['cavity_pulse_len'],
                                               pulse_type=self.expt_cfg['tomography_pulse_type'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()

            elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
                for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                    self.prep_optimal_control_pulse(sequencer,ii=ii, print_times=print_times)
                    print_times = False
                    self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                               phase=0, pulse_type=self.expt_cfg['pulse_type'],
                               add_freq=df)
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()

            elif self.expt_cfg['measurement'] == 'photon_number_distribution_measurement':
                for n in np.arange(self.expt_cfg['N_max']):
                    self.prep_optimal_control_pulse(sequencer,ii=ii, print_times=print_times)
                    print_times = False
                    self.pi_q_resolved(sequencer, qubit_id,
                                       pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                       add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                                       [self.expt_cfg['mode_index']] * n)
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
            else:
                print("!!!Experiment not recognized!!!")

        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_test_1step(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
    
        if self.expt_cfg['measurement'] == 'wigner_tomography_1d':
            for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.idle_all(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'],
                                           amp=self.expt_cfg['amp'], phase=x, len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                else:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=x,
                                           phase=self.expt_cfg['phase'], len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'wigner_tomography_2d':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])

            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2),
                                                              channel='cavity_amp_vs_freq_list')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=False)
                sequencer.sync_channels_time(self.channels)

                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    self.pi_q(sequencer, qubit_id, pulse_type = 'gauss',
                              add_freq=self.expt_cfg['add_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df)
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                            add_freq= df)

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'photon_number_distribution_measurement':
            for n in np.arange(self.expt_cfg['N_max']):
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                                   [self.expt_cfg['mode_index']] * n,use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'rabi':
            for ii in range(self.expt_cfg['rabi_steps'] + 1):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac']*ii/
                                                                            self.expt_cfg['rabi_steps'],
                                                      print_times=print_times)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'qubit_tomography':
            for ii in range(3):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    if not self.expt_cfg['resolved']:
                        self.pi_q(sequencer, qubit_id, pulse_type = 'gauss',
                              add_freq=self.expt_cfg['add_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                    else:
                        self.pi_q_resolved(sequencer, qubit_id, 
                              add_freq=self.expt_cfg['add_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']],use_weak_drive=self.expt_cfg['use_weak_drive'])
                    sequencer.sync_channels_time(self.channels)
                if ii == 0:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   add_freq=self.expt_cfg['tomography_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                elif ii == 1:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = -np.pi/2,
                                   add_freq=self.expt_cfg['tomography_freq_end'] * 2 *self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                else:pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'resolved_qubit_tomography':
            for ii in range(3):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    self.pi_q(sequencer, qubit_id, pulse_type = 'gauss',
                              add_freq=self.expt_cfg['add_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)
                if ii == 0:
                    self.half_pi_q_resolved(sequencer, qubit_id,
                                   add_freq=self.expt_cfg['tomography_freq_end']*2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                elif ii  == 1:
                    self.half_pi_q_resolved(sequencer, qubit_id, phase = -np.pi/2,
                                   add_freq=self.expt_cfg['tomography_freq_end'] * 2 *self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
                else:pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'readout_cross_kerr_debug':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        else:
            print("!!!Experiment not recognized!!!")
        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_test_1step_partial_2D_wt(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        if self.expt_cfg['measurement'] == 'wigner_tomography_2d':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()])[self.expt_cfg['wt_start_index'] : self.expt_cfg['wt_end_index']] / \
                         (self.expt_cfg['cavity_pulse_len'])
                    ys = np.array(f['alphay'][()])[self.expt_cfg['wt_start_index'] : self.expt_cfg['wt_end_index']] / \
                         (self.expt_cfg['cavity_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])[self.expt_cfg['wt_start_index'] : self.expt_cfg['wt_end_index']]
                    ys = np.array(f['ay'])[self.expt_cfg['wt_start_index'] : self.expt_cfg['wt_end_index']]

            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2),
                                                              channel='cavity_amp_vs_freq_list')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=False)
                sequencer.sync_channels_time(self.channels)

                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        else:
            print("!!!Experiment not recognized!!!")
        return sequencer.complete(self, plot=self.plot_visdom)
    
    def optimal_control_test_ef_spectroscopy(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                  print_times=print_times)
            print_times = False
            if self.expt_cfg['measure_e']:
                self.pi_q(sequencer, qubit_id, pulse_type='gauss', add_freq=0)
            sequencer.sync_channels_time(self.channels)
            self.pi_q_resolved_ef(sequencer, qubit_id, add_freq=df, phase=0.0,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
            sequencer.sync_channels_time(self.channels)
            self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'], add_freq=0)
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_test_1step_recovery(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True

        if self.expt_cfg['measurement'] == 'wigner_tomography_1d':
            for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                print_times = False
                self.idle_all(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'],
                                           amp=self.expt_cfg['amp'], phase=x, len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                else:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=x,
                                           phase=self.expt_cfg['phase'], len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'wigner_tomography_2d':

            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

            for ii, y in enumerate(ys):
                x = xs[ii]

                tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=False)
                sequencer.sync_channels_time(self.channels)

                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()


        elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    self.pi_q(sequencer, qubit_id, pulse_type='gauss',
                              add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                       self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                           self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)

                if self.expt_cfg['recovery_at_end']:
                    self.gen_q(sequencer, qubit_id, pulse_type=self.expt_cfg['recovery_pulse_type'],
                               phase = self.expt_cfg['recovery_angle'], 
                               len=self.expt_cfg['recovery_pulse_length'], amp=self.expt_cfg['recovery_amp'],
                              add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                       self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                           self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)
                    
                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],
                                        amp=self.expt_cfg['amp'],
                                        phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                        add_freq=df)
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df)

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'photon_number_distribution_measurement':
            for n in np.arange(self.expt_cfg['N_max']):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                print_times = False
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                                   [self.expt_cfg['mode_index']] * n,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'rabi':
            for ii in range(self.expt_cfg['rabi_steps'] + 1):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'] * ii /
                                                                            self.expt_cfg['rabi_steps'],
                                                      print_times=print_times)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'qubit_tomography':
            for ii in range(3):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    if not self.expt_cfg['resolved']:
                        self.pi_q(sequencer, qubit_id, pulse_type='gauss',
                                  add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                           self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                               self.expt_cfg['mode_index']])
                    else:
                        self.pi_q_resolved(sequencer, qubit_id,
                                           add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                                    self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                        self.expt_cfg['mode_index']],
                                           use_weak_drive=self.expt_cfg['use_weak_drive'])
                    sequencer.sync_channels_time(self.channels)
                    
                if self.expt_cfg['recovery_at_end']:
                    self.gen_q(sequencer, qubit_id, pulse_type=self.expt_cfg['recovery_pulse_type'],
                               phase = self.expt_cfg['recovery_angle'], 
                               len=self.expt_cfg['recovery_pulse_length'], amp=self.expt_cfg['recovery_amp'],
                              add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                       self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                           self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)

                if ii == 0:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   add_freq=self.expt_cfg['tomography_freq_end'] * 2 *
                                            self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                self.expt_cfg['mode_index']])
                elif ii == 1:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=-np.pi / 2,
                                   add_freq=self.expt_cfg['tomography_freq_end'] * 2 *
                                            self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                self.expt_cfg['mode_index']])
                else:
                    pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'resolved_qubit_tomography':
            for ii in range(3):
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False
                if self.expt_cfg['pi_at_end']:
                    self.pi_q(sequencer, qubit_id, pulse_type='gauss',
                              add_freq=self.expt_cfg['add_freq_end'] * 2 *
                                       self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                           self.expt_cfg['mode_index']])
                    sequencer.sync_channels_time(self.channels)
                if ii == 0:
                    self.half_pi_q_resolved(sequencer, qubit_id,
                                            add_freq=self.expt_cfg['tomography_freq_end'] * 2 *
                                                     self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                         self.expt_cfg['mode_index']])
                elif ii == 1:
                    self.half_pi_q_resolved(sequencer, qubit_id, phase=-np.pi / 2,
                                            add_freq=self.expt_cfg['tomography_freq_end'] * 2 *
                                                     self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                         self.expt_cfg['mode_index']])
                else:
                    pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        else:
            print("!!!Experiment not recognized!!!")
        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_repeated_pi_pulses(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True

        count = 1 + self.expt_cfg['readout_before_oct'] + self.expt_cfg['num_pi_pulses_at_n'] + self.expt_cfg['num_pi_pulses_at_m']

        if self.expt_cfg['cavity_amp'] != 0.0:
            count += 1

        for j in np.arange(count):
            sequencer.new_sequence(self)

            if j < self.expt_cfg['readout_before_oct']:
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            elif j == self.expt_cfg['readout_before_oct']:
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times, new=False)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                
            elif j <= self.expt_cfg['num_pi_pulses_at_n'] + self.expt_cfg['readout_before_oct']:
                add_freq = 2 * self.expt_cfg['pi_at_n'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                          'chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['pi_at_n'] **2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chi_2_by2pi'][self.expt_cfg['mode_index']]
                self.pi_q_resolved(sequencer, qubit_id, add_freq=add_freq, phase=0.0,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                
            elif self.expt_cfg['cavity_amp'] != 0.0 and j == 1 + self.expt_cfg['num_pi_pulses_at_n'] + self.expt_cfg['readout_before_oct']:
                if self.expt_cfg['use_weak_cavity_for_n_to_m']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                                    amp=self.expt_cfg['cavity_amp'], add_freq=0, phase=0,
                                    pulse_type=self.expt_cfg['cavity_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                           amp=self.expt_cfg['cavity_amp'], add_freq=0, phase=0, pulse_type=self.expt_cfg['cavity_pulse_type'])
                    """Included a resolved pi pulse at |m> after coherent drive to project the cavity state"""
                    add_freq = 2 * self.expt_cfg['pi_at_m'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                        'chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['pi_at_m'] **2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                   'chi_2_by2pi'][self.expt_cfg['mode_index']]
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=add_freq, phase=0.0,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                    """Edit ends here"""
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            else:
                add_freq = 2 * self.expt_cfg['pi_at_m'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['pi_at_m'] **2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                    'chi_2_by2pi'][self.expt_cfg['mode_index']]
                self.pi_q_resolved(sequencer, qubit_id, add_freq=add_freq, phase=0.0,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_repeated_pi_pulses_wt(self, sequencer):
        """Performing WT after executing the first part of stim Em experiment. What state the cavity is left in after coherent drive!"""
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True

        # need to update count to account for the number of Wigner tomography experiments (different cavity displacements)
        count = 1 + self.expt_cfg['readout_before_oct'] + self.expt_cfg['num_pi_pulses_at_n'] + 1

        if self.expt_cfg['cavity_amp'] != 0.0:
            count += 1

        for j in np.arange(count):
            sequencer.new_sequence(self)

            if j < self.expt_cfg['readout_before_oct']:
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            elif j == self.expt_cfg['readout_before_oct']:
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times, new=False)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            elif j <= self.expt_cfg['num_pi_pulses_at_n'] + self.expt_cfg['readout_before_oct']:
                add_freq = 2 * self.expt_cfg['pi_at_n'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['pi_at_n'] **2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                               'chi_2_by2pi'][self.expt_cfg['mode_index']]
                self.pi_q_resolved(sequencer, qubit_id, add_freq=add_freq, phase=0.0,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            # still want to do this cavity displacement? Yes - Ankur
            elif self.expt_cfg['cavity_amp'] != 0.0 and j == 1 + self.expt_cfg['num_pi_pulses_at_n'] + self.expt_cfg['readout_before_oct']:
                if self.expt_cfg['use_weak_cavity_for_n_to_m']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                                    amp=self.expt_cfg['cavity_amp'], add_freq=0, phase=0,
                                    pulse_type=self.expt_cfg['cavity_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                               amp=self.expt_cfg['cavity_amp'], add_freq=0, phase=0, pulse_type=self.expt_cfg['cavity_pulse_type'])
                    # add_freq = 2 * self.expt_cfg['pi_at_m'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    #     'chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg['pi_at_m'] **2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    #                'chi_2_by2pi'][self.expt_cfg['mode_index']]
                    # self.pi_q_resolved(sequencer, qubit_id, add_freq=add_freq, phase=0.0,
                    #                    use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            else:
                print("Doing the WT")
                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=self.expt_cfg['tom_amp'],
                                       phase=self.expt_cfg['tom_phase'], len=self.expt_cfg['cavity_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def optimal_control_histogram(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        count = 2 + self.expt_cfg['readout_before_oct'] 

        for j in np.arange(count):

            sequencer.new_sequence(self)

            if j < self.expt_cfg['readout_before_oct']:
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                # print("dummy readout %.f"%j)

            elif j == self.expt_cfg['readout_before_oct']:
                self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times, new=False)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                print("readout after oct %.f"%j)
            elif j < 2 + self.expt_cfg['readout_before_oct'] and self.expt_cfg['num_pi_pulses_at_n']==0:
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                # print("2nd readout after oct %.f"%j)

            elif j < 2 + self.expt_cfg['readout_before_oct'] and self.expt_cfg['num_pi_pulses_at_n']>0:
                add_freq = 2 * self.expt_cfg['pi_at_n'] * self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                    'chiby2pi_e'][self.expt_cfg['mode_index']]
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'], add_freq=add_freq)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                # print("2nd readout after pi %.f"%j)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    ### in progress ###
    def heeres_snap_phase_evolution_measurement(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            print_times = True
            for delta_T in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['excite_qubit_at_beginning']:
                    self.pi_q(sequencer, qubit_id=qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['use_weak_for_displace1']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len1'],
                                    amp=self.expt_cfg['displace_amp1'], phase=self.expt_cfg['displace_phase1'],
                                    pulse_type=self.expt_cfg['displace_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len1'],
                               amp=self.expt_cfg['displace_amp1'], phase=self.expt_cfg['displace_phase1'],
                               pulse_type=self.expt_cfg['displace_pulse_type'])
                self.idle_all(sequencer, qubit_id=qubit_id, time=delta_T)
                # # below is incorrect. should be a 2pi pulse on n+1?
                # self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                #                                       print_times=print_times, new=False)
                print_times = False
                if self.expt_cfg['use_weak_for_2pi']:
                    self.gen_q_weak(sequencer, qubit_id, len=self.pulse_info[qubit_id]['2pi_len_resolved_weak'],
                               amp=self.pulse_info[qubit_id]['2pi_amp_resolved_weak'], phase=0,
                               add_freq=(self.expt_cfg['n'] + 1)*self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                   'chiby2pi_e'][self.expt_cfg['mode_index']],
                               pulse_type=self.pulse_info[qubit_id]['resolved_pulse_type_weak'])
                else:
                    self.gen_q(sequencer, qubit_id, len=self.pulse_info[qubit_id]['2pi_len_resolved'],
                            amp=self.pulse_info[qubit_id]['2pi_amp_resolved'], phase=0,
                               add_freq=(self.expt_cfg['n'] + 1)*self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                   'chiby2pi_e'][self.expt_cfg['mode_index']],
                               pulse_type=self.pulse_info[qubit_id]['resolved_pulse_type'])

                if self.expt_cfg['use_weak_for_displace2']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len2'],
                                    amp=self.expt_cfg['displace_amp2'], phase=self.expt_cfg['displace_phase1'],
                                    pulse_type=self.expt_cfg['displace_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len2'],
                               amp=self.expt_cfg['displace_amp2'], phase=self.expt_cfg['displace_phase1'],
                               pulse_type=self.expt_cfg['displace_pulse_type'])
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                                   [self.expt_cfg['mode_index']] * self.expt_cfg['n'],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def heeres_snap_pop_vs_phase_measurement(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            print_times = True
            for theta in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['excite_qubit_at_beginning']:
                    self.pi_q(sequencer, qubit_id=qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['use_weak_for_displace1']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len1'],
                                    amp=self.expt_cfg['displace_amp1'], phase=self.expt_cfg['displace_phase1'],
                                    pulse_type=self.expt_cfg['displace_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len1'],
                               amp=self.expt_cfg['displace_amp1'], phase=self.expt_cfg['displace_phase1'],
                               pulse_type=self.expt_cfg['displace_pulse_type'])
                self.idle_all(sequencer, qubit_id=qubit_id, time=self.expt_cfg['deltaT'])
                # # below is incorrect. should be a 2pi pulse on n+1?
                # self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                #                                       print_times=print_times, new=False)
                print_times = False
                if self.expt_cfg['use_weak_for_2pi']:
                    self.gen_q_weak(sequencer, qubit_id, len=self.pulse_info[qubit_id]['2pi_len_resolved_weak'],
                               amp=self.pulse_info[qubit_id]['2pi_amp_resolved_weak'], phase=theta,
                               add_freq=(self.expt_cfg['n'] + 1)*self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                   'chiby2pi_e'][self.expt_cfg['mode_index']],
                               pulse_type=self.pulse_info[qubit_id]['resolved_pulse_type_weak'])
                else:
                    self.gen_q(sequencer, qubit_id, len=self.pulse_info[qubit_id]['2pi_len_resolved'],
                            amp=self.pulse_info[qubit_id]['2pi_amp_resolved'], phase=theta,
                               add_freq=(self.expt_cfg['n'] + 1)*self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                   'chiby2pi_e'][self.expt_cfg['mode_index']],
                               pulse_type=self.pulse_info[qubit_id]['resolved_pulse_type'])

                if self.expt_cfg['use_weak_for_displace2']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len2'],
                                    amp=self.expt_cfg['displace_amp2'], phase=self.expt_cfg['displace_phase1'],
                                    pulse_type=self.expt_cfg['displace_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['displace_len2'],
                               amp=self.expt_cfg['displace_amp2'], phase=self.expt_cfg['displace_phase1'],
                               pulse_type=self.expt_cfg['displace_pulse_type'])
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']
                                   [self.expt_cfg['mode_index']] * self.expt_cfg['n'],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

    def weak_rabi(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q_weak(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=self.expt_cfg['add_freq'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_photon_resolved_qubit_rabi(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['add_photon_with_blockade']:
                    blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                    # print (blockade_pulse_info)
                    mode_index = self.expt_cfg['mode_index']
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                    self.gen_q_weak(sequencer, qubit_id, len=rabi_len, amp=self.expt_cfg['amp'], phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                    add_freq=2*self.expt_cfg['rabi_level']*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq_weak(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'], phase=0,
                                pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq)


                # sequencer.append('charge%s_Q' % qubit_id,
                #                  Square(max_amp=self.expt_cfg['amp'], flat_len=self.expt_cfg['pulse_length'],
                #                         ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                #                         phase=self.pulse_info[qubit_id]['Q_phase']))
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_temp_rabi(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['use_weak_drive']:
                    if self.expt_cfg['drive_1']:
                        self.gen_q_weak(sequencer, qubit_id, len=rabi_len,
                                   amp=self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved_weak'], phase=0,
                                   add_freq=2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']],
                                   pulse_type=self.expt_cfg['pulse_type'])
                    else:
                        self.gen_q_weak(sequencer, qubit_id, len=rabi_len, amp=self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved_weak'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'])
                else:
                    if self.expt_cfg['drive_1']:
                        self.gen_q(sequencer, qubit_id, len=rabi_len,
                                   amp=self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved'], phase=0,
                                   add_freq=2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']],
                                   pulse_type=self.expt_cfg['pulse_type'])
                    else:
                        self.gen_q(sequencer, qubit_id, len=rabi_len, amp=self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def blockaded_cavity_rabi(self, sequencer):
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['cavity_pulse_type'] == "gauss":tau = rabi_len*4.0
                else:tau=rabi_len
                if self.expt_cfg["use_weak_drive_for_dressing"]:
                    self.gen_q_weak(sequencer, qubit_id, len=tau,
                               amp=self.expt_cfg['dressing_amp'], phase=0,
                               add_freq=2 * self.expt_cfg['blockade_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],
                               pulse_type=self.expt_cfg['dressing_pulse_type'])
                else:
                    self.gen_q(sequencer, qubit_id, len=rabi_len,
                           amp=self.expt_cfg['dressing_amp'], phase=0,
                           add_freq=2*self.expt_cfg['blockade_level']* self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],
                           pulse_type=self.expt_cfg['dressing_pulse_type'])


                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=rabi_len,
                           amp=self.expt_cfg['cavity_amp'], phase=self.expt_cfg['cavity_phase'],
                           pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def blockaded_cavity_rabi_wigner_tomography_2d_sweep(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']
        rabi_len = self.expt_cfg['rabi_len']

        if self.expt_cfg['wigner_points_from_file']:
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

            for ii,y in enumerate(ys):
                x = xs[ii]

                tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                sequencer.new_sequence(self)

                for qubit_id in self.expt_cfg['on_qubits']:
                    if self.expt_cfg['cavity_pulse_type'] == "gauss":
                        tau = rabi_len * 4.0
                    else:
                        tau = rabi_len
                    if self.expt_cfg["use_weak_drive_for_dressing"]:
                        self.gen_q_weak(sequencer, qubit_id, len=tau,
                                        amp=self.expt_cfg['dressing_amp'], phase=0,
                                        add_freq=2 * self.expt_cfg['blockade_level'] *
                                                 self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                     0],
                                        pulse_type=self.expt_cfg['dressing_pulse_type'])
                    else:
                        self.gen_q(sequencer, qubit_id, len=rabi_len,
                                   amp=self.expt_cfg['dressing_amp'], phase=0,
                                   add_freq=2 * self.expt_cfg['blockade_level'] *
                                            self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],
                                   pulse_type=self.expt_cfg['dressing_pulse_type'])

                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=rabi_len,
                               amp=self.expt_cfg['cavity_amp'], phase=self.expt_cfg['cavity_phase'],
                               pulse_type=self.expt_cfg['cavity_pulse_type'])
                    sequencer.sync_channels_time(self.channels)

                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                           phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        else:
            ys  = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)
            xs  = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)

            for y in ys:
                for x in xs:

                    tom_amp = np.sqrt(x**2+y**2)
                    tom_phase = np.arctan2(y,x)
                    sequencer.new_sequence(self)

                    for qubit_id in self.expt_cfg['on_qubits']:
                        if self.expt_cfg['cavity_pulse_type'] == "gauss":
                            tau = rabi_len * 4.0
                        else:
                            tau = rabi_len
                        if self.expt_cfg["use_weak_drive_for_dressing"]:
                            self.gen_q_weak(sequencer, qubit_id, len=tau,
                                            amp=self.expt_cfg['dressing_amp'], phase=0,
                                            add_freq=2 * self.expt_cfg['blockade_level'] *
                                                     self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                         0],
                                            pulse_type=self.expt_cfg['dressing_pulse_type'])
                        else:
                            self.gen_q(sequencer, qubit_id, len=rabi_len,
                                       amp=self.expt_cfg['dressing_amp'], phase=0,
                                       add_freq=2 * self.expt_cfg['blockade_level'] *
                                                self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],
                                       pulse_type=self.expt_cfg['dressing_pulse_type'])

                        self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=rabi_len,
                                   amp=self.expt_cfg['cavity_amp'], phase=self.expt_cfg['cavity_phase'],
                                   pulse_type=self.expt_cfg['cavity_pulse_type'])
                        sequencer.sync_channels_time(self.channels)
                        if self.expt_cfg['sweep_phase']:self.wigner_tomography(sequencer, qubit_id,mode_index=self.expt_cfg['mode_index'],amp = self.expt_cfg['amp'],phase=x,len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])
                        else:self.wigner_tomography(sequencer, qubit_id,mode_index=self.expt_cfg['mode_index'],amp = tom_amp,phase=tom_phase,len = self.expt_cfg['cavity_pulse_len'],pulse_type = self.expt_cfg['tomography_pulse_type'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def blockade_experiments_with_optimal_control_wt(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']
        rabi_len = self.expt_cfg['rabi_len']

        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

        for ii, y in enumerate(ys):
            x = xs[ii]
            if self.expt_cfg['transfer_fn_wt']:
                tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq_list')
            else:
                tom_amp = np.sqrt(x ** 2 + y ** 2)
            tom_phase = np.arctan2(y, x)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                    phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=self.expt_cfg['phase'], add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=self.expt_cfg['weak_cavity_for_blockade_expt'])
                sequencer.sync_channels_time(self.channels)
                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['tomography_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def blockade_experiments_with_optimal_control_generalized_wt(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']
        rabi_len = self.expt_cfg['rabi_len']

        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

        for ii, y in enumerate(ys):
            x = xs[ii]
            if self.expt_cfg['transfer_fn_wt']:
                tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq_list')
            else:
                tom_amp = np.sqrt(x ** 2 + y ** 2)
            tom_phase = np.arctan2(y, x)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                    phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=self.expt_cfg['phase'], add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=self.expt_cfg['weak_cavity_for_blockade_expt'])
                sequencer.sync_channels_time(self.channels)
                self.generalized_wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                                   phase=tom_phase, len=self.expt_cfg['tomography_pulse_len'],
                                                   pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                   ramsey_time=self.expt_cfg['wigner_ramsey_time'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def blockade_pulse_segment(self,sequencer,qubit_id='1',mode_index=0,len=10000,cavity_pulse_type="square",use_weak_for_dressing=False,
                               dressing_amp=0.024,blockade_levels=[2],dressing_pulse_type = "square",cavity_amp=0.024,phase=0,add_freq=0.0,
                               add_dressing_drive_offset=0.0, weak_cavity=False):
        if cavity_pulse_type == "gauss" and dressing_pulse_type == "square": tau = len*4.0
        elif cavity_pulse_type == "square" and dressing_pulse_type == "gauss": tau = len*0.25
        else: tau = len
        if use_weak_for_dressing:
            if not weak_cavity:
                self.gen_q_weak_multitone(sequencer, qubit_id, len=tau,
                                          amp=dressing_amp,
                                          add_freqs=2*np.array(blockade_levels)*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index]
                                                    +add_dressing_drive_offset,
                                          phases=0*np.array(blockade_levels),
                                          pulse_type=dressing_pulse_type)
                self.gen_c(sequencer, mode_index=mode_index, len=len,
                           amp=cavity_amp, phase=phase,
                           pulse_type=cavity_pulse_type, add_freq=add_freq)

            else:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index] * np.array(
                    blockade_levels) + add_dressing_drive_offset
                self.gen_c_and_blockade_weak(sequencer, mode_index=mode_index, len=len,
                                amp=cavity_amp, phase=phase,
                                pulse_type=cavity_pulse_type, add_freq=add_freq,blockade_amp = dressing_amp,blockade_freqs=freqlist)
        else:
            self.gen_q_multitone(sequencer, qubit_id, len=tau,
                                      amp=dressing_amp,
                                      add_freqs=2 *np.array(blockade_levels)*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index]
                                                +add_dressing_drive_offset,
                                      phases=0*np.array(blockade_levels),
                                      pulse_type=dressing_pulse_type)
            if weak_cavity:
                self.gen_c_weak(sequencer, mode_index=mode_index,len=len,
                       amp=cavity_amp, phase=phase,
                       pulse_type=cavity_pulse_type,add_freq=add_freq)
            else:
                self.gen_c(sequencer, mode_index=mode_index,len=len,
                           amp=cavity_amp, phase=phase,
                           pulse_type=cavity_pulse_type,add_freq=add_freq)
        return
    def multimode_blockade_pulse_segment(self,sequencer,qubit_id='1',blockade_mode_indices=[0],cavity_drive_mode_indices = [0],length=10000,cavity_pulse_type="square",use_weak_for_dressing=False,
                                         dressing_amps=[0.024],blockade_levels=[[2]],dressing_pulse_type = "square",cavity_amps=[0.024],
                                         phases=[0],add_freqs=[0.0], add_dressing_drive_offsets=[0.0], weak_cavity=False):
        if cavity_pulse_type == "gauss" and dressing_pulse_type == "square": tau = length*4.0
        elif cavity_pulse_type == "square" and dressing_pulse_type == "gauss": tau = length*0.25
        else: tau = length

        add_freqs_q = []
        freq_ref = []  # used to keep track of the mode and level to which each frequency corresponds
        for i, mode_index in enumerate(blockade_mode_indices):
            for j, level in enumerate(blockade_levels[mode_index]):
                add_freq = 2 * level * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index] + \
                           add_dressing_drive_offsets[mode_index][j]
                if add_freq not in add_freqs_q:
                    add_freqs_q.append(add_freq)
                    freq_ref.append((i, j))

        ref_cutoff = len(add_freqs_q)  # keep track of which frequencies to ignore if only want to blockade pair sums
        dressing_amps = np.array(dressing_amps)
        if self.expt_cfg['blockade_pair_sums'] and not self.expt_cfg['blockade_only_triple_sums']:
            for indices in list(combinations(range(len(add_freqs_q)), 2)):
                add_freq = (add_freqs_q[indices[0]] + add_freqs_q[indices[1]])/2.0
                if add_freq not in add_freqs_q:
                    add_freqs_q.append(add_freq)
                    # dressing_amps = np.append(dressing_amps, [(dressing_amps[freq_ref[indices[0]][0]][freq_ref[indices[0]][1]]
                    #                                           + dressing_amps[freq_ref[indices[1]][0]][freq_ref[indices[1]][1]])/2.0])
                    dressing_amps = np.append(dressing_amps, self.expt_cfg['pair_dressing_amps'])
        if self.expt_cfg['blockade_triple_sums'] and not self.expt_cfg['blockade_only_pair_sums']:
            for indices in list(combinations(range(len(add_freqs_q)), 3)):
                add_freq = (add_freqs_q[indices[0]] + add_freqs_q[indices[1]] + add_freqs_q[indices[2]])/3.0
                if add_freq not in add_freqs_q:
                    add_freqs_q.append(add_freq)
                    # dressing_amps = np.append(dressing_amps, [(dressing_amps[freq_ref[indices[0]][0]][freq_ref[indices[0]][1]]
                    #                                           + dressing_amps[freq_ref[indices[1]][0]][freq_ref[indices[1]][1]])/2.0])
                    dressing_amps = np.append(dressing_amps, self.expt_cfg['triple_dressing_amps'])
        if self.expt_cfg['blockade_only_pair_sums'] or self.expt_cfg['blockade_only_triple_sums']:
            if self.expt_cfg['blockade_only_pair_sums'] and self.expt_cfg['blockade_only_triple_sums']:
                print("!!!CANNOT BLOCKADE ONLY PAIR AND ONLY TRIPLE!!!")
            add_freqs_q = add_freqs_q[ref_cutoff:]
            dressing_amps = dressing_amps[ref_cutoff:]
        if use_weak_for_dressing:
            if not weak_cavity:
                # test this case, hopefully modified correctly
                self.gen_q_weak_multitone(sequencer, qubit_id, len=tau,
                                          amp=dressing_amps,
                                          add_freqs=add_freqs_q,
                                          phases=[0] * len(add_freqs_q),
                                          pulse_type=dressing_pulse_type)

                self.gen_c_multitone(sequencer, mode_indices=cavity_drive_mode_indices, len=length,
                           amps=cavity_amps, phases=phases,
                           pulse_type=cavity_pulse_type, add_freqs=add_freqs)
            else:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + np.array(add_freqs_q)
                self.gen_c_multitone_and_blockade_weak(sequencer, mode_indices=cavity_drive_mode_indices, len=length,
                                amps=cavity_amps, phases=phases,
                                pulse_type=cavity_pulse_type, add_freqs=add_freqs, blockade_amp=dressing_amps, blockade_freqs=freqlist)
        else:
            # test this case, hopefully modified correctly
            self.gen_q_multitone(sequencer, qubit_id, len=tau,
                                      amp=dressing_amps,
                                      add_freqs=add_freqs_q,
                                      phases=[0] * len(add_freqs_q),
                                      pulse_type=dressing_pulse_type)
            if weak_cavity:
                self.gen_c_weak_multitone(sequencer, mode_indices=cavity_drive_mode_indices,len=length,
                       amps=cavity_amps, phases=phases,
                       pulse_type=cavity_pulse_type,add_freqs=add_freqs)
            else:
                self.gen_c_multitone(sequencer, mode_indices=cavity_drive_mode_indices,len=length,
                           amps=cavity_amps, phases=phases,
                           pulse_type=cavity_pulse_type,add_freqs=add_freqs)
        return add_freqs_q, dressing_amps

    def multitone_blockaded_cavity_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])

                else:sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_blockaded_weak_cavity_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])

                else:sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=True)
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['probe_mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)
    
    def multitone_blockaded_weak_cavity_pnrqs(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                else:sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=self.expt_cfg['cavity_pulse_len'],
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=True)
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['blockade_during_spec']:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
                        resolved_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                        resolved_amp = self.pulse_info[qubit_id]['pi_amp_resolved_weak']
                    else:
                        pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
                        resolved_len = self.pulse_info[qubit_id]['pi_len_resolved']
                        resolved_amp = self.pulse_info[qubit_id]['pi_amp_resolved']

                    # get proper length of blockade pulse
                    if self.expt_cfg['dressing_pulse_type'] == "gauss" and pulse_type == "square":
                        blockade_during_spec_len = resolved_len / 4.0
                    elif self.expt_cfg['dressing_pulse_type'] == "square" and pulse_type == "gauss":
                        blockade_during_spec_len = resolved_len * 4.0

                    # calculate blockade frequencies
                    freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + 2 * \
                               self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                   self.expt_cfg['mode_index']] * np.array(
                        self.expt_cfg['blockade_levels']) + self.expt_cfg['dressing_drive_offset_freq']

                    # need to handle special case if pulses on same channel
                    if self.expt_cfg['use_weak_drive_for_probe'] and self.expt_cfg['use_weak_drive_for_dressing']:
                        self.gen_q_and_blockade_weak(sequencer, qubit_id=qubit_id, len=resolved_len, amp=resolved_amp,
                                                     add_freq=df, phase=0, pulse_type=pulse_type, blockade_amp=self.expt_cfg['dressing_amp'],
                                                     blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['dressing_pulse_type'])
                    elif not (self.expt_cfg['use_weak_drive_for_probe'] or self.expt_cfg['use_weak_drive_for_dressing']):
                        self.gen_q_and_blockade(sequencer, qubit_id=qubit_id, len=resolved_len, amp=resolved_amp,
                                                     add_freq=df, phase=0, pulse_type=pulse_type, blockade_amp=self.expt_cfg['dressing_amp'],
                                                     blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['dressing_pulse_type'])
                    else:
                        # blockade without cavity, purposely set cavity_pulse_type to same as dressing pulse to avoid length
                        # changes that are unwanted; cavity amp 0.0
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                                len=blockade_during_spec_len,
                                                cavity_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                                dressing_amp=self.expt_cfg['dressing_amp'],
                                                blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                cavity_amp=0.0,
                                                phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                                add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                                weak_cavity=True)
                        self.pi_q_resolved(sequencer, qubit_id,
                                           add_freq=df,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    self.pi_q_resolved(sequencer, qubit_id,
                                       add_freq=df,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_blockaded_weak_cavity_change_dressing_freq(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])

                else:sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=self.expt_cfg['cavity_pulse_len'],
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'] + df,
                                            weak_cavity=True)
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=0,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multimode_blockade_experiments_cavity_spectroscopy(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer,qubit_id=qubit_id,mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2],dressing_pulse_type = "square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params']['use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id, blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg['cavity_drive_mode_indices'],
                                                      length=self.expt_cfg['cavity_pulse_len'],
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']) + df,
                                                      add_dressing_drive_offsets=self.expt_cfg['dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*self.expt_cfg['probe_level'] *
                                        self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['probe_mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_multimode_blockaded_cavity_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=rabi_len,
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.expt_cfg['probe_level'] *
                                            self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                self.expt_cfg['probe_mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_multimode_blockaded_cavity_rabi_multiprobe(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=rabi_len,
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                add_freq_probe = 0
                for jj, mode in enumerate(self.expt_cfg['probe_mode_indices']):
                    add_freq_probe += 2 * self.expt_cfg['probe_mode_levels'][jj] * self.quantum_device_cfg['flux_pulse_info'][
                        qubit_id]['chiby2pi_e'][mode]
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=add_freq_probe,
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_multimode_blockaded_cavity_pnrqs(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                freqlist, amplist = self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=self.expt_cfg['cavity_pulse_len'],
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                freqlist = np.array(freqlist) + self.quantum_device_cfg['qubit'][qubit_id]['freq']
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['blockade_during_spec']:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type_weak']
                        resolved_len = self.pulse_info[qubit_id]['pi_len_resolved_weak']
                        resolved_amp = self.pulse_info[qubit_id]['pi_amp_resolved_weak']
                    else:
                        pulse_type = self.pulse_info[qubit_id]['resolved_pulse_type']
                        resolved_len = self.pulse_info[qubit_id]['pi_len_resolved']
                        resolved_amp = self.pulse_info[qubit_id]['pi_amp_resolved']

                    # get proper length of blockade pulse
                    if self.expt_cfg['dressing_pulse_type'] == "gauss" and pulse_type == "square":
                        blockade_during_spec_len = resolved_len / 4.0
                    elif self.expt_cfg['dressing_pulse_type'] == "square" and pulse_type == "gauss":
                        blockade_during_spec_len = resolved_len * 4.0

                    # need to handle special case if pulses on same channel
                    if self.expt_cfg['use_weak_drive_for_probe'] and self.expt_cfg['use_weak_drive_for_dressing']:
                        self.gen_q_and_blockade_weak_multitone(sequencer, qubit_id=qubit_id, len=resolved_len, amp=resolved_amp,
                                                     add_freqs=np.array([df]), phases=[0], pulse_type=pulse_type,
                                                     blockade_amp=amplist,
                                                     blockade_freqs=freqlist,
                                                     blockade_pulse_type=self.expt_cfg['dressing_pulse_type'])
                    elif not (
                            self.expt_cfg['use_weak_drive_for_probe'] or self.expt_cfg['use_weak_drive_for_dressing']):
                        self.gen_q_and_blockade_multitone(sequencer, qubit_id=qubit_id, len=resolved_len, amp=resolved_amp,
                                                add_freqs=np.array([df]), phases=[0], pulse_type=pulse_type,
                                                blockade_amp=amplist,
                                                blockade_freqs=freqlist,
                                                blockade_pulse_type=self.expt_cfg['dressing_pulse_type'])
                    else:
                        # blockade without cavity, purposely set cavity_pulse_type to same as dressing pulse to avoid length
                        # changes that are unwanted; cavity amp 0.0
                        self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                              blockade_mode_indices=self.expt_cfg[
                                                                  'blockade_mode_indices'],
                                                              cavity_drive_mode_indices=self.expt_cfg[
                                                                  'cavity_drive_mode_indices'],
                                                              length=blockade_during_spec_len,
                                                              cavity_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                              use_weak_for_dressing=self.expt_cfg[
                                                                  "use_weak_drive_for_dressing"],
                                                              dressing_amps=self.expt_cfg['dressing_amps'],
                                                              blockade_levels=np.array(
                                                                  self.expt_cfg['blockade_levels']),
                                                              dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                              cavity_amps=np.zeros(len(self.expt_cfg['cavity_amps'])),
                                                              phases=self.expt_cfg['cavity_phases'],
                                                              add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                              add_dressing_drive_offsets=self.expt_cfg[
                                                                  'dressing_drive_offset_freqs'],
                                                              weak_cavity=self.expt_cfg['use_weak_cavity'])
                        self.pi_q_resolved(sequencer, qubit_id,
                                           add_freq=df,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    self.pi_q_resolved(sequencer, qubit_id,
                                       add_freq=df,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_multimode_blockaded_cavity_tomography_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=rabi_len,
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.gen_two_mode_correlator(sequencer, qubit_id="1", mode1=self.expt_cfg['tomography_mode_indices'][0],
                                             mode2=self.expt_cfg['tomography_mode_indices'][1],
                                             op1=self.expt_cfg['measure_correlator'][0], op2=self.expt_cfg['measure_correlator'][1],
                                             use_weak_resolved_pulse_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                             simultaneous_rotations=False, qubit_phase=0)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multitone_multimode_blockaded_cavity_full_tomography(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for correlator in self.expt_cfg['correlator_list']:
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=self.expt_cfg['cavity_pulse_len'],
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.gen_two_mode_correlator(sequencer, qubit_id="1", mode1=self.expt_cfg['tomography_mode_indices'][0],
                                             mode2=self.expt_cfg['tomography_mode_indices'][1],
                                             op1=correlator[0], op2=correlator[1],
                                             use_weak_resolved_pulse_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                             simultaneous_rotations=self.expt_cfg['simultaneous_rotations'], qubit_phase=0)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multimode_blockade_experiments_wt(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['use_wigner_points_file']:  # pull from file of wigner points
                with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                    if self.expt_cfg['transfer_fn_wt']:
                        # Kevin edit testing a transfer function
                        xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        # end edit
                    else:
                        xs = np.array(f['ax'])
                        ys = np.array(f['ay'])

                for ii, y1 in enumerate(ys):
                    sequencer.new_sequence(self)
                    x1 = xs[ii]
                    y2 = ys[self.expt_cfg['tom2_index_pt']]
                    x2 = xs[self.expt_cfg['tom2_index_pt']]
                    if len(self.expt_cfg['cavity_amps']) > 2:
                        y3 = ys[self.expt_cfg['tom3_index_pt']]
                        x3 = xs[self.expt_cfg['tom3_index_pt']]
                        tom_phase3 = np.arctan2(y3, x3)
                        tom_amp3 = np.sqrt(x3 ** 2 + y3 ** 2)
                        if self.expt_cfg['transfer_fn_wt']:
                            tom_amp3 = self.transfer_function_blockade(tom_amp3, channel='cavity_amp_vs_freq_joint' +
                                                                               str(self.expt_cfg['probe_mode_indices'][
                                                                                       2]))
                    if self.expt_cfg['transfer_fn_wt']:
                        tom_amp1 = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                                   channel='cavity_amp_vs_freq_joint' +
                                                                           str(self.expt_cfg['probe_mode_indices'][0]))
                        tom_amp2 = self.transfer_function_blockade(np.sqrt(x2 ** 2 + y2 ** 2),
                                                                   channel='cavity_amp_vs_freq_joint' +
                                                                           str(self.expt_cfg['probe_mode_indices'][1]))
                    else:
                        tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                        tom_amp2 = np.sqrt(x2 ** 2 + y2 ** 2)
                    tom_phase1 = np.arctan2(y1, x1)
                    tom_phase2 = np.arctan2(y2, x2)

                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                              blockade_mode_indices=self.expt_cfg[
                                                                                  'blockade_mode_indices'],
                                                                              cavity_drive_mode_indices=self.expt_cfg[
                                                                                  'cavity_drive_mode_indices'],
                                                                              length=self.expt_cfg['cavity_pulse_len'],
                                                                              cavity_pulse_type=self.expt_cfg[
                                                                                  'cavity_pulse_type'],
                                                                              use_weak_for_dressing=self.expt_cfg[
                                                                                  "use_weak_drive_for_dressing"],
                                                                              dressing_amps=self.expt_cfg[
                                                                                  'dressing_amps'],
                                                                              blockade_levels=np.array(
                                                                                  self.expt_cfg['blockade_levels']),
                                                                              dressing_pulse_type=self.expt_cfg[
                                                                                  'dressing_pulse_type'],
                                                                              cavity_amps=self.expt_cfg['cavity_amps'],
                                                                              phases=self.expt_cfg['cavity_phases'],
                                                                              add_freqs=np.array(
                                                                                  self.expt_cfg['cavity_offset_freqs']),
                                                                              add_dressing_drive_offsets=self.expt_cfg[
                                                                                  'dressing_drive_offset_freqs'],
                                                                              weak_cavity=self.expt_cfg[
                                                                                  'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    if not len(self.expt_cfg['cavity_amps']) > 2:
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id, modes=self.expt_cfg['probe_mode_indices'],
                                             use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                             sequential=self.expt_cfg['sequential_cav_displacements'],
                                             n_max=self.expt_cfg['n_max'],
                                             lengths=[self.expt_cfg['tomography_pulse_len'] for num in range(len(self.expt_cfg['probe_mode_indices']))],
                                             pulse_type=self.expt_cfg['tomography_pulse_type'],
                                             amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                             pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'])
                    else:
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                     modes=self.expt_cfg['probe_mode_indices'],
                                                     use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                     sequential=self.expt_cfg['sequential_cav_displacements'],
                                                     n_max=self.expt_cfg['n_max'],
                                                     lengths=[self.expt_cfg['tomography_pulse_len'] for num in
                                                              range(len(self.expt_cfg['probe_mode_indices']))],
                                                     pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                     amps=[tom_amp1, tom_amp2, tom_amp3], phases=[tom_phase1, tom_phase2, tom_phase3],
                                                     pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                     ramsey_parity=self.expt_cfg['ramsey_parity'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
            else:
                x0 = self.expt_cfg['offset_x']
                y0 = self.expt_cfg['offset_y']
                for y1 in (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0):
                    for x1 in (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0):
                        tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                        x2 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[self.expt_cfg['tom2_index_pt']]
                        y2 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[self.expt_cfg['tom2_index_pt']]
                        tom_amp2 = np.sqrt(y2 ** 2 + x2 ** 2)
                        tom_phase1 = np.arctan2(y1, x1)
                        tom_phase2 = np.arctan2(y2, x2)
                        sequencer.new_sequence(self)
                        if self.expt_cfg['prep_state_before_blockade']:
                            if self.expt_cfg['prep_using_blockade']:
                                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                                # print (blockade_pulse_info)
                                mode_index = self.expt_cfg['prep_mode_index']
                                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                            len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                            cavity_pulse_type=
                                                            blockade_pulse_info['blockade_cavity_pulse_type'][
                                                                mode_index],
                                                            use_weak_for_dressing=
                                                            blockade_pulse_info['use_weak_for_blockade'][
                                                                mode_index],
                                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                                mode_index],
                                                            blockade_levels=[2], dressing_pulse_type="square",
                                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                                mode_index], phase=0,
                                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                                mode_index],
                                                            weak_cavity=
                                                            self.quantum_device_cfg['blockade_pulse_params'][
                                                                'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                        freqlist, amplist = self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                                  blockade_mode_indices=self.expt_cfg[
                                                                                      'blockade_mode_indices'],
                                                                                  cavity_drive_mode_indices=
                                                                                  self.expt_cfg[
                                                                                      'cavity_drive_mode_indices'],
                                                                                  length=self.expt_cfg[
                                                                                      'cavity_pulse_len'],
                                                                                  cavity_pulse_type=self.expt_cfg[
                                                                                      'cavity_pulse_type'],
                                                                                  use_weak_for_dressing=self.expt_cfg[
                                                                                      "use_weak_drive_for_dressing"],
                                                                                  dressing_amps=self.expt_cfg[
                                                                                      'dressing_amps'],
                                                                                  blockade_levels=np.array(
                                                                                      self.expt_cfg['blockade_levels']),
                                                                                  dressing_pulse_type=self.expt_cfg[
                                                                                      'dressing_pulse_type'],
                                                                                  cavity_amps=self.expt_cfg[
                                                                                      'cavity_amps'],
                                                                                  phases=self.expt_cfg['cavity_phases'],
                                                                                  add_freqs=np.array(self.expt_cfg[
                                                                                                         'cavity_offset_freqs']),
                                                                                  add_dressing_drive_offsets=
                                                                                  self.expt_cfg[
                                                                                      'dressing_drive_offset_freqs'],
                                                                                  weak_cavity=self.expt_cfg[
                                                                                      'use_weak_cavity'])
                        sequencer.sync_channels_time(self.channels)
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                 modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len']] * len(self.expt_cfg['probe_mode_indices']),
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                     ramsey_parity=self.expt_cfg['ramsey_parity'])
                        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multimode_blockade_experiments_1wt_pnrqs(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['use_wigner_points_file']:  # pull from file of wigner points
                with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                    if self.expt_cfg['transfer_fn_wt']:
                        # Kevin edit testing a transfer function
                        xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        # end edit
                    else:
                        xs = np.array(f['ax'])
                        ys = np.array(f['ay'])

                x1 = xs[self.expt_cfg['tom1_index_pt']]
                y1 = ys[self.expt_cfg['tom1_index_pt']]
                print(x1 * self.expt_cfg['tomography_pulse_len'], y1 * self.expt_cfg['tomography_pulse_len'])
                y2 = ys[self.expt_cfg['tom2_index_pt']]
                x2 = xs[self.expt_cfg['tom2_index_pt']]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp1 = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][0]))
                    tom_amp2 = self.transfer_function_blockade(np.sqrt(x2 ** 2 + y2 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][1]))
                else:
                    tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    tom_amp2 = np.sqrt(x2 ** 2 + y2 ** 2)
                tom_phase1 = np.arctan2(y1, x1)
                tom_phase2 = np.arctan2(y2, x2)

                for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                    sequencer.new_sequence(self)
                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                              blockade_mode_indices=self.expt_cfg[
                                                                                  'blockade_mode_indices'],
                                                                              cavity_drive_mode_indices=self.expt_cfg[
                                                                                  'cavity_drive_mode_indices'],
                                                                              length=self.expt_cfg['cavity_pulse_len'],
                                                                              cavity_pulse_type=self.expt_cfg[
                                                                                  'cavity_pulse_type'],
                                                                              use_weak_for_dressing=self.expt_cfg[
                                                                                  "use_weak_drive_for_dressing"],
                                                                              dressing_amps=self.expt_cfg[
                                                                                  'dressing_amps'],
                                                                              blockade_levels=np.array(
                                                                                  self.expt_cfg['blockade_levels']),
                                                                              dressing_pulse_type=self.expt_cfg[
                                                                                  'dressing_pulse_type'],
                                                                              cavity_amps=self.expt_cfg['cavity_amps'],
                                                                              phases=self.expt_cfg['cavity_phases'],
                                                                              add_freqs=np.array(
                                                                                  self.expt_cfg['cavity_offset_freqs']),
                                                                              add_dressing_drive_offsets=self.expt_cfg[
                                                                                  'dressing_drive_offset_freqs'],
                                                                              weak_cavity=self.expt_cfg[
                                                                                  'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id, modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_parity'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len'] for num in range(len(self.expt_cfg['probe_mode_indices']))],
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'],
                                                 displace_cav=self.expt_cfg['do_cavity_displacement'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_q_resolved(sequencer, qubit_id,
                                       add_freq=df,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
            else:
                x0 = self.expt_cfg['offset_x']
                y0 = self.expt_cfg['offset_y']
                for df in (np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']) + y0):
                    x1 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[
                        self.expt_cfg['tom1_index_pt']]
                    y1 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[
                        self.expt_cfg['tom1_index_pt']]
                    tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    x2 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[self.expt_cfg['tom2_index_pt']]
                    y2 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[self.expt_cfg['tom2_index_pt']]
                    tom_amp2 = np.sqrt(y2 ** 2 + x2 ** 2)
                    tom_phase1 = np.arctan2(y1, x1)
                    tom_phase2 = np.arctan2(y2, x2)
                    sequencer.new_sequence(self)
                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=
                                                        self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    freqlist, amplist = self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                              blockade_mode_indices=self.expt_cfg[
                                                                                  'blockade_mode_indices'],
                                                                              cavity_drive_mode_indices=
                                                                              self.expt_cfg[
                                                                                  'cavity_drive_mode_indices'],
                                                                              length=self.expt_cfg[
                                                                                  'cavity_pulse_len'],
                                                                              cavity_pulse_type=self.expt_cfg[
                                                                                  'cavity_pulse_type'],
                                                                              use_weak_for_dressing=self.expt_cfg[
                                                                                  "use_weak_drive_for_dressing"],
                                                                              dressing_amps=self.expt_cfg[
                                                                                  'dressing_amps'],
                                                                              blockade_levels=np.array(
                                                                                  self.expt_cfg['blockade_levels']),
                                                                              dressing_pulse_type=self.expt_cfg[
                                                                                  'dressing_pulse_type'],
                                                                              cavity_amps=self.expt_cfg[
                                                                                  'cavity_amps'],
                                                                              phases=self.expt_cfg['cavity_phases'],
                                                                              add_freqs=np.array(self.expt_cfg[
                                                                                                     'cavity_offset_freqs']),
                                                                              add_dressing_drive_offsets=
                                                                              self.expt_cfg[
                                                                                  'dressing_drive_offset_freqs'],
                                                                              weak_cavity=self.expt_cfg[
                                                                                  'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                 modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len']] * len(self.expt_cfg['probe_mode_indices']),
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'])
                    self.pi_q_resolved(sequencer, qubit_id,
                                       add_freq=df,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multimode_blockade_experiments_1wt_parity_rabi(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['use_wigner_points_file']:  # pull from file of wigner points
                with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                    if self.expt_cfg['transfer_fn_wt']:
                        # Kevin edit testing a transfer function
                        xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        # end edit
                    else:
                        xs = np.array(f['ax'])
                        ys = np.array(f['ay'])

                x1 = xs[self.expt_cfg['tom1_index_pt']]
                y1 = ys[self.expt_cfg['tom1_index_pt']]
                y2 = ys[self.expt_cfg['tom2_index_pt']]
                x2 = xs[self.expt_cfg['tom2_index_pt']]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp1 = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][0]))
                    tom_amp2 = self.transfer_function_blockade(np.sqrt(x2 ** 2 + y2 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][1]))
                else:
                    tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    tom_amp2 = np.sqrt(x2 ** 2 + y2 ** 2)
                tom_phase1 = np.arctan2(y1, x1)
                tom_phase2 = np.arctan2(y2, x2)

                for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                    sequencer.new_sequence(self)
                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                              blockade_mode_indices=self.expt_cfg[
                                                                                  'blockade_mode_indices'],
                                                                              cavity_drive_mode_indices=self.expt_cfg[
                                                                                  'cavity_drive_mode_indices'],
                                                                              length=rabi_len,
                                                                              cavity_pulse_type=self.expt_cfg[
                                                                                  'cavity_pulse_type'],
                                                                              use_weak_for_dressing=self.expt_cfg[
                                                                                  "use_weak_drive_for_dressing"],
                                                                              dressing_amps=self.expt_cfg[
                                                                                  'dressing_amps'],
                                                                              blockade_levels=np.array(
                                                                                  self.expt_cfg['blockade_levels']),
                                                                              dressing_pulse_type=self.expt_cfg[
                                                                                  'dressing_pulse_type'],
                                                                              cavity_amps=self.expt_cfg['cavity_amps'],
                                                                              phases=self.expt_cfg['cavity_phases'],
                                                                              add_freqs=np.array(
                                                                                  self.expt_cfg['cavity_offset_freqs']),
                                                                              add_dressing_drive_offsets=self.expt_cfg[
                                                                                  'dressing_drive_offset_freqs'],
                                                                              weak_cavity=self.expt_cfg[
                                                                                  'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id, modes=self.expt_cfg['probe_mode_indices'],
                                             use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_parity'],
                                             sequential=self.expt_cfg['sequential_cav_displacements'],
                                             n_max=self.expt_cfg['n_max'],
                                             lengths=[self.expt_cfg['tomography_pulse_len'] for
                                                      num in range(len(self.expt_cfg['probe_mode_indices']))],
                                             pulse_type=self.expt_cfg['tomography_pulse_type'],
                                             amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                             pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                             ramsey_parity=self.expt_cfg['ramsey_parity'],
                                             displace_cav=self.expt_cfg['do_cavity_displacement'])
                    sequencer.sync_channels_time(self.channels)
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
            else:
                x0 = self.expt_cfg['offset_x']
                y0 = self.expt_cfg['offset_y']
                for df in (np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']) + y0):
                    x1 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[
                        self.expt_cfg['tom1_index_pt']]
                    y1 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[
                        self.expt_cfg['tom1_index_pt']]
                    tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    x2 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[self.expt_cfg['tom2_index_pt']]
                    y2 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[self.expt_cfg['tom2_index_pt']]
                    tom_amp2 = np.sqrt(y2 ** 2 + x2 ** 2)
                    tom_phase1 = np.arctan2(y1, x1)
                    tom_phase2 = np.arctan2(y2, x2)
                    sequencer.new_sequence(self)
                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=
                                                        self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    freqlist, amplist = self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                              blockade_mode_indices=self.expt_cfg[
                                                                                  'blockade_mode_indices'],
                                                                              cavity_drive_mode_indices=
                                                                              self.expt_cfg[
                                                                                  'cavity_drive_mode_indices'],
                                                                              length=self.expt_cfg[
                                                                                  'cavity_pulse_len'],
                                                                              cavity_pulse_type=self.expt_cfg[
                                                                                  'cavity_pulse_type'],
                                                                              use_weak_for_dressing=self.expt_cfg[
                                                                                  "use_weak_drive_for_dressing"],
                                                                              dressing_amps=self.expt_cfg[
                                                                                  'dressing_amps'],
                                                                              blockade_levels=np.array(
                                                                                  self.expt_cfg['blockade_levels']),
                                                                              dressing_pulse_type=self.expt_cfg[
                                                                                  'dressing_pulse_type'],
                                                                              cavity_amps=self.expt_cfg[
                                                                                  'cavity_amps'],
                                                                              phases=self.expt_cfg['cavity_phases'],
                                                                              add_freqs=np.array(self.expt_cfg[
                                                                                                     'cavity_offset_freqs']),
                                                                              add_dressing_drive_offsets=
                                                                              self.expt_cfg[
                                                                                  'dressing_drive_offset_freqs'],
                                                                              weak_cavity=self.expt_cfg[
                                                                                  'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                 modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_parity'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len'] for
                                                          num in range(len(self.expt_cfg['probe_mode_indices']))],
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'],
                                                 displace_cav=self.expt_cfg['do_cavity_displacement'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def sideband_chi_dressing_simultaneous_wt(self, sequencer):
        # wigner tomography with both cavity and sideband drive from tek2
        x0 = self.expt_cfg['offset_x']
        y0 = self.expt_cfg['offset_y']

        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

        for ii, y in enumerate(ys):
            tom_amps = []
            tom_phases = []
            x = xs[ii]
            if self.expt_cfg['transfer_fn_wt']:
                tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel=self.expt_cfg['wigner_transfer_channels'])
            else:
                tom_amp = np.sqrt(x ** 2 + y ** 2)
            tom_phase = np.arctan2(y, x)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.new_sequence(self)
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                                    phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                      blockade_mode_indices=self.expt_cfg['blockade_mode_indices'],
                                                      cavity_drive_mode_indices=self.expt_cfg[
                                                          'cavity_drive_mode_indices'],
                                                      length=self.expt_cfg['cavity_pulse_len'],
                                                      cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                                      use_weak_for_dressing=self.expt_cfg[
                                                          "use_weak_drive_for_dressing"],
                                                      dressing_amps=self.expt_cfg['dressing_amps'],
                                                      blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                                      dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                      cavity_amps=self.expt_cfg['cavity_amps'],
                                                      phases=self.expt_cfg['cavity_phases'],
                                                      add_freqs=np.array(self.expt_cfg['cavity_offset_freqs']),
                                                      add_dressing_drive_offsets=self.expt_cfg[
                                                          'dressing_drive_offset_freqs'],
                                                      weak_cavity=self.expt_cfg['use_weak_cavity'])
                if self.expt_cfg['chi_dressing_on_during_blockade']:
                    length = self.expt_cfg['cavity_pulse_len']
                    if self.expt_cfg['cavity_pulse_type'] == "gauss":
                        length *= 4
                    self.gen_sideband(sequencer,qubit_id,len= length, amp=self.expt_cfg['chi_dressing_amp'],add_freq=0.0,
                                      iq_freq = self.sideband_pulse_info['iq_freq'])
                sequencer.sync_channels_time(self.channels)
                self.wigner_tomography_with_sideband(sequencer, qubit_id, mode_index=self.expt_cfg['probe_mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['tomography_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def blockade_experiments_cavity_spectroscopy(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['use_optimal_control']:
                        self.prep_optimal_control_pulse_1step(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                              print_times=self.expt_cfg['print_times'])
                        sequencer.sync_channels_time(self.channels)
                    elif self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        prep_mode_index = self.expt_cfg['prep_mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=prep_mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][prep_mode_index],
                                                    cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        prep_mode_index],
                                                    use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                        prep_mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][prep_mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][prep_mode_index],
                                                    phase=0, add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][prep_mode_index],
                                                    weak_cavity=blockade_pulse_info['use_weak_for_cavity'][prep_mode_index])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=self.expt_cfg['cavity_pulse_len'],
                                            cavity_pulse_type=self.expt_cfg['cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=0, add_freq=df+self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=self.expt_cfg['weak_cavity'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['prep_state_before_blockade'] and self.expt_cfg['prep_using_blockade']: # and \
                   #self.expt_cfg['prep_mode_index'] != self.expt_cfg['mode_index']:
                    self.pi_q_resolved(sequencer, qubit_id,
                                       add_freq=2 * self.expt_cfg['probe_level'] *
                                                self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                    self.expt_cfg['mode_index']] +
                                                2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][
                                                    self.expt_cfg['prep_mode_index']],
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2 * self.expt_cfg['probe_level'] *
                                            self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def prep_optimal_control_pulse_with_blockade(self, sequencer, pulse_frac=1.0, print_times=False):
        if self.expt_cfg['filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(self.expt_cfg['filename'])
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file

        if not self.expt_cfg['carrier_freqs']:
            carrier_freqs = {"cavity": self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][
                                 self.expt_cfg['mode_index']]}
        else:
            carrier_freqs = self.expt_cfg['carrier_freqs']

        num_pulses = len(pulses)
        mode_index = self.expt_cfg['mode_index']

        if pulse_frac != 0:
            # write pulses to their appropriate channels
            for qubit_id in self.expt_cfg['on_qubits']:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + 2 * \
                           self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index] * \
                           np.array(self.expt_cfg['blockade_params']['levels'])
                if self.expt_cfg['use_weak_cavity_drive']:
                    # print("Using weak drive port for cavity")
                    cav_scale = self.expt_cfg['calibrations']['cavity_weak']
                    if self.expt_cfg['use_weak_drive_for_dressing']:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB_with_blockade(A_list=self.transfer_function(np.real(pulses[self.expt_cfg['pulse_number_map']['cavity'][0]]), channel='cavity_list_weak',
                                                                                             mode_index=self.expt_cfg['mode_index']),
                                                 B_list=self.transfer_function(np.real(pulses[self.expt_cfg['pulse_number_map']['cavity'][1]]), channel='cavity_list_weak',
                                                                               mode_index=self.expt_cfg['mode_index']),
                                                 len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                                               blockade_amp=self.expt_cfg['blockade_params']['amp'],
                                                               blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['blockade_params']['pulse_type']))
                        else:
                            sequencer.append('qubitweak', ARB_with_blockade(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
                                                                            B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                                           len=total_time*pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                                           scale=cav_scale, blockade_amp=self.expt_cfg['blockade_params']['amp'],
                                                           blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['blockade_params']['pulse_type']))
                    else:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB(A_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], channel='cavity_list_weak'),
                                                 B_list=self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity'][1]], channel='cavity_list_weak'),
                                                 len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0))
                        else:
                            sequencer.append('qubitweak', ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
                                                              B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                                              len=total_time*pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                                              scale=cav_scale))
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id,
                                             Square_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                              flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                              cutoff_sigma=2, freqs=freqlist,
                                                              phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id,
                                             Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                             sigma_len=total_time * pulse_frac / (2*2.0), cutoff_sigma=2,
                                                             freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                else:
                    cav_scale = self.expt_cfg['calibrations']['cavity']
                    if self.expt_cfg['try_transfer_function']:
                        sequencer.append('cavity',
                                         ARB(A_list=self.transfer_function(
                                             pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], channel='cavity_list',
                                             mode_index=self.expt_cfg['mode_index']),
                                             B_list=self.transfer_function(
                                             pulses[self.expt_cfg['pulse_number_map']['cavity'][1]], channel='cavity_list',
                                             mode_index=self.expt_cfg['mode_index']),
                                             len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0))
                    else:
                        sequencer.append('cavity',
                                         ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]],
                                             B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                             len=total_time * pulse_frac, freq=carrier_freqs["cavity"], phase=0,
                                             scale=cav_scale, blockade_amp=self.expt_cfg['blockade_params']['amp']))
                    if self.expt_cfg['use_weak_drive_for_dressing']:
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('qubitweak', Square_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freqs=freqlist, phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('qubitweak', Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                    else:
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freqs=freqlist, phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
        sequencer.sync_channels_time(self.channels)

    ### in progress ###
    def prep_optimal_control_pulse_with_blockade_multimode(self, sequencer, pulse_frac=1.0, print_times=False):
        if self.expt_cfg['filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times: print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(self.expt_cfg['filename'])
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file

        if not self.expt_cfg['carrier_freqs']:
            carrier_freqs = {"cavity": np.array(self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'])[
                                 self.expt_cfg['mode_indices']]}
        else:
            carrier_freqs = self.expt_cfg['carrier_freqs']

        num_pulses = len(pulses)
        mode_index = self.expt_cfg['mode_index']
        if pulse_frac != 0:
            # write pulses to their appropriate channels
            for qubit_id in self.expt_cfg['on_qubits']:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + np.array(self.expt_cfg['blockade_params']['freq'])
                if isinstance(freqlist, (int, float)):
                    freqlist = [freqlist]
                if self.expt_cfg['use_weak_cavity_drive']:
                    # print("Using weak drive port for cavity")
                    # collect correct scale list if not using transfer function
                    counter = 1
                    cav_scale_list = []
                    cav_scale_name = 'cavity'
                    while (cav_scale_name + str(counter) + '_weak') in self.expt_cfg['calibrations']:
                        cav_scale_list.append(self.expt_cfg['calibrations'][cav_scale_name + str(counter) + '_weak'])
                        counter += 1
                    # build correct pulse lists
                    A_list_list = []
                    B_list_list = []
                    A_list_list_linear = []
                    B_list_list_linear = []
                    for ii, mode in enumerate(self.expt_cfg['mode_indices']):
                        A_list_list.append(self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][0]], channel='cavity_list_weak',mode_index=mode))
                        B_list_list.append(self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][1]], channel='cavity_list_weak', mode_index=mode))
                        A_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][0]])
                        B_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][1]])
                    if self.expt_cfg['use_weak_drive_for_dressing']:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB_Sum_with_blockade(A_list_list=A_list_list, B_list_list=B_list_list,
                                                                len=total_time * pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                                blockade_amp=self.expt_cfg['blockade_params']['amp'],
                                                                blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['blockade_params']['pulse_type']))
                        else:
                            sequencer.append('qubitweak', ARB_Sum_with_blockade(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                                len=total_time*pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                                blockade_amp=self.expt_cfg['blockade_params']['amp'],
                                                                blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['blockade_params']['pulse_type'],
                                                                scale_list=cav_scale_list))
                    else:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB_Sum(A_list_list=A_list_list, B_list_list=B_list_list,
                                                    length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']]))
                        else:
                            sequencer.append('qubitweak', ARB_Sum(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                              length=total_time*pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                              scale=cav_scale))
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id,
                                             Square_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                              flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                              cutoff_sigma=2, freqs=freqlist,
                                                              phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id,
                                             Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                             sigma_len=total_time * pulse_frac / (2*2.0), cutoff_sigma=2,
                                                             freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                else:
                    # collect correct scale list if not using transfer function
                    counter = 1
                    cav_scale_list = []
                    cav_scale_name = 'cavity'
                    while (cav_scale_name + str(counter)) in self.expt_cfg['calibrations']:
                        cav_scale_list.append(self.expt_cfg['calibrations'][cav_scale_name + str(counter)])
                        counter += 1
                    # build correct pulse lists
                    A_list_list = []
                    B_list_list = []
                    A_list_list_linear = []
                    B_list_list_linear = []
                    for ii, mode in enumerate(self.expt_cfg['mode_indices']):
                        A_list_list.append(
                            self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii + 1)][0]],
                                                   channel='cavity_list', mode_index=mode))
                        B_list_list.append(
                            self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii + 1)][1]],
                                                   channel='cavity_list', mode_index=mode))
                        A_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii + 1)][0]])
                        B_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii + 1)][1]])

                    if self.expt_cfg['try_transfer_function']:
                        sequencer.append('cavity',
                                         ARB_Sum(A_list_list=A_list_list, B_list_list=B_list_list,
                                                length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"],
                                                phase_list=[0.0 for freq in carrier_freqs['cavity']]))
                    else:
                        sequencer.append('cavity',
                                         ARB_Sum(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"],
                                                phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                scale_list=cav_scale_list))
                    if self.expt_cfg['use_weak_drive_for_dressing']:
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            # print ("blockade freqlist - ", freqlist)
                            sequencer.append('qubitweak', Square(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freq=freqlist[0], phase=np.zeros(len(freqlist))[0]))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('qubitweak', Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                    else:
                        if self.expt_cfg['blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freqs=freqlist, phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=self.expt_cfg['blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
        sequencer.sync_channels_time(self.channels)

    def prep_optimal_control_pulse_before_blockade_multimode(self, sequencer, pulse_frac=1.0, print_times=False):
        if self.expt_cfg['filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(self.expt_cfg['filename'])
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file

        if not self.expt_cfg['carrier_freqs']:
            carrier_freqs = {"cavity": self.quantum_device_cfg['flux_pulse_info']['1']['cavity_freqs'][
                                 self.expt_cfg['prep_mode_indices_multimode']]}
        else:
            carrier_freqs = self.expt_cfg['carrier_freqs']

        num_pulses = len(pulses)
        mode_index = self.expt_cfg['mode_index']

        if pulse_frac != 0:
            # write pulses to their appropriate channels
            for qubit_id in self.expt_cfg['on_qubits']:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + np.array(self.expt_cfg['prep_blockade_params']['freq'])
                if isinstance(freqlist, (int, float)):
                    freqlist = [freqlist]
                if self.expt_cfg['prep_use_weak_cavity_drive']:
                    print("Using weak drive port for cavity prep")
                    # collect correct scale list if not using transfer function
                    counter = 1
                    cav_scale_list = []
                    cav_scale_name = 'cavity'
                    while (cav_scale_name + str(counter) + '_weak') in self.expt_cfg['prep_calibrations_multimode']:
                        cav_scale_list.append(self.expt_cfg['prep_calibrations'][cav_scale_name + str(counter) + '_weak'])
                        counter += 1
                    # build correct pulse lists
                    A_list_list = []
                    B_list_list = []
                    A_list_list_linear = []
                    B_list_list_linear = []
                    for ii, mode in enumerate(self.expt_cfg['mode_indices']):
                        A_list_list.append(self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][0]], channel='cavity_list_weak',mode_index=mode))
                        B_list_list.append(self.transfer_function(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][1]], channel='cavity_list_weak', mode_index=mode))
                        A_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][0]])
                        B_list_list_linear.append(pulses[self.expt_cfg['pulse_number_map']['cavity' + str(ii+1)][1]])

                    if self.expt_cfg['prep_use_weak_drive_for_dressing']:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB_Sum_with_blockade(A_list_list=A_list_list, B_list_list=B_list_list,
                                                                len=total_time * pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                                blockade_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['prep_blockade_params']['pulse_type']))
                        else:
                            sequencer.append('qubitweak', ARB_Sum_with_blockade(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                                len=total_time*pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                                blockade_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                blockade_freqs=freqlist, blockade_pulse_type=self.expt_cfg['prep_blockade_params']['pulse_type'],
                                                                scale_list=cav_scale_list))
                    else:
                        if self.expt_cfg['try_transfer_function']:
                            sequencer.append('qubitweak',
                                             ARB_Sum(A_list_list=A_list_list, B_list_list=B_list_list,
                                                    length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']]))
                        else:
                            sequencer.append('qubitweak', ARB_Sum(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                              length=total_time*pulse_frac, freq_list=carrier_freqs["cavity"], phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                              scale=cav_scale))
                        if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id,
                                             Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                              flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                              cutoff_sigma=2, freqs=freqlist,
                                                              phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id,
                                             Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                             sigma_len=total_time * pulse_frac / (2*2.0), cutoff_sigma=2,
                                                             freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                else:
                    # collect correct scale list if not using transfer function
                    counter = 1
                    cav_scale_list = []
                    cav_scale_name = 'cavity'
                    while (cav_scale_name + str(counter)) in self.expt_cfg['prep_calibrations']:
                        cav_scale_list.append(self.expt_cfg['prep_calibrations'][cav_scale_name + str(counter)])
                        counter += 1
                    # build correct pulse lists
                    A_list_list = []
                    B_list_list = []
                    A_list_list_linear = []
                    B_list_list_linear = []
                    for ii, mode in enumerate(self.expt_cfg['prep_mode_indices']):
                        A_list_list.append(
                            self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map_multimode']['cavity' + str(ii + 1)][0]],
                                                   channel='cavity_list', mode_index=mode))
                        B_list_list.append(
                            self.transfer_function(pulses[self.expt_cfg['prep_pulse_number_map_multimode']['cavity' + str(ii + 1)][1]],
                                                   channel='cavity_list', mode_index=mode))
                        A_list_list_linear.append(pulses[self.expt_cfg['prep_pulse_number_map_multimode']['cavity' + str(ii + 1)][0]])
                        B_list_list_linear.append(pulses[self.expt_cfg['prep_pulse_number_map_multimode']['cavity' + str(ii + 1)][1]])

                    if self.expt_cfg['try_transfer_function']:
                        sequencer.append('cavity',
                                         ARB_Sum(A_list_list=A_list_list, B_list_list=B_list_list,
                                                length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"],
                                                phase_list=[0.0 for freq in carrier_freqs['cavity']]))
                    else:
                        sequencer.append('cavity',
                                         ARB_Sum(A_list_list=A_list_list_linear, B_list_list=B_list_list_linear,
                                                length=total_time * pulse_frac, freq_list=carrier_freqs["cavity"],
                                                phase_list=[0.0 for freq in carrier_freqs['cavity']],
                                                scale_list=cav_scale_list))
                    if self.expt_cfg['prep_use_weak_drive_for_dressing']:
                        if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('qubitweak', Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freqs=freqlist, phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('qubitweak', Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
                    else:
                        if self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'square':
                            sequencer.append('charge%s' % qubit_id, Square_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                 flat_len=total_time * pulse_frac, ramp_sigma_len=0.001,
                                                                cutoff_sigma=2, freqs=freqlist, phases=np.zeros(len(freqlist))))
                        elif self.expt_cfg['prep_blockade_params']['pulse_type'].lower() == 'gauss':
                            sequencer.append('charge%s' % qubit_id, Gauss_multitone(max_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                                                 sigma_len=total_time*pulse_frac/4.0, cutoff_sigma=2,
                                                                          freqs=freqlist, phases=np.zeros(len(freqlist))))
                        else:
                            print("blockade pulse type not recognized, not blockading")
        sequencer.sync_channels_time(self.channels)

    def optimal_control_test_with_blockade_1step(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        mode_index = self.expt_cfg['mode_index']
        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
        if self.expt_cfg['measurement'] == 'wigner_tomography_1d':
            for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index],
                                                    phase=self.expt_cfg['cavity_phase_during_blockade_drive'],
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.idle_all(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'],
                                           amp=self.expt_cfg['amp'], phase=x, len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                else:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=x,
                                           phase=self.expt_cfg['phase'], len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'wigner_tomography_2d':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])

            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq_list')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=False)

                sequencer.sync_channels_time(self.channels)

                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                else:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df)
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                            add_freq= df)

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'parity_meas_debug_pnrqs':  # writing in progress
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False

                self.pi_resolved_joint_parity_measurement(self, sequencer)

                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                else:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df)
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                            add_freq= df)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        elif self.expt_cfg['measurement'] == 'photon_number_distribution_measurement':
            for n in np.arange(self.expt_cfg['N_max']):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=1.0, print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*n*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] +
                                   n*(n-1.0)/2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chi_2_by2pi'][self.expt_cfg['mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'rabi':
            for ii in range(self.expt_cfg['rabi_steps'] + 1):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=1.0, print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer, pulse_frac=self.expt_cfg['pulse_frac']*ii/
                                                                            self.expt_cfg['rabi_steps'],
                                                      print_times=print_times)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'qubit_tomography':
            for ii in range(3):
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=1.0, print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False

                if ii == 0:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif ii  ==1:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                else:pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'parity_bandwidth_wt_calibration':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['cavity_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])

            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq_list')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_blockade']:
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                    else:
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                else:
                    sequencer.new_sequence(self)
                self.prep_optimal_control_pulse_with_blockade(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=False)
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                           amp=tom_amp, phase=tom_phase,
                           pulse_type=self.expt_cfg['parity_cavity_pulse_type'], add_freq=0.0)
                sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=self.expt_cfg['second_half_pi_phase'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
            return sequencer.complete(self, plot=True)

        else:
            print("!!!Experiment not recognized!!!")

        return sequencer.complete(self, plot=self.plot_visdom)

    ### in progress ###
    def optimal_control_test_with_blockade_multimode(self, sequencer):
        # assumes that first pair of pulses is x/y of qubit ge pulse, then ef, cavity, sideband
        qubit_id = '1'
        print_times = True
        if self.expt_cfg['measurement'] == 'wigner_tomography_1d':
            for x in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer, pulse_frac=self.expt_cfg['prep_pulse_frac'], print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.idle_all(sequencer, time=5.0)
                if self.expt_cfg['sweep_phase']:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'],
                                           amp=self.expt_cfg['amp'], phase=x, len=self.expt_cfg['tomography_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                else:
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=x,
                                           phase=self.expt_cfg['1d_tomography_phase'], len=self.expt_cfg['tomography_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'wigner_tomography_2d':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])
            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)

                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)

                sequencer.sync_channels_time(self.channels)
                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=False)
                sequencer.sync_channels_time(self.channels)

                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['tomography_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'multimode_wigner_tomography_2d':
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])

            for ii, y1 in enumerate(ys):
                sequencer.new_sequence(self)
                x1 = xs[ii]
                y2 = ys[self.expt_cfg['tom2_index_pt']]
                x2 = xs[self.expt_cfg['tom2_index_pt']]
                if len(self.expt_cfg['probe_mode_indices']) > 2:
                    y3 = ys[self.expt_cfg['tom3_index_pt']]
                    x3 = xs[self.expt_cfg['tom3_index_pt']]
                    tom_phase3 = np.arctan2(y3, x3)
                    tom_amp3 = np.sqrt(x3 ** 2 + y3 ** 2)
                    if self.expt_cfg['transfer_fn_wt']:
                        tom_amp3 = self.transfer_function_blockade(tom_amp3, channel='cavity_amp_vs_freq_joint' +
                                                                                     str(self.expt_cfg[
                                                                                             'probe_mode_indices'][
                                                                                             2]))
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp1 = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][0]))
                    tom_amp2 = self.transfer_function_blockade(np.sqrt(x2 ** 2 + y2 ** 2),
                                                               channel='cavity_amp_vs_freq_joint' +
                                                                       str(self.expt_cfg['probe_mode_indices'][1]))
                else:
                    tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    tom_amp2 = np.sqrt(x2 ** 2 + y2 ** 2)
                tom_phase1 = np.arctan2(y1, x1)
                tom_phase2 = np.arctan2(y2, x2)

                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)

                sequencer.sync_channels_time(self.channels)
                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                sequencer.sync_channels_time(self.channels)
                if not len(self.expt_cfg['probe_mode_indices']) > 2:
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                 modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len'] for num in
                                                          range(len(self.expt_cfg['probe_mode_indices']))],
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'])
                else:
                    self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                 modes=self.expt_cfg['probe_mode_indices'],
                                                 use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                 sequential=self.expt_cfg['sequential_cav_displacements'],
                                                 n_max=self.expt_cfg['n_max'],
                                                 lengths=[self.expt_cfg['tomography_pulse_len'] for num in
                                                          range(len(self.expt_cfg['probe_mode_indices']))],
                                                 pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                 amps=[tom_amp1, tom_amp2, tom_amp3],
                                                 phases=[tom_phase1, tom_phase2, tom_phase3],
                                                 pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                 ramsey_parity=self.expt_cfg['ramsey_parity'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                       use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'parity_meas_debug_pnrqs':  # writing in progress
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False

                self.pi_resolved_joint_parity_measurement(self, sequencer)

                if self.expt_cfg['use_spec_pulse_from_pulse_info']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=df, phase=0.0,
                                           use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])

                else:
                    if self.expt_cfg['use_weak_drive_for_probe']:
                        self.gen_q_weak(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], amp=self.expt_cfg['amp'],
                                   phase=0, pulse_type=self.expt_cfg['pulse_type'],
                                   add_freq=df)
                    else:
                        self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'],
                                            add_freq= df)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'photon_number_distribution_measurement':
            for n in np.arange(self.expt_cfg['N_max']):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
                print_times = False
                self.pi_q_resolved(sequencer, qubit_id,
                                   add_freq=2*n*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] +
                                   n*(n-1.0)/2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chi_2_by2pi'][self.expt_cfg['mode_index']],
                                   use_weak_drive=self.expt_cfg['use_weak_drive_for_probe'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'rabi':
            for ii in range(self.expt_cfg['rabi_steps'] + 1):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer, pulse_frac=self.expt_cfg['pulse_frac']*ii/
                                                                            self.expt_cfg['rabi_steps'],
                                                      print_times=print_times)
                print_times = False
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        elif self.expt_cfg['measurement'] == 'qubit_tomography':
            for ii in range(3):
                # State preparation options, before running the main blockade pulse
                if self.expt_cfg['prep_state_first']:
                    if self.expt_cfg['prep_using_optimal_control_single']:  # single mode optimal control
                        self.prep_optimal_control_pulse_before_blockade(sequencer,
                                                                        pulse_frac=self.expt_cfg['prep_pulse_frac'],
                                                                        print_times=print_times)
                    elif self.expt_cfg['prep_using_optimal_control_multimode']:
                        # multimode optimal control
                        self.prep_optimal_control_pulse_before_blockade_multimode(sequencer,
                                                                                  pulse_frac=self.expt_cfg[
                                                                                      'prep_pulse_frac'],
                                                                                  print_times=print_times)
                    elif self.expt_cfg['prep_using_blockade']:
                        # put a photon in using blockade
                        sequencer.new_sequence()
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['prep_mode_index']
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index], phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)

                self.prep_optimal_control_pulse_with_blockade_multimode(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                      print_times=print_times)
                # sequencer.new_sequence(self)
                # self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                print_times = False

                if ii == 0:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                elif ii  ==1:
                    self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = np.pi/2)
                else:pass

                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        else:
            print("!!!Experiment not recognized!!!")

        return sequencer.complete(self, plot=self.plot_visdom)

    def prep_alpha_scattering_state(self, sequencer):
        # prep a coherent state in the cavity
        if self.expt_cfg['use_weak_cavity_for_prep']:
            self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                            amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                            pulse_type=self.expt_cfg['cavity_pulse_type'])
        else:
            self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                       amp=self.expt_cfg['prep_cav_amp'], phase=self.expt_cfg['prep_cav_phase'],
                       pulse_type=self.expt_cfg['cavity_pulse_type'])
        sequencer.sync_channels_time(self.channels)
        mode_index = self.expt_cfg['mode_index']
        blockade_levels = self.expt_cfg['blockade_levels']
        for qubit_id in self.expt_cfg['on_qubits']:
            # drive for the given evolution length, with blockade on
            if self.expt_cfg['evol_cav_len'] != 0:
                freqlist = self.quantum_device_cfg['qubit'][qubit_id]['freq'] + 2 * \
                           self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][mode_index] * np.array(
                    blockade_levels)
                blockade_len = self.expt_cfg['evol_cav_len']
                if self.expt_cfg['dressing_pulse_type'] == "gauss" and self.expt_cfg['cavity_pulse_type'] == "square":
                    blockade_len /= 4.0
                elif self.expt_cfg['dressing_pulse_type'] == "square" and self.expt_cfg['cavity_pulse_type'] == "gauss":
                    blockade_len *= 4.0

                if self.expt_cfg['use_weak_cavity_for_evol'] and self.expt_cfg['use_weak_drive_for_dressing']:
                    self.gen_c_and_blockade_weak(sequencer, mode_index=self.expt_cfg['mode_index'],
                                                 len=self.expt_cfg['evol_cav_len'],
                                                 amp=self.expt_cfg['evol_cav_amp'],
                                                 phase=self.expt_cfg['evol_cav_phase'],
                                                 pulse_type=self.expt_cfg['evol_cav_pulse_type'],
                                                 blockade_amp=self.expt_cfg['dressing_amp'],
                                                 blockade_freqs=freqlist,
                                                 blockade_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                                 add_freq=self.expt_cfg['cavity_offset_freq'])

                elif self.expt_cfg['use_weak_cavity_for_evol']:
                    self.gen_c_weak(sequencer, mode_index=self.expt_cfg['mode_index'],
                                    len=self.expt_cfg['evol_cav_len'],
                                    amp=self.expt_cfg['evol_cav_amp'], phase=self.expt_cfg['evol_cav_phase'],
                                    pulse_type=self.expt_cfg['evol_cav_pulse_type'],
                                    add_freq=self.expt_cfg['cavity_offset_freq'])
                    # blockade pulse
                    blockade_len = self.expt_cfg['evol_cav_len']
                    if self.expt_cfg['dressing_pulse_type'] == "gauss":
                        blockade_len /= 4.0
                    self.gen_q(sequencer, len=blockade_len, amp=self.expt_cfg['dressing_amp'], phase=0,
                               pulse_type=self.expt_cfg['dressing_pulse_type'])
                else:
                    self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['evol_cav_len'],
                               amp=self.expt_cfg['evol_cav_amp'], phase=self.expt_cfg['evol_cav_phase'],
                               pulse_type=self.expt_cfg['evol_cav_pulse_type'],
                               add_freq=self.expt_cfg['cavity_offset_freq'])
                    # blockade pulse
                    if self.expt_cfg['use_weak_drive_for_dressing']:
                        self.gen_q_weak(sequencer, len=blockade_len, amp=self.expt_cfg['dressing_amp'], phase=0,
                                        pulse_type=self.expt_cfg['dressing_pulse_type'])
                    else:
                        self.gen_q(sequencer, len=blockade_len, amp=self.expt_cfg['dressing_amp'], phase=0,
                                   pulse_type=self.expt_cfg['dressing_pulse_type'])
            sequencer.sync_channels_time(self.channels)

    def alpha_scattering_off_blockade(self, sequencer):
        qubit_id = self.expt_cfg['on_qubits'][0]
        # set up wigner tomography measurement
        if self.expt_cfg['use_wigner_points_file']:  # pull from file of wigner points
            with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                if self.expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                    # end edit
                else:
                    xs = np.array(f['ax'])
                    ys = np.array(f['ay'])
            for ii, y in enumerate(ys):
                x = xs[ii]
                if self.expt_cfg['transfer_fn_wt']:
                    tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2),
                                                              channel='cavity_amp_vs_freq_list')
                else:
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                tom_phase = np.arctan2(y, x)
                if self.expt_cfg['print_wigner_amp_phase']:
                    print(tom_amp, tom_phase)
                sequencer.new_sequence(self)
                self.prep_alpha_scattering_state(sequencer)
                self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                       phase=tom_phase, len=self.expt_cfg['tomography_pulse_len'],
                                       pulse_type=self.expt_cfg['tomography_pulse_type'])
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()

        else:  # otherwise do a 2d sweep
            x0 = self.expt_cfg['offset_x']
            y0 = self.expt_cfg['offset_y']
            for y in (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0):
                for x in (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0):
                    tom_amp = np.sqrt(x ** 2 + y ** 2)
                    tom_phase = np.arctan2(y, x)
                    sequencer.new_sequence(self)
                    self.prep_alpha_scattering_state(sequencer)
                    self.wigner_tomography(sequencer, qubit_id, mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                           phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                           pulse_type=self.expt_cfg['tomography_pulse_type'])
                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
        return sequencer.complete(self, plot=self.plot_visdom)

    def transfer_function(self, omegas_in, channel='qubitweak', mode_index=0):
        # takes input array of omegas and converts them to output array of amplitudes,
        # using a calibration h5 file defined in the experiment config
        # pull calibration data from file, handling properly in case of multimode cavity
        if channel == 'cavity_list':
            fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel][mode_index]
        elif channel == 'cavity_list_weak':
            fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel][mode_index]
        else:
            fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel]
        with File(fn_file, 'r') as f:
            omegas = f['omegas'][()]
            amps = f['amps'][()]
        # assume zero frequency at zero amplitude, used for interpolation function
        omegas = np.append(omegas, -omegas)
        amps = np.append(amps, -amps)
        omegas = np.append(omegas, 0.0)
        amps = np.append(amps, 0.0)
        o_s = [x for y, x in sorted(zip(amps, omegas))]
        a_s = np.sort(amps)

        # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
        transfer_fn = interpolate.interp1d(o_s, a_s)
        output_amps = []
        max_interp_index = np.argmax(omegas)
        for i in range(len(omegas_in)):
            # if frequency greater than calibrated range, assume a proportional relationship (high amp)
            if np.abs(omegas_in[i]) > omegas[max_interp_index]:
                output_amps.append(omegas_in[i] * amps[max_interp_index] / omegas[max_interp_index])
            else:  # otherwise just use the interpolated transfer function
                output_amps.append(transfer_fn((omegas_in[i])))
        return np.array(output_amps)

    def transfer_function_blockade(self, amp, channel='cavity_amp_vs_amp'):
        # pull calibration data from file
        if channel == 'cavity_amp_vs_freq_list':
            fn_file = self.experiment_cfg['transfer_function_blockade_calibration_files'][channel][self.expt_cfg['mode_index']]
            channel = 'cavity_amp_vs_freq'
        elif 'cavity_amp_vs_freq_joint' in channel:
            fn_file = self.experiment_cfg['transfer_function_blockade_calibration_files']['cavity_amp_vs_freq_list'][int(channel[-1])]
            channel = 'cavity_amp_vs_freq'
        else:
            fn_file = self.experiment_cfg['transfer_function_blockade_calibration_files'][channel]
        if channel == 'cavity_amp_vs_amp':
            with File(fn_file, 'r') as f:
                amps_desired = f['amps_desired'][()]
                amps_awg = f['amps_awg'][()]
            # assume zero amp at zero amplitude, used for interpolation function
            amps_desired = np.append(amps_desired, -amps_desired)
            amps_awg = np.append(amps_awg, -amps_awg)
            amps_desired = np.append(amps_desired, 0.0)
            amps_awg = np.append(amps_awg, 0.0)
            amps_desired_s = [x for y, x in sorted(zip(amps_awg, amps_desired))]
            amps_awg_s = np.sort(amps_awg)

            # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
            transfer_fn = interpolate.interp1d(amps_desired_s, amps_awg_s)
            max_interp_index = np.argmax(amps_desired)
            if np.abs(amp) > amps_desired[max_interp_index]:
                print("interpolating beyond max range")
                output_amp = amp * amps_awg[max_interp_index] / amps_desired[max_interp_index]
            else:  # otherwise just use the interpolated transfer function
                output_amp = transfer_fn(amp)
        elif channel == "cavity_amp_vs_freq":
            with File(fn_file, 'r') as f:
                omegas = f['omegas'][()]
                amps = f['amps'][()]
            # assume zero frequency at zero amplitude, used for interpolation function
            omegas = np.append(omegas, -omegas)
            amps = np.append(amps, -amps)
            omegas = np.append(omegas, 0.0)
            amps = np.append(amps, 0.0)
            o_s = [x for y, x in sorted(zip(amps, omegas))]
            a_s = np.sort(amps)

            # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
            transfer_fn = interpolate.interp1d(o_s, a_s)
            output_amps = []
            max_interp_index = np.argmax(omegas)
            if np.abs(amp) > omegas[max_interp_index]:
                print("interpolating beyond max range")
                output_amp = amp * amps[max_interp_index] / omegas[max_interp_index]
            else:  # otherwise just use the interpolated transfer function
                output_amp = transfer_fn(amp)
        else:
            print("transfer function channel not found, using original input amp")
            output_amp = amp
        return output_amp

    def drag_beta_calibration_qubit(self, sequencer):
        beta_list = np.arange(self.expt_cfg['beta_start'], self.expt_cfg['beta_stop'], self.expt_cfg['beta_step'])
        for qubit_id in self.expt_cfg['on_qubits']:
            freq = self.qubit_freq[qubit_id]
            if self.expt_cfg['calibrate_ef']:
                freq += self.quantum_device_cfg['qubit'][qubit_id]["anharmonicity"]
                A_pi = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_ef_amp']
                A_piby2 = self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_ef_amp']
                sigma_len_pi = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_ef_len']
                sigma_len_piby2 = self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_ef_len']
            else:
                A_pi = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp']
                A_piby2 = self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_amp']
                sigma_len_pi = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_len']
                sigma_len_piby2 = self.quantum_device_cfg['pulse_info'][qubit_id]['half_pi_len']
            for beta in beta_list:
                sequencer.new_sequence(self)
                if self.expt_cfg['calibrate_ef']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                # state prep, Xpi Ypi/2
                sequencer.append('charge%s' % qubit_id, DRAG(A=A_pi, beta=beta, sigma_len=sigma_len_pi, cutoff_sigma=2,
                                                             freq=freq, phase=np.pi/2))
                sequencer.append('charge%s' % qubit_id, DRAG(A=A_piby2, beta=beta, sigma_len=sigma_len_piby2,
                                                             cutoff_sigma=2, freq=freq, phase=0))
                if self.expt_cfg['calibrate_ef']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                # end state prep
                self.readout_pxi(sequencer, self.expt_cfg["on_qubits"])
                sequencer.end_sequence()

            for beta in beta_list:
                sequencer.new_sequence(self)
                # state prep, Ypi Xpi/2
                if self.expt_cfg['calibrate_ef']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.append('charge%s' % qubit_id, DRAG(A=A_pi, beta=beta, sigma_len=sigma_len_pi, cutoff_sigma=2,
                                                             freq=freq, phase=0))
                sequencer.append('charge%s' % qubit_id, DRAG(A=A_piby2, beta=beta, sigma_len=sigma_len_piby2,
                                                             cutoff_sigma=2, freq=freq, phase=np.pi/2))
                if self.expt_cfg['calibrate_ef']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                # end state prep
                self.readout_pxi(sequencer, self.expt_cfg["on_qubits"])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def parity_measurement_bandwidth_calibration(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            for cavity_amp in np.arange(self.expt_cfg['start_amp'], self.expt_cfg['stop_amp'], self.expt_cfg['amp_step']):
                sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['prep_cav_len'],
                           amp=cavity_amp, phase=0,
                           pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=self.expt_cfg['add_freq_cav'])
                sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def multimode_parity_measurement_bandwidth_calibration(self, sequencer):
        for qubit_id in self.expt_cfg['on_qubits']:
            if self.expt_cfg['use_wigner_points_file']:  # pull from file of wigner points
                with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
                    if self.expt_cfg['transfer_fn_wt']:
                        # Kevin edit testing a transfer function
                        xs = np.array(f['alphax'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        ys = np.array(f['alphay'][()]) / (self.expt_cfg['tomography_pulse_len'])
                        # end edit
                    else:
                        xs = np.array(f['ax'])
                        ys = np.array(f['ay'])

                for ii, y1 in enumerate(ys):
                    sequencer.new_sequence(self)
                    x1 = xs[ii]
                    y2 = ys[self.expt_cfg['tom2_index_pt']]
                    x2 = xs[self.expt_cfg['tom2_index_pt']]
                    if len(self.expt_cfg['probe_mode_indices']) > 2:
                        y3 = ys[self.expt_cfg['tom3_index_pt']]
                        x3 = xs[self.expt_cfg['tom3_index_pt']]
                        tom_phase3 = np.arctan2(y3, x3)
                        tom_amp3 = np.sqrt(x3 ** 2 + y3 ** 2)
                        if self.expt_cfg['transfer_fn_wt']:
                            tom_amp3 = self.transfer_function_blockade(tom_amp3, channel='cavity_amp_vs_freq_joint' +
                                                                               str(self.expt_cfg['probe_mode_indices'][
                                                                                       2]))
                    if self.expt_cfg['transfer_fn_wt']:
                        tom_amp1 = self.transfer_function_blockade(np.sqrt(x1 ** 2 + y1 ** 2),
                                                                   channel='cavity_amp_vs_freq_joint' +
                                                                           str(self.expt_cfg['probe_mode_indices'][0]))

                        tom_amp2 = self.transfer_function_blockade(np.sqrt(x2 ** 2 + y2 ** 2),
                                                                   channel='cavity_amp_vs_freq_joint' +
                                                                           str(self.expt_cfg['probe_mode_indices'][1]))
                    else:
                        tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                        tom_amp2 = np.sqrt(x2 ** 2 + y2 ** 2)
                    tom_phase1 = np.arctan2(y1, x1)
                    tom_phase2 = np.arctan2(y2, x2)

                    if self.expt_cfg['prep_state_before_blockade']:
                        if self.expt_cfg['prep_using_blockade']:
                            blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                            # print (blockade_pulse_info)
                            mode_index = self.expt_cfg['prep_mode_index']
                            self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                        len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                        cavity_pulse_type=
                                                        blockade_pulse_info['blockade_cavity_pulse_type'][
                                                            mode_index],
                                                        use_weak_for_dressing=
                                                        blockade_pulse_info['use_weak_for_blockade'][
                                                            mode_index],
                                                        dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                            mode_index],
                                                        blockade_levels=[2], dressing_pulse_type="square",
                                                        cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                            mode_index], phase=0,
                                                        add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                            mode_index],
                                                        weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                            'use_weak_for_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                          blockade_mode_indices=self.expt_cfg[
                                                              'blockade_mode_indices'],
                                                          cavity_drive_mode_indices=self.expt_cfg[
                                                              'cavity_drive_mode_indices'],
                                                          length=self.expt_cfg['cavity_pulse_len'],
                                                          cavity_pulse_type=self.expt_cfg[
                                                              'cavity_pulse_type'],
                                                          use_weak_for_dressing=self.expt_cfg[
                                                              "use_weak_drive_for_dressing"],
                                                          dressing_amps=self.expt_cfg[
                                                              'dressing_amps'],
                                                          blockade_levels=np.array(
                                                              self.expt_cfg['blockade_levels']),
                                                          dressing_pulse_type=self.expt_cfg[
                                                              'dressing_pulse_type'],
                                                          cavity_amps=self.expt_cfg['cavity_amps'],
                                                          phases=self.expt_cfg['cavity_phases'],
                                                          add_freqs=np.array(
                                                              self.expt_cfg['cavity_offset_freqs']),
                                                          add_dressing_drive_offsets=self.expt_cfg[
                                                              'dressing_drive_offset_freqs'],
                                                          weak_cavity=self.expt_cfg[
                                                              'use_weak_cavity'])
                    sequencer.sync_channels_time(self.channels)
                    if not len(self.expt_cfg['probe_mode_indices']) > 2:
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                     modes=self.expt_cfg['probe_mode_indices'],
                                                     use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                     sequential=self.expt_cfg['sequential_cav_displacements'],
                                                     n_max=self.expt_cfg['n_max'],
                                                     lengths=[self.expt_cfg['tomography_pulse_len'] for num in
                                                              range(len(self.expt_cfg['probe_mode_indices']))],
                                                     pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                     amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                     pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                     ramsey_parity=self.expt_cfg['ramsey_parity'])
                    else:
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                     modes=self.expt_cfg['probe_mode_indices'],
                                                     use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                     sequential=self.expt_cfg['sequential_cav_displacements'],
                                                     n_max=self.expt_cfg['n_max'],
                                                     lengths=[self.expt_cfg['tomography_pulse_len'] for num in
                                                              range(len(self.expt_cfg['probe_mode_indices']))],
                                                     pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                     amps=[tom_amp1, tom_amp2, tom_amp3],
                                                     phases=[tom_phase1, tom_phase2, tom_phase3],
                                                     pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                     ramsey_parity=self.expt_cfg['ramsey_parity'])

                    self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                    sequencer.end_sequence()
            else:
                x0 = self.expt_cfg['offset_x']
                y0 = self.expt_cfg['offset_y']
                for y1 in (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0):
                    for x1 in (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0):
                        tom_amp1 = np.sqrt(x1 ** 2 + y1 ** 2)
                        x2 = (np.arange(self.expt_cfg['starty'], self.expt_cfg['stopy'], self.expt_cfg['stepy']) + y0)[
                            self.expt_cfg['tom2_index_pt']]
                        y2 = (np.arange(self.expt_cfg['startx'], self.expt_cfg['stopx'], self.expt_cfg['stepx']) + x0)[
                            self.expt_cfg['tom2_index_pt']]
                        tom_amp2 = np.sqrt(y2 ** 2 + x2 ** 2)
                        tom_phase1 = np.arctan2(y1, x1)
                        tom_phase2 = np.arctan2(y2, x2)
                        sequencer.new_sequence(self)
                        if self.expt_cfg['prep_state_before_blockade']:
                            if self.expt_cfg['prep_using_blockade']:
                                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                                # print (blockade_pulse_info)
                                mode_index = self.expt_cfg['prep_mode_index']
                                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                            len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                            cavity_pulse_type=
                                                            blockade_pulse_info['blockade_cavity_pulse_type'][
                                                                mode_index],
                                                            use_weak_for_dressing=
                                                            blockade_pulse_info['use_weak_for_blockade'][
                                                                mode_index],
                                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                                mode_index],
                                                            blockade_levels=[2], dressing_pulse_type="square",
                                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                                mode_index], phase=0,
                                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                                mode_index],
                                                            weak_cavity=
                                                            self.quantum_device_cfg['blockade_pulse_params'][
                                                                'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                        freqlist, amplist = self.multimode_blockade_pulse_segment(sequencer, qubit_id=qubit_id,
                                                                                  blockade_mode_indices=self.expt_cfg[
                                                                                      'blockade_mode_indices'],
                                                                                  cavity_drive_mode_indices=
                                                                                  self.expt_cfg[
                                                                                      'cavity_drive_mode_indices'],
                                                                                  length=self.expt_cfg[
                                                                                      'cavity_pulse_len'],
                                                                                  cavity_pulse_type=self.expt_cfg[
                                                                                      'cavity_pulse_type'],
                                                                                  use_weak_for_dressing=self.expt_cfg[
                                                                                      "use_weak_drive_for_dressing"],
                                                                                  dressing_amps=self.expt_cfg[
                                                                                      'dressing_amps'],
                                                                                  blockade_levels=np.array(
                                                                                      self.expt_cfg['blockade_levels']),
                                                                                  dressing_pulse_type=self.expt_cfg[
                                                                                      'dressing_pulse_type'],
                                                                                  cavity_amps=self.expt_cfg[
                                                                                      'cavity_amps'],
                                                                                  phases=self.expt_cfg['cavity_phases'],
                                                                                  add_freqs=np.array(self.expt_cfg[
                                                                                                         'cavity_offset_freqs']),
                                                                                  add_dressing_drive_offsets=
                                                                                  self.expt_cfg[
                                                                                      'dressing_drive_offset_freqs'],
                                                                                  weak_cavity=self.expt_cfg[
                                                                                      'use_weak_cavity'])
                        sequencer.sync_channels_time(self.channels)
                        self.joint_wigner_tomography(sequencer, qubit_id=qubit_id,
                                                     modes=self.expt_cfg['probe_mode_indices'],
                                                     use_weak_qubit_drive=self.expt_cfg['use_weak_drive_for_probe'],
                                                     sequential=self.expt_cfg['sequential_cav_displacements'],
                                                     n_max=self.expt_cfg['n_max'],
                                                     lengths=[self.expt_cfg['tomography_pulse_len']] * len(
                                                         self.expt_cfg['probe_mode_indices']),
                                                     pulse_type=self.expt_cfg['tomography_pulse_type'],
                                                     amps=[tom_amp1, tom_amp2], phases=[tom_phase1, tom_phase2],
                                                     pi_resolved_parity=self.expt_cfg['pi_resolved_parity'],
                                                     ramsey_parity=self.expt_cfg['ramsey_parity'],
                                                     add_final_piby2_phase=self.expt_cfg['add_final_piby2_phase'])
                        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)
    
    def parity_measurement_bandwidth_calibration_wigner_pts_state_prep(self, sequencer):
        rabi_len = self.expt_cfg['rabi_len']

        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['cavity_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['cavity_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

        for ii, y in enumerate(ys):
            x = xs[ii]
            if self.expt_cfg['transfer_fn_wt']:
                tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2),
                                                          channel='cavity_amp_vs_freq_list') * self.expt_cfg['scale_alphas']
            else:
                tom_amp = np.sqrt(x ** 2 + y ** 2) * self.expt_cfg['scale_alphas']
            tom_phase = np.arctan2(y, x)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['prep_state_before_blockade']:
                    if self.expt_cfg['prep_using_blockade']:
                        blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                        # print (blockade_pulse_info)
                        mode_index = self.expt_cfg['mode_index']
                        sequencer.new_sequence(self)
                        self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                                    len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                                    cavity_pulse_type=
                                                    blockade_pulse_info['blockade_cavity_pulse_type'][
                                                        mode_index],
                                                    use_weak_for_dressing=
                                                    blockade_pulse_info['use_weak_for_blockade'][
                                                        mode_index],
                                                    dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][
                                                        mode_index],
                                                    blockade_levels=[2], dressing_pulse_type="square",
                                                    cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][
                                                        mode_index],
                                                    phase=0,
                                                    add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][
                                                        mode_index],
                                                    weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                        'use_weak_for_cavity'])
                        sequencer.sync_channels_time(self.channels)
                else:
                    sequencer.new_sequence(self)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=self.expt_cfg['mode_index'],
                                            len=rabi_len,
                                            cavity_pulse_type=self.expt_cfg['rabi_cavity_pulse_type'],
                                            use_weak_for_dressing=self.expt_cfg["use_weak_drive_for_dressing"],
                                            dressing_amp=self.expt_cfg['dressing_amp'],
                                            blockade_levels=np.array(self.expt_cfg['blockade_levels']),
                                            dressing_pulse_type=self.expt_cfg['dressing_pulse_type'],
                                            cavity_amp=self.expt_cfg['cavity_amp'],
                                            phase=self.expt_cfg['phase'],
                                            add_freq=self.expt_cfg['cavity_offset_freq'],
                                            add_dressing_drive_offset=self.expt_cfg['dressing_drive_offset_freq'],
                                            weak_cavity=self.expt_cfg['weak_cavity_for_blockade_expt'])
                sequencer.sync_channels_time(self.channels)
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'],
                           amp=tom_amp, phase=tom_phase,
                           pulse_type=self.expt_cfg['cavity_pulse_type'], add_freq=self.expt_cfg['add_freq_cav'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["check_readout_fluctuations_alone"]:
                    pass
                else:
                    self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.half_pi_q(sequencer, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                                   phase=self.expt_cfg['second_half_pi_phase'])
                    sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
                sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def cavity_blockade_t1(self, sequencer):
        for length in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                # print (blockade_pulse_info)
                mode_index = self.expt_cfg['mode_index']
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                            len=blockade_pulse_info['blockade_pi_length'][mode_index],
                                            cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                mode_index],
                                            use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                mode_index],
                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                            blockade_levels=[2], dressing_pulse_type="square",
                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                            phase=0,
                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                            weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=length)
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['probe_zero']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=0.0, phase=0.0,
                                       use_weak_drive=self.expt_cfg['use_weak_resolved_pulse'])
                else:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=2*self.quantum_device_cfg['flux_pulse_info']['1']['chiby2pi_e'][mode_index],
                                       phase=0.0, use_weak_drive=self.expt_cfg['use_weak_resolved_pulse'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def cavity_blockade_ramsey(self, sequencer):
        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                blockade_pulse_info = self.quantum_device_cfg['blockade_pulse_params']
                # print (blockade_pulse_info)
                mode_index = self.expt_cfg['mode_index']
                prep_mode_index = self.expt_cfg['prep_mode_index']
                if self.expt_cfg['prep_photon_with_blockade']:
                    self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=prep_mode_index,
                                                len=blockade_pulse_info['blockade_pi_length'][prep_mode_index],
                                                cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                    prep_mode_index],
                                                use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                    prep_mode_index],
                                                dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][prep_mode_index],
                                                blockade_levels=[2], dressing_pulse_type="square",
                                                cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][prep_mode_index],
                                                phase=0,
                                                add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][prep_mode_index],
                                                weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                    'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                            len=blockade_pulse_info['blockade_half_pi_length'][mode_index],
                                            cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                mode_index],
                                            use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                mode_index],
                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                            blockade_levels=[2], dressing_pulse_type="square",
                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                            phase=0,
                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                            weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=ramsey_len)
                sequencer.sync_channels_time(self.channels)
                self.blockade_pulse_segment(sequencer, qubit_id=qubit_id, mode_index=mode_index,
                                            len=blockade_pulse_info['blockade_half_pi_length'][mode_index],
                                            cavity_pulse_type=blockade_pulse_info['blockade_cavity_pulse_type'][
                                                mode_index],
                                            use_weak_for_dressing=blockade_pulse_info['use_weak_for_blockade'][
                                                mode_index],
                                            dressing_amp=blockade_pulse_info['blockade_pi_amp_qubit'][mode_index],
                                            blockade_levels=[2], dressing_pulse_type="square",
                                            cavity_amp=blockade_pulse_info['blockade_pi_amp_cavity'][mode_index],
                                            phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'],
                                            add_freq=blockade_pulse_info['blockade_cavity_offset_freq'][mode_index],
                                            weak_cavity=self.quantum_device_cfg['blockade_pulse_params'][
                                                'use_weak_for_cavity'])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg['probe_zero']:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=0.0, phase=0.0,
                                       use_weak_drive=self.expt_cfg['use_weak_resolved_pulse'])
                else:
                    self.pi_q_resolved(sequencer, qubit_id, add_freq=2*self.quantum_device_cfg['flux_pulse_info']['1']['chiby2pi_e'][mode_index],
                                       phase=0.0, use_weak_drive=self.expt_cfg['use_weak_resolved_pulse'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    # work in progress, goal: prepare a state with the blockade on, then perform wigner tomography on it to find it,
    # then act an optimal control blockade pulse and perform wigner tomography on that,
    # for full gate characterization this function will be called in a sequential experiment
    def blockade_gate_tomography(self, sequencer):

        # load set of wigner points used for reconstructing the state
        with File(self.expt_cfg['wigner_points_file_name'], 'r') as f:
            if self.expt_cfg['transfer_fn_wt']:
                # Kevin edit testing a transfer function
                xs = np.array(f['alphax'][()]) / (self.expt_cfg['cavity_pulse_len'])
                ys = np.array(f['alphay'][()]) / (self.expt_cfg['cavity_pulse_len'])
                # end edit
            else:
                xs = np.array(f['ax'])
                ys = np.array(f['ay'])

        # add option to either define drive preparation points or pull from wigner points file
        prep_drive_phase = self.expt_cfg['prep_drive_params']['phase']
        prep_drive_amp = self.expt_cfg['prep_drive_params']['amp']

        for ii, y in enumerate(ys):
            x = xs[ii]
            if self.expt_cfg['transfer_fn_wt']:
                tom_amp = self.transfer_function_blockade(np.sqrt(x ** 2 + y ** 2), channel='cavity_amp_vs_freq')
            else:
                tom_amp = np.sqrt(x ** 2 + y ** 2)
            tom_phase = np.arctan2(y, x)

            # prepare the initial state on which to act the gate
            sequencer.new_sequence(self)
            self.blockade_pulse_segment(sequencer, mode_index=self.expt_cfg['mode_index'],
                                        len=self.expt_cfg['prep_drive_params']['len'],
                                        cavity_pulse_type=self.expt_cfg['prep_drive_params']['pulse_type'],
                                        use_weak_for_dressing=self.expt_cfg['prep_blockade_params']['use_weak_drive'],
                                        dressing_amp=self.expt_cfg['prep_blockade_params']['amp'],
                                        blockade_levels=self.expt_cfg['prep_blockade_params']['levels'],
                                        dressing_pulse_type=self.expt_cfg['prep_blockade_params']['pulse_type'],
                                        cavity_amp=prep_drive_amp, phase=prep_drive_phase,
                                        weak_cavity=self.expt_cfg['prep_drive_params']['use_weak_drive'])
            sequencer.sync_channels_time(self.channels)

            # act the gate
            self.prep_optimal_control_pulse_with_blockade(sequencer, pulse_frac=self.expt_cfg['pulse_frac'],
                                                          print_times=False)
            sequencer.sync_channels_time(self.channels)

            # perform the measurement
            self.wigner_tomography(sequencer, '1', mode_index=self.expt_cfg['mode_index'], amp=tom_amp,
                                   phase=tom_phase, len=self.expt_cfg['cavity_pulse_len'],
                                   pulse_type=self.expt_cfg['tomography_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=self.plot_visdom)

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