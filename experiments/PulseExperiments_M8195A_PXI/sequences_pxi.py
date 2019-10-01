try:
    from .sequencer_pxi import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone,Double_Square, ARB, ARB_Sum
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a, ARB, ARB_Sum
# from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom
import os
import pickle

import copy

from h5py import File

class PulseSequences:
    # channels and awgs

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg



        self.pulse_info = self.quantum_device_cfg['pulse_info']

        self.multimodes = self.quantum_device_cfg['flux_pulse_info']

        try:self.cavity_pulse_info = self.quantum_device_cfg['cavity']
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



        self.multimodes = self.quantum_device_cfg['multimodes']

        self.communication = self.quantum_device_cfg['communication']
        self.sideband_cooling = self.quantum_device_cfg['sideband_cooling']

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,plot_visdom=True):
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

    def gen_q_weak(self,sequencer,qubit_id = '1',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
        if pulse_type.lower() == 'square':
            sequencer.append('qubitweak', Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.qubit_freq[qubit_id]+add_freq,
                                    phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('qubitweak', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['pi_amp'], flat_len=self.pulse_info[qubit_id]['pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s' % qubit_id, Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'], sigma_len=self.pulse_info[qubit_id]['pi_len'],cutoff_sigma=2,
                            freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))

    def half_pi_q(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square', add_freq=0):
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s' % qubit_id, Square(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], flat_len=self.pulse_info[qubit_id]['half_pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.qubit_freq[qubit_id]+add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
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

    def pi_q_resolved(self, sequencer, qubit_id='1',phase=0,add_freq = 0,use_weak_drive=False):
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
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'][mode_index],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_len'][mode_index],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][mode_index]+add_freq, phase=phase,fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][mode_index],plot=False))

    def pi_e0g2_sb(self,sequencer,qubit_id = '1',phase = 0,pulse_type = 'square',add_freq = 0,mode_index = 0):
        sequencer.append('sideband2',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_e0g2_amp'][mode_index],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_e0g2_len'][mode_index],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['e0g2_freq'][mode_index]+add_freq, phase=phase,fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                dc_offset=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['e0g2_dc_offset'][mode_index],plot=False))

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
        sequencer.append('sideband', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_all(self, sequencer, qubit_id='1', time=0):
        sequencer.sync_channels_time(self.channels)
        sequencer.append('charge%s' % qubit_id, Idle(time=time))
        sequencer.append('sideband', Idle(time=time))
        sequencer.append('cavity', Idle(time=time))
        sequencer.append('qubitweak', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_sb(self,sequencer,time=0.0):
        sequencer.append('sideband', Idle(time=time))

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
        # self.idle_q(sequencer, qubit_id, time=np.abs(1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0))
        self.idle_q(sequencer, qubit_id,time= self.quantum_device_cfg['flux_pulse_info'][qubit_id]['parity_time_e'][self.expt_cfg['mode_index']])
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sequencer.sync_channels_time(self.channels)

    def gen_c(self, sequencer, mode_index = 0, len=10, amp=1, add_freq=0, phase=0, pulse_type='square'):
        if pulse_type.lower() == 'square':
            sequencer.append('cavity', Square(max_amp=amp, flat_len=len,
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.cavity_freq[mode_index] + add_freq,phase=phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('cavity', Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2, freq=self.cavity_freq[mode_index] + add_freq,
                                                             phase=phase))

    def wigner_tomography(self, sequencer, qubit_id='1',mode_index = 0,amp = 0, phase=0,len = 0,pulse_type = 'square'):
        self.gen_c(sequencer,mode_index=mode_index, len=len, amp = amp, add_freq=0, phase=phase + np.pi, pulse_type=pulse_type)
        sequencer.sync_channels_time(self.channels)
        self.idle_all(sequencer, time=10)
        self.parity_measurement(sequencer,qubit_id)
    #
    def resonator_spectroscopy(self, sequencer):
        sequencer.new_sequence(self)
        if self.expt_cfg['pi_qubit']:
            for qubit_id in self.expt_cfg['on_qubits']:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                if self.expt_cfg['pi_qubit_ef']:
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

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
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

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


            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer,qubit_id,amp = self.expt_cfg['amp'],len=rabi_len,phase=0,pulse_type=self.expt_cfg['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
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
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+ self.quantum_device_cfg['qubit']['1']['anharmonicity']+self.quantum_device_cfg['qubit']['1']['anharmonicity_fh'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

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
        sideband_freq = self.expt_cfg['freq']
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
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=length,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=2)
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    # self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])  # ge pulse for testing
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

        sideband_freq = self.expt_cfg['freq']
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
                sequencer.append('sideband',
                             Square(max_amp=self.expt_cfg['amp'], flat_len=length, ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq + dfreq, phase=0,
                                    fix_phase=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                    dc_offset=dc_offset,plot=False))
                if self.expt_cfg['f0g1']:
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q_sb(sequencer, qubit_id, time=20)
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

    def chi_dressing_calibration(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:

                add_phase = 0
                if self.expt_cfg['add_photon']:
                    phase_freq = 2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']] + self.expt_cfg[
                        'ramsey_freq']
                else:
                    phase_freq =self.expt_cfg['ramsey_freq']
                if self.expt_cfg["h0e1"]:
                    sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['h0e1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                elif self.expt_cfg["f0g1"]:  sideband_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'][self.expt_cfg['mode_index']]+self.expt_cfg['detuning']
                else:
                    sideband_freq = 0.0
                    print ("what transition do you want to use for dressing good sir/madam")


                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

                if self.expt_cfg['add_photon']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                    sequencer.sync_channels_time(self.channels)
                    self.pi_f0g1_sb(sequencer, qubit_id, pulse_type='square',phase = offset_phase/2.0,mode_index=self.expt_cfg['mode_index'])
                    sequencer.sync_channels_time(self.channels)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_all(sequencer, qubit_id, time=2)
                dc_offset = 0
                sequencer.append('sideband',
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len,
                                        ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id][
                                            'ramp_sigma_len'], cutoff_sigma=2, freq=sideband_freq, phase=0,
                                        fix_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['fix_phase'],
                                        dc_offset=dc_offset,
                                        plot=False))
                self.idle_all(sequencer, qubit_id, time=2)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase = add_phase+2*np.pi*ramsey_len*phase_freq)

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
                else:phase_freq = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_dc_offset'][self.expt_cfg['mode_index']] + self.expt_cfg['ramsey_freq']

                offset_phase = self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_pi_pi_offset'][self.expt_cfg['mode_index']]

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

    def cavity_spectroscopy_resolved_qubit_pulse(self, sequencer):
        # transmon ef rabi with tek2
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['excite_qubit']:
                    self.pi_q(sequencer, qubit_id,
                                                pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                self.gen_c(sequencer,mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_pulse_len'], amp=self.expt_cfg['cavity_amp'], add_freq=df, phase=0, pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                        pulse_type=self.expt_cfg['resolved_qubit_pulse_type'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'],overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def photon_number_resolved_qubit_spectroscopy(self, sequencer):
        for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):

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

                elif self.expt_cfg['state'] in ['2', '3', '4']:
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

                self.pi_q_resolved(sequencer, qubit_id,add_freq = 2*self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][self.expt_cfg['mode_index']]*n,use_weak_drive=self.expt_cfg['use_weak_drive'])

            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'], overlap=False)

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
                self.idle_all(sequencer, qubit_id, time=ramsey_len )
                self.gen_c(sequencer, mode_index=self.expt_cfg['mode_index'], len=self.expt_cfg['cavity_drive_len'],
                           amp=self.expt_cfg['cavity_drive_amp'], phase=np.pi + 2*np.pi*phase_freq*ramsey_len,
                           pulse_type=self.expt_cfg['cavity_pulse_type'])
                sequencer.sync_channels_time(self.channels)
                self.pi_q_resolved(sequencer, qubit_id,
                                            pulse_type=self.pulse_info[qubit_id]['pulse_type'],add_freq = 0)

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
            if kwargs['save']: self.save_sequences(multiple_sequences,kwargs['filename'])
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
                                                len=total_time,
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
                                                len=total_time,
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
                                                len=total_time,
                                                freq_list=[carrier_freqs["pi_ge"], carrier_freqs["pi_ef"]],
                                                phase_list=[0, 0], scale=1.0))
            counter += 2
            sequencer.append('cavity', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                           len=total_time, freq=carrier_freqs["cavity"], phase=0, scale=1.0))

    def prep_optimal_control_pulse(self, sequencer,ii=0, print_times=False):
        if self.expt_cfg['filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:
                    print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(self.expt_cfg['filename'])
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file
        num_pulses = len(pulses)
        sequencer.new_sequence(self)
        # write pulses to their appropriate channels
        counter = 0
        # currently only works for 1 qubit, kept for loop for convention / similarity to other experiments
        for qubit_id in self.expt_cfg['on_qubits']:
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
            # combine qubit ge and ef pulses
            if self.expt_cfg['ge_on'] and self.expt_cfg['ef_on']:
                sequencer.append('charge%s' % qubit_id, ARB_Sum(A_list_list=[pulses[2 * counter],
                                                                             pulses[2 * counter + 2]],
                                                                B_list_list=[pulses[2 * counter + 1],
                                                                             pulses[2 * counter + 3]],
                                                                len=total_time * (ii + 1.0) / self.expt_cfg['steps'],
                                                                freq_list=[carrier_freqs["pi_ge"],
                                                                           carrier_freqs["pi_ef"]],
                                                                phase_list=[0, 0],
                                                                scale_list=[2*self.expt_cfg['calibrations']['qubit_ge'],
                                                                            2*self.expt_cfg['calibrations']['qubit_ef']]))
                counter += 2
            elif self.expt_cfg['ge_on']:
                sequencer.append('charge%s' % qubit_id, ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                               len=total_time * (ii + 1.0) / self.expt_cfg['steps'],
                                               freq=carrier_freqs["pi_ge"], phase=0,
                                               scale=2*self.expt_cfg['calibrations']['qubit_ge']))
                counter += 1
            elif self.expt_cfg['ef_on']:
                sequencer.append('charge%s' % qubit_id, ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                               len=total_time * (ii + 1.0) / self.expt_cfg['steps'],
                                               freq=carrier_freqs["pi_ef"], phase=0,
                                               scale=2*self.expt_cfg['calibrations']['qubit_ef']))
                counter += 1
        if self.expt_cfg['cavity_on']:  # cavity drive
            sequencer.append('cavity', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                           len=total_time * (ii + 1.0) / self.expt_cfg['steps'],
                                           freq=carrier_freqs["cavity"], phase=0,
                                           scale=2*self.expt_cfg['calibrations']['cavity']))
            # maybe want to multiply/divide scale by g/delta, 125 MHz / (cav - qubit freqs)
            counter += 1
        if self.expt_cfg['sideband_on']:  # sideband drive
            sequencer.append('sideband', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                             len=total_time * (ii + 1.0) / self.expt_cfg['steps'],
                                             freq=carrier_freqs["f0g1"], phase=0,
                                             scale=2*self.expt_cfg['calibrations']['sideband']))
            counter += 1
        sequencer.sync_channels_time(self.channels)

    def prep_optimal_control_pulse_1step(self, sequencer, pulse_frac=1.0, print_times=False):
        if self.expt_cfg['filename'].split(".")[-1] == 'h5':  # detect if file is an h5 file
            with File(self.expt_cfg['filename'], 'r') as f:
                pulses = f['uks'][-1]
                total_time = f['total_time'][()] + 0.0
                dt = total_time / f['steps'][()]
                if print_times:
                    print(total_time, dt)
        else:  # if not h5, read from a text file
            pulses = np.genfromtxt(self.expt_cfg['filename'])
            total_time = 1200.0  # currently hard coded, total pulse time, unnecessary if using h5 file
        num_pulses = len(pulses)
        sequencer.new_sequence(self)
        # write pulses to their appropriate channels
        counter = 0
        # currently only works for 1 qubit, kept for loop for convention / similarity to other experiments
        for qubit_id in self.expt_cfg['on_qubits']:

            if self.expt_cfg['use_weak_drive']:
                qub_channel = 'qubitweak'
                print ("Using weak drive port")
                qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge_weak']
                qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef_weak']
            else:
                qub_channel = 'charge%s' % qubit_id
                qub_scale_ge = self.expt_cfg['calibrations']['qubit_ge']
                qub_scale_ef = self.expt_cfg['calibrations']['qubit_ef']

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
                # combine qubit ge and ef pulses
                if self.expt_cfg['ge_on'] and self.expt_cfg['ef_on']:

                    sequencer.append(qub_channel, ARB_Sum(A_list_list=[pulses[self.expt_cfg['pulse_number_map']['ge'][0]],
                                                                                 pulses[self.expt_cfg['pulse_number_map']['ge'][1]]],
                                                                    B_list_list=[pulses[self.expt_cfg['pulse_number_map']['ef'][0]],
                                                                                 pulses[self.expt_cfg['pulse_number_map']['ef'][1]]],
                                                                    len=total_time * pulse_frac,
                                                                    freq_list=[carrier_freqs["pi_ge"],
                                                                               carrier_freqs["pi_ef"]],
                                                                    phase_list=[0, 0],
                                                                    scale_list=[qub_scale_ge,qub_scale_ef]))

                elif self.expt_cfg['ge_on']:
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
                    sequencer.append('cavity', ARB(A_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][0]], B_list=pulses[self.expt_cfg['pulse_number_map']['cavity'][1]],
                                                   len=total_time * pulse_frac,
                                                   freq=carrier_freqs["cavity"], phase=0,
                                                   scale=self.expt_cfg['calibrations']['cavity']))
                    # maybe want to multiply/divide scale by g/delta, 125 MHz / (cav - qubit freqs)
        
                if self.expt_cfg['sideband_on']:  # sideband drive
                    sequencer.append('sideband', ARB(A_list=pulses[2 * counter], B_list=pulses[2 * counter + 1],
                                                     len=total_time * pulse_frac,
                                                     freq=carrier_freqs["f0g1"], phase=0,
                                                     scale=2*self.expt_cfg['calibrations']['sideband']))
                    counter += 1
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

        elif self.expt_cfg['measurement'] == 'photon_number_resolved_qubit_spectroscopy':
            for df in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
                self.prep_optimal_control_pulse_1step(sequencer,pulse_frac=self.expt_cfg['pulse_frac'], print_times=print_times)
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

    def weak_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                self.gen_q_weak(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def cavity_temp_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['drive_1']:
                    self.gen_q(sequencer, qubit_id, len=rabi_len,
                               amp=self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved'], phase=0,
                               add_freq=2 * self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e'][0],
                               pulse_type=self.expt_cfg['pulse_type'])
                else:
                    self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.quantum_device_cfg['pulse_info'][qubit_id]['pi_amp_resolved'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

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