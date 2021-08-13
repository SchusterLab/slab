try:
    from .sequencer_pxi import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone, linear_ramp, adb_ramp
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

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.lattice_cfg = lattice_cfg



        self.pulse_info = self.quantum_device_cfg['pulse_info']

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg'] #which awg each channel correspons to

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay_array_roll_ns']
        
        self.on_qubits = list(quantum_device_cfg["on_qubits"].keys())

        # pulse params
        for qb in self.on_qubits:
            self.qubit_freq = {}
            self.qubit_freq[qb] = self.quantum_device_cfg['qubit'][qb]['freq']

            self.qubit_ef_freq = {}
            self.qubit_ef_freq[qb] = self.quantum_device_cfg['qubit'][qb]['freq']+self.quantum_device_cfg[
                'qubit'][qb]['anharmonicity']

            self.qubit_pi_I = {}
            self.qubit_pi_I[qb]= Square(max_amp=self.pulse_info[qb]['pi_amp'], flat_len=self.pulse_info[qb]['pi_len'],
                            ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qb]['iq_freq'],phase=0)

            self.qubit_pi_Q = {}
            self.qubit_pi_Q[qb]= Square(max_amp=self.pulse_info[qb]['pi_amp'], flat_len=self.pulse_info[qb][
                'pi_len'], ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qb]['iq_freq'],
                                             phase=self.pulse_info[qb]['Q_phase'])

            self.qubit_half_pi_I = {}
            self.qubit_half_pi_I[qb]= Square(max_amp=self.pulse_info[qb]['half_pi_amp'], flat_len=self.pulse_info[
                qb]['half_pi_len'],ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qb]['iq_freq'], phase=0)

            self.qubit_half_pi_Q = {}
            self.qubit_half_pi_Q[qb]= Square(max_amp=self.pulse_info[qb]['half_pi_amp'], flat_len=self.pulse_info[qb]['half_pi_len'],
                            ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.pulse_info[qb]['iq_freq'], phase=self.pulse_info[qb]['Q_phase'])

            self.qubit_ef_pi = {}
            self.qubit_ef_pi[qb]= Gauss(max_amp=self.pulse_info[qb]['pi_ef_amp'], sigma_len=self.pulse_info[qb][
                'pi_ef_len'], cutoff_sigma=2,
                       freq=self.qubit_ef_freq[qb], phase=0, plot=False)

            self.qubit_ef_half_pi = {}
            self.qubit_ef_half_pi[qb]= Gauss(max_amp=self.pulse_info[qb]['half_pi_ef_amp'],
                                             sigma_len=self.pulse_info[qb]['half_pi_ef_len'],
                       cutoff_sigma=2, freq=self.qubit_ef_freq[qb], phase=0, plot=False)

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)


    def  __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=True):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
        self.plot_visdom = plot_visdom


    def gen_q(self,sequencer,qubit_id = 'A',len = 10,amp = 1,add_freq = 0,phase = 0,pulse_type = 'square'):
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

    def pi_q(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
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

    def half_pi_q(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
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

    def pi_q_ef(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']
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

    def half_pi_q_ef(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']
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

    def pi_q_fh(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit']['A']['anharmonicity']+self.quantum_device_cfg['qubit']['A']['anharmonicity_fh']
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

    def half_pi_q_fh(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
        freq = self.pulse_info[qubit_id]['iq_freq'] + self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity']+self.quantum_device_cfg['qubit']['A']['anharmonicity_fh']
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

    def pi_f0g1_sb(self,sequencer,qubit_id = 'A',phase = 0,pulse_type = 'square'):
        sequencer.append('sideband',Square(max_amp=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_amp'],flat_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['pi_f0g1_len'],
                                ramp_sigma_len=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'],
                                cutoff_sigma=2, freq=self.quantum_device_cfg['flux_pulse_info'][qubit_id]['f0g1_freq'], phase=phase,
                                plot=False))

    def idle_q(self,sequencer,qubit_id = 'A',time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))

    def idle_q_sb(self, sequencer, qubit_id='A', time=0):
        sequencer.append('charge%s_I' % qubit_id, Idle(time=time))
        sequencer.append('charge%s_Q' % qubit_id, Idle(time=time))
        sequencer.append('sideband', Idle(time=time))
        sequencer.sync_channels_time(self.channels)

    def idle_sb(self,sequencer,time=0):
        sequencer.append('sideband', Idle(time=time))

    def ff_pulse(self, sequencer, ff_len, pulse_type, flux_vec, flip_amp=False):
        area_vec = []
        for qb, flux in enumerate(flux_vec):
            if flip_amp:
                flux = -flux
            if pulse_type=="square":
                pulse = Square(max_amp=flux, flat_len=ff_len[qb],
                                    ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb], cutoff_sigma=2, freq=0,
                                    phase=0)
            if pulse_type=="linear":
                pulse = linear_ramp(max_amp=flux, flat_len=ff_len[qb],
                                    ramp1_len=self.lattice_cfg['ff_info']['ff_linear_ramp_len'],
                                    ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                    cutoff_sigma=2, freq=0, phase=0)
            if pulse_type=="adb":
                pulse = adb_ramp(max_amp=flux, flat_len=ff_len[qb],
                                 adb_ramp1_sig=self.lattice_cfg['ff_info']['ff_adb_ramp_sig'][qb],
                                 ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                 cutoff_sigma=2, freq=0, phase=0)

            sequencer.append('ff_Q%s' % qb, pulse)
            area_vec.append(pulse.get_area())
        return np.asarray(area_vec)

    def ff_comp(self, sequencer, area_vec):
        for qb in range(len(self.expt_cfg['ff_vec'])):
            ff_len = area_vec[qb]/self.lattice_cfg['ff_info']['comp_pulse_amp'][qb]
            sequencer.append('ff_Q%s' % qb,
                             Square(max_amp=self.lattice_cfg['ff_info']['comp_pulse_amp'][qb], flat_len=ff_len,
                                    ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb], cutoff_sigma=2, freq=0,
                                    phase=0))

    def ff_square_and_comp(self, sequencer, ff_len, flux_vec):
        area_vec = self.ff_pulse(sequencer, ff_len, pulse_type="square", flux_vec = flux_vec, flip_amp=False)

        # COMPENSATION PULSE
        if self.quantum_device_cfg["ff_info"]["ff_comp_sym"]:
            self.ff_pulse(sequencer, ff_len, pulse_type="square", flux_vec = flux_vec, flip_amp=True)
        else:
            self.ff_comp(self, sequencer, area_vec)

    def pad_start_pxi(self,sequencer,on_qubits=None, time = 500):
        # Need 500 ns of padding for the sequences to work reliably. Not sure exactly why.
        for channel in self.channels:
            sequencer.append(channel,
                             Square(max_amp=0.0, flat_len= time, ramp_sigma_len=0.001, cutoff_sigma=2, freq=0.0,
                                    phase=0))

    def pad_start_pxi_tek2(self,sequencer,on_qubits=None, time = 500):
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=time)
        sequencer.append('tek2_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

    def readout(self, sequencer, on_qubits=None, sideband = False):
        if on_qubits == None:
            on_qubits = ["A", "B"]

        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_trig

        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5

        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        sequencer.sync_channels_time(self.channels)


        for qubit_id in on_qubits:
            sequencer.append('readout%s' % qubit_id,
                             Square(max_amp=self.quantum_device_cfg['readout'][qubit_id]['amp'],
                                    flat_len=self.quantum_device_cfg['readout'][qubit_id]['length'],
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.quantum_device_cfg['readout'][qubit_id]['freq'],
                                    phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def readout_pxi(self, sequencer, on_qubits=None, sideband = False, overlap = False, synch_channels=None):
        if on_qubits == None:
            on_qubits = ["A", "B"]

        if synch_channels==None:
            synch_channels = self.channels
        sequencer.sync_channels_time(synch_channels)
        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(synch_channels)

        for qubit_id in on_qubits:
            sequencer.append('readout%s' % qubit_id,
                             Square(max_amp=self.quantum_device_cfg['readout'][qubit_id]['amp'],
                                    flat_len=self.quantum_device_cfg['readout'][qubit_id]['length'],
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                    phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def excited_readout_pxi(self, sequencer, on_qubits=None, sideband = False, overlap = False):
        if on_qubits == None:
            on_qubits = ["A", "B"]

        sequencer.sync_channels_time(self.channels)
        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(self.channels)

        for qubit_id in on_qubits:
            self.gen_q(sequencer, qubit_id, len=2000, amp=1, phase=0, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            sequencer.append('readout%s' % qubit_id,
                             Square(max_amp=self.quantum_device_cfg['readout'][qubit_id]['amp'],
                                    flat_len=self.quantum_device_cfg['readout'][qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=0,
                                    phase=0, phase_t0=readout_time_5ns_multiple))

        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    # def readout_pxi_pi(self, sequencer, on_qubits=None, sideband = False, overlap = False):
    #     if on_qubits == None:
    #         on_qubits = ["A", "B"]
    #
    #     sequencer.sync_channels_time(self.channels)
    #     readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_tri
    #     readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
    #     sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
    #     if overlap:
    #         pass
    #     else:
    #         sequencer.sync_channels_time(self.channels)
    #
    #     qubit_id = on_qubits[0]
    #     self.pi_q(sequencer, qubit_id, phase=0, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
    #     sequencer.append('drive',)
    #     sequencer.append('readout',
    #                      Square(max_amp=self.quantum_device_cfg['readout']['amp'],
    #                             flat_len=self.quantum_device_cfg['readout']['length'],
    #                             ramp_sigma_len=20, cutoff_sigma=2, freq=0,
    #                             phase=0, phase_t0=readout_time_5ns_multiple))
    #     sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))
    #
    #     return readout_time

    def parity_measurement(self, sequencer, qubit_id='A'):
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        self.idle_q(sequencer, qubit_id, time=-1/self.quantum_device_cfg['flux_pulse_info'][qubit_id]['chiby2pi_e']/4.0)
        self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=np.pi)
        sequencer.sync_channels_time(self.channels)

    def resonator_spectroscopy(self, sequencer):

        sequencer.new_sequence(self)
        self.readout_pxi(sequencer, self.on_qubits)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def resonator_spectroscopy_pi(self, sequencer):
        qubit_id = self.on_qubits[0]
        res_nb = self.quantum_device_cfg["qubit"][qubit_id]["id"]
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        if len(self.expt_cfg["pi_qb"])==0:
            self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        else:
            for qb in self.expt_cfg["pi_qb"]:
                pulse_info = self.lattice_cfg["pulse_info"]
                qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][res_nb]
                self.gen_q(sequencer, qubit_id=qubit_id, len=pulse_info[qubit_id]["pi_len"][qb],
                           amp=pulse_info[qubit_id]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                           pulse_type=pulse_info["pulse_type"][qb])
        self.idle_q(sequencer, time=self.expt_cfg["delay"])
        self.readout_pxi(sequencer, self.on_qubits)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def resonator_spectroscopy_ef_pi(self, sequencer):

        qubit_id = self.on_qubits[0]
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
        self.readout_pxi(sequencer, self.on_qubits)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def melting_single_readout_full_ramp(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        qb_freq_list = [self.lattice_cfg["qubit"]["freq"][i] for i in qb_list]
        lo_qb_temp_ind = np.argmax(qb_freq_list)
        lo_qb = qb_list[lo_qb_temp_ind]

        setup = self.on_qubits[0]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                pulse_info = self.lattice_cfg["pulse_info"]
                qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][lo_qb]
                self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                           pulse_type=pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()
        # ############################## PI CAL ###########################################
        # pulse_info = self.lattice_cfg["pulse_info"]
        # rd_qb = self.expt_cfg["rd_qb"]
        # if self.expt_cfg['pi_calibration']:
        #
        #     sequencer.new_sequence(self)
        #     self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        #     sequencer.sync_channels_time(self.channels)
        #     self.readout_pxi(sequencer, setup, overlap=False)
        #     sequencer.sync_channels_time(self.channels)
        #     sequencer.end_sequence()
        #
        #     rd_qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][self.expt_cfg["rd_qb"]] - self.lattice_cfg["qubit"]["freq"][lo_qb]
        #     sequencer.new_sequence(self)
        #     self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        #     self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][rd_qb],
        #                amp=pulse_info[setup]["pi_amp"][rd_qb], add_freq=rd_qb_iq_freq_dif, phase=0,
        #                pulse_type=pulse_info["pulse_type"][rd_qb])
        #     sequencer.sync_channels_time(self.channels)
        #     self.readout_pxi(sequencer, setup, overlap=False)
        #     sequencer.sync_channels_time(self.channels)
        #     sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pi_cal(self, sequencer):
        setup = self.on_qubits[0]
        pulse_info = self.pulse_info[setup]

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        sequencer.sync_channels_time(self.channels)
        self.readout_pxi(sequencer, setup, overlap=False)
        sequencer.sync_channels_time(self.channels)
        sequencer.end_sequence()

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        self.pi_q(sequencer, qubit_id=setup, phase=0, pulse_type=pulse_info["pulse_type"])
        sequencer.sync_channels_time(self.channels)
        self.readout_pxi(sequencer, setup, overlap=False)
        sequencer.sync_channels_time(self.channels)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def ff_sweep_j(self, sequencer, perc_flux_vec):
        qb_list = self.expt_cfg["Mott_qbs"]
        qb_freq_list = [self.lattice_cfg["qubit"]["freq"][i] for i in qb_list]
        lo_qb_temp_ind = np.argmax(qb_freq_list)
        lo_qb = qb_list[lo_qb_temp_ind]

        setup = self.on_qubits[0]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                pulse_info = self.lattice_cfg["pulse_info"]
                qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][lo_qb]
                self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                           pulse_type=pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = np.array(self.expt_cfg["ff_vec"]) * perc_flux_vec
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()
        ############################## PI CAL ###########################################
        pulse_info = self.lattice_cfg["pulse_info"]
        rd_qb = self.expt_cfg["rd_qb"]
        if self.expt_cfg['pi_calibration']:

            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

            rd_qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][self.expt_cfg["rd_qb"]] - self.lattice_cfg["qubit"]["freq"][lo_qb]
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][rd_qb],
                       amp=pulse_info[setup]["pi_amp"][rd_qb], add_freq=rd_qb_iq_freq_dif, phase=0,
                       pulse_type=pulse_info["pulse_type"][rd_qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def measure_lattice_state(self, sequencer, rd_qb):
        qb_list = self.expt_cfg["Mott_qbs"]
        qb_freq_list = [self.lattice_cfg["qubit"]["freq"][i] for i in qb_list]
        lo_qb_temp_ind = np.argmax(qb_freq_list)
        lo_qb = qb_list[lo_qb_temp_ind]

        setup = self.on_qubits[0]
        evolution_t = self.expt_cfg["evolution_t"]

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

        ##################################GENERATE PI PULSES ################################################
        for i, qb in enumerate(qb_list):
            pulse_info = self.lattice_cfg["pulse_info"]
            qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][lo_qb]
            self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                       pulse_type=pulse_info["pulse_type"][qb])
            self.idle_q(sequencer, time=20)

        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=100)
        ##############################GENERATE RAMP###########################################
        flux_vec = self.expt_cfg["ff_vec"]
        self.ff_pulse(sequencer, [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
        ############################## readout ###########################################
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
        sequencer.sync_channels_time(self.channels)
        self.readout_pxi(sequencer, setup, overlap=False)
        sequencer.sync_channels_time(self.channels)

        ############################## generate compensation ###########################################
        flux_vec = self.expt_cfg["ff_vec"]
        self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
        sequencer.end_sequence()
        ############################## PI CAL ###########################################
        pulse_info = self.lattice_cfg["pulse_info"]
        if self.expt_cfg['pi_calibration']:

            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

            rd_qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][rd_qb] - self.lattice_cfg["qubit"]["freq"][lo_qb]
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            self.gen_q(sequencer, qubit_id=setup, len=pulse_info[setup]["pi_len"][rd_qb],
                       amp=pulse_info[setup]["pi_amp"][rd_qb], add_freq=rd_qb_iq_freq_dif, phase=0,
                       pulse_type=pulse_info["pulse_type"][rd_qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, setup, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            for qubit_id in self.on_qubits:
                self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['pulse_length'],
                           amp=self.expt_cfg['amp'], add_freq=dfreq, phase=0,
                           pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=self.expt_cfg['delay'])
            self.readout_pxi(sequencer, self.on_qubits,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq_gauss(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            for qubit_id in self.on_qubits:
                sequencer.append('charge%s_I' % qubit_id,
                                 Gauss(max_amp=self.expt_cfg['amp'], sigma_len=self.expt_cfg['pulse_length'],
                                        cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=0))
                sequencer.append('charge%s_Q' % qubit_id,
                                 Gauss(max_amp=self.expt_cfg['amp'], sigma_len=self.expt_cfg['pulse_length'],
                                       cutoff_sigma=2, freq=self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=self.pulse_info[qubit_id]['Q_phase']))
                self.idle_q(sequencer, time=self.expt_cfg['delay'])
            self.readout_pxi(sequencer, self.on_qubits,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_resonator_spectroscopy(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
        if self.expt_cfg["pulse_inside_flux"]:
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time - pre_flux_time +self.lattice_cfg['ff_info']['ff_pulse_padding']] * 8
            else:
                ff_len = np.asarray(
                    np.array(self.lattice_cfg['ff_info']["ff_len"]) + self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        else:
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = np.asarray(
                    np.array(self.lattice_cfg['ff_info']["ff_len"])) + self.lattice_cfg['ff_info']['ff_pulse_padding']
            else:
                ff_len = self.expt_cfg["ff_len"]

            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_resonator_spectroscopy_pi(self, sequencer):

        qubit_id = self.on_qubits[0]
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

        if self.expt_cfg["pulse_inside_flux"]:
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            res_nb = self.quantum_device_cfg["qubit"][qubit_id]["id"]
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            if len(self.expt_cfg["pi_qb"]) == 0:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            else:
                for qb in self.expt_cfg["pi_qb"]:
                    pulse_info = self.lattice_cfg["pulse_info"]
                    qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][res_nb]
                    self.gen_q(sequencer, qubit_id=qubit_id, len=pulse_info[qubit_id]["pi_len"][qb],
                               amp=pulse_info[qubit_id]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                               pulse_type=pulse_info["pulse_type"][qb])
            self.idle_q(sequencer, time=self.expt_cfg["delay"])
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time - pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']] * 8
            else:
                ff_len = np.asarray(
                    self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)
        else:
            sequencer.sync_channels_time(self.channels)
            qubit_id = self.on_qubits[0]
            res_nb = self.quantum_device_cfg["qubit"][qubit_id]["id"]
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            if len(self.expt_cfg["pi_qb"]) == 0:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            else:
                for qb in self.expt_cfg["pi_qb"]:
                    pulse_info = self.lattice_cfg["pulse_info"]
                    qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][res_nb]
                    self.gen_q(sequencer, qubit_id=qubit_id, len=pulse_info[qubit_id]["pi_len"][qb],
                               amp=pulse_info[qubit_id]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
                               pulse_type=pulse_info["pulse_type"][qb])
            self.idle_q(sequencer, time=self.expt_cfg["delay"])
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = np.asarray(
                    self.lattice_cfg['ff_info']["ff_len"]) + self.lattice_cfg['ff_info']['ff_pulse_padding']
            else:
                ff_len = self.expt_cfg["ff_len"]

            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def ff_pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            #add IQ pulse
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
            for qubit_id in self.on_qubits:
                self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
            self.idle_q(sequencer, time=self.expt_cfg['delay'])

            #synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

            #add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time-pre_flux_time]*8
            else:
                ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
            self.ff_pulse(sequencer, ff_len=ff_len,flux_vec =self.expt_cfg['ff_vec'], pulse_type= self.expt_cfg['ff_pulse_type'])
            self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'], pulse_type=self.expt_cfg['ff_pulse_type'],
                          flip_amp=True)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramp_cal_ppiq(self, sequencer):
        delt = self.expt_cfg['delt']
        pulse_type = self.expt_cfg['pulse_type']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            if delt<0:
                #ppiq first if dt less than zero
                for qubit_id in self.on_qubits:
                    self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'],
                               amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

                # wait time dt, the apply flux pulse
                for qb in range(8):
                    sequencer.append('ff_Q%s' % qb, Idle(time=-delt))

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = 8 * [
                        self.expt_cfg['qb_pulse_length'] + delt + self.quantum_device_cfg['readout'][qubit_id[0]][
                            'length']]
                else:
                    ff_len = self.lattice_cfg['ff_info']["ff_len"]

                # add flux pulse
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

            else:
                #flux pulse first if dt greater than zero
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = 8*[self.expt_cfg['qb_pulse_length']+delt+self.quantum_device_cfg['readout'][self.on_qubits[0]][
                        'length']]
                else:
                    ff_len = self.lattice_cfg['ff_info']["ff_len"]

                #add flux pulse
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

                #wait time dt, the apply ppiq
                self.idle_q(sequencer, time=delt)
                for qubit_id in self.on_qubits:
                    self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info[qubit_id]['pulse_type'])

            #synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

            #add compensation flux pulse
            fluxch = [ch for ch in self.channels if 'ff' in ch]
            sequencer.sync_channels_time(fluxch + ['readoutA'])
            if self.quantum_device_cfg["ff_info"]["ff_comp_sym"]:
                self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramp_down_cal_ppiq(self, sequencer):
        delt = self.expt_cfg['delt']
        pulse_type = self.expt_cfg['pulse_type']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            # add fast flux pulse
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = 8 * [
                    self.expt_cfg['qb_pulse_length'] + delt + self.quantum_device_cfg['readout'][qubit_id[0]][
                        'length']]
            else:
                ff_len = self.lattice_cfg['ff_info']["ff_len"]

            #add ppiq and readout
            if delt<0:
                for qubit_id in self.on_qubits:
                    self.idle_q(sequencer=sequencer, qubit_id=qubit_id, time=ff_len[0]+delt)
                    self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'],
                               amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type='square')
            else:
                for qubit_id in self.on_qubits:
                    self.idle_q(sequencer=sequencer, qubit_id=qubit_id, time=delt)
                    self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'],
                               amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type='square')

            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=self.channels)

            #add compensation flux pulse
            if self.quantum_device_cfg["ff_info"]["ff_comp_sym"]:
                self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_track_traj(self, sequencer, perc_flux_vec):
        ramp_type = self.expt_cfg['ramp_type']
        flux_vec = np.array(self.expt_cfg["ff_vec"]) * perc_flux_vec
        for dfreq in np.arange(self.expt_cfg['qbf_start'], self.expt_cfg['qbf_stop'], self.expt_cfg['qbf_step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            #flux pulse first if dt greater than zero
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = 8*[self.expt_cfg['qb_pulse_length']+self.expt_cfg["delay_post_ramp"]+self.quantum_device_cfg['readout'][self.on_qubits[0]][
                    'length']]
            else:
                ff_len = self.lattice_cfg['ff_info']["ff_len"]

            #add flux pulse
            area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= ramp_type, flux_vec=flux_vec, flip_amp=False)

            #wait time dt, the apply ppiq
            self.idle_q(sequencer, time=self.expt_cfg["delay_post_ramp"])
            for qubit_id in self.on_qubits:
                self.gen_q(sequencer=sequencer, qubit_id=qubit_id, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type='square')

            #synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

            #add compensation flux pulse
            fluxch = [ch for ch in self.channels if 'ff' in ch]
            sequencer.sync_channels_time(fluxch + ['readoutA'])
            if self.quantum_device_cfg["ff_info"]["ff_comp_sym"]:
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= ramp_type, flux_vec=flux_vec, flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            if self.expt_cfg["Qpulse_in_flux"]:
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                # add rabi pulse
                for qubit_id in self.on_qubits:
                    self.gen_q(sequencer, qubit_id, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

            else:
                # add rabi pulse
                for qubit_id in self.on_qubits:
                    self.gen_q(sequencer, qubit_id, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])

                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"], flux_vec= self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_qubits, overlap=False, synch_channels=self.channels)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)


            #end sequence
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_pulse_padding'])

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #self.gen_q(sequencer,qubit_id,len=2000,amp=1,phase=0,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            # synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False,
                             synch_channels=channels_excluding_fluxch)

            #add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time-pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']]*8
            else:
                ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"]+ self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_pulse_padding'])

            for qubit_id in self.on_qubits:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            # synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_qubits, overlap=False,
                             synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time-pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']]*8
            else:
                ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"]+ self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fast_flux_flux_fact_finding_mission(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            #add flux to pulse
            for target in self.expt_cfg['target_qb']:
                #since we haven't been synching the flux channels, this should still be back at the beginning
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=self.expt_cfg['ff_amp'], flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            #COMPENSATION PULSE
            for target in self.expt_cfg['target_qb']:
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=-self.expt_cfg['ff_amp'],
                                        flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            sequencer.sync_channels_time(self.channels)
            for qubit_id in self.on_qubits:
                sequencer.append('charge%s_I' % qubit_id,
                                 Square(max_amp=self.expt_cfg['qb_amp'], flat_len=self.expt_cfg['qb_pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2,
                                        freq=self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=0))

                sequencer.append('charge%s_Q' % qubit_id,
                                 Square(max_amp=self.expt_cfg['qb_amp'], flat_len=self.expt_cfg['qb_pulse_length'],
                                        ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[qubit_id]['iq_freq'] + dfreq,
                                        phase=self.pulse_info[qubit_id]['Q_phase']))
                self.idle_q(sequencer, time=self.expt_cfg['delay'])

            #add readout and readout trig (ie dig) pulse
            self.readout_pxi(sequencer, self.on_qubits,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fast_flux_pulse(self, sequencer):
        for i in range(self.expt_cfg['nb_reps']):
            sequencer.new_sequence(self)

            #add flux to pulse
            for target in self.expt_cfg['target_qb']:
                #since we haven't been synching the flux channels, this should still be back at the beginning
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=self.expt_cfg['ff_amp'], flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            #COMPENSATION PULSE
            for target in self.expt_cfg['target_qb']:
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=-self.expt_cfg['ff_amp'],
                                        flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            for qubit_id in self.on_qubits:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi_chevron(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['len_start'], self.expt_cfg['len_stop'], self.expt_cfg['len_step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            for qubit_id in self.on_qubits:
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #self.gen_q(sequencer,qubit_id,len=2000,amp=1,phase=0,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
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

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def t1rho(self,sequencer):

        for t1rho_len in np.arange(self.expt_cfg['start'],self.expt_cfg['stop'],self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                #self.idle_q(sequencer, time= self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['half_pi_len'])
                self.gen_q(sequencer, qubit_id, len=t1rho_len, amp=self.expt_cfg['amp'], phase=np.pi/2,
                           pulse_type=self.expt_cfg['pulse_type'])
                #self.idle_q(sequencer, time=self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[qubit_id]['half_pi_len'])
                self.half_pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'],
                               phase=0)


                self.readout_pxi(sequencer, self.on_qubits)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_ef_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'],amp = self.expt_cfg['amp'] ,phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+self.quantum_device_cfg['qubit']['A']['anharmonicity'])
            self.readout_pxi(sequencer, self.on_qubits,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            for qubit_id in self.on_qubits:
                if self.expt_cfg['ge_pi']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.gen_q(sequencer,qubit_id,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],add_freq = self.quantum_device_cfg['qubit'][qubit_id]['anharmonicity'] )
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_t1(self, sequencer):
        # t1 for the e and f level

        for ef_t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:

                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ef_t1_len)

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.idle_q(sequencer, time=ramsey_len)
                self.half_pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
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

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_fh_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.gen_q(sequencer, qubit_id, len=self.expt_cfg['pulse_length'], phase=0,pulse_type=self.expt_cfg['pulse_type'],add_freq=dfreq+2*self.quantum_device_cfg['qubit']['A']['anharmonicity'])
            self.readout_pxi(sequencer, self.on_qubits,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            for qubit_id in self.on_qubits:
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

            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fh_ramsey(self, sequencer):


        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)

            for qubit_id in self.on_qubits:
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


            self.readout_pxi(sequencer, self.on_qubits)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


        return sequencer.complete(self, plot=True)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            for qubit_id in self.on_qubits:
                
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=qubit_id, time=500)
                self.readout_pxi(sequencer,qubit_id)
                sequencer.end_sequence()
                
                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
                for qubit_id in self.on_qubits:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.readout_pxi(sequencer, qubit_id)
                sequencer.end_sequence()
                
                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=self.on_qubits, time=500)
                for qubit_id in self.on_qubits:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.pi_q_ef(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['ef_pulse_type'])
                self.readout_pxi(sequencer, qubit_id)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def qp_pumping_t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=self.on_qubits,time=500)

            N_pump = self.expt_cfg['N_pump']
            pump_wait = self.expt_cfg['pump_wait']

            for pump in range(N_pump):
                for qubit_id in self.on_qubits:
                    self.pi_q(sequencer, qubit_id, pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                    self.idle_q(sequencer, time=pump_wait)

            for qubit_id in self.on_qubits:
                self.pi_q(sequencer,qubit_id,pulse_type=self.pulse_info[qubit_id]['pulse_type'])
                self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.on_qubits)
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