from slab.experiments.PulseExperiments_M8195A_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_M8195A_PXI.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_M8195A_PXI import keysight_pxi_load_m8195a as ks_pxi
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin
try:from skopt import Optimizer
except:print("No optimizer")
from slab.experiments.PulseExperiments_M8195A_PXI.PostExperimentAnalysis import PostExperiment
from slab.instruments import InstrumentManager
import time
from h5py import File
from scipy import interpolate


# from slab.experiments.PulseExperiments.get_data import get_singleshot_data_two_qubits_4_calibration_v2,\
#     get_singleshot_data_two_qubits, data_to_correlators, two_qubit_quantum_state_tomography,\
#     density_matrix_maximum_likelihood

import pickle

class SequentialExperiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, path,
                 analyze = False,show=True,P = 'Q', return_val=False, hvi=False):

        self.Is = []
        self.Qs = []
        self.show = show
        self.hvi = hvi

        eval('self.' + experiment_name)(quantum_device_cfg, experiment_cfg, hardware_cfg,path)
        if analyze:
            try:
                if return_val:
                    return self.analyze(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,self.show,self.Is,self.Qs,P = 'I', return_val=True)
                else:
                    self.analyze(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, self.show, self.Is, self.Qs, P='I')
            except: print ("No post expt analysis")
        else:pass

        # if experiment_name in ['adc_calibration']:
        #     im = InstrumentManager()

    def histogram_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        expt_cfg = experiment_cfg['histogram_sweep']
        sweep_amp = expt_cfg['sweep_amp']
        attens = np.arange(expt_cfg['atten_start'],expt_cfg['atten_stop'],expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'],expt_cfg['freq_stop'],expt_cfg['freq_step'])

        experiment_name = 'histogram'
        
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram_sweep', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        ii = 0
        if sweep_amp:
            for att in attens:
                quantum_device_cfg['readout']['dig_atten'] = att
                print ("Expt num = ",ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii+=1
        else:
            for freq in freqs:
                quantum_device_cfg['readout']['freq'] = freq
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii+=1

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def histogram_amp_and_freq_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['histogram_amp_and_freq_sweep']
        attens = np.arange(expt_cfg['atten_start'],expt_cfg['atten_stop'],expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'],expt_cfg['freq_stop'],expt_cfg['freq_step'])

        experiment_name = 'histogram'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram_amp_and_freq_sweep', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        ii = 0
        for freq in freqs:
            for att in attens:
                quantum_device_cfg['readout']['dig_atten'] = att
                print ("Attenuation = ",att,"dB")
                quantum_device_cfg['readout']['freq'] = freq
                print ("Frequency = ",freq,"GHz")
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii+=1

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def qp_pumping_t1_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['qp_pumping_t1_sweep']
        sweep_N_pump = np.arange(expt_cfg['N_pump_start'], expt_cfg['N_pump_stop'], expt_cfg['N_pump_step'])
        sweep_pump_wait = np.arange(expt_cfg['pump_wait_start'], expt_cfg['pump_wait_stop'], expt_cfg['pump_wait_step'])

        experiment_name = 'qp_pumping_t1'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qp_pumping_t1_sweep', suffix='.h5'))

        for N_pump in sweep_N_pump:
            experiment_cfg['qp_pumping_t1']['N_pump'] = int(N_pump)
            print ("Number of pi pulses: " + str(N_pump) )
            for pump_wait in sweep_pump_wait:
                experiment_cfg['qp_pumping_t1']['pump_wait'] = int(pump_wait)
                print("pi pulse delay: " + str(pump_wait))
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def resonator_spectroscopy(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

        for freq in np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step']):
            quantum_device_cfg['readout']['freq'] = freq
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def resonator_spectroscopy_weak_qubit_drive(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy_weak_qubit_drive'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_weak_qubit_drive', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

        for freq in np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step']):
            quantum_device_cfg['readout']['freq'] = freq
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def resonator_spectroscopy_power_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        sweep_expt_name = 'resonator_spectroscopy_power_sweep'
        swp_cfg = experiment_cfg[sweep_expt_name]

        experiment_name = 'resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, sweep_expt_name, suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

        for atten in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            print("Attenuation set to ", atten, 'dB')
            quantum_device_cfg['readout']['dig_atten'] = atten
            Is_t = []
            Qs_t = []
            for freq in np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step']):
                quantum_device_cfg['readout']['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                Is_t.append(I)
                Qs_t.append(Q)

            self.Is.append(np.array(Is_t))
            self.Qs.append(np.array(Qs_t))

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qubit_temperature', suffix='.h5'))

        for ge_pi in [True,False]:
            experiment_cfg['ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['ef_rabi']['acquisition_num'] = experiment_cfg['ef_rabi']['avgs_without_pi']
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def coherence_and_qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'coherence_and_qubit_temperature', suffix='.h5'))

        experiment_name = 't1'
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
        I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
        self.Is.append(I)
        self.Qs.append(Q)

        experiment_name = 'ramsey'
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
        I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
        self.Is.append(I)
        self.Qs.append(Q)

        experiment_name = 'echo'
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
        I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
        self.Is.append(I)
        self.Qs.append(Q)


        experiment_name = 'ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]


        for ge_pi in [True,False]:

            experiment_cfg['ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['ef_rabi']['acquisition_num'] = 10000
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_reset_qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'sideband_transmon_reset'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_reset_qubit_temperature', suffix='.h5'))

        for ge_pi in [True,False]:

            experiment_cfg['sideband_transmon_reset']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['sideband_transmon_reset']['acquisition_num'] = 50000
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_reset_qubit_temperature_wait_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        experiment_name = 'sideband_transmon_reset'
        expt_cfg = experiment_cfg[experiment_name]
        swp_cfg = experiment_cfg['sideband_reset_qubit_temperature_wait_sweep']
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,get_next_filename(data_path, 'sideband_reset_qubit_temperature_wait_sweep', suffix='.h5'))

        wait_times = np.arange(swp_cfg['wait_start'], swp_cfg['wait_stop'], swp_cfg['wait_step'])

        for wait in wait_times:

            for ge_pi in [True, False]:

                experiment_cfg['sideband_transmon_reset']['ge_pi'] = ge_pi
                if ge_pi:
                    pass
                else:
                    experiment_cfg['sideband_transmon_reset']['acquisition_num'] = 2000
                experiment_cfg['sideband_transmon_reset']['wait_after_reset'] = wait
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def two_cavity_swap_qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'two_cavity_swap_qubit_temperature',
                                                                  suffix='.h5'))
        for ge_pi in [True,False]:
            experiment_cfg['two_cavity_swap_ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['two_cavity_swap_ef_rabi']['acquisition_num'] = experiment_cfg['two_cavity_swap_ef_rabi']['avgs_without_pi']
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def two_cavity_swap_before_qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_before_ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'two_cavity_swap_before_qubit_temperature',
                                                                  suffix='.h5'))
        for ge_pi in [True,False]:
            experiment_cfg['two_cavity_swap_before_ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['two_cavity_swap_before_ef_rabi']['acquisition_num'] = experiment_cfg[
                'two_cavity_swap_before_ef_rabi']['avgs_without_pi']
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_drive_qubit_after(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_drive_qubit_after'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_two_cavity_swap_drive_qubit_after',
                                                                  suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_two_cavity_swap_drive_qubit_after']
        
        for time_len in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            expt_cfg['qub_drive_after_len'] = time_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        

    def sequential_pulse_probe_ef_iq(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        experiment_name = 'pulse_probe_ef_iq'
        expt_cfg = experiment_cfg[experiment_name]
        swp_cfg = experiment_cfg['sequential_pulse_probe_ef_iq']
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,get_next_filename(data_path, 'sequential_pulse_probe_ef_iq', suffix='.h5'))

        alpha_centers = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for alpha in alpha_centers:
            quantum_device_cfg['qubit'][expt_cfg['on_qubits'][0]]['anharmonicity'] = alpha
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_rabi_freq_scan_length_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        expt_cfg = experiment_cfg['sideband_rabi_freq_scan_length_sweep']
        lengths = np.arange(expt_cfg['start'],expt_cfg['stop'],expt_cfg['step'])

        experiment_name = 'sideband_rabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_freq_scan_length_sweep', suffix='.h5'))

        for length in lengths:

            experiment_cfg[experiment_name]['length'] = length
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_rabi_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sideband_rabi_sweep']
        freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'sideband_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_sweep', suffix='.h5'))

        for freq in freqs:
            print("Sideband frequency set to", freq, "GHz")

            experiment_cfg[experiment_name]['freq'] = freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_rabi_freq_scan_amp_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sideband_rabi_freq_scan_amp_sweep']
        amps = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        length_sc = swp_cfg['length_scale']
        amp_sc = swp_cfg['amp_scale']


        experiment_name = 'sideband_rabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_freq_scan_amp_sweep', suffix='.h5'))

        for amp in amps:
            experiment_cfg[experiment_name]['length'] = length_sc*amp_sc/amp
            experiment_cfg[experiment_name]['amp'] = amp
            print("Sideband amplitude set to", amp)
            print("Sideband length set to", length_sc*amp_sc/amp,'ns')
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_pulse_probe_iq_amp_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['cavity_drive_pulse_probe_iq_amp_sweep']
        amps = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'cavity_drive_pulse_probe_iq'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_pulse_probe_iq_amp_sweep', suffix='.h5'))

        for amp in amps:
            print("Cavity drive amplitude set to ", amp)

            experiment_cfg[experiment_name]['cavity_amp'] = amp
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_test_phase_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_test_phase_sweep']
        phases = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'wigner_tomography_test_cavity_drive_sideband'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'wigner_tomography_test_phase_sweep', suffix='.h5'))

        for phase in phases:
            print("Cavity drive phase set to", phase,"rad")

            experiment_cfg[experiment_name]['phase'] = phase
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_sideband_only_phase_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_sideband_only_phase_sweep']
        phases = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'wigner_tomography_test_sideband_only'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'wigner_tomography_sideband_only_phase_sweep', suffix='.h5'))

        for phase in phases:
            print("Cavity drive phase set to", phase,"rad")

            experiment_cfg[experiment_name]['phase'] = phase
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_sideband_one_pulse_phase_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_sideband_one_pulse_phase_sweep']
        phases = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'wigner_tomography_test_sideband_one_pulse'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'wigner_tomography_sideband_one_pulse_phase_sweep', suffix='.h5'))

        for phase in phases:
            print("Cavity drive phase set to", phase,"rad")

            experiment_cfg[experiment_name]['phase'] = phase
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_alltek2_phase_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_alltek2_phase_sweep']
        phases = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'wigner_tomography_test_sideband_alltek2'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'wigner_tomography_alltek2_phase_sweep', suffix='.h5'))

        for phase in phases:
            print("Cavity drive phase set to", phase,"rad")

            experiment_cfg[experiment_name]['phase'] = phase
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_2d_offset_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_2d_offset_sweep']
        xoffsets = np.arange(swp_cfg['startx'], swp_cfg['stopx'], swp_cfg['stepx'])
        yoffsets = np.arange(swp_cfg['starty'], swp_cfg['stopy'], swp_cfg['stepy'])

        experiment_name = "wigner_tomography_2d_sweep"

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'wigner_tomography_2d_offset_sweep', suffix='.h5'))

        for offset_y in yoffsets:
            for offset_x in xoffsets:
                print("xoffset = ", offset_x)
                print("yoffset = ", offset_y)

                experiment_cfg[experiment_name]['offset_x'] = offset_x
                experiment_cfg[experiment_name]['offset_y'] = offset_y
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_ramsey(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_ramsey']
        stops = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])[1:]

        experiment_name = 'sideband_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_ramsey', suffix='.h5'))

        for ii in range(len(stops)):
            experiment_cfg[experiment_name]['stop'] = stops[ii]
            if ii is 0:experiment_cfg[experiment_name]['start'] = swp_cfg['start']
            else:experiment_cfg[experiment_name]['start'] = stops[ii-1]

            print ("Sideband Ramsey start,stop,step = ",experiment_cfg[experiment_name]['start'],experiment_cfg[experiment_name]['stop'],experiment_cfg[experiment_name]['step'])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_fnm1gnrabi_freq_scan_varyn(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sideband_fnm1gnrabi_freq_scan_varyn']
        ns = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'sideband_fnm1gnrabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_fnm1gnrabi_freq_scan_varyn', suffix='.h5'))

        for ii,nn in enumerate(ns):

            experiment_cfg[experiment_name]['n'] = int(nn)
            if swp_cfg['pulse_params_from_quantum_device_cfg']:
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['length'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_lens'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][expt_cfg['mode_index']][ii]

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sideband_fnm1gnrabi_varyn(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sideband_fnm1gnrabi_varyn']
        ns = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'sideband_fnm1gnrabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_fnm1gnrabi_varyn', suffix='.h5'))

        for ii,nn in enumerate(ns):

            experiment_cfg[experiment_name]['n'] = int(nn)
            if swp_cfg['pulse_params_from_quantum_device_cfg']:
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][expt_cfg['mode_index']][ii]

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_rabi_freq_scan(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_rabi_freq_scan']
        amplist = swp_cfg['amplist']
        freqlist = swp_cfg['freqlist']
        lengthlist = swp_cfg['lengthlist']

        experiment_name = 'sideband_rabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_rabi_freq_scan', suffix='.h5'))

        for ii,freq in enumerate(freqlist):
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)

            experiment_cfg[experiment_name]['freq'] = freq
            experiment_cfg[experiment_name]['amp'] = amplist[ii]
            experiment_cfg[experiment_name]['length'] = lengthlist[ii]
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_rabis(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_rabis']
        amplist = swp_cfg['amplist']
        freqlist = swp_cfg['freqlist']
        stoplist  = swp_cfg['stoplist']

        experiment_name = 'sideband_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_rabis', suffix='.h5'))

        for ii,freq in enumerate(freqlist):
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            experiment_cfg[experiment_name]['freq'] = freq
            experiment_cfg[experiment_name]['amp'] = amplist[ii]
            experiment_cfg[experiment_name]['stop'] = stoplist[ii]
            experiment_cfg[experiment_name]['step'] = stoplist[ii]/100.0

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_t1(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_t1']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_t1'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_t1', suffix='.h5'))

        for ii,mode in enumerate(modelist):

            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_ramsey_overmodes(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_ramsey_overmodes']
        modelist = swp_cfg['modelist']

        if swp_cfg['f0g1']:
            experiment_name = 'sideband_ramsey'
        elif swp_cfg['e0g2']:
            experiment_name = 'sideband_e0g2_ramsey'
        elif swp_cfg['h0e1']:
            experiment_name = 'sideband_h0e1_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_ramsey_overmodes', suffix='.h5'))

        for ii,mode in enumerate(modelist):
            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_pi_pi_offset(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_pi_pi_offset']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_pi_pi_offset'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_sideband_pi_pi_offset', suffix='.h5'))

        for ii, mode in enumerate(modelist):
            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_chi_ge_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_ge_calibration']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_chi_ge_calibration'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_sideband_chi_ge_calibration', suffix='.h5'))

        for ii, mode in enumerate(modelist):
            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_chi_gf_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_gf_calibration']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_chi_gf_calibration'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_sideband_chi_gf_calibration',
                                                       suffix='.h5'))
        for ii, mode in enumerate(modelist):
            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_sideband_chi_ef_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_ef_calibration']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_chi_ef_calibration'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_sideband_chi_ef_calibration',
                                                       suffix='.h5'))

        for ii, mode in enumerate(modelist):
            experiment_cfg[experiment_name]['mode_index'] = int(mode)
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_chi_dressing_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_chi_dressing_calibration']

        varlist =  np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])


        experiment_name = 'sideband_chi_dressing_calibration'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_chi_dressing_calibration',
                                                       suffix='.h5'))

        for ii, x in enumerate(varlist):
            if swp_cfg['sweep_detuning']:experiment_cfg[experiment_name]['detuning'] = x
            elif  swp_cfg['sweep_amp']: experiment_cfg[experiment_name]['amp'] = x
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_qubit_calibration(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        pass

    def adc_calibration(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'adc_calibration', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        im = InstrumentManager()
        swp_cfg = experiment_cfg['adc_calibration']
        yoko = im['YOKO1']
        print ("Connected to YOKO1:",str(yoko.get_id()))
        yoko.set_output(True)
        yoko.set_voltage_limit(10)

        for voltage in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):

            yoko.set_volt(voltage)
            print ("YOKO voltage set to: ",voltage)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def test_xlnk_awg(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name='rabi'
        experiment_cfg['rabi']['start'] = 250.0
        experiment_cfg['rabi']['stop'] = 251.0
        experiment_cfg['rabi']['step'] = 1.0
        experiment_cfg['rabi']['use_pi_calibration'] = False

        hardware_cfg['channels_delay']['readout'] = -2400.0
        hardware_cfg['channels_delay']['readout_trig'] = -2500.0
        hardware_cfg['channels_delay']['charge1_I'] = 0.0
        hardware_cfg['channels_delay']['charge1_Q'] = 0.0

        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'rabi_xilinx_test', suffix='.h5'))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        
        im = InstrumentManager()
        im['RFsoc'].generate_pulses(num_pulses=101, iqfreq=200)
        
        for jj in range(101):
            im['RFsoc'].load_rabi_pulse(jj)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def test_xlnk_awg2(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'ramsey'
        experiment_cfg['ramsey']['start'] = 800.0
        experiment_cfg['ramsey']['stop'] = 801.0
        experiment_cfg['ramsey']['step'] = 1.0
        experiment_cfg['ramsey']['use_pi_calibration'] = False

        hardware_cfg['channels_delay']['readout'] = -2400.0
        hardware_cfg['channels_delay']['readout_trig'] = -2500.0
        hardware_cfg['channels_delay']['charge1_I'] = 0.0
        hardware_cfg['channels_delay']['charge1_Q'] = 0.0

        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'ramsey_xilinx_test', suffix='.h5'))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)

        im = InstrumentManager()
        im['RFsoc'].generate_pulses(num_pulses=101, iqfreq=200)

        for jj in range(101):
            im['RFsoc'].load_ramsey_pulse(jj)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def analyze(self,quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,Is,Qs,P='Q', return_val=False):
        PA = PostExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, Is,Qs,P,show)
        return PA

    def sequential_photon_number_resolved_qubit_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_photon_number_resolved_qubit_spectroscopy']
        amplist = np.arange(swp_cfg['start_amp'], swp_cfg['stop_amp'], swp_cfg['amp_step'])
        phaselist = np.arange(swp_cfg['start_phase'], swp_cfg['stop_phase'], swp_cfg['phase_step'])

        experiment_name = 'photon_number_resolved_qubit_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_photon_number_resolved_qubit_spectroscopy', suffix='.h5'))

        if swp_cfg['sweep_phase']:
            for phase in phaselist:
                experiment_cfg[experiment_name]['snap_phase'] = phase
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        elif swp_cfg['sweep_mode_index']:
            for ii in range(len(swp_cfg['mode_list'])):
                experiment_cfg[experiment_name]['mode_index'] = swp_cfg['mode_list'][ii]
                experiment_cfg[experiment_name]['prep_cav_len'] = swp_cfg['len_list'][ii]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for amp in amplist:
                experiment_cfg[experiment_name]['prep_cav_amp'] = amp
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_quadratic_dispersive_shift_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_quadratic_dispersive_shift_calibration']
        nlist = np.arange(swp_cfg['N_min'], swp_cfg['N_max'],swp_cfg['N_step'])
        sigmalist = np.sqrt(nlist)/swp_cfg['cavity_drive_cal']*swp_cfg['cal_cavity_pulse_len']
        nu_q = quantum_device_cfg['qubit']['1']['freq']
        experiment_name = 'photon_number_resolved_qubit_spectroscopy'
        span = swp_cfg['span']
        step = swp_cfg['step']
        expt_cfg = experiment_cfg[experiment_name]
        expt_cfg['state']='alpha'
        chi = quantum_device_cfg["flux_pulse_info"]['1']['chiby2pi_e'][expt_cfg['mode_index']]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_quadratic_dispersive_shift_calibration', suffix='.h5'))

        experiment_cfg[experiment_name]['amp'] = swp_cfg['amp']
        experiment_cfg[experiment_name]['cavity_pulse_type'] = swp_cfg['cavity_pulse_type']
        for ii,n in enumerate(nlist):
            expt_cfg['start'] = -span/2.0
            expt_cfg['stop'] = span/2.0
            expt_cfg['step'] = step
            expt_cfg['add_freq'] = 2*chi*n
            expt_cfg['prep_cav_len'] = sigmalist[ii]
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_photon_number_resolved_qubit_spectroscopy_over_states(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_photon_number_resolved_qubit_spectroscopy_over_states']
        statelist = swp_cfg['states_list']

        experiment_name = 'photon_number_resolved_qubit_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_photon_number_resolved_qubit_spectroscopy_over_states', suffix='.h5'))

        for state in statelist:
            experiment_cfg[experiment_name]['state'] = state
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_photon_number_distribution_measurement(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_photon_number_distribution_measurement']

        experiment_name = 'photon_number_distribution_measurement'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_photon_number_distribution_measurement', suffix='.h5'))

        if swp_cfg['sweep_phase']:
            phaselist = np.arange(swp_cfg['start_phase'], swp_cfg['stop_phase'], swp_cfg['phase_step'])
            for phase in phaselist:
                experiment_cfg[experiment_name]['snap_phase'] = phase
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        elif swp_cfg['sweep_prep_cav_len']:
            lenlist = np.arange(swp_cfg['start_len'], swp_cfg['stop_len'], swp_cfg['len_step'])
            for cav_len in lenlist:
                expt_cfg['prep_cav_len'] = float(cav_len)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            amplist = np.arange(swp_cfg['start_amp'], swp_cfg['stop_amp'], swp_cfg['amp_step'])
            for amp in amplist:
                experiment_cfg[experiment_name]['prep_cav_amp'] = amp
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
    
    def cavity_transfer_calibration_pnds(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['cavity_transfer_calibration_pnds']

        experiment_name = 'photon_number_distribution_measurement'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'cavity_transfer_calibration_pnds', suffix='.h5'))

        expt_cfg['cavity_pulse_type'] = swp_cfg['pulse_type']
        expt_cfg['prep_cav_amp'] = swp_cfg['prep_cav_amp']
        expt_cfg['prep_cav_len'] = swp_cfg['prep_cav_len']
        expt_cfg['state'] = "alpha"

        freqlist = np.arange(swp_cfg['start_freq'], swp_cfg['stop_freq'], swp_cfg['step_freq']) + \
                   quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][
                       expt_cfg['mode_index']]

        for freq in freqlist:
            quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][expt_cfg['mode_index']] = freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

    def sequential_optimal_control_cavity_transfer_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_cavity_transfer_calibration']

        experiment_name = 'optimal_control_test_1step'
        
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_optimal_control_cavity_transfer_calibration', suffix='.h5'))

        freqlist = np.arange(swp_cfg['start_freq'], swp_cfg['stop_freq'], swp_cfg['step_freq']) + \
                   quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][
                       expt_cfg['mode_index']]

        for freq in freqlist:
            quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][expt_cfg['mode_index']] = freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

    def sequential_optimal_control_cavity_transfer_calibration_blockade(self, quantum_device_cfg, experiment_cfg, hardware_cfg,path):
        swp_cfg = experiment_cfg['sequential_optimal_control_cavity_transfer_calibration_blockade']

        experiment_name = 'optimal_control_test_with_blockade_1step'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_optimal_control_cavity_transfer_calibration_blockade',
                                                                  suffix='.h5'))

        freqlist = np.arange(swp_cfg['start_freq'], swp_cfg['stop_freq'], swp_cfg['step_freq']) + \
                   quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][
                       expt_cfg['mode_index']]

        for freq in freqlist:
            quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['cavity_freqs'][
                expt_cfg['mode_index']] = freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

    def sequential_photon_number_distribution_measurement_beta_cal(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_photon_number_distribution_measurement_beta_cal']
        amplist1 = np.arange(swp_cfg['start_amp1'], swp_cfg['stop_amp1'], swp_cfg['amp_step1'])
        amplist2 = np.arange(swp_cfg['start_amp2'], swp_cfg['stop_amp2'], swp_cfg['amp_step2'])

        experiment_name = 'photon_number_distribution_measurement'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_photon_number_distribution_measurement_beta_cal', suffix='.h5'))

        for amp1 in amplist1:
            for amp2 in amplist2:
                experiment_cfg[experiment_name]['snap_cav_amps'][0] = amp1
                experiment_cfg[experiment_name]['snap_cav_amps'][1] = amp2
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_rabi_amp_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_rabi_amp_sweep']
        amplist = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        data_path = os.path.join(path, 'data/')

        if swp_cfg["calibrate_weak_drive"]:
            experiment_name = 'weak_rabi'
            seq_data_file = os.path.join(data_path,
                                         get_next_filename(data_path, 'sequential_weak_rabi_amp_sweep', suffix='.h5'))
        else:
            experiment_name = 'rabi'
            seq_data_file = os.path.join(data_path,
                                         get_next_filename(data_path, 'sequential_rabi_amp_sweep', suffix='.h5'))
        expt_cfg = experiment_cfg[experiment_name]

        for amp in amplist:
            experiment_cfg[experiment_name]['amp'] = amp
            #experiment_cfg[experiment_name]['step'] = int((0.3 / amp * 1.0 * 3)/0.0625)*0.0625
            #experiment_cfg[experiment_name]['stop'] = int((0.3 / amp * 101.0 * 3)/0.0625)*0.0625
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_ef_rabi_amp_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_ef_rabi_amp_sweep']
        amplist = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'ef_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_ef_rabi_amp_sweep', suffix='.h5'))

        for amp in amplist:
            experiment_cfg[experiment_name]['amp'] = amp
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_optimal_control_test(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_test']
        steplist = np.arange(swp_cfg['steps'] + 1)

        experiment_name = 'optimal_control_test_1step'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_optimal_control_test', suffix='.h5'))

        for step in steplist:
            experiment_cfg[experiment_name]['pulse_frac'] = (step + 0.0) / (len(steplist) - 1)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("got sequences")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            print("past experiment")
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            print("got I, Q")
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_optimal_control_test_with_blockade(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_test_with_blockade']
        steplist = np.arange(swp_cfg['steps'] + 1)

        experiment_name = 'optimal_control_test_with_blockade_1step'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_optimal_control_test_with_blockade', suffix='.h5'))
        if swp_cfg['define_pulse_fracs']:
            print("defined_pulse_Fracs" + str(len(pulse_fracs)))
            for frac in swp_cfg['pulse_fracs']:
                experiment_cfg[experiment_name]['pulse_frac'] = frac
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                print("past experiment")
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            print("using equal number of steps")
            for step in steplist:
                experiment_cfg[experiment_name]['pulse_frac'] = (step + 0.0) / (len(steplist) - 1)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                print("past experiment")
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_blockade_gate_tomography(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_blockade_gate_tomography']
        experiment_name = 'blockade_gate_tomography'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_blockade_gate_tomography',
                                                       suffix='.h5'))

        if swp_cfg['prep_only']:
            expt_cfg['pulse_frac'] = 0.0

        if swp_cfg['pull_prep_from_wigner_points_file']:
            with File(expt_cfg['wigner_points_file_name'], 'r') as f:
                if expt_cfg['transfer_fn_wt']:
                    # Kevin edit testing a transfer function
                    xs = np.array(f['alphax'][()])[1:swp_cfg['num_prep_points']+1]
                    ys = np.array(f['alphay'][()])[1:swp_cfg['num_prep_points']+1]
                    # end edit
                else:
                    xs = np.array(f['ax'])[:swp_cfg['num_prep_points']]
                    ys = np.array(f['ay'])[:swp_cfg['num_prep_points']]
            prep_lens = []
            prep_phases = []
            conversion_const = self.transfer_function_blockade_inv(expt_cfg['prep_drive_params']['amp'], experiment_cfg,
                                                               channel='cavity_amp_vs_freq').tolist()
            print("conversion scale factor: ", conversion_const)
            for ii, y in enumerate(ys):  # convert pulled points to amplitudes and phases
                x = xs[ii]
                if expt_cfg['transfer_fn_wt']:
                    prep_lens.append(np.sqrt(x ** 2 + y ** 2) / conversion_const)
                else:
                    prep_lens.append(np.sqrt(x ** 2 + y ** 2)) * expt_cfg['prep_drive_params']['len'] / expt_cfg['prep_drive_params']['amp']
                prep_phases.append(np.arctan2(y, x))
            print(prep_lens, prep_phases)
            for i in range(len(prep_lens)):
                expt_cfg['prep_drive_params']['len'] = prep_lens[i]
                expt_cfg['prep_drive_params']['phase'] = prep_phases[i]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                print("past experiment")
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)

        else:  # custom defined prep drive amps and phases
            for i in range(len(swp_cfg['prep_drive_lens'])):
                expt_cfg['prep_drive_params']['len'] = swp_cfg['prep_drive_lens'][i]
                expt_cfg['prep_drive_params']['phase'] = swp_cfg['prep_drive_phases'][i]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                print("past experiment")
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_alpha_scattering_off_blockade(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_alpha_scattering_off_blockade']

        experiment_name = 'alpha_scattering_off_blockade'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_alpha_scattering_off_blockade', suffix='.h5'))
        for evol_len in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            if swp_cfg['sweep_prep_time']:
                experiment_cfg[experiment_name]['prep_cav_len'] = evol_len
            else:
                experiment_cfg[experiment_name]['evol_cav_len'] = evol_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            print("got I, Q")
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_optimal_control_scale_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_scale_sweep']
        scale =  np.arange(swp_cfg['start'],swp_cfg['stop'],swp_cfg['step'])

        experiment_name = 'optimal_control_test_1step'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_optimal_control_scale_sweep', suffix='.h5'))

        for sc in scale:
            if swp_cfg['sweep_ge_scale']:
                if expt_cfg['use_weak_drive']:
                    experiment_cfg[experiment_name]['calibrations']['qubit_ge_weak'] = sc
                else:
                    experiment_cfg[experiment_name]['calibrations']['qubit_ge'] = sc
            elif swp_cfg['sweep_cavity_scale']:
                experiment_cfg[experiment_name]['calibrations']['cavity'] = sc
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_self_kerr_calibration(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_self_kerr_calibration']
        amps = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])[1:]

        experiment_name = 'cavity_drive_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_self_kerr_calibration', suffix='.h5'))

        for amp in amps:
            experiment_cfg[experiment_name]['cavity_drive_amp'] = amp

            print ("Cavity drive ramsey amp = ",amp)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_cavity_drive_ramsey(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_cavity_drive_ramsey']
        experiment_name = 'cavity_drive_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_cavity_drive_ramsey', suffix='.h5'))

        if swp_cfg['vary_len']:
            for ii, mode in enumerate(swp_cfg['mode_indices']):
                experiment_cfg[experiment_name]['cavity_drive_len'] = swp_cfg['cavity_lens'][ii]
                experiment_cfg[experiment_name]['mode_index'] = mode
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for ii, mode in enumerate(swp_cfg['mode_indices']):
                experiment_cfg[experiment_name]['cavity_drive_amp'] = swp_cfg['cavity_amps'][ii]
                experiment_cfg[experiment_name]['mode_index'] = mode
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_repetitive_parity_measurement(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'repetitive_parity_measurement'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_repetitive_parity_measurement', suffix='.h5'))

        for ii in range (80):
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
    def sequential_cavity_temp_rabi(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_temp_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_cavity_temp_rabi', suffix='.h5'))

        for drive in [False, True]:
            experiment_cfg['cavity_temp_rabi']['drive_1'] = drive
            if drive:
                experiment_cfg['cavity_temp_rabi']['acquisition_num'] = 20000
            else:
                experiment_cfg['cavity_temp_rabi']['acquisition_num'] = 2000
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_blockaded_cavity_rabi_vary_probe_level(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_blockaded_cavity_rabi_vary_probe_level', suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_blockaded_cavity_rabi_vary_probe_level']
        probe_levels = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])


        for probe_level in probe_levels:
            experiment_cfg[experiment_name]['probe_level'] = probe_level
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_multitone_blockaded_cavity_rabi_vary_probe_level(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_blockaded_cavity_rabi_vary_probe_level']
        if swp_cfg['weak_cavity']:
            experiment_name = 'multitone_blockaded_weak_cavity_rabi'
        else:
            experiment_name = 'multitone_blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_blockaded_cavity_rabi_vary_probe_level', suffix='.h5'))

        probe_levels = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for probe_level in probe_levels:
            experiment_cfg[experiment_name]['probe_level'] = probe_level
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_blockaded_cavity_rabi_split(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_blockaded_cavity_rabi_split']
        if swp_cfg['weak_cavity']:
            experiment_name = 'multitone_blockaded_weak_cavity_rabi'
        else:
            experiment_name = 'multitone_blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_blockaded_cavity_rabi_split', suffix='.h5'))

        stepnumber = swp_cfg['stepnumber']
        stopstep = swp_cfg['stopstep']
        step = swp_cfg['step']
        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep * (ii + 1)
            experiment_cfg[experiment_name]['start'] = stopstep * (ii)
            experiment_cfg[experiment_name]['step'] = step
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_multimode_blockaded_cavity_rabi_vary_probe_level(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_rabi_vary_probe_level']
        experiment_name = 'multitone_multimode_blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_multimode_blockaded_cavity_rabi_vary_probe_level', suffix='.h5'))

        probe_levels = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for probe_level in probe_levels:
            experiment_cfg[experiment_name]['probe_level'] = probe_level
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_multimode_blockaded_cavity_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_pnrqs']
        experiment_name = 'multitone_multimode_blockaded_cavity_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_multimode_blockaded_cavity_pnrqs', suffix='.h5'))

        cav_pulse_lens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for cav_pulse_len in cav_pulse_lens:
            experiment_cfg[experiment_name]['cavity_pulse_len'] = cav_pulse_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_multimode_blockaded_cavity_beamsplitter_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'multitone_multimode_blockaded_cavity_beamsplitter_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_multitone_multimode_blockaded_cavity_beamsplitter_pnrqs',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_beamsplitter_pnrqs']

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_multimode_blockaded_cavity_beamsplitter_freq_scan(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'multitone_multimode_blockaded_cavity_beamsplitter_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_multitone_multimode_blockaded_cavity_beamsplitter_freq_scan',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_beamsplitter_freq_scan']

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multimode_blockade_experiments_wt(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                     path):
        swp_cfg = experiment_cfg['sequential_multimode_blockade_experiments_wt']
        experiment_name = 'multimode_blockade_experiments_wt'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multimode_blockade_experiments_wt',
                                                                  suffix='.h5'))
        for index in np.arange(swp_cfg['wt_pt_start_index'], swp_cfg['wt_pt_stop_index'] + 1, 1):
            experiment_cfg[experiment_name]['tom2_index_pt'] = int(index)
            print ("tomgraphy point 2 index  = ",index)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_blockaded_weak_cavity_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_blockaded_weak_cavity_pnrqs']
        experiment_name = 'multitone_blockaded_weak_cavity_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_blockaded_weak_cavity_pnrqs', suffix='.h5'))

        cav_pulse_lens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for cav_pulse_len in cav_pulse_lens:
            experiment_cfg[experiment_name]['cavity_pulse_len'] = cav_pulse_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_chi_dressing_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_chi_dressing_pnrqs']
        experiment_name = 'chi_dressing_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_chi_dressing_pnrqs', suffix='.h5'))

        detunings = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for detuning in detunings:
            experiment_cfg[experiment_name]['detuning'] = detuning
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_sideband_chi_dressing_2mode_overlap_measurement(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_dressing_2mode_overlap_measurement']
        experiment_name = 'sideband_chi_dressing_2mode_overlap_measurement'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_chi_dressing_2mode_overlap_measurement', 
                                                                  suffix='.h5'))

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_blockaded_cavity_rabi_vary_dressing_freq(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_blockaded_cavity_rabi_vary_probe_level']
        if swp_cfg['weak_cavity']:
            experiment_name = 'multitone_blockaded_weak_cavity_rabi'
        else:
            experiment_name = 'multitone_blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_blockaded_cavity_rabi_vary_dressing_freq', suffix='.h5'))

        dressing_freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for add_dressing_freq in dressing_freqs:
            experiment_cfg[experiment_name]['dressing_drive_offset_freq'] = add_dressing_freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_blockaded_cavity_rabi_wigner_tomography(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'blockaded_cavity_rabi_wigner_tomography_2d_sweep'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_blockaded_cavity_rabi_wigner_tomography', suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_blockaded_cavity_rabi_wigner_tomography']
        rabi_lens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        
        


        for rabi_len in rabi_lens:
            print ("Rabi length = ",rabi_len/1e3,"us")
            experiment_cfg[experiment_name]['rabi_len'] = rabi_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_blockade_experiments_with_optimal_control_wt(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                           path):
        experiment_name = 'blockade_experiments_with_optimal_control_wt'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_blockade_experiments_with_optimal_control_wt',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_blockade_experiments_with_optimal_control_wt']
        rabi_lens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        for rabi_len in rabi_lens:
            print("Rabi length = ", rabi_len / 1e3, "us")
            experiment_cfg[experiment_name]['rabi_len'] = rabi_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_blockade_experiments_cavity_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'blockade_experiments_cavity_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_blockade_experiments_cavity_spectroscopy',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_blockade_experiments_cavity_spectroscopy']

        if swp_cfg['vary_dressing_freq']:
            change_freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
            for df in change_freqs:
                experiment_cfg[experiment_name]['dressing_drive_offset_freq'] = df
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for ii, mode_index in enumerate(swp_cfg['mode_indices']):
                experiment_cfg[experiment_name]['mode_index'] = mode_index
                experiment_cfg[experiment_name]['cavity_amp'] = swp_cfg['cavity_amps'][ii]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_blockade_spectroscopy_amp_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'blockade_experiments_cavity_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_blockade_spectroscopy_amp_sweep', suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_blockade_spectroscopy_amp_sweep']
        amps = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_cfg[experiment_name]['use_weak_drive_for_dressing'] = swp_cfg['use_weak_drive_for_dressing']
        for amp in amps:
            experiment_cfg[experiment_name]['dressing_amp'] = amp
            
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_cavity_blockade_ramsey(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_blockade_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_cavity_blockade_ramsey',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_cavity_blockade_ramsey']
        stopstep = swp_cfg['stopstep']
        stepnumber = swp_cfg['stepnumber']
        step = swp_cfg['step']


        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep*(ii+1)
            experiment_cfg[experiment_name]['start'] = stopstep*(ii)
            experiment_cfg[experiment_name]['step'] = step

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_cavity_blockade_t1(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_blockade_t1'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_cavity_blockade_t1',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_cavity_blockade_t1']
        stopstep = swp_cfg['stopstep']
        stepnumber = swp_cfg['stepnumber']
        step = swp_cfg['step']

        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep * (ii + 1)
            experiment_cfg[experiment_name]['start'] = stopstep * (ii)
            experiment_cfg[experiment_name]['step'] = step

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_split_sideband_t1(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'sideband_t1'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_split_sideband_t1',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_split_sideband_t1']
        stopstep = swp_cfg['stopstep']
        stepnumber = swp_cfg['stepnumber']
        step = swp_cfg['step']

        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep * (ii + 1)
            experiment_cfg[experiment_name]['start'] = stopstep * (ii)
            experiment_cfg[experiment_name]['step'] = step

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_split_sideband_ramsey(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'sideband_ramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_split_sideband_ramsey',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_split_sideband_ramsey']
        stopstep = swp_cfg['stopstep']
        stepnumber = swp_cfg['stepnumber']
        step = swp_cfg['step']

        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep * (ii + 1)
            experiment_cfg[experiment_name]['start'] = stopstep * (ii)
            experiment_cfg[experiment_name]['step'] = step

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_split_sideband_t1_cavity_swap_tones(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'sideband_t1_cavity_swap_tones'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_split_sideband_t1_cavity_swap_tones',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_split_sideband_t1_cavity_swap_tones']
        stopstep = swp_cfg['stopstep']
        stepnumber = swp_cfg['stepnumber']
        step = swp_cfg['step']

        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = stopstep * (ii + 1)
            experiment_cfg[experiment_name]['start'] = stopstep * (ii)
            experiment_cfg[experiment_name]['step'] = step

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_freq_scan(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_two_cavity_swap_freq_scan',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_two_cavity_swap_freq_scan']

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_over_time(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_over_time'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_two_cavity_swap_over_time',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_two_cavity_swap_over_time']

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_two_cavity_swap_pnrqs',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_two_cavity_swap_pnrqs']

        for time in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            experiment_cfg[experiment_name]['drive_len'] = time

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_over_time_double(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_over_time_double'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_two_cavity_swap_over_time_double',
                                                       suffix='.h5'))

        swp_cfg = experiment_cfg['sequential_two_cavity_swap_over_time_double']

        for expt in range(swp_cfg['expt_num']):
            experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                       expt * swp_cfg['points_per_expt'] * expt_cfg['step']
            experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                      (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - 1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_two_cavity_swap_over_time_double_sequential(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'two_cavity_swap_over_time_double_sequential'
    
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
    
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_two_cavity_swap_over_time_double_sequential',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_two_cavity_swap_over_time_double_sequential']
        experiment_cfg[experiment_name]['sweep_drive_times_separately'] = True
        for expt in range(swp_cfg['expt_num']):
            if swp_cfg['sweep_a']:
                experiment_cfg[experiment_name]['start'] = swp_cfg['start'] + \
                                                           expt * swp_cfg['points_per_expt'] * expt_cfg['step']
                experiment_cfg[experiment_name]['stop'] = swp_cfg['start'] + \
                                                          (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step'] - \
                                                          1e-10
            else:
                experiment_cfg[experiment_name]['start_b'] = swp_cfg['start'] + \
                                                           expt * swp_cfg['points_per_expt'] * expt_cfg['step_b']
                experiment_cfg[experiment_name]['stop_b'] = swp_cfg['start'] + \
                                                          (expt + 1) * swp_cfg['points_per_expt'] * expt_cfg['step_b'] - \
                                                          1e-10
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=self.hvi)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def transfer_function_blockade_inv(self, amp, experiment_cfg, channel='cavity_amp_vs_amp'):
        # pull calibration data from file
        fn_file = experiment_cfg['transfer_function_blockade_calibration_files'][channel]
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
        elif channel == 'cavity_amp_vs_freq':
            with File(fn_file, 'r') as f:  # in this case want inverse of original function
                omegas = f['amps'][()]     # just relabeling out of laziness
                amps = f['omegas'][()]
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

        