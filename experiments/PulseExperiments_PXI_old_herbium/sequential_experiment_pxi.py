from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.instruments.keysight import keysight_pxi_load as ks_pxi
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin
try:from skopt import Optimizer
except:print("No optimizer")
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperiment
from slab.instruments import InstrumentManager
import time


# from slab.experiments.PulseExperiments.get_data import get_singleshot_data_two_qubits_4_calibration_v2,\
#     get_singleshot_data_two_qubits, data_to_correlators, two_qubit_quantum_state_tomography,\
#     density_matrix_maximum_likelihood

import pickle

class SequentialExperiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, path,analyze = False,show=True,P = 'Q'):

        self.Is = []
        self.Qs = []

        eval('self.' + experiment_name)(quantum_device_cfg, experiment_cfg, hardware_cfg,path)
        if analyze:
            try:
                self.analyze(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,self.Is,self.Qs,P = 'I')
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii+=1
        else:
            for freq in freqs:
                quantum_device_cfg['readout']['freq'] = freq
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
        self.Is.append(I)
        self.Qs.append(Q)

        experiment_name = 'ramsey'
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
        self.Is.append(I)
        self.Qs.append(Q)

        experiment_name = 'echo'
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def wigner_tomography_2d_offset_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['wigner_tomography_2d_offset_sweep']
        xoffsets = np.arange(swp_cfg['startx'], swp_cfg['stopx'], swp_cfg['stepx'])
        yoffsets = np.arange(swp_cfg['starty'], swp_cfg['stopy'], swp_cfg['stepy'])

        experiment_name = "wigner_tomography_2d_sideband_alltek2"

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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][ii]
                experiment_cfg[experiment_name]['length'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_lens'][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][ii]

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][ii]

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_ramsey_overmodes(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_ramsey_overmodes']
        modelist = swp_cfg['modelist']

        experiment_name = 'sideband_ramsey'

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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def analyze(self,quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,Is,Qs,P='Q'):
        PA = PostExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, Is,Qs,P,show)
