from slab.experiments.PulseExperiments_M8195A_PXI_sideband_with_IQ.sequences_pxi_stimEm import PulseSequences
from slab.experiments.PulseExperiments_M8195A_PXI_sideband_with_IQ.pulse_experiment_stimEm import Experiment
from slab.instruments.keysight import keysight_pxi_load as ks_pxi
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin
try:from skopt import Optimizer
except:print("No optimizer")
from slab.experiments.PulseExperiments_M8195A_PXI_sideband_with_IQ.PostExperimentAnalysis_StimEm import PostExperiment
from slab.instruments import InstrumentManager
import time
from h5py import File
from scipy import interpolate

import pickle

class SequentialExperiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, path, analyze = False,
                 show=True,P = 'Q', return_val=False, fock_number=-1):

        self.Is = []
        self.Qs = []
        self.show = show
        self.fock_number = fock_number

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

    def oct_histogram_amp_and_freq_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['oct_histogram_amp_and_freq_sweep']
        attens = np.arange(expt_cfg['atten_start'],expt_cfg['atten_stop'],expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'],expt_cfg['freq_stop'],expt_cfg['freq_step'])

        experiment_name = 'oct_histogram'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'oct_histogram_amp_and_freq_sweep', suffix='.h5'))
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

    def resonator_spectroscopy_weak_qubit_drive(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy_weak_qubit_drive'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_weak_qubit_drive', suffix='.h5'))
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
    
    
    def cavity_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_photon_resolved_qubit_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_temperature', suffix='.h5'))

        for rabi_level in [1,0]:
            experiment_cfg[experiment_name]['rabi_level'] = rabi_level
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
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['length'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_lens'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][expt_cfg['mode_index']][ii]

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
                experiment_cfg[experiment_name]['freq'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['fnm1gn_freqs'][expt_cfg['mode_index']][ii]
                experiment_cfg[experiment_name]['amp'] = quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['pi_fnm1gn_amps'][expt_cfg['mode_index']][ii]

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

        for ii,mode in enumerate(modelist):
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            experiment_cfg[experiment_name]['mode_index'] = mode
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


    def sideband_rabi_freq_scan_sweepLO(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sideband_rabi_freq_scan_sweepLO']

        experiment_name = 'sideband_rabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_freq_scan_sweepLO', suffix='.h5'))
        
        dfreqs = np.arange(swp_cfg['start'],swp_cfg['stop'],swp_cfg['step'])
        if expt_cfg['use_freq_from_expt_cfg']:
            freq_center = expt_cfg['sideband_freq']
        else:
            if expt_cfg['f0g1']:
                freq_center = quantum_device_cfg['flux_pulse_info']['1']['f0g1_freq'][expt_cfg['mode_index']]
            elif expt_cfg['h0e1']:
                freq_center = quantum_device_cfg['flux_pulse_info']['1']['h0e1_freq'][expt_cfg['mode_index']]
        
        experiment_cfg[experiment_name]['start'] = 0.0
        experiment_cfg[experiment_name]['stop'] = 0.001
        experiment_cfg[experiment_name]['step'] = 0.001
            
        for ii,dfreq in enumerate(dfreqs):
            expt_cfg['sideband_freq'] = freq_center + dfreq
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
        # amplist = swp_cfg['amplist']
        freqlist = swp_cfg['freqlist']
        modelist = swp_cfg['modelist']
        stoplist  = swp_cfg['stoplist']
        lolist = swp_cfg['lolist']

        experiment_name = 'sideband_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_rabis', suffix='.h5'))

        if expt_cfg['use_freq_from_expt_cfg']:
            uselist = freqlist
        else:
            uselist = modelist
        for ii, use in enumerate(uselist):
            if swp_cfg['take_trigger_from_t1']:
                hardware_cfg['trigger']['period_us'] = int(
                    quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][ii] * 5)
            if expt_cfg['use_freq_from_expt_cfg']:
                freq = use
                quantum_device_cfg['sideband_drive_lo_powers']['1'] = int(lolist[ii])
                experiment_cfg[experiment_name]['sideband_freq'] = freq

            else:
                mode = use
                experiment_cfg[experiment_name]['mode_index'] = mode
            # experiment_cfg[experiment_name]['amp'] = amplist[ii]
            experiment_cfg[experiment_name]['stop'] = stoplist[ii]
            experiment_cfg[experiment_name]['step'] = stoplist[ii] / 100.0
            print ("lo power = ",int(lolist[ii]),"freq = ",freq,"GHz")

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_sideband_chi_dressing_calibration_vary_power(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_dressing_calibration_vary_power']
        freqlist = swp_cfg['freqlist']
        modelist = swp_cfg['modelist']
        lolist = swp_cfg['lolist']

        experiment_name = 'sideband_chi_dressing_calibration'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_chi_dressing_calibration_vary_power', suffix='.h5'))

        if expt_cfg['use_freq_from_expt_cfg']:
            uselist = freqlist
        else:
            uselist = modelist
        for ii, use in enumerate(uselist):
            if expt_cfg['use_freq_from_expt_cfg']:
                freq = use
                quantum_device_cfg['sideband_drive_lo_powers']['1'] = int(lolist[ii])
                experiment_cfg[experiment_name]['sideband_freq'] = freq
            else:
                mode = use
                experiment_cfg[experiment_name]['mode_index'] = mode
            print ("lo power = ",int(lolist[ii]),"freq = ",freq,"GHz")

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_photon_number_expt_fit_wigner_alphas(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_photon_number_expt_fit_wigner_alphas']
        experiment_name = 'photon_number_expt_fit_wigner_alphas'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_photon_number_expt_fit_wigner_alphas',
                                                                  suffix='.h5'))
        for index in np.arange(swp_cfg['wigner_pt_start_index'], swp_cfg['wigner_pt_stop_index'] + 1, 1):
            experiment_cfg[experiment_name]['tom_index_pt'] = int(index)
            print ("tomgraphy point index  = ",index)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_pnrqs_sideband_on_vary_power(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_pnrqs_sideband_on_vary_power']
        powers = np.arange(swp_cfg['start'],swp_cfg['stop'],swp_cfg['step'])

        experiment_name = 'photon_number_resolved_qubit_spectroscopy_with_sideband_on'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_pnrqs_sideband_on_vary_power', suffix='.h5'))

       
        for ii, power in enumerate(powers):

            quantum_device_cfg['sideband_drive_lo_powers']['1'] = int(power)
       
            print ("lo power = ",int(power))
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_pnrqs_sideband_on_vary_length(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_pnrqs_sideband_on_vary_length']
        lengths = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'photon_number_resolved_qubit_spectroscopy_with_sideband_on'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_pnrqs_sideband_on_vary_length',
                                                                  suffix='.h5'))
        for ii, length in enumerate(lengths):
            if swp_cfg['use_weak_drive']:
                expt_cfg['use_weak_drive'] = True
                quantum_device_cfg['pulse_info']['1']['pi_len_resolved_weak'] = float(length)
            else:
                expt_cfg['use_weak_drive'] = False
                quantum_device_cfg['pulse_info']['1']['pi_len_resolved'] = float(length)
            print("resolved pulse length = ", length)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_sideband_chi_dressing_t1test_vary_power(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_dressing_t1test_vary_power']
        freqlist = swp_cfg['freqlist']
        modelist = swp_cfg['modelist']
        lolist = swp_cfg['lolist']

        experiment_name = 'sideband_chi_dressing_t1test'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_chi_dressing_t1test_vary_power', suffix='.h5'))

        if expt_cfg['use_freq_from_expt_cfg']:
            uselist = freqlist
        else:
            uselist = modelist
        for ii, use in enumerate(uselist):
            if expt_cfg['use_freq_from_expt_cfg']:
                freq = use
                quantum_device_cfg['sideband_drive_lo_powers']['1'] = int(lolist[ii])
                experiment_cfg[experiment_name]['sideband_freq'] = freq
            else:
                mode = use
                experiment_cfg[experiment_name]['mode_index'] = mode
            print ("lo power = ",int(lolist[ii]),"freq = ",freq,"GHz")

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
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

    def sequential_chi_dressing_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_chi_dressing_calibration']

        varlist =  np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])


        experiment_name = 'chi_dressing_calibration'

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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        elif swp_cfg['sweep_mode_index']:
            for ii in range(len(swp_cfg['mode_list'])):
                experiment_cfg[experiment_name]['mode_index'] = swp_cfg['mode_list'][ii]
                experiment_cfg[experiment_name]['prep_cav_len'] = swp_cfg['len_list'][ii]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for amp in amplist:
                experiment_cfg[experiment_name]['prep_cav_amp'] = amp
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_pnrqs_vs_detuning(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_pnrqs_vs_detuning']
        freqlist = np.arange(swp_cfg['start_freq'], swp_cfg['stop_freq'], swp_cfg['freq_step'])

        experiment_name = 'photon_number_resolved_qubit_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_pnrqs_vs_detuning', suffix='.h5'))


        for offset_freq in freqlist:
            experiment_cfg[experiment_name]['add_cavity_drive_detuning'] = offset_freq
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        elif swp_cfg['sweep_prep_cav_len']:
            lenlist = np.arange(swp_cfg['start_len'], swp_cfg['stop_len'], swp_cfg['len_step'])
            for cav_len in lenlist:
                expt_cfg['prep_cav_len'] = float(cav_len)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            amplist = np.arange(swp_cfg['start_amp'], swp_cfg['stop_amp'], swp_cfg['amp_step'])
            for amp in amplist:
                experiment_cfg[experiment_name]['prep_cav_amp'] = amp
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_ramsey_after_readout(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_ramsey_after_readout']
        amplist = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        data_path = os.path.join(path, 'data/')

        experiment_name = 'ramsey_after_readout'
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_ramsey_after_readout', suffix='.h5'))
        expt_cfg = experiment_cfg[experiment_name]

        for time in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            experiment_cfg[experiment_name]['wait_time'] = time
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            print("using defined pulse fracs")
            for frac in swp_cfg['pulse_fracs']:
                experiment_cfg[experiment_name]['pulse_frac'] = frac
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                print("past experiment")
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for step in steplist:
                experiment_cfg[experiment_name]['pulse_frac'] = (step + 0.0) / (len(steplist) - 1)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for ii, mode in enumerate(swp_cfg['mode_indices']):
                experiment_cfg[experiment_name]['cavity_drive_amp'] = swp_cfg['cavity_amps'][ii]
                experiment_cfg[experiment_name]['mode_index'] = mode
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multitone_multimode_blockaded_cavity_rabi_split(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_rabi_split']
        experiment_name = 'multitone_multimode_blockaded_cavity_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_multitone_multimode_blockaded_cavity_rabi_split', suffix='.h5'))

        stepnumber = swp_cfg['stepnumber']
        stopstep = swp_cfg['stopstep']
        step = swp_cfg['step']
        for ii in np.arange(stepnumber):
            experiment_cfg[experiment_name]['stop'] = float(stopstep * (ii + 1))
            experiment_cfg[experiment_name]['start'] = float(stopstep * (ii))
            experiment_cfg[experiment_name]['step'] = float(step)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_single_photon_with_blockade_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_single_photon_with_blockade_pnrqs']
        experiment_name = 'multitone_blockaded_weak_cavity_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_single_photon_with_blockade_pnrqs', suffix='.h5'))
        blockade_pulse_info = quantum_device_cfg['blockade_pulse_params']
        bp = blockade_pulse_info
        modelist = swp_cfg['modelist']
        expt_cfg['cavity_pulse_len'] = 0
        expt_cfg['prep_state_before_blockade'] = True
        expt_cfg['use_optimal_control'] = False
        expt_cfg['prep_using_blockade'] = True
        for mode in modelist:
            # expt_cfg['dressing_amp'] = bp['blockade_pi_amp_qubit'][mode]
            # expt_cfg['cavity_amp'] = bp['blockade_pi_amp_cavity'][mode]
            # expt_cfg['cavity_offset_freq'] = bp['blockade_cavity_offset_freq'][mode]
            # expt_cfg['cavity_pulse_len'] = bp['blockade_pi_length'][mode]
            # expt_cfg['use_weak_drive_for_dressing'] = bp['use_weak_for_blockade'][mode]
            # expt_cfg['mode_index'] = mode
            expt_cfg['prep_mode_index'] = mode
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_multitone_multimode_blockaded_cavity_full_tomography(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_multitone_multimode_blockaded_cavity_full_tomography']
        experiment_name = 'multitone_multimode_blockaded_cavity_full_tomography'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multitone_multimode_blockaded_cavity_full_tomography',
                                                                  suffix='.h5'))
        cav_pulse_lens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        for cav_pulse_len in cav_pulse_lens:
            experiment_cfg[experiment_name]['cavity_pulse_len'] = cav_pulse_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
    
    def sequential_multimode_blockade_experiments_wt_3modes(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_multimode_blockade_experiments_wt_3modes']
        experiment_name = 'multimode_blockade_experiments_wt'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multimode_blockade_experiments_wt_3modes',
                                                                  suffix='.h5'))
        wg_pts_index_array = np.arange(swp_cfg['wt_pt_start_index'], swp_cfg['wt_pt_stop_index'] + 1, 1)
        for index2 in wg_pts_index_array:
            for index in wg_pts_index_array:
                if len(wg_pts_index_array) * index2 + index >= swp_cfg['absolute_start_index']:
                    experiment_cfg[experiment_name]['tom2_index_pt'] = int(index)
                    print ("tomgraphy point 2 index  = ",index)
                    experiment_cfg[experiment_name]['tom3_index_pt'] = int(index2)
                    print("tomgraphy point 3 index  = ", index2)
                    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                    sequences = ps.get_experiment_sequences(experiment_name)
                    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                    I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                    self.Is.append(I)
                    self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_parity_measurement_bandwidth_calibration_wigner_pts_state_prep(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_parity_measurement_bandwidth_calibration_wigner_pts_state_prep']
        experiment_name = 'parity_measurement_bandwidth_calibration_wigner_pts_state_prep'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_parity_measurement_bandwidth_calibration_wigner_pts_state_prep',
                                                                  suffix='.h5'))
        for length in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
            experiment_cfg[experiment_name]['rabi_len'] = int(length)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

 
    def sequential_multimode_parity_measurement_bandwidth_calibration(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_multimode_parity_measurement_bandwidth_calibration']
        experiment_name = 'multimode_parity_measurement_bandwidth_calibration'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multimode_parity_measurement_bandwidth_calibration',
                                                                  suffix='.h5'))
        for index in np.arange(swp_cfg['wt_pt_start_index'], swp_cfg['wt_pt_stop_index'] + 1, 1):
            experiment_cfg[experiment_name]['tom2_index_pt'] = int(index)
            print ("tomgraphy point 2 index  = ",index)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_multimode_parity_measurement_bandwidth_calibration_3modes(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_multimode_parity_measurement_bandwidth_calibration_3modes']
        experiment_name = 'multimode_parity_measurement_bandwidth_calibration'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multimode_parity_measurement_bandwidth_calibration_3modes',
                                                                  suffix='.h5'))
        wg_pts_index_array = np.arange(swp_cfg['wt_pt_start_index'], swp_cfg['wt_pt_stop_index'] + 1, 1)
        for index2 in wg_pts_index_array:
            for index in wg_pts_index_array:
                if len(wg_pts_index_array) * index2 + index >= swp_cfg['absolute_start_index']:
                    experiment_cfg[experiment_name]['tom2_index_pt'] = int(index)
                    print("tomgraphy point 2 index  = ", index)
                    experiment_cfg[experiment_name]['tom3_index_pt'] = int(index2)
                    print("tomgraphy point 3 index  = ", index2)
                    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                    sequences = ps.get_experiment_sequences(experiment_name)
                    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                    I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                    self.Is.append(I)
                    self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
        
    def sequential_multimode_blockade_experiments_wt_with_parity_times(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                              path):
        swp_cfg = experiment_cfg['sequential_multimode_blockade_experiments_wt_with_parity_times']
        experiment_name = 'multimode_blockade_experiments_wt'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path,
                                                                  'sequential_multimode_blockade_experiments_wt_with_parity_times',
                                                                  suffix='.h5'))
        for ramsey_time in swp_cfg['ramsey_parity_times']:
            experiment_cfg[experiment_name]['ramsey_parity_time'] = ramsey_time
            experiment_cfg[experiment_name]['ramsey_parity'] = True
            experiment_cfg[experiment_name]['pi_resolved_parity'] = False
            for index in np.arange(swp_cfg['wt_pt_start_index'], swp_cfg['wt_pt_stop_index'] + 1, 1):
                experiment_cfg[experiment_name]['tom2_index_pt'] = int(index)
                print ("tomgraphy point 2 index  = ",index)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_sideband_chi_dressing_pnrqs(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_sideband_chi_dressing_pnrqs']
        experiment_name = 'sideband_chi_dressing_pnrqs'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_chi_dressing_pnrqs', suffix='.h5'))

        detunings = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])
        original_freq = experiment_cfg[experiment_name]['sideband_freq']
        for detuning in detunings:
            experiment_cfg[experiment_name]['sideband_freq'] = original_freq + detuning
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_blockade_experiments_with_optimal_control_generalized_wt(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                           path):
        experiment_name = 'blockade_experiments_with_optimal_control_generalized_wt'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_blockade_experiments_with_optimal_control_generalized_wt',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_blockade_experiments_with_optimal_control_generalized_wt']
        for ramsey_time in swp_cfg['wigner_ramsey_times']:
            print("Ramsey Time = ", ramsey_time, "ns")
            experiment_cfg[experiment_name]['wigner_ramsey_time'] = ramsey_time
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for ii, mode_index in enumerate(swp_cfg['mode_indices']):
                experiment_cfg[experiment_name]['mode_index'] = mode_index
                experiment_cfg[experiment_name]['cavity_amp'] = swp_cfg['cavity_amps'][ii]
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_cross_kerr_calibration_with_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'blockade_experiments_cavity_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_cross_kerr_calibration_with_spectroscopy',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_cross_kerr_calibration_with_spectroscopy']
        for ii, mode_index in enumerate(swp_cfg['ramsey_mode_indices']):
            if expt_cfg['prep_state_before_blockade']:
                experiment_cfg[experiment_name]['prep_mode_index'] = swp_cfg['prep_mode_indices'][ii]
            experiment_cfg[experiment_name]['mode_index'] = mode_index
            experiment_cfg[experiment_name]['cavity_amp'] = swp_cfg['cavity_amps'][ii]
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
        
    def cross_kerr_with_readout_spectroscopy(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cross_kerr_with_readout_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cross_kerr_with_readout_spectroscopy', suffix='.h5'))
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
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
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_cross_kerr_calibration_with_ramsey(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_blockade_ramsey'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_cross_kerr_calibration_with_ramsey',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_cross_kerr_calibration_with_ramsey']
        prep_modes = swp_cfg['prep_mode_indices']
        modes = swp_cfg['ramsey_mode_indices']
        for ii, mode in enumerate(modes):
            if expt_cfg['prep_photon_with_blockade']:
                experiment_cfg[experiment_name]['prep_mode_index'] = prep_modes[ii]
            experiment_cfg[experiment_name]['mode_index'] = mode
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_cavity_spectroscopy_resolved_qubit_pulse(self, quantum_device_cfg, experiment_cfg, hardware_cfg,
                                                            path):
        experiment_name = 'cavity_spectroscopy_resolved_qubit_pulse'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_cavity_spectroscopy_resolved_qubit_pulse',
                                                       suffix='.h5'))
        swp_cfg = experiment_cfg['sequential_cavity_spectroscopy_resolved_qubit_pulse']
        for ii, mode_index in enumerate(swp_cfg['modelist']):
            experiment_cfg[experiment_name]['mode_index'] = mode_index
            experiment_cfg[experiment_name]['cavity_pulse_len'] = swp_cfg['lenlist'][ii]
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def sequential_optimal_control_test_ef_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_test_ef_spectroscopy']
        steplist = np.arange(swp_cfg['steps'] + 1)

        experiment_name = 'optimal_control_test_ef_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_optimal_control_test_ef_spectroscopy', suffix='.h5'))

        if swp_cfg['define_pulse_fracs']:
            print("using defined pulse fracs")
            for frac in swp_cfg['pulse_fracs']:
                experiment_cfg[experiment_name]['pulse_frac'] = frac
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                print("past experiment")
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for step in steplist:
                experiment_cfg[experiment_name]['pulse_frac'] = (step + 0.0) / (len(steplist) - 1)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("got sequences")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                print("past experiment")
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                print("got I, Q")
                self.Is.append(I)
                self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def sequential_optimal_control_test_1step_recovery(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_test_1step_recovery']
        lengths = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'optimal_control_test_1step_recovery'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_optimal_control_test_1step_recovery', suffix='.h5'))

        for length in lengths:
            experiment_cfg[experiment_name]['recovery_pulse_length'] = length
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
    
    def sequential_optimal_control_repeated_pi_pulses(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_repeated_pi_pulses']
        experiment_name = 'optimal_control_repeated_pi_pulses'
        expt_cfg = experiment_cfg[experiment_name]
        fock_number = self.fock_number
        # data_path = os.path.join(path, 'data/test')
        data_path = os.path.join(path, "data/g" + str(fock_number) + "/")
        expt_cfg['pi_at_n'] = fock_number
        expt_cfg['pi_at_m'] = fock_number + 1

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_optimal_control_repeated_pi_pulses', suffix='.h5'))
        for temp in np.arange(swp_cfg['rep_number']):
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
    #Added 0n 07/28/2020 - Ankur for stimulated emission experiment. This loads the M8195 and PXI AWG only once.
    def sequential_optimal_control_repeated_pi_pulses_noload(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_repeated_pi_pulses']
        experiment_name = 'optimal_control_repeated_pi_pulses'
        expt_cfg = experiment_cfg[experiment_name]
        fock_number = self.fock_number
        camp = expt_cfg['cavity_amp']
        l = expt_cfg['cavity_pulse_len']
        # data_path = os.path.join(path, 'data/g0')
        # data_path = os.path.join(path, "data/g" + str(fock_number))
        # if camp!=0.012:
        #     data_path = os.path.join(path, "data/g" + str(fock_number) + "/camp_" + str(camp))
        # else:
        #     data_path = os.path.join(path, "data/g" + str(fock_number) + "/camp_0.003")
        # data_path = os.path.join(path, "data/g" + str(fock_number) + "/camp_" + str(camp))
        data_path = os.path.join(path, "data/g" + str(fock_number) + "/camp_" + str(camp)+"_"+str(int(l)))

        expt_cfg['pi_at_n'] = fock_number
        expt_cfg['pi_at_m'] = fock_number + 1

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_optimal_control_repeated_pi_pulses', suffix='.h5'))
        for temp in range(swp_cfg['rep_number']):
            if temp is 0:
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                load = True
            else:
                load = False
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)

            print("$$$$$$$$$$$$$&&&&&&&&&&&& \n Current iteration: %d, %s"%(temp, load))

            I, Q = exp.run_experiment_pxi_repeated_noload(sequences, path, experiment_name, seq_data_file=seq_data_file, load=load)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        
    def sequential_optimal_control_histogram(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['sequential_optimal_control_histogram']
        experiment_name = 'optimal_control_histogram'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'sequential_optimal_control_histogram', suffix='.h5'))
        for temp in range(swp_cfg['rep_number']):
            if temp is 0:
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                load = True
            else:
                load = False
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)

            print("$$$$$$$$$$$$$&&&&&&&&&&&& \n Current iteration: %d, %s"%(temp, load))

            I, Q = exp.run_experiment_pxi_repeated_noload(sequences, path, experiment_name, seq_data_file=seq_data_file, load=load)
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

        