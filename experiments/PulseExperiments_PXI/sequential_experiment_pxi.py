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

    def t1rho_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        expt_cfg = experiment_cfg['t1rho_sweep']
        data_path = os.path.join(path, 'data/')


        amparray = np.arange(expt_cfg['ampstart'],expt_cfg['ampstop'],expt_cfg['ampstep'])
        amparray = [ 0.01,        0.0129155,   0.01668101,  0.02154435,  0.02782559,  0.03593814,
  0.04641589,  0.05994843,  0.07742637,  0.1       ]
        print(amparray)

        experiment_name = 't1rho_sweep'
        print("Sequences generated")
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 't1rho_sweep', suffix='.h5'))
        for ampval in amparray:
            experiment_name = 't1rho'
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            experiment_cfg['t1rho']['amp']=ampval
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)


    def histogram_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['histogram_sweep']
        sweep_amp = expt_cfg['sweep_amp']
        attens = np.arange(expt_cfg['atten_start'],expt_cfg['atten_stop'],expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'],expt_cfg['freq_stop'],expt_cfg['freq_step'])

        experiment_name = 'histogram'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram_sweep', suffix='.h5'))

        if sweep_amp:
            for att in attens:
                quantum_device_cfg['readout']['dig_atten'] = att
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I,Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
        else:
            for freq in freqs:
                quantum_device_cfg['readout']['freq'] = freq
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)


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

    def rabisweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'rabisweep'
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
            quantum_device_cfg['readout_drive_digital_attenuation']= atten
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
            else:experiment_cfg['ef_rabi']['acquisition_num'] = 5000
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

    def pulse_probe_delay_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['pulse_probe_delay_sweep']
        delays = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'pulse_probe_iq'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_delay_sweep', suffix='.h5'))

        for delay in delays:

            experiment_cfg["pulse_probe_iq"]["delay"] = delay
            print("delay set to", delay)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def pulse_probe_atten_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['pulse_probe_atten_sweep']
        attens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'pulse_probe_iq'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_atten_sweep', suffix='.h5'))

        for atten in attens:

            quantum_device_cfg["qubit_drive_digital_attenuation"] = atten
            print("atten set to", atten)
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


    def sequential_qubit_calibration(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        pass


    def analyze(self,quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,Is,Qs,P='Q'):
        PA = PostExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, Is,Qs,P,show)
