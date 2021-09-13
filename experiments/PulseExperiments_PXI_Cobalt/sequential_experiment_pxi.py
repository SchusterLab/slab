from slab.experiments.PulseExperiments_PXI_Cobalt.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI_Cobalt.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI_Cobalt import keysight_pxi_load as ks_pxi
# from slab.instruments.keysight import keysight_pxi_load as ks_pxi
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin
try:from skopt import Optimizer
except:print("No optimizer")
from slab.experiments.PulseExperiments_PXI_Cobalt.PostExperimentAnalysis import PostExperiment
# from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperiment
import copy
import time

# from slab.experiments.PulseExperiments.get_data import get_singleshot_data_two_qubits_4_calibration_v2,\
#     get_singleshot_data_two_qubits, data_to_correlators, two_qubit_quantum_state_tomography,\
#     density_matrix_maximum_likelihood

import pickle

class SequentialExperiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,experiment_name, path, analyze = False,show=True,P = 'Q'):
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.quantum_device_cfg = quantum_device_cfg

        self.data = []
        if len(self.experiment_cfg[experiment_name]['on_qubits']) == 2:
            self.two_qubits = True
        else:
            self.two_qubits = False

        if self.two_qubits:
            self.IAs = []
            self.IBs = []
            self.QAs = []
            self.QBs = []
            self.seq_data = []
        else:
            self.Is = []
            self.Qs = []
            self.seq_data = []

        eval('self.' + experiment_name)(quantum_device_cfg, experiment_cfg, hardware_cfg,path)


        # Analyze option will not work with two qubits
        # Unless is modified, should not be very hard 7/6/21
        if analyze:
            try:
                self.analyze(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,self.Is,self.Qs,P = 'I')
            except: print ("No post expt analysis")
        else:pass

    #
    def resonator_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy'
        # data_path = os.path.join(path, 'data/')
        # seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy', suffix='.h5'))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)

        exp.run_experiment_pxi_resspec(sequences=sequences, path=path, name=experiment_name)

# This one works as well as the above one! Pick which one you're using but either should go just fine [MGP] 7/7/2021
    # def resonator_spectroscopy(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    #     experiment_name = 'resonator_spectroscopy'
    #     data_path = os.path.join(path, 'data/')
    #     self.seq_data_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
    #     self.expt_cfg = experiment_cfg[experiment_name]
    #     # self.quantum_device_cfg = quantum_device_cfg
    #
    #     ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
    #     sequences = ps.get_experiment_sequences(experiment_name)
    #     exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
    #
    #     exp.generate_datafile(path, experiment_name, self.seq_data_file)
    #     exp.set_trigger()
    #     exp.initiate_drive_LOs()
    #     exp.initiate_cavity_drive_LOs(experiment_name)
    #     exp.initiate_readout_attenuators()
    #     exp.initiate_drive_attenuators()
    #     exp.initiate_pxi(experiment_name, sequences)
    #     exp.initiate_readout_LOs()
    #     exp.awg_run(run_pxi=True, name=experiment_name)
    #
    #     time.sleep(0.1)
    #
    #     for qubit in self.expt_cfg['on_qubits']:
    #         read_freq = copy.deepcopy(self.quantum_device_cfg['readout'][qubit]['freq'])
    #         for freq in np.arange(self.expt_cfg['start'] + read_freq, self.expt_cfg['stop'] + read_freq, self.expt_cfg['step']):
    #
    #             self.quantum_device_cfg['readout'][qubit]['freq'] = freq
    #             exp.initiate_readout_LOs()
    #
    #             # Put more data options in here
    #             if self.expt_cfg['singleshot']:
    #                 self.data = exp.get_ss_data_pxi(self.expt_cfg, seq_data_file=seq_data_file)
    #             elif self.expt_cfg['traj_data']:
    #                 self.data = exp.get_traj_data_pxi(seq_data_file=seq_data_file)
    #             elif self.expt_cfg['traj_data_nowindow']:
    #                 self.data = exp.get_traj_data_pxi_nowindow(seq_data_file=seq_data_file)
    #             else:
    #                 self.data = exp.get_avg_data_pxi(self.expt_cfg, seq_data_file=self.seq_data_file)
    #
    #             exp.pxi.DIG_module.stopAll()
    #             #self.pxi.DIG_module.clearAll()
    #
    #             # self.awg_stop(experiment_name)
    #             # self.pxi = ks_pxi.KeysightSingleQubit(self.experiment_cfg, self.hardware_cfg, self.quantum_device_cfg,
    #             #                                       sequences, name)
    #             exp.pxi.configureDigitizerChannels(self.hardware_cfg, self.experiment_cfg, experiment_name)
    #
    #             if self.two_qubits:
    #                 exp.pxi.DIG_ch_1.clear()
    #                 exp.pxi.DIG_ch_1.start()
    #                 exp.pxi.DIG_ch_2.clear()
    #                 exp.pxi.DIG_ch_2.start()
    #                 exp.pxi.DIG_ch_3.clear()
    #                 exp.pxi.DIG_ch_3.start()
    #                 exp.pxi.DIG_ch_4.clear()
    #                 exp.pxi.DIG_ch_4.start()
    #             elif '1' == self.expt_cfg['on_qubits'][0]:
    #                 exp.pxi.DIG_ch_1.clear()
    #                 exp.pxi.DIG_ch_1.start()
    #                 exp.pxi.DIG_ch_2.clear()
    #                 exp.pxi.DIG_ch_2.start()
    #             elif "2" == self.expt_cfg['on_qubits'][0]:
    #                 exp.pxi.DIG_ch_3.clear()
    #                 exp.pxi.DIG_ch_3.start()
    #                 exp.pxi.DIG_ch_4.clear()
    #                 exp.pxi.DIG_ch_4.start()
    #             else:
    #                 print("yikes o'clock qubit addresing error")
    #             time.sleep(0.1)
    #     exp.awg_stop(experiment_name)
    #     return self.data


############## things below this line do not work unless edited as of 7/21 [MGP] ###########################

    def histogram_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['histogram_sweep']
        sweep_amp = expt_cfg['sweep_amp']
        attens = np.arange(expt_cfg['atten_start'], expt_cfg['atten_stop'], expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])

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
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii += 1
        else:
            for freq in freqs:
                quantum_device_cfg['readout']['freq'] = freq
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii += 1

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def histogram_amp_and_freq_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['histogram_amp_and_freq_sweep']
        attens = np.arange(expt_cfg['atten_start'], expt_cfg['atten_stop'], expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])

        experiment_name = 'histogram'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'histogram_amp_and_freq_sweep', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        ii = 0
        for freq in freqs:
            for att in attens:
                quantum_device_cfg['readout']['dig_atten'] = att
                print("Attenuation = ", att, "dB")
                quantum_device_cfg['readout']['freq'] = freq
                print("Frequency = ", freq, "GHz")
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii += 1

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_histogram_amp_and_freq_sweep_mixedtones(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['cavity_drive_histogram_amp_and_freq_sweep_mixedtones']
        attens = np.arange(expt_cfg['atten_start'], expt_cfg['atten_stop'], expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])
        experiment_name = 'cavity_drive_histogram_mixedtones'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'cavity_drive_histogram_amp_and_freq_sweep_mixedtones', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)
        ii = 0
        for freq in freqs:
            for att in attens:
                quantum_device_cfg['readout']['dig_atten'] = att
                print("Attenuation = ", att, "dB")
                quantum_device_cfg['readout']['freq'] = freq
                print("Frequency = ", freq, "GHz")
                print("Expt num = ", ii)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.Is.append(I)
                self.Qs.append(Q)
                ii += 1

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_direct_spectroscopy_freq_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):

        expt_cfg = experiment_cfg['cavity_drive_direct_spectroscopy_freq_sweep']
        freq_starts = np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'] - expt_cfg['freq_step'], expt_cfg['freq_step'])
        freq_step = expt_cfg['freq_step']
        freq_substep = expt_cfg['freq_substep']
        acquisition_num = expt_cfg['acquisition_num']


        experiment_name = 'cavity_drive_direct_spectroscopy'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_direct_spectroscopy_freq_sweep', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences(experiment_name)

        for freq in freq_starts:

            experiment_cfg[experiment_name]['start'] = freq
            experiment_cfg[experiment_name]['stop'] = freq + freq_step
            experiment_cfg[experiment_name]['step'] = freq_substep
            experiment_cfg[experiment_name]['acquisition_num'] = acquisition_num

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



    def resonator_spectroscopy_mixedtones(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'resonator_spectroscopy_mixedtones'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_mixedtones', suffix='.h5'))
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


    def cavity_drive_resonator_spectroscopy_mixedtones(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'cavity_drive_resonator_spectroscopy_mixedtones'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_resonator_spectroscopy_mixedtones', suffix='.h5'))
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

        # maybe initialize Is, Qs here

        for atten in np.arange(swp_cfg['pwr_start'], swp_cfg['pwr_stop'], swp_cfg['pwr_step']):
            print("Attenuation set to ", atten, 'dB')
            Is_t = []
            Qs_t = []
            for freq in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
                quantum_device_cfg['readout']['freq'] = freq
                quantum_device_cfg['readout']['dig_atten'] = atten
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                Is_t.append(I)
                Qs_t.append(Q)

            self.Is.append(np.array(Is_t))
            self.Qs.append(np.array(Qs_t))

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        self.freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])


    def qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        experiment_name = 'ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qubit_temperature', suffix='.h5'))

        for ge_pi in [True,False]:

            experiment_cfg['ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['ef_rabi']['acquisition_num'] = 15000
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

    def amplitude_time_rabi(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['amplitude_time_rabi']
        amps = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'amplitude_time_rabi', suffix='.h5'))

        for amp in amps:
            expt_cfg['amp'] = amp
            expt_cfg['start'] = swp_cfg['starttime']
            expt_cfg['stop'] = swp_cfg['stoptime']
            expt_cfg['step'] = swp_cfg['steptime']
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_rabi_freq_scan_vstime(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['cavity_drive_rabi_freq_scan_vstime']
        freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])



        experiment_name = 'cavity_drive_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_rabi_freq_scan_vstime', suffix='.h5'))

        for dfreq in freqs:

            quantum_device_cfg['cavity']['1']['freq'] = dfreq
            print("Sideband freq set to", quantum_device_cfg['cavity']['1']['freq'])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)
        # Added by MGP, 1/15/2020
        print(seq_data_file, "data file name")

    def cavity_drive_rabi_freq_scan_vstime_mixedtones(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        import time
        swp_cfg = experiment_cfg['cavity_drive_rabi_freq_scan_vstime_mixedtones']
        freqs = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'cavity_drive_rabi_mixedtones_withf0g1_fromfreqscan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'cavity_drive_rabi_freq_scan_vstime_mixedtones', suffix='.h5'))

        for dfreq in freqs:
            #quantum_device_cfg['cavity']['1']['freq'] = dfreq
            #print("Sideband freq set to", quantum_device_cfg['cavity']['1']['freq'])
            quantum_device_cfg['cavity']['1']['f0g1AddFreq']=dfreq
            print('f0g1 AddFreq set to', quantum_device_cfg['cavity']['1']['f0g1AddFreq'])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
            #time.sleep(30)
            # added sleept time to try to address tqdm error where thread joining failed, possibly due to some overlap?

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_twotonerabi_geramsey_sequential(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['cavity_drive_twotonerabi_geramsey_sequential']
        times = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])



        experiment_name = 'cavity_drive_twotonerabi_geramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_twotonerabi_geramsey_sequential', suffix='.h5'))

        t = swp_cfg['start']
        while t < swp_cfg['stop']:
            expt_cfg['start'] = t
            expt_cfg['stop'] = t + 75*swp_cfg['step']
            expt_cfg['step'] = swp_cfg['step']

            print("Sideband freq set to", quantum_device_cfg['cavity']['1']['freq'])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
            t = t + 75*swp_cfg['step'] + swp_cfg['step']

        self.Is = np.array(self.Is)
        self.Qs = np.array(self.Qs)

    def cavity_drive_f0g1_geramsey_sequential(self, quantum_device_cfg, experiment_cfg, hardware_cfg, path):
        swp_cfg = experiment_cfg['cavity_drive_f0g1_geramsey_sequential']
        times = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])



        experiment_name = 'cavity_drive_rabi_geramsey'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'cavity_drive_f0g1_geramsey_sequential', suffix='.h5'))

        t = swp_cfg['start']
        while t < swp_cfg['stop']:
            expt_cfg['start'] = t
            if t + 80*swp_cfg['step']<swp_cfg['stop']:
                expt_cfg['stop'] = t + 80*swp_cfg['step']
            else:
                expt_cfg['stop'] = swp_cfg['stop']
            expt_cfg['step'] = swp_cfg['step']

            print("Sideband freq set to", quantum_device_cfg['cavity']['1']['freq'])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            I, Q = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.Is.append(I)
            self.Qs.append(Q)
            t = expt_cfg['stop'] + swp_cfg['step']

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

        experiment_name = 'sideband_rabi_freq_scan'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_rabi_freq_scan', suffix='.h5'))

        for ii,freq in enumerate(freqlist):

            experiment_cfg[experiment_name]['freq'] = freq
            experiment_cfg[experiment_name]['amp'] = amplist[ii]
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

        experiment_name = 'sideband_rabi'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sequential_sideband_rabis', suffix='.h5'))

        for ii,freq in enumerate(freqlist):

            experiment_cfg[experiment_name]['freq'] = freq
            experiment_cfg[experiment_name]['amp'] = amplist[ii]
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
            hardware_cfg['trigger']['period_us'] = int(
                quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode] * 5)
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
            hardware_cfg['trigger']['period_us'] = int(
                quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode] * 5)
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
            hardware_cfg['trigger']['period_us'] = int(quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode]*5)
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
            hardware_cfg['trigger']['period_us'] = int(quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode]*5)
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
            hardware_cfg['trigger']['period_us'] = int(
                quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode] * 5)
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
            hardware_cfg['trigger']['period_us'] = int(
                quantum_device_cfg['flux_pulse_info'][expt_cfg['on_qubits'][0]]['t1s'][mode] * 5)
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


    def analyze(self,quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,Is,Qs,P='Q'):
        PA = PostExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, Is,Qs,P,show)
