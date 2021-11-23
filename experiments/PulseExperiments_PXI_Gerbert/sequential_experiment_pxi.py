import copy

from slab.experiments.PulseExperiments_PXI_Gerbert.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI_Gerbert import keysight_pxi_load as ks_pxi
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin
try:from skopt import Optimizer
except:print("No optimizer")
from slab.experiments.PulseExperiments_PXI_Gerbert.PostExperimentAnalysis import PostExperimentAnalyzeAndSave


# from slab.experiments.PulseExperiments.get_data import get_singleshot_data_two_qubits_4_calibration_v2,\
#     get_singleshot_data_two_qubits, data_to_correlators, two_qubit_quantum_state_tomography,\
#     density_matrix_maximum_likelihood

import pickle

class SequentialExperiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,lattice_cfg,experiment_name, path,analyze = False,show=True,P = 'Q'):

        self.seq_data = []
        
        eval('self.' + experiment_name)(quantum_device_cfg, experiment_cfg, hardware_cfg,lattice_cfg,path)
        if analyze:
            try:
                #haven't fixed post-experiment analysis
                self.analyze(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,self.data,P = 'I')
            except: print ("No post expt analysis")
        else:pass

    def t1rho_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        expt_cfg = experiment_cfg['t1rho_sweep']
        data_path = os.path.join(path, 'data/')


        amparray = np.arange(expt_cfg['ampstart'],expt_cfg['ampstop'],expt_cfg['ampstep'])
        print(amparray)

        experiment_name = 't1rho_sweep'
        print("Sequences generated")
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 't1rho_sweep', suffix='.h5'))
        for ampval in amparray:
            experiment_name = 't1rho'
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            experiment_cfg['t1rho']['amp']=ampval
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
            self.seq_data.append(data)

        self.seq_data = np.array(self.data)

    def ff_ramp_down_cal_ppiq(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        expt_cfg = experiment_cfg['ff_ramp_down_cal_ppiq']
        data_path = os.path.join(path, 'data/')


        dt_array = np.arange(expt_cfg['dt_start'],expt_cfg['dt_stop'],expt_cfg['dt_step'])
        experiment_name = 'ff_ramp_down_cal_ppiq'
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'ff_ramp_down_cal_ppiq', suffix='.h5'))

        lattice_temp = copy.copy(lattice_cfg)
        on_qb = expt_cfg["on_qbs"][0]
        on_qb_setup = lattice_cfg["qubit"]["setup"][on_qb]
        if "on_rds" in expt_cfg.keys():
            on_rd = expt_cfg["on_rds"][0]
        else:
            on_rd = on_qb
        on_rd_setup = lattice_cfg["qubit"]["setup"][on_rd]

        no_flx_rd = lattice_temp["readout"][on_rd_setup]["freq"][on_rd]
        no_flx_qb = lattice_temp["qubit"]["freq"][on_qb]
        flx_rd = expt_cfg["rd_freq_flux"]
        flx_qb = expt_cfg["qb_freq_flux"]

        if lattice_cfg["pulse_info"]["pulse_type"][on_qb]=="square":
            zero_pt = expt_cfg["qb_pulse_length"]/2
        else:
            zero_pt = expt_cfg["qb_pulse_length"]*2
        for delt in dt_array:
            if delt<-zero_pt:
                lattice_cfg["readout"][on_rd_setup]["freq"][on_rd] = flx_rd
                lattice_cfg["qubit"]["freq"][on_qb] = flx_qb
            else:
                lattice_cfg["readout"][on_rd_setup]["freq"][on_rd] = no_flx_rd
                lattice_cfg["qubit"]["freq"][on_qb] = no_flx_qb

            experiment_cfg[experiment_name]['delt'] = float(delt)
            # print(experiment_cfg[experiment_name])
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def ff_track_traj(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'ff_track_traj'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')


        per_array = np.arange(expt_cfg['perc_flux_vec_start'],expt_cfg['perc_flux_vec_stop'],expt_cfg['perc_flux_vec_step'])

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))

        for per in per_array:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name, perc_flux_vec=per)
            print("Sequences generated")
            #
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def ff_sweep_j(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'ff_sweep_j'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')


        per_array = np.arange(expt_cfg['per_start'],expt_cfg['per_stop'],expt_cfg['per_step'])

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))

        for per in per_array:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name, perc_flux_vec=per)
            print("Sequences generated")
            #
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def measure_lattice_state(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'measure_lattice_state'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))

        Mott_qbs = expt_cfg["Mott_qbs"]
        qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
        lo_qb_temp_ind = np.argmax(qb_freq_list)
        lo_qb = Mott_qbs[lo_qb_temp_ind]

        for rd_qb in expt_cfg["rd_qbs"]:
            quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
            quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name, rd_qb=rd_qb)
            print("Sequences generated")
            #
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def histogram_sweep(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):

        expt_cfg = experiment_cfg['histogram_sweep']
        sweep_amp = expt_cfg['sweep_amp']
        attens = np.arange(expt_cfg['atten_start'],expt_cfg['atten_stop'],expt_cfg['atten_step'])
        freqs = np.arange(expt_cfg['freq_start'],expt_cfg['freq_stop'],expt_cfg['freq_step'])

        experiment_name = 'histogram'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram_sweep', suffix='.h5'))

        for qb in quantum_device_cfg["setups"]:
            if sweep_amp:
                for att in attens:
                    quantum_device_cfg['powers'][qb]['readout_drive_digital_attenuation'] = att
                    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
                    sequences = ps.get_experiment_sequences(experiment_name)
                    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                    data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                    self.seq_data.append(data)
            else:
                for freq in freqs:
                    quantum_device_cfg['readout'][qb]['freq'] = freq
                    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
                    sequences = ps.get_experiment_sequences(experiment_name)
                    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                    data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                    self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def qp_pumping_t1_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):

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
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)


        self.seq_data = np.array(self.seq_data)

    def resonator_spectroscopy(self,quantum_device_cfg, experiment_cfg, hardware_cfg,lattice_cfg, path):
        experiment_name = 'resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg,lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            read_freq = copy.deepcopy(quantum_device_cfg['readout'][qb]['freq'])
            for freq in np.arange(expt_cfg['start']+read_freq, expt_cfg['stop']+read_freq, expt_cfg['step']):
                quantum_device_cfg['readout'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def resonator_spectroscopy_pi(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        """Runs res spec with pi pulse before resonator pulse, so you can measure chi by comparing to normal res_spec"""
        experiment_name = 'resonator_spectroscopy_pi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_pi', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            read_freq = quantum_device_cfg['readout'][qb]['efreq']
            for freq in np.arange(expt_cfg['start'] + read_freq, expt_cfg['stop'] + read_freq, expt_cfg['step']):
                quantum_device_cfg['readout'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def ff_resonator_spectroscopy(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'ff_resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            read_freq = quantum_device_cfg['readout'][qb]['freq']
            for freq in np.arange(expt_cfg['start']+read_freq, expt_cfg['stop']+read_freq, expt_cfg['step']):
                quantum_device_cfg['readout'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def ff_resonator_spectroscopy_pi(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        """Runs res spec with pi pulse before resonator pulse, so you can measure chi by comparing to normal res_spec"""
        experiment_name = 'ff_resonator_spectroscopy_pi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_pi', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            read_freq = quantum_device_cfg['readout'][qb]['efreq']
            for freq in np.arange(expt_cfg['start'] + read_freq, expt_cfg['stop'] + read_freq, expt_cfg['step']):
                quantum_device_cfg['readout'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def resonator_spectroscopy_ef_pi(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        """Runs res spec with pi pulse on e/f before resonator pulse, so you can measure chi/contrast by comparing to normal res_spec"""
        experiment_name = 'resonator_spectroscopy_ef_pi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy_ef_pi', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            read_freq = quantum_device_cfg['readout'][qb]['ffreq']
            for freq in np.arange(expt_cfg['start'] + read_freq, expt_cfg['stop'] + read_freq, expt_cfg['step']):
                quantum_device_cfg['readout'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def rabi_chevron(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'rabi_chevron'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'rabi_chevron', suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            qb_freq = copy.deepcopy(quantum_device_cfg['qubit'][qb]['freq'])
            for freq in np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])+qb_freq:
                quantum_device_cfg['qubit'][qb]['freq'] = freq
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def resonator_spectroscopy_power_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        sweep_expt_name = 'resonator_spectroscopy_power_sweep'
        swp_cfg = experiment_cfg[sweep_expt_name]

        experiment_name = 'resonator_spectroscopy'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, sweep_expt_name, suffix='.h5'))
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)

        for qb in quantum_device_cfg["setups"]:
            for atten in np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step']):
                print("Attenuation set to ", atten, 'dB')
                quantum_device_cfg['powers'][qb]['readout_drive_digital_attenuation']= atten
                data_t = []
                for freq in np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step']):
                    quantum_device_cfg['readout'][qb]['freq'] = freq
                    sequences = ps.get_experiment_sequences(experiment_name)
                    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                    data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                    data_t.append(data)

            self.seq_data.append(np.array(data_t))

        self.seq_data = np.array(self.seq_data)


    def qubit_temperature(self,quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        experiment_name = 'ef_rabi'
        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qubit_temperature', suffix='.h5'))

        for ge_pi in [True,False]:

            experiment_cfg['ef_rabi']['ge_pi'] = ge_pi
            if ge_pi:pass
            else:experiment_cfg['ef_rabi']['acquisition_num'] = 5000
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def pulse_probe_delay_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        swp_cfg = experiment_cfg['pulse_probe_delay_sweep']
        delays = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'pulse_probe_iq'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_delay_sweep', suffix='.h5'))

        for delay in delays:

            experiment_cfg["pulse_probe_iq"]["delay"] = delay
            print("delay set to", delay)
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
            sequences = ps.get_experiment_sequences(experiment_name)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
            self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)

    def pulse_probe_atten_sweep(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, path):
        swp_cfg = experiment_cfg['pulse_probe_atten_sweep']
        attens = np.arange(swp_cfg['start'], swp_cfg['stop'], swp_cfg['step'])

        experiment_name = 'pulse_probe_iq'

        expt_cfg = experiment_cfg[experiment_name]
        data_path = os.path.join(path, 'data/')

        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_atten_sweep', suffix='.h5'))

        for qb in quantum_device_cfg["setups"]:
            for atten in attens:

                quantum_device_cfg['powers'][qb]["qubit_drive_digital_attenuation"] = atten
                print("atten set to", atten)
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
                sequences = ps.get_experiment_sequences(experiment_name)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                data = exp.run_experiment_pxi(sequences, path, experiment_name, seq_data_file=seq_data_file)
                self.seq_data.append(data)

        self.seq_data = np.array(self.seq_data)


    #TODO: haven't fixed post-analysis
    def analyze(self,quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name,show,Is,Qs,P='Q'):
        PA = PostExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, experiment_name, Is,Qs,P,show)
