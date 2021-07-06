from slab.experiments.PulseExperiments.sequences import PulseSequences
from slab.experiments.PulseExperiments.pulse_experiment import Experiment
import numpy as np
import os
import time
from tqdm import tqdm
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin

from skopt import Optimizer

from slab.experiments.PulseExperiments.get_data import get_singleshot_data_two_qubits_4_calibration_v2,\
    get_singleshot_data_two_qubits, data_to_correlators, two_qubit_quantum_state_tomography,\
    density_matrix_maximum_likelihood

import pickle

def histogram(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['histogram']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    lo_freq = {"1": quantum_device_cfg['heterodyne']['1']['lo_freq'],
               "2": quantum_device_cfg['heterodyne']['2']['lo_freq']}

    for amp in np.arange(expt_cfg['amp_start'], expt_cfg['amp_stop'], expt_cfg['amp_step']):
        for qubit_id in on_qubits:
            quantum_device_cfg['heterodyne'][qubit_id]['amp'] = amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')
        update_awg = True
        for lo_freq_delta in np.arange(expt_cfg['lo_freq_delta_start'], expt_cfg['lo_freq_delta_stop'], expt_cfg['lo_freq_delta_step']):
            for qubit_id in on_qubits:
                quantum_device_cfg['heterodyne'][qubit_id]['lo_freq'] = lo_freq[qubit_id] + lo_freq_delta
                print('lo_freq_delta: ',lo_freq_delta,' GHz, amp: ',amp)

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'histogram', seq_data_file, update_awg)

            update_awg = False

def readout_time_optimization(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['readout_time_optimization']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'readout_time_optimization', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['histogram']['states'] = expt_cfg['states']
    experiment_cfg['histogram']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['histogram']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['histogram']['singleshot'] = expt_cfg['singleshot']
    experiment_cfg['histogram']['flux_probe'] = expt_cfg['flux_probe']

    for readout_time in np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step']):
        print('readout time is',readout_time)
        quantum_device_cfg['alazar_readout']['width'] = int(readout_time)
        quantum_device_cfg['alazar_readout']['window'] = [0,int(readout_time)]
        for qubit_id in on_qubits:
            quantum_device_cfg['heterodyne'][qubit_id]['length'] = readout_time
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'readout_time_optimization', seq_data_file, update_awg)

        update_awg = False

def readout_iq_freq_optimization(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['readout_iq_freq_optimization']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'readout_iq_freq_optimization', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    lo_freq = {"1": quantum_device_cfg['heterodyne']['1']['lo_freq'],
               "2": quantum_device_cfg['heterodyne']['2']['lo_freq']}

    iq_freq = {"1": quantum_device_cfg['heterodyne']['1']['freq'],
               "2": quantum_device_cfg['heterodyne']['2']['freq']}

    experiment_cfg['histogram']['states'] = expt_cfg['states']
    experiment_cfg['histogram']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['histogram']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['histogram']['singleshot'] = expt_cfg['singleshot']
    experiment_cfg['histogram']['flux_probe'] = expt_cfg['flux_probe']

    for iq_freq_delta in np.arange(expt_cfg['iq_freq_delta_start'], expt_cfg['iq_freq_delta_stop'], expt_cfg['iq_freq_delta_step']):
        print('iq_freq_delta is',iq_freq_delta)
        for qubit_id in on_qubits:
            quantum_device_cfg['heterodyne'][qubit_id]['freq'] = iq_freq[qubit_id]+iq_freq_delta
            quantum_device_cfg['heterodyne'][qubit_id]['lo_freq'] = lo_freq[qubit_id]+iq_freq_delta
            print('readout freq is', quantum_device_cfg['heterodyne'][qubit_id]['lo_freq']-quantum_device_cfg['heterodyne'][qubit_id]['freq'])
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'readout_iq_freq_optimization', seq_data_file, update_awg)

        update_awg = False

def readout_lo_power_optimization(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['readout_lo_power_optimization']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'readout_lo_power_optimization', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    lo_power = {"1": quantum_device_cfg['heterodyne']['1']['lo_power'],
               "2": quantum_device_cfg['heterodyne']['2']['lo_power']}


    experiment_cfg['histogram']['states'] = expt_cfg['states']
    experiment_cfg['histogram']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['histogram']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['histogram']['singleshot'] = expt_cfg['singleshot']
    experiment_cfg['histogram']['flux_probe'] = expt_cfg['flux_probe']

    for lo_power_delta in np.arange(expt_cfg['lo_power_delta_start'], expt_cfg['lo_power_delta_stop'], expt_cfg['lo_power_delta_step']):
        print('lo_power_delta is',lo_power_delta)
        for qubit_id in on_qubits:
            quantum_device_cfg['heterodyne'][qubit_id]['lo_power'] = lo_power[qubit_id]+lo_power_delta
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'readout_lo_power_optimization', seq_data_file, update_awg)

        update_awg = False

def qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
    sequences = ps.get_experiment_sequences('ramsey')
    expt_cfg = experiment_cfg['ramsey']
    uncalibrated_qubits = list(expt_cfg['on_qubits'])
    for ii in range(3):
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'ramsey')
        expt_cfg = experiment_cfg['ramsey']
        on_qubits = expt_cfg['on_qubits']
        expt_pts = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
        with SlabFile(data_file) as a:
            for qubit_id in on_qubits:
                data_list = np.array(a['expt_avg_data_ch%s' % qubit_id])
                fitdata = fitdecaysin(expt_pts, data_list, showfit=False)
                qubit_freq = quantum_device_cfg['qubit']['%s' % qubit_id]['freq']
                ramsey_freq = experiment_cfg['ramsey']['ramsey_freq']
                real_qubit_freq = qubit_freq + ramsey_freq - fitdata[1]
                possible_qubit_freq = qubit_freq + ramsey_freq + fitdata[1]
                flux_offset = -(real_qubit_freq - qubit_freq) / quantum_device_cfg['freq_flux'][qubit_id][
                    'freq_flux_slope']
                suggested_flux = round(quantum_device_cfg['freq_flux'][qubit_id]['current_mA'] + flux_offset, 4)
                print('qubit %s' %qubit_id)
                print('original qubit frequency:' + str(qubit_freq) + " GHz")
                print('Decay Time: %s ns' % (fitdata[3]))
                print("Oscillation frequency: %s GHz" % str(fitdata[1]))
                print("Suggested qubit frequency: %s GHz" % str(real_qubit_freq))
                print("possible qubit frequency: %s GHz" % str(possible_qubit_freq))
                print("Suggested flux: %s mA" % str(suggested_flux))
                print("Max contrast: %s" % str(max(data_list)-min(data_list)))

                freq_offset = ramsey_freq - fitdata[1]

                if (abs(freq_offset) < 100e-6):
                    print("Frequency is within expected value. No further calibration required.")
                    if qubit_id in uncalibrated_qubits: uncalibrated_qubits.remove(qubit_id)
                elif (abs(flux_offset) < 0.01):
                    print("Changing flux to the suggested flux: %s mA" % str(suggested_flux))
                    quantum_device_cfg['freq_flux'][qubit_id]['current_mA'] = suggested_flux
                else:
                    print("Large change in flux is required; please do so manually")
                    return

                if uncalibrated_qubits == []:
                    print("All qubits frequency calibrated.")
                    with open(os.path.join(path, 'quantum_device_config.json'), 'w') as f:
                        json.dump(quantum_device_cfg, f)
                    return

def pulse_probe_flux_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['pulse_probe_flux_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_flux_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['pulse_probe']['pulse_amp'] = expt_cfg['pulse_amp']
    experiment_cfg['pulse_probe']['pulse_length'] = expt_cfg['pulse_length']
    experiment_cfg['pulse_probe']['start'] = expt_cfg['freq_start']
    experiment_cfg['pulse_probe']['stop'] = expt_cfg['freq_stop']
    experiment_cfg['pulse_probe']['step'] = expt_cfg['freq_step']
    experiment_cfg['pulse_probe']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['pulse_probe']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['pulse_probe']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['pulse_probe']['singleshot'] = expt_cfg['singleshot']
    experiment_cfg['pulse_probe']['flux_probe'] = expt_cfg['flux_probe']
    experiment_cfg['pulse_probe']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['pulse_probe']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['pulse_probe']['flux_pi_calibration'] = expt_cfg['flux_pi_calibration']

    if expt_cfg['use_qubit_spec'][0]:
        for qubit_id in on_qubits:
            p5 = np.poly1d(np.polyfit(quantum_device_cfg['qubit_spec'][qubit_id]['current_mA'], quantum_device_cfg['qubit_spec'][qubit_id]['freq'], 5))

    for flux2 in np.arange(expt_cfg['flux2_start'], expt_cfg['flux2_stop'], expt_cfg['flux2_step']):
        for flux1 in np.arange(expt_cfg['flux1_start'], expt_cfg['flux1_stop'], expt_cfg['flux1_step']):

            if expt_cfg['use_qubit_spec'][0]:
                qubit_freq = p5(flux1)
                experiment_cfg['pulse_probe']['start'] = qubit_freq - expt_cfg['use_qubit_spec'][1]/2
                experiment_cfg['pulse_probe']['stop'] = qubit_freq + expt_cfg['use_qubit_spec'][1]/2

            quantum_device_cfg['freq_flux']['1']['current_mA'] = flux1
            quantum_device_cfg['freq_flux']['2']['current_mA'] = flux2

            print('flux is', [flux1, flux2])
            print('freq range is', [experiment_cfg['pulse_probe']['start'], experiment_cfg['pulse_probe']['stop']])

            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('pulse_probe')
            update_awg = True

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'pulse_probe_flux_sweep', seq_data_file, update_awg)

            update_awg = False

def pulse_probe_power_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['pulse_probe_power_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'pulse_probe_power_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    for amp in np.arange(expt_cfg['amp_start'], expt_cfg['amp_stop'], expt_cfg['amp_step']):
        experiment_cfg['pulse_probe']['pulse_amp'] = amp
        experiment_cfg['pulse_probe']['pulse_length'] = expt_cfg['pulse_length']
        experiment_cfg['pulse_probe']['start'] = expt_cfg['freq_start']
        experiment_cfg['pulse_probe']['stop'] = expt_cfg['freq_stop']
        experiment_cfg['pulse_probe']['step'] = expt_cfg['freq_step']
        experiment_cfg['pulse_probe']['acquisition_num'] = expt_cfg['acquisition_num']
        experiment_cfg['pulse_probe']['on_qubits'] = expt_cfg['on_qubits']
        experiment_cfg['pulse_probe']['ge_pi'] = expt_cfg['ge_pi']
        experiment_cfg['pulse_probe']['singleshot'] = expt_cfg['singleshot']
        experiment_cfg['pulse_probe']['flux_probe'] = expt_cfg['flux_probe']
        experiment_cfg['pulse_probe']['pi_calibration'] = expt_cfg['pi_calibration']
        experiment_cfg['pulse_probe']['calibration_qubit'] = expt_cfg['calibration_qubit']
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('pulse_probe')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'pulse_probe_power_sweep', seq_data_file, update_awg)

        update_awg = False

def sideband_rabi_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['sideband_rabi']['amp'] = expt_cfg['amp']
    experiment_cfg['sideband_rabi']['start'] = expt_cfg['time_start']
    experiment_cfg['sideband_rabi']['stop'] = expt_cfg['time_stop']
    experiment_cfg['sideband_rabi']['step'] = expt_cfg['time_step']
    experiment_cfg['sideband_rabi']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['sideband_rabi']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['sideband_rabi']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['sideband_rabi']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['sideband_rabi']['flux_pi_calibration'] = expt_cfg['flux_pi_calibration']
    experiment_cfg['sideband_rabi']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['sideband_rabi']['ef_pi'] = expt_cfg['ef_pi']
    experiment_cfg['sideband_rabi']['flux_line'] = expt_cfg['flux_line']

    for freq in np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step']):

        experiment_cfg['sideband_rabi']['freq'] = freq

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_sweep', seq_data_file, update_awg)

        update_awg = False


def sideband_rabi_drive_both_flux_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_drive_both_flux_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_drive_both_flux_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['sideband_rabi_drive_both_flux']['amp'] = expt_cfg['amp']
    experiment_cfg['sideband_rabi_drive_both_flux']['start'] = expt_cfg['time_start']
    experiment_cfg['sideband_rabi_drive_both_flux']['stop'] = expt_cfg['time_stop']
    experiment_cfg['sideband_rabi_drive_both_flux']['step'] = expt_cfg['time_step']
    experiment_cfg['sideband_rabi_drive_both_flux']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['sideband_rabi_drive_both_flux']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['sideband_rabi_drive_both_flux']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['sideband_rabi_drive_both_flux']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_pi_calibration'] = expt_cfg['flux_pi_calibration']
    experiment_cfg['sideband_rabi_drive_both_flux']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['sideband_rabi_drive_both_flux']['ef_pi'] = expt_cfg['ef_pi']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_line'] = expt_cfg['flux_line']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_pulse'] = expt_cfg['flux_pulse']
    experiment_cfg['sideband_rabi_drive_both_flux']['phase'] = expt_cfg['phase']

    for index, freq in enumerate(np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        experiment_cfg['sideband_rabi_drive_both_flux']['freq'] = freq

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi_drive_both_flux')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_drive_both_flux_sweep', seq_data_file, update_awg)

        update_awg = False


def sideband_rabi_drive_both_flux_phase_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_drive_both_flux_phase_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_drive_both_flux_phase_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['sideband_rabi_drive_both_flux']['amp'] = expt_cfg['amp']
    experiment_cfg['sideband_rabi_drive_both_flux']['start'] = expt_cfg['time_start']
    experiment_cfg['sideband_rabi_drive_both_flux']['stop'] = expt_cfg['time_stop']
    experiment_cfg['sideband_rabi_drive_both_flux']['step'] = expt_cfg['time_step']
    experiment_cfg['sideband_rabi_drive_both_flux']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['sideband_rabi_drive_both_flux']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['sideband_rabi_drive_both_flux']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['sideband_rabi_drive_both_flux']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_pi_calibration'] = expt_cfg['flux_pi_calibration']
    experiment_cfg['sideband_rabi_drive_both_flux']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['sideband_rabi_drive_both_flux']['ef_pi'] = expt_cfg['ef_pi']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_line'] = expt_cfg['flux_line']
    experiment_cfg['sideband_rabi_drive_both_flux']['flux_pulse'] = expt_cfg['flux_pulse']
    experiment_cfg['sideband_rabi_drive_both_flux']['freq'] = expt_cfg['freq']

    for index, phase in enumerate(np.arange(expt_cfg['phase_start'], expt_cfg['phase_stop'], expt_cfg['phase_step'])):

        phase1 = quantum_device_cfg['flux_pulse_info']['1']['pulse_phase'] + phase
        phase2 = quantum_device_cfg['flux_pulse_info']['2']['pulse_phase'] + phase + 0
        experiment_cfg['sideband_rabi_drive_both_flux']['phase'][0] = phase1
        experiment_cfg['sideband_rabi_drive_both_flux']['phase'][1] = phase2
        print('Index: %s Phase1: %s' %(index, phase1))
        print('Index: %s Phase2: %s' %(index, phase2))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi_drive_both_flux')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_drive_both_flux_phase_sweep', seq_data_file, update_awg)

        update_awg = False


def t1rho_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['t1rho_amp_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 't1rho_amp_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['t1rho']['start'] = expt_cfg['time_start']
    experiment_cfg['t1rho']['stop'] = expt_cfg['time_stop']
    experiment_cfg['t1rho']['step'] = expt_cfg['time_step']
    experiment_cfg['t1rho']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['t1rho']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['t1rho']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['t1rho']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['t1rho']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['t1rho']['ef_pi'] = expt_cfg['ef_pi']
    experiment_cfg['t1rho']['flux_pi'] = expt_cfg['flux_pi']
    experiment_cfg['t1rho']['use_freq_amp_halfpi'] = expt_cfg['use_freq_amp_halfpi']
    experiment_cfg['t1rho']['mid_phase'] = expt_cfg['mid_phase']

    if expt_cfg['log_spacing']:
        num_pts = (expt_cfg['amp_stop']-expt_cfg['amp_start'])/expt_cfg['amp_step']  + 1
        amp_arr = np.linspace(np.log10(expt_cfg['amp_start']), np.log10(expt_cfg['amp_stop']), num_pts)
        amp_arr = 10**amp_arr
    else:
        amp_arr = np.arange(expt_cfg['amp_start'], expt_cfg['amp_stop'], expt_cfg['amp_step'])

    for index, amp in enumerate(amp_arr):

        experiment_cfg['t1rho']['amp'] = amp
        print('Index: %s amp: %s' %(index,amp))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('t1rho')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 't1rho_amp_sweep', seq_data_file, update_awg)

        update_awg = False


def t1rho_mid_phase_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['t1rho_mid_phase_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 't1rho_mid_phase_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['t1rho']['amp'] = expt_cfg['amp']
    experiment_cfg['t1rho']['start'] = expt_cfg['time_start']
    experiment_cfg['t1rho']['stop'] = expt_cfg['time_stop']
    experiment_cfg['t1rho']['step'] = expt_cfg['time_step']
    experiment_cfg['t1rho']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['t1rho']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['t1rho']['calibration_qubit'] = expt_cfg['calibration_qubit']
    experiment_cfg['t1rho']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['t1rho']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['t1rho']['ef_pi'] = expt_cfg['ef_pi']
    experiment_cfg['t1rho']['flux_pi'] = expt_cfg['flux_pi']
    experiment_cfg['t1rho']['use_freq_amp_halfpi'] = expt_cfg['use_freq_amp_halfpi']

    for index, phase in enumerate(np.arange(expt_cfg['phase_start'], expt_cfg['phase_stop'], expt_cfg['phase_step'])):

        experiment_cfg['t1rho']['mid_phase'] = phase
        print('Index: %s amp: %s' %(index,phase))

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('t1rho')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 't1rho_mid_phase_sweep', seq_data_file, update_awg)

        update_awg = False


def rabi_while_flux_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['rabi_while_flux_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'rabi_while_flux_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    experiment_cfg['rabi_while_flux']['amp'] = expt_cfg['amp']
    experiment_cfg['rabi_while_flux']['start'] = expt_cfg['time_start']
    experiment_cfg['rabi_while_flux']['stop'] = expt_cfg['time_stop']
    experiment_cfg['rabi_while_flux']['step'] = expt_cfg['time_step']
    experiment_cfg['rabi_while_flux']['acquisition_num'] = expt_cfg['acquisition_num']
    experiment_cfg['rabi_while_flux']['on_qubits'] = expt_cfg['on_qubits']
    experiment_cfg['rabi_while_flux']['pi_calibration'] = expt_cfg['pi_calibration']
    experiment_cfg['rabi_while_flux']['flux_pi_calibration'] = expt_cfg['flux_pi_calibration']
    experiment_cfg['rabi_while_flux']['ge_pi'] = expt_cfg['ge_pi']
    experiment_cfg['rabi_while_flux']['flux_line'] = expt_cfg['flux_line']
    experiment_cfg['rabi_while_flux']['flux_freq'] = expt_cfg['flux_freq']
    experiment_cfg['rabi_while_flux']['phase'] = expt_cfg['phase']

    for ii, freq in enumerate(np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step'])):

        print('Index %s: rabi_drive_freq: %s' %(ii, freq))
        experiment_cfg['rabi_while_flux']['qubit_freq'] = freq

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('rabi_while_flux')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'rabi_while_flux_sweep', seq_data_file, update_awg)

        update_awg = False


def sideband_rabi_around_mm(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_around_mm']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_around_mm', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    mm_freq_list_1 = [quantum_device_cfg['multimodes']['1']['freq'][-1]]
    freq_list_all_1 = []
    for mm_freq in mm_freq_list_1:
        freq_list_all_1 += [np.arange(mm_freq-expt_cfg['freq_range'],mm_freq+expt_cfg['freq_range'],expt_cfg['step'])]

    freq_array_1 = np.hstack(np.array(freq_list_all_1))


    mm_freq_list_2 = [quantum_device_cfg['multimodes']['2']['freq'][-1]]
    freq_list_all_2 = []
    for mm_freq in mm_freq_list_2:
        freq_list_all_2 += [np.arange(mm_freq-expt_cfg['freq_range'],mm_freq+expt_cfg['freq_range'],expt_cfg['step'])]

    freq_array_2 = np.hstack(np.array(freq_list_all_2))
    print(freq_array_2)
    last_freq_1 = 0

    for freq_1, freq_2 in zip(freq_array_1,freq_array_2):

        # calibrate qubit frequency everytime changing target mm
        if freq_1 - last_freq_1 > 2*expt_cfg['step']:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)

        last_freq_1 = freq_1

        experiment_cfg['sideband_rabi_2_freq']['freq_1'] = freq_1
        experiment_cfg['sideband_rabi_2_freq']['freq_2'] = freq_2
        experiment_cfg['sideband_rabi_2_freq']['amp'] = expt_cfg['amp']
        experiment_cfg['sideband_rabi_2_freq']['start'] = expt_cfg['time_start']
        experiment_cfg['sideband_rabi_2_freq']['stop'] = expt_cfg['time_stop']
        experiment_cfg['sideband_rabi_2_freq']['step'] = expt_cfg['time_step']
        experiment_cfg['sideband_rabi_2_freq']['acquisition_num'] = expt_cfg['acquisition_num']
        experiment_cfg['sideband_rabi_2_freq']['on_qubits'] = expt_cfg['on_qubits']
        experiment_cfg['sideband_rabi_2_freq']['pi_calibration'] = expt_cfg['pi_calibration']
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi_2_freq')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_around_mm', seq_data_file, update_awg)

        update_awg = False


def multimode_rabi_pi_calibrate(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['multimode_rabi']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']

    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            experiment_cfg['multimode_rabi']['on_mms'][qubit_id] = int(on_mms)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('multimode_rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'multimode_rabi')
        expt_pts = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
        with SlabFile(data_file) as a:
            for qubit_id in on_qubits:
                expt_avg_data = np.array(a['expt_avg_data_ch%s'%qubit_id])

                data_list = np.array(a['expt_avg_data_ch%s'%qubit_id])
                fitdata = fitdecaysin(expt_pts[1:],data_list[1:],showfit=True)

                pi_length = round((-fitdata[2]%180 + 90)/(360*fitdata[1]),5)
                print("Flux pi length ge: %s" %pi_length)

                quantum_device_cfg['multimodes'][qubit_id]['pi_len'][on_mms] = pi_length
                with open(os.path.join(path, 'quantum_device_config.json'), 'w') as f:
                    json.dump(quantum_device_cfg, f)


def multimode_ef_rabi_pi_calibrate(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['multimode_ef_rabi']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']

    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            experiment_cfg['multimode_ef_rabi']['on_mms'][qubit_id] = int(on_mms)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('multimode_ef_rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'multimode_ef_rabi')
        expt_pts = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
        with SlabFile(data_file) as a:
            for qubit_id in on_qubits:
                expt_avg_data = np.array(a['expt_avg_data_ch%s'%qubit_id])

                data_list = np.array(a['expt_avg_data_ch%s'%qubit_id])
                fitdata = fitdecaysin(expt_pts[1:],data_list[1:],showfit=True)

                pi_length = round((-fitdata[2]%180 + 90)/(360*fitdata[1]),5)
                print("Flux pi length ge: %s" %pi_length)

                quantum_device_cfg['multimodes'][qubit_id]['ef_pi_len'][on_mms] = pi_length
                with open(os.path.join(path, 'quantum_device_config.json'), 'w') as f:
                    json.dump(quantum_device_cfg, f)


def multimode_t1(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['multimode_t1']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']

    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            experiment_cfg['multimode_t1']['on_mms'][qubit_id] = int(on_mms)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('multimode_t1')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'multimode_t1')


def multimode_dc_offset(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['multimode_ramsey']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']

    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            experiment_cfg['multimode_ramsey']['on_mms'][qubit_id] = int(on_mms)
            experiment_cfg['multimode_ramsey']['ramsey_freq'] = 0
            quantum_device_cfg['multimodes'][qubit_id]['dc_offset'][on_mms] = 0

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('multimode_ramsey')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'multimode_ramsey')
        expt_pts = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
        with SlabFile(data_file) as a:
            for qubit_id in on_qubits:
                data_list = np.array(a['expt_avg_data_ch%s' % qubit_id])
                fitdata = fitdecaysin(expt_pts, data_list, showfit=False)
                print("Oscillation frequency: %s GHz" % str(fitdata[1]))

                quantum_device_cfg['multimodes'][qubit_id]['dc_offset'][on_mms] = round(-fitdata[1],5)
                with open(os.path.join(path, 'quantum_device_config.json'), 'w') as f:
                    json.dump(quantum_device_cfg, f)

def multimode_ramsey(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['multimode_ramsey']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']

    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            experiment_cfg['multimode_ramsey']['on_mms'][qubit_id] = int(on_mms)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('multimode_ramsey')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'multimode_ramsey')


def sideband_cooling_test(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['ef_rabi']
    data_path = os.path.join(path, 'data/')
    on_qubits = expt_cfg['on_qubits']


    on_mms_list = np.arange(1,9)

    for on_mms in on_mms_list:

        for qubit_id in expt_cfg['on_qubits']:
            quantum_device_cfg['sideband_cooling'][qubit_id]['cool'] = True
            quantum_device_cfg['sideband_cooling'][qubit_id]['mode_id'] = int(on_mms)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('ef_rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'ef_rabi')


        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('ramsey')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'ramsey')


def photon_transfer_optimize(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer_arb']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'photon_transfer_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    init_iteration_num = 1

    iteration_num = 20000

    sequence_num = 10
    expt_num = sequence_num*expt_cfg['repeat']

    A_list_len = 10

    max_a = {"1":0.6, "2":0.65}
    max_len = 1000

    limit_list = []
    limit_list += [(0.0, max_a[expt_cfg['sender_id']])]*A_list_len
    limit_list += [(0.0, max_a[expt_cfg['receiver_id']])]*A_list_len
    limit_list += [(10.0,max_len)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    if use_prev_model:
        with open(os.path.join(path,'optimizer/00041_photon_transfer_optimize.pkl'), 'rb') as f:
            opt = pickle.load(f)
    else:
        opt = Optimizer(limit_list, "GBRT", acq_optimizer="lbfgs")

        gauss_z = np.linspace(-2,2,A_list_len)
        gauss_envelop = np.exp(-gauss_z**2)
        init_send_A_list = list(quantum_device_cfg['communication'][expt_cfg['sender_id']]['pi_amp'] * gauss_envelop)
        init_rece_A_list = list(quantum_device_cfg['communication'][expt_cfg['receiver_id']]['pi_amp'] * gauss_envelop)
        init_send_len = [300]
        init_rece_len = [300]

        init_x = [init_send_A_list + init_rece_A_list + init_send_len + init_rece_len] * sequence_num



        x_array = np.array(init_x)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]


        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)
            print(f_val_list[::2])
            print(f_val_list[1::2])

        f_val_all = []

        for ii in range(sequence_num):
            f_val_all += [np.mean(f_val_list[ii::sequence_num])]

        print(f_val_all)
        opt.tell(init_x, f_val_all)

    if use_prev_model:
        init_iteration_num = 0


    for iteration in range(init_iteration_num):

        next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        x_array = np.array(next_x_list)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]
        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)
            print(f_val_list[::2])
            print(f_val_list[1::2])

        f_val_all = []

        for ii in range(sequence_num):
            f_val_all += [np.mean(f_val_list[ii::sequence_num])]

        print(f_val_all)

        opt.tell(next_x_list, f_val_all)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 20
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)


    for iteration in range(iteration_num):

        experiment_cfg['photon_transfer_arb']['repeat'] = 1

        X_cand = opt.space.transform(opt.space.rvs(n_samples=1000000))
        X_cand_predict = opt.models[-1].predict(X_cand)
        X_cand_argsort = np.argsort(X_cand_predict)
        X_cand_sort = np.array([X_cand[ii] for ii in X_cand_argsort])
        X_cand_top = X_cand_sort[:expt_num]

        next_x_list = opt.space.inverse_transform(X_cand_top)

        # do the experiment
        x_array = np.array(next_x_list)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]
        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = expt_num,
                                                send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)


        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 20
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)

def photon_transfer_optimize_v2(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer_arb']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'photon_transfer_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 100
    expt_num = sequence_num

    A_list_len = 6

    max_a = {"1":0.5, "2":0.65}
    max_len = 300

    limit_list = []
    limit_list += [(0.0, max_a[expt_cfg['sender_id']])]*A_list_len
    limit_list += [(0.0, max_a[expt_cfg['receiver_id']])]*A_list_len
    limit_list += [(10.0,max_len)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = True

    if use_prev_model:
        with open(os.path.join(path,'optimizer/00051_photon_transfer_optimize.pkl'), 'rb') as f:
            opt = pickle.load(f)
    else:
        opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

        gauss_z = np.linspace(-2,2,A_list_len)
        gauss_envelop = np.exp(-gauss_z**2)
        init_send_A_list = list(quantum_device_cfg['communication'][expt_cfg['sender_id']]['pi_amp'] * gauss_envelop)
        init_rece_A_list = list(quantum_device_cfg['communication'][expt_cfg['receiver_id']]['pi_amp'] * gauss_envelop)
        init_send_len = [300]
        init_rece_len = [300]

        init_x = [init_send_A_list + init_rece_A_list + init_send_len + init_rece_len] * sequence_num



        x_array = np.array(init_x)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]


        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)

        opt.tell(init_x, f_val_list)


    for iteration in range(iteration_num):

        next_x_list = opt.ask(sequence_num,strategy='cl_min')

        # do the experiment
        x_array = np.array(next_x_list)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]
        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 20
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)


def photon_transfer_optimize_v3(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer_arb']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'photon_transfer_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 100
    expt_num = sequence_num


    max_a = {"1":0.5, "2":0.65}
    max_len = 300
    # max_delta_freq = 0.0005

    limit_list = []
    limit_list += [(0.60, max_a[expt_cfg['sender_id']])]
    limit_list += [(0.3, max_a[expt_cfg['receiver_id']])]
    limit_list += [(100.0,max_len)] * 2
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    if use_prev_model:
        with open(os.path.join(path,'optimizer/00057_photon_transfer_optimize.pkl'), 'rb') as f:
            opt = pickle.load(f)
    else:
        opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

        init_send_a = [quantum_device_cfg['communication'][expt_cfg['sender_id']]['pi_amp']]
        init_rece_a = [quantum_device_cfg['communication'][expt_cfg['receiver_id']]['pi_amp']]
        init_send_len = [quantum_device_cfg['communication'][expt_cfg['sender_id']]['pi_len']]
        init_rece_len = [quantum_device_cfg['communication'][expt_cfg['receiver_id']]['pi_len']]
        # init_delta_freq_send = [0.00025]
        # init_delta_freq_rece = [0]

        init_x = [init_send_a + init_rece_a + init_send_len + init_rece_len] * sequence_num



        x_array = np.array(init_x)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_len = x_array[:,2]
        rece_len = x_array[:,3]
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)

        opt.tell(init_x, f_val_list)


    for iteration in range(iteration_num):

        next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_len = x_array[:,2]
        rece_len = x_array[:,3]
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 20
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)

def photon_transfer_optimize_gp_v4(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer_arb']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'photon_transfer_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 50
    expt_num = sequence_num


    max_a = {"1":0.3, "2":0.7}
    max_len = 400
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.10, max_a[sender_id])]
    limit_list += [(0.1, max_a[receiver_id])]
    limit_list += [(100.0,max_len)]
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):

        if use_prev_model:
            with open(os.path.join(path,'optimizer/00057_photon_transfer_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)

        if iteration == 0 and not use_prev_model:
            opt = Optimizer(limit_list, "GP", acq_optimizer="lbfgs")

            init_send_a = [quantum_device_cfg['communication'][sender_id]['pi_amp']]
            init_rece_a = [quantum_device_cfg['communication'][receiver_id]['pi_amp']]
            init_send_len = [quantum_device_cfg['communication'][sender_id]['pi_len']]

            next_x_list = [init_send_a + init_rece_a + init_send_len]

            for ii in range(sequence_num-1):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)

        else:
            next_x_list = []
            gp_best = opt.ask()
            next_x_list.append(gp_best)

            random_sample_num = 10
            x_from_model_num = sequence_num-random_sample_num-1

            X_cand = opt.space.transform(opt.space.rvs(n_samples=100000))
            X_cand_predict = opt.models[-1].predict(X_cand)
            X_cand_argsort = np.argsort(X_cand_predict)
            X_cand_sort = np.array([X_cand[ii] for ii in X_cand_argsort])
            X_cand_top = X_cand_sort[:x_from_model_num]

            gp_sample = opt.space.inverse_transform(X_cand_top)
            next_x_list += gp_sample

            for ii in range(random_sample_num):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)


        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        transfer_len = x_array[:,2]

        send_len = transfer_len
        rece_len = transfer_len
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'][-1])
            single_data2 = np.array(a['single_data2'][-1])
            single_data_list = [single_data1, single_data2]

            state_norm = get_singleshot_data_two_qubits_4_calibration_v2(single_data_list)
            if receiver_id == 2:
                f_val_list = list(1-state_norm[1])
            else:
                f_val_list = list(1-state_norm[2])
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 20
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)


def bell_entanglement_by_half_sideband_optimize(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'bell_entanglement_by_half_sideband_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 20
    expt_num = sequence_num


    max_a = {"1":0.6, "2":0.7}
    max_len = 200
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.30, max_a[sender_id])]
    limit_list += [(0.3, max_a[receiver_id])]
    limit_list += [(50.0,max_len)] * 2
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):
        if use_prev_model:
            with open(os.path.join(path,'optimizer/00057_photon_transfer_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)
            next_x_list = opt.ask(sequence_num,strategy='cl_min')
        else:

            if iteration == 0:
                opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

                init_send_a = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']]
                init_rece_a = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']]
                init_send_len = [quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
                init_rece_len = [quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]
                # init_delta_freq_send = [0.00025]
                # init_delta_freq_rece = [0]

                next_x_list = [init_send_a + init_rece_a + init_send_len + init_rece_len]

                for ii in range(sequence_num-1):
                    x_list = []
                    for limit in limit_list:
                        sample = np.random.uniform(low=limit[0],high=limit[1])
                        x_list.append(sample)
                    next_x_list.append(x_list)



            else:
                next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_len = x_array[:,2]
        rece_len = x_array[:,3]
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'])[-1]
            single_data2 = np.array(a['single_data2'])[-1]

            # single_data_list = [single_data1, single_data2]

            f_val_list = []
            for expt_id in range(sequence_num):
                elem_list = list(range(expt_id*9,(expt_id+1)*9)) + [-2,-1]
                single_data_list = [single_data1[:,:,elem_list,:], single_data2[:,:,elem_list,:]]
                state_norm = get_singleshot_data_two_qubits(single_data_list, expt_cfg['pi_calibration'])
                # print(state_norm.shape)
                state_data = data_to_correlators(state_norm)
                den_mat = two_qubit_quantum_state_tomography(state_data)
                perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
                perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)

                fidelity = np.trace(np.dot(np.transpose(np.conjugate(perfect_bell_den_mat)),np.abs(den_mat) ))

                f_val_list.append(1-fidelity)
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 10
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)



def bell_entanglement_by_half_sideband_optimize_gp_v4(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    # use a * len as a variable, and not use opt.ask
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'bell_entanglement_by_half_sideband_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 10
    expt_num = sequence_num


    max_a = {"1":0.4, "2":0.6}
    max_len_send = 80
    max_len_rece = 200
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.35, max_a[sender_id])]
    limit_list += [(0.55, max_a[receiver_id])]
    limit_list += [(60.0*0.35,max_len_send*max_a[sender_id])]
    limit_list += [(170.0*0.55,max_len_rece*max_a[receiver_id])]
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):
        if iteration == 0 and use_prev_model:
            with open(os.path.join(path,'optimizer/00031_bell_entanglement_by_half_sideband_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)

        if iteration == 0 and not use_prev_model:
            opt = Optimizer(limit_list, "GP", acq_optimizer="lbfgs")

            init_send_a = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']]
            init_rece_a = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']]
            init_send_len = [quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
            init_rece_len = [quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]

            init_send_area = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']*quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
            init_rece_area = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']*quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]
            # init_delta_freq_send = [0.00025]
            # init_delta_freq_rece = [0]

            next_x_list = [init_send_a + init_rece_a + init_send_area + init_rece_area]

            for ii in range(sequence_num-1):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)



        else:
            next_x_list = []
            gp_best = opt.ask()
            next_x_list.append(gp_best)

            random_sample_num = 2
            x_from_model_num = sequence_num-random_sample_num-1

            X_cand = opt.space.transform(opt.space.rvs(n_samples=100000))
            X_cand_predict = opt.models[-1].predict(X_cand)
            X_cand_argsort = np.argsort(X_cand_predict)
            X_cand_sort = np.array([X_cand[ii] for ii in X_cand_argsort])
            X_cand_top = X_cand_sort[:x_from_model_num]

            gp_sample = opt.space.inverse_transform(X_cand_top)
            next_x_list += gp_sample

            for ii in range(random_sample_num):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)

        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_area = x_array[:,2]
        rece_area = x_array[:,3]


        send_len = send_area/send_a
        rece_len = rece_area/rece_a
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'])[-1]
            single_data2 = np.array(a['single_data2'])[-1]

            # single_data_list = [single_data1, single_data2]

            f_val_list = []
            for expt_id in range(sequence_num):
                elem_list = list(range(expt_id*17,(expt_id+1)*17)) + [-4,-3,-2,-1]
                single_data_list = [single_data1[:,:,elem_list,:], single_data2[:,:,elem_list,:]]
                state_norm = get_singleshot_data_two_qubits_4_calibration_v2(single_data_list)

                state_data = data_to_correlators(state_norm[:9])
                den_mat_guess = two_qubit_quantum_state_tomography(state_data)

                ew, ev = np.linalg.eigh(den_mat_guess)
                pos_ew = [max(w,0) for w in ew]
                pos_D = np.diag(pos_ew)
                inv_ev = np.linalg.inv(ev)
                pos_den_mat_guess = np.dot(np.dot(ev,pos_D),inv_ev)

                den_mat_guess_input = pos_den_mat_guess[::-1,::-1]
                den_mat_guess_input = np.real(den_mat_guess_input) - 1j*np.imag(den_mat_guess_input)

                optimized_rho = density_matrix_maximum_likelihood(state_norm,den_mat_guess_input)

                perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
                perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)
                fidelity = np.trace(np.dot(np.transpose(np.conjugate(perfect_bell_den_mat)),np.abs(optimized_rho).clip(max=0.5) ))

                f_val_list.append(1-fidelity)
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        # only save the latest model
        opt.models = [opt.models[-1]]

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 10
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)



def bell_entanglement_by_half_sideband_optimize_v4(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    # use a * len as a variable, and not use opt.ask
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'bell_entanglement_by_half_sideband_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 10
    expt_num = sequence_num


    max_a = {"1":0.6, "2":0.7}
    max_len = 200
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.30, max_a[sender_id])]
    limit_list += [(0.3, max_a[receiver_id])]
    limit_list += [(10.0,max_len*max_a[sender_id])]
    limit_list += [(10.0,max_len*max_a[receiver_id])]
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):
        if iteration == 0 and use_prev_model:
            with open(os.path.join(path,'optimizer/00031_bell_entanglement_by_half_sideband_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)

        if iteration == 0 and not use_prev_model:
            opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

            init_send_a = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']]
            init_rece_a = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']]
            init_send_len = [quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
            init_rece_len = [quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]

            init_send_area = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']*quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
            init_rece_area = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']*quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]
            # init_delta_freq_send = [0.00025]
            # init_delta_freq_rece = [0]

            next_x_list = [init_send_a + init_rece_a + init_send_area + init_rece_area]

            for ii in range(sequence_num-1):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)



        else:
            x_from_model_num = int(sequence_num/2)

            X_cand = opt.space.transform(opt.space.rvs(n_samples=100000))
            X_cand_predict = opt.models[-1].predict(X_cand)
            X_cand_argsort = np.argsort(X_cand_predict)
            X_cand_sort = np.array([X_cand[ii] for ii in X_cand_argsort])
            X_cand_top = X_cand_sort[:x_from_model_num]

            next_x_list = opt.space.inverse_transform(X_cand_top)

            for ii in range(sequence_num-x_from_model_num):
                x_list = []
                for limit in limit_list:
                    sample = np.random.uniform(low=limit[0],high=limit[1])
                    x_list.append(sample)
                next_x_list.append(x_list)

        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_area = x_array[:,2]
        rece_area = x_array[:,3]


        send_len = send_area/send_a
        rece_len = rece_area/rece_a
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'])[-1]
            single_data2 = np.array(a['single_data2'])[-1]

            # single_data_list = [single_data1, single_data2]

            f_val_list = []
            for expt_id in range(sequence_num):
                elem_list = list(range(expt_id*17,(expt_id+1)*17)) + [-4,-3,-2,-1]
                single_data_list = [single_data1[:,:,elem_list,:], single_data2[:,:,elem_list,:]]
                state_norm = get_singleshot_data_two_qubits_4_calibration(single_data_list)

                state_data = data_to_correlators(state_norm[:9])
                den_mat_guess = two_qubit_quantum_state_tomography(state_data)

                ew, ev = np.linalg.eigh(den_mat_guess)
                pos_ew = [max(w,0) for w in ew]
                pos_D = np.diag(pos_ew)
                inv_ev = np.linalg.inv(ev)
                pos_den_mat_guess = np.dot(np.dot(ev,pos_D),inv_ev)

                den_mat_guess_input = pos_den_mat_guess[::-1,::-1]
                den_mat_guess_input = np.real(den_mat_guess_input) - 1j*np.imag(den_mat_guess_input)

                optimized_rho = density_matrix_maximum_likelihood(state_norm,den_mat_guess_input)

                perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
                perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)
                fidelity = np.trace(np.dot(np.transpose(np.conjugate(perfect_bell_den_mat)),np.abs(optimized_rho) ))

                f_val_list.append(1-fidelity)
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        # only save the latest model
        opt.models = [opt.models[-1]]

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 10
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)



def bell_entanglement_by_half_sideband_optimize_v3(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    # use a * len as a variable
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'bell_entanglement_by_half_sideband_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 20
    expt_num = sequence_num


    max_a = {"1":0.6, "2":0.7}
    max_len = 200
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.30, max_a[sender_id])]
    limit_list += [(0.3, max_a[receiver_id])]
    limit_list += [(20.0,max_len*max_a[sender_id])]
    limit_list += [(20.0,max_len*max_a[receiver_id])]
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):
        if use_prev_model:
            with open(os.path.join(path,'optimizer/00057_photon_transfer_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)
            next_x_list = opt.ask(sequence_num,strategy='cl_min')
        else:

            if iteration == 0:
                opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

                init_send_a = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']]
                init_rece_a = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']]
                init_send_len = [quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
                init_rece_len = [quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]

                init_send_area = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']*quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
                init_rece_area = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']*quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]
                # init_delta_freq_send = [0.00025]
                # init_delta_freq_rece = [0]

                next_x_list = [init_send_a + init_rece_a + init_send_area + init_rece_area]

                for ii in range(sequence_num-1):
                    x_list = []
                    for limit in limit_list:
                        sample = np.random.uniform(low=limit[0],high=limit[1])
                        x_list.append(sample)
                    next_x_list.append(x_list)



            else:
                next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        send_a = x_array[:,0]
        rece_a = x_array[:,1]
        send_area = x_array[:,2]
        rece_area = x_array[:,3]


        send_len = send_area/send_a
        rece_len = rece_area/rece_a
        # delta_freq_send = x_array[:,4]
        # delta_freq_rece = x_array[:,5]

        send_A_list = np.outer(send_a, np.ones(10))
        rece_A_list = np.outer(rece_a, np.ones(10))


        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'])[-1]
            single_data2 = np.array(a['single_data2'])[-1]

            # single_data_list = [single_data1, single_data2]

            f_val_list = []
            for expt_id in range(sequence_num):
                elem_list = list(range(expt_id*9,(expt_id+1)*9)) + [-2,-1]
                single_data_list = [single_data1[:,:,elem_list,:], single_data2[:,:,elem_list,:]]
                state_norm = get_singleshot_data_two_qubits(single_data_list, expt_cfg['pi_calibration'])
                # print(state_norm.shape)
                state_data = data_to_correlators(state_norm)
                den_mat = two_qubit_quantum_state_tomography(state_data)
                perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
                perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)

                fidelity = np.trace(np.dot(np.transpose(np.conjugate(perfect_bell_den_mat)),np.abs(den_mat) ))

                f_val_list.append(1-fidelity)
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 10
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)

def bell_entanglement_by_half_sideband_optimize_v2(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'bell_entanglement_by_half_sideband_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 20
    expt_num = sequence_num

    A_list_len = 2

    max_a = {"1":0.6, "2":0.7}
    max_len = 200
    # max_delta_freq = 0.0005

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    limit_list = []
    limit_list += [(0.30, max_a[sender_id])]*A_list_len
    limit_list += [(0.2, max_a[receiver_id])]*A_list_len
    limit_list += [(50.0,max_len)] * 2
    # limit_list += [(-max_delta_freq,max_delta_freq)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    use_prev_model = False

    for iteration in range(iteration_num):
        if use_prev_model:
            with open(os.path.join(path,'optimizer/00057_photon_transfer_optimize.pkl'), 'rb') as f:
                opt = pickle.load(f)
            next_x_list = opt.ask(sequence_num,strategy='cl_max')
        else:

            if iteration == 0:
                opt = Optimizer(limit_list, "GBRT", acq_optimizer="auto")

                init_send_a = [quantum_device_cfg['communication'][sender_id]['half_transfer_amp']]*A_list_len
                init_rece_a = [quantum_device_cfg['communication'][receiver_id]['half_transfer_amp']]*A_list_len
                init_send_len = [quantum_device_cfg['communication'][sender_id]['half_transfer_len']]
                init_rece_len = [quantum_device_cfg['communication'][receiver_id]['half_transfer_len']]
                # init_delta_freq_send = [0.00025]
                # init_delta_freq_rece = [0]

                next_x_list = [init_send_a + init_rece_a + init_send_len + init_rece_len]

                for ii in range(sequence_num-1):
                    x_list = []
                    for limit in limit_list:
                        sample = np.random.uniform(low=limit[0],high=limit[1])
                        x_list.append(sample)
                    next_x_list.append(x_list)



            else:
                next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        print(next_x_list)
        x_array = np.array(next_x_list)

        # send_a = x_array[:,0]
        # rece_a = x_array[:,1]
        # send_len = x_array[:,2]
        # rece_len = x_array[:,3]
        # # delta_freq_send = x_array[:,4]
        # # delta_freq_rece = x_array[:,5]
        #
        # send_A_list = np.outer(send_a, np.ones(10))
        # rece_A_list = np.outer(rece_a, np.ones(10))


        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]


        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography', sequence_num = sequence_num,
                                                    send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                    send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', seq_data_file)

        with SlabFile(data_file) as a:
            single_data1 = np.array(a['single_data1'])[-1]
            single_data2 = np.array(a['single_data2'])[-1]

            # single_data_list = [single_data1, single_data2]

            f_val_list = []
            for expt_id in range(sequence_num):
                elem_list = list(range(expt_id*9,(expt_id+1)*9)) + [-2,-1]
                single_data_list = [single_data1[:,:,elem_list,:], single_data2[:,:,elem_list,:]]
                state_norm = get_singleshot_data_two_qubits(single_data_list, expt_cfg['pi_calibration'])
                # print(state_norm.shape)
                state_data = data_to_correlators(state_norm)
                den_mat = two_qubit_quantum_state_tomography(state_data)
                perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
                perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)

                fidelity = np.trace(np.dot(np.transpose(np.conjugate(perfect_bell_den_mat)),np.abs(den_mat) ))

                f_val_list.append(1-fidelity)
            print(f_val_list)

        opt.tell(next_x_list, f_val_list)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 10
        if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)



def photon_transfer_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'photon_transfer_sweep', suffix='.h5'))

    sweep = 'rece_a'

    sender_id = quantum_device_cfg['communication']['sender_id']
    receiver_id = quantum_device_cfg['communication']['receiver_id']

    if sweep == 'delay':
        delay_len_start = -100
        delay_len_stop = 100
        delay_len_step = 4.0

        for delay_len in np.arange(delay_len_start, delay_len_stop,delay_len_step):
            experiment_cfg['photon_transfer']['rece_delay'] = delay_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('photon_transfer')

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'photon_transfer', seq_data_file)

    elif sweep == 'sender_a':
        start = 0.41
        stop = 0.39
        step = -0.002

        for amp in np.arange(start, stop,step):
            quantum_device_cfg['communication'][sender_id]['pi_amp'] = amp
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('photon_transfer')

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'photon_transfer', seq_data_file)
    elif sweep == 'rece_a':
        start = 0.5
        stop = 0.3
        step = -0.01

        for amp in np.arange(start, stop,step):
            quantum_device_cfg['communication'][receiver_id]['pi_amp'] = amp
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('photon_transfer')

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'photon_transfer', seq_data_file)


def communication_rabi_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['communication_rabi']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'communication_rabi_amp_sweep', suffix='.h5'))

    amp_start = 0.7
    amp_stop = 0.0
    amp_step = -0.01

    on_qubit = "2"

    for amp in np.arange(amp_start, amp_stop,amp_step):
        quantum_device_cfg['communication'][on_qubit]['pi_amp'] = amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('communication_rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'communication_rabi', seq_data_file)


def sideband_rabi_freq_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_freq']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_freq_amp_sweep', suffix='.h5'))

    amp_start = 0.70
    amp_stop = 0.0
    amp_step = -0.005

    for amp in np.arange(amp_start, amp_stop,amp_step):
        experiment_cfg['sideband_rabi_freq']['amp'] = amp
        experiment_cfg['sideband_rabi_freq']['pulse_len'] = 90*amp_start/amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi_freq')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_freq', seq_data_file)

def ef_sideband_rabi_freq_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['ef_sideband_rabi_freq']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'ef_sideband_rabi_freq_amp_sweep', suffix='.h5'))

    amp_start = 0.4
    amp_stop = 0.0
    amp_step = -0.005

    for amp in np.arange(amp_start, amp_stop,amp_step):
        experiment_cfg['ef_sideband_rabi_freq']['amp'] = amp
        experiment_cfg['ef_sideband_rabi_freq']['pulse_len'] = 50*amp_start/amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('ef_sideband_rabi_freq')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'ef_sideband_rabi_freq', seq_data_file)


def rabi_repeat(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['rabi']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'rabi_repeat', suffix='.h5'))

    repeat = 100

    update_awg = True

    for ii in range(repeat):
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'rabi', seq_data_file, update_awg = update_awg)

        update_awg = False


def bell_repeat(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['bell_entanglement_by_half_sideband_tomography']
    data_path = os.path.join(path, 'data/')
    #seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'bell_entanglement_by_half_sideband_tomography_repeat', suffix='.h5'))

    repeat = 10

    update_awg = True

    for iteration in range(repeat):
        qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('bell_entanglement_by_half_sideband_tomography')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'bell_entanglement_by_half_sideband_tomography', update_awg = update_awg)

        # frequency_recalibrate_cycle = 10
        # if iteration % frequency_recalibrate_cycle == frequency_recalibrate_cycle-1:
        #     qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)
