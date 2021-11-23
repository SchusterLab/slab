import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json
import numpy as np

import os
path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

'''
General Experiments:
=======================
resonator_spectroscopy
pulse_probe_iq
rabi
t1
ramsey
echo
pulse_probe_ef_iq
ef_rabi
ef_ramsey
ef_t1
ef_echo
pulse_probe_fh_iq
fh_rabi
fh_ramsey
histogram
=======================

Charge sideband Experiments
=======================
sideband_rabi
sideband_rabi_freq_scan
sideband_rabi_two_tone
sideband_t1
sideband_ramsey
sideband_pi_pi_offset
sideband_rabi_two_tone_freq_scan
sideband_rabi_two_tone
sideband_histogram
sideband_pulse_probe_iq
sideband_chi_ge_calibration
sideband_chi_ef_calibration
sideband_chi_gf_calibration
sideband_transmon_reset
sideband_parity_measurement
sideband_repetitive_parity_measurement
sideband_cavity_photon_number
=======================
'''


with open('210301_quantum_device_config_wff.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('210301_experiment_config_wff.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('210226_hardware_config_wff.json', 'r') as f:
    hardware_cfg = json.load(f)

show = 'I'

nb_amps = [0.200]
experiment_names = ['fast_flux_pulse', 'ramsey']

for i in nb_amps:
    while True:
        experiment_cfg['fast_flux_pulse']['ff_amp'] = i
        #experiment_cfg['fast_flux_pulse']['acquisition_num'] = i

        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


