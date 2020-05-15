
from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
import json

'''
General Sequential Experiments:
=======================
resonator_spectroscopy
qubit_temperature
coherence_and_qubit_temperature
histogram_sweep
=======================
'''

'''
Sequential Charge sideband Experiments:
=======================
sideband_rabi_sweep
sideband_rabi_freq_scan_length_sweep
sideband_rabi_freq_scan_amp_sweep
sequential_sideband_ramsey
sideband_rabi_two_tone_detuning_sweep
sideband_reset_qubit_temperature
sideband_reset_qubit_temperature_wait_sweep
cavity_drive_pulse_probe_iq_amp_sweep
wigner_tomography_test_phase_sweep
wigner_tomography_sideband_only_phase_sweep
wigner_tomography_sideband_one_pulse_phase_sweep
wigner_tomography_alltek2_phase_sweep
wigner_tomography_2d_offset_sweep
=======================
'''

import os
path = os.getcwd()

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)




experiment_name = 'qubit_temperature'
sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=True,show=True)
