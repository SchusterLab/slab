from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
import json

'''
General Sequential Experiments:
=======================
resonator_spectroscopy
qubit_temperature
histogram_sweep
=======================
'''

'''
Sequential Charge sideband Experiments:
=======================
sideband_rabi_sweep
sideband_rabi_freq_scan_length_sweep
sequential_sideband_ramsey
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

experiment_name = 'sideband_rabi_two_tone_freq_scan'

sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=True,show=True)
