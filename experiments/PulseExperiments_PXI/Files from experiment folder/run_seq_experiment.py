from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
import json

'''
Sequential Experiments:
=======================
resonator_spectroscopy
qubit_temperature
histogram_sweep
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

experiment_name = 'histogram_sweep'

sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=True,show=True)
