from slab.experiments.PulseExperiments_PXI.sequences import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import slab.experiments.PulseExperiments.sequential_experiment_pxi as seq_exp
import json


'''
Sequential Experiments:
=======================
resonator_spectroscopy_sweep
qubit_temperature
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


experiment_name = 'resonator_spectroscopy_sweep'

eval('seq_exp.' + experiment_name)(quantum_device_cfg,experiment_cfg,hardware_cfg, path)
