import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json
import numpy as np

import os
path = os.getcwd()

with open('210301_quantum_device_config_wff.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('210301_experiment_config_wff.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('210226_hardware_config_wff.json', 'r') as f:
    hardware_cfg = json.load(f)

for V_ramp in [3,0.5]:
    for qb_ind in range(0,8):
        experiment_names = ['pulse_probe_iq']

        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

