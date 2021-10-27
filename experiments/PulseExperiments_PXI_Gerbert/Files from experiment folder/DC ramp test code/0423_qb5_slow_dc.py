import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json
import numpy as np
import os
from DACInterface import AD5780_serial
import time

path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

with open('210301_quantum_device_config_wff.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('210301_experiment_config_wff.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('210226_hardware_config_wff.json', 'r') as f:
    hardware_cfg = json.load(f)

dac = AD5780_serial()
quantum_device_cfg['qubit']['1']['freq'] = 6.13649
quantum_device_cfg['readout']['freq'] = 6.7305
experiment_names = ['ramsey']

V_target = 0.001

for qb_ind in [5]:
    #run ramsey
    for experiment_name in experiment_names:
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

    # ramp qb
    dac.ramp3(qb_ind + 1, V_target, 1, 7)
    time.sleep(5)
    dac.ramp3(qb_ind + 1, -V_target, 1, 7)
    time.sleep(5)
    # ramp qb back
    dac.ramp3(qb_ind + 1, 0, 1, 7)
    time.sleep(2)

    # run ramsey
    for experiment_name in experiment_names:
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


