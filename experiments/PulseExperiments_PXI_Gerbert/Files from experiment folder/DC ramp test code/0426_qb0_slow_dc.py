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
experiment_names = ['ramsey']

V_target = [3.0, 0, 0, 0, 0, 0, 0.0, 0]
V0 = [0.0]*8
qb_ind = [0]
#run ramsey
# for experiment_name in experiment_names:
#     ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
#     sequences = ps.get_experiment_sequences(experiment_name)
#     print("Sequences generated")
#     exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#     exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# ramp qb
dac.parallelramp(V_target,step = 1,speed = 1)
# ramp qb back
dac.parallelramp(V0,step = 1,speed = 1)

# run ramsey
for experiment_name in experiment_names:
    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
    sequences = ps.get_experiment_sequences(experiment_name)
    print("Sequences generated")
    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
    exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


