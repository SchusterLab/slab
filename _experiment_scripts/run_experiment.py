import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice_v2
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice

import json
import numpy as np

import os
path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

show = 'I'

lattice_cfg_name = '210526_sawtooth_lattice_device_config.json'

with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

setup = 'A'
experiment_cfg['on_qubits'] = setup

# experiment_names = ['resonator_spectroscopy']
# for i in [0,1,2,3,4,5,6,7]:
#     for jj in [10,15,20]:
#         if setup == 'A':
#             quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[i, (i+1)%8], setups=['A','B'])
#         else:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[(i + 1) % 8,i],setups=['A', 'B'])
#         quantum_device_cfg['powers'][setup]["readout_drive_digital_attenuation"] = jj
#         for experiment_name in experiment_names:
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

experiment_names = ['pulse_probe_iq']
for i in [0,1,2,3,4,5,6,7]:
    for jj in [5,10,15]:
        if setup == 'A':
            quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[i, (i+1)%8], setups=['A','B'])
        else:
            quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[(i + 1) % 8,i],setups=['A', 'B'])
        quantum_device_cfg['powers'][setup]["drive_digital_attenuation"] = jj
        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)

            if experiment_name == 'resonator_spectroscopy':
                exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
            else:
                exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['rabi']
# for i in [0]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[i, (i+1)%8], setups=['A','B'])
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         import pickle
#         pickle.dump(sequences, open("rabi.p", "wb"))
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)



