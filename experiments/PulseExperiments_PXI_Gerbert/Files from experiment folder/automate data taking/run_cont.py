
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile

import os
path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"
import time
import numpy as np
import pickle



## Remember to run python -m visdom.server
## Remember to start instrumentmanager with .cfg file configured correctly

lattice_cfg_name = '210510_sawtooth_lattice_device_config_wff.json'

with open('S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\'+lattice_cfg_name, 'r') as f:
    lattice_cfg  = json.load(f)
with open('S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

check_sync = False

path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"
data_path = os.path.join(path, 'data\\')

experiment_names = ['pulse_probe_iq']
qb_list = [0,1,2,3,4,5,6,7]
cont_name = ["cont_ppiq_%s"%qb for qb in qb_list]
cont_data_file = [os.path.join(data_path, "00000_cont_ppiq_%s.h5"%qb) for qb in qb_list]

slab_file = [SlabFile(file) for file in cont_data_file]

for i in [0,1,2,3,4,5,6,7]:
    quantum_device_cfg = generate_quantum_device_from_lattice(lattice_cfg_name, qb_ids=[i], setups=['A', 'B'])

    phi = 0
    with slab_file[i] as file:
        file.append_line('phi', [phi])

    print("Start a CONT experiment! Cont_name is: " + cont_data_file[i])

    for experiment_name in experiment_names:
        #f = open("t1_sequences.pkl", "rb")
        #sequences = pickle.load(f)
        #f.close()

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,sequences, experiment_name)
        exp.run_experiment_pxi(sequences, path, experiment_name, check_sync=check_sync)
        print("about to begin post-analysis")
        exp.post_analysisandsave(path, experiment_name, cont_name[i], P='Q', phi=phi, cont_data_file=cont_data_file[i])

    print("End a CONT experiment! Cont_name is: " + cont_data_file)
