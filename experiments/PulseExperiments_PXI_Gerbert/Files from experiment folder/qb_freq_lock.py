import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI_Gerbert.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import generate_quantum_device_from_lattice_v3
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import melting_update
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile

import copy
import json
import numpy as np
import os
path = os.getcwd()
path = "C:\\210801 - PHMIV3_56 - BF4 cooldown 4"

show = 'I'
setup = 'B'
# setup_list = ["A", "B"]
lattice_cfg_name = '210923_large_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

# ##################################################################################
# #################### Pulse Probe IQ ###################
# ##################################################################################
# amplist = np.linspace(0.5,1.0,6)
# for jj,amp in enumerate(amplist):


experiment_names = ['pulse_probe_iq']
ppiq_avgs = [200,200,200,100,100,100,100,1000]
# #ppiq_avgs = [250,250,250,250,250,250,250,1000]
for i in range(7):
    # for j in np.linspace(-0.0005,0.0005,10):
    setup = lattice_cfg["qubit"]["setup"][i]
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
    # quantum_device_cfg['readout'][setup]['amp'] = amp
    #quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i]
    # quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq'][i] + j
    experiment_cfg['pulse_probe_iq']['acquisition_num'] = ppiq_avgs[i]
    experiment_cfg['pulse_probe_iq']['pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]

    for experiment_name in experiment_names:
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        if experiment_name == 'resonator_spectroscopy':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
        else:
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

        exp.post_analysisandsave(path, experiment_name, cont_name[i], P='Q', phi=phi, cont_data_file=cont_data_file[i])

