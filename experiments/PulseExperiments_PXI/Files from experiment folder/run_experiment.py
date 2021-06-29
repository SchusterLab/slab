import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice_v3

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

# experiment_names = ['resonator_spectroscopy']
# for i in [5,6]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


experiment_names = ['pulse_probe_iq']
ppiq_avgs = [250,250,250,250,250,250,250,2000]
for i in [0, 1, 2,3,4, 5, 6, 7]:
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
    experiment_cfg['pulse_probe_iq']['acquisition_num'] = ppiq_avgs[i]
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
# ## gauss
# ## rabi_start_array = [0,0,0,0,0,0,0,0]
# ## rabi_stop_array = [2000,1500,1500,2000,1500,1500,1000,500]
# ## rabi_step_array = [20,15,15,20,15,15,15,5]
# ## rabi_step_avgs = [500,500,500,500,500,500,500,1500]
# ###square wave
# rabi_start_array = [0,0,0,0,0,0,0,0]
# rabi_stop_array = [4000,3000,3000,4000,3000,3000,2000,500]
# rabi_step_array = [40,30,30,40,30,30,20,10]
# rabi_step_avgs = [500,500,500,500,500,500,500,2000]
# for i in [7]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['rabi']['start'] = rabi_start_array[i]
#     experiment_cfg['rabi']['stop'] = rabi_stop_array[i]
#     experiment_cfg['rabi']['step'] = rabi_step_array[i]
#     experiment_cfg['rabi']['acquisition_num'] = rabi_step_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['t1']
# t1_start_array = [0,0,0,0,0,0,0,0]
# t1_stop_array = [250000,450000,350000,300000,450000,450000,450000,20000]
# t1_step_array = [20000,40000,30000,30000,40000,40000,40000,1000]
# t1_avgs = [1000,1000,1000,1000,1000,1000,1000,3000]
# for i in [0, 1, 2, 3, 4, 5, 6, 7]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['t1']['start'] = t1_start_array[i]
#     experiment_cfg['t1']['stop'] = t1_stop_array[i]
#     experiment_cfg['t1']['step'] = t1_step_array[i]
#     experiment_cfg['t1']['acquisition_num'] = t1_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['ramsey']
# ramsey_start_array = [0,0,0,0,0,0,0,0]
# ramsey_stop_array = [2000,2000,2000,2000,2000,2000,2000,1000]
# ramsey_step_array = [20,20,20,20,20,20,20,10]
# ramsey_freq_array = [0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.005,]
# ramsey_avgs = [1500,1500,1500,1500,1500,1500,1500,8000]
# for i in [0,1,2,3,4,5,6,7]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['ramsey']['start'] = ramsey_start_array[i]
#     experiment_cfg['ramsey']['stop'] = ramsey_stop_array[i]
#     experiment_cfg['ramsey']['step'] = ramsey_step_array[i]
#     experiment_cfg['ramsey']['ramsey_freq'] = ramsey_freq_array[i]
#     experiment_cfg['ramsey']['acquisition_num'] = ramsey_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['echo']
# echo_start_array = [0,0,0,0,0,0,0,0]
# echo_stop_array = [2000,2000,2000,2000,2000,2000,2000,1000]
# echo_step_array = [20,20,20,20,20,20,20,10]
# echo_freq_array = [0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.005,]
# echo_avgs = [1000,1000,1000,1000,1000,1000,1000,8000]
# for i in [0, 1, 2, 3, 4, 5, 6, 7]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['echo']['start'] = echo_start_array[i]
#     experiment_cfg['echo']['stop'] = echo_stop_array[i]
#     experiment_cfg['echo']['step'] = echo_step_array[i]
#     experiment_cfg['echo']['ramsey_freq'] = echo_freq_array[i]
#     experiment_cfg['echo']['acquisition_num'] = echo_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['resonator_spectroscopy_pi']
# for i in [0,1,2,3,4,5,6]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         if experiment_name == 'resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec_pi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# experiment_names = ['histogram']
# for i in [0,1,2,3,4,5,6]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         if experiment_name == 'resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec_pi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#

# experiment_names = ['ff_ramp_cal_ppiq']
# ff_ramp_cal_ppiq_avgs = [250,250,250,250,250,250,250,8000]
# for i in [0]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['ff_ramp_cal_ppiq']['acquisition_num'] = ff_ramp_cal_ppiq_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         # exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         # if experiment_name == 'resonator_spectroscopy':
#         #     exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         # else:
#         #     exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)