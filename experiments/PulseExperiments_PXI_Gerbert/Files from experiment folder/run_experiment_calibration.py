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
lattice_cfg_name = '211008_2qb_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

##################################################################################
#################### Res Spec  - this code block needs some cleaning up ###################
##################################################################################
V_list = [[ 0.633,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.001,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.905,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.56 ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   , -0.351,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.938,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.54 ]]

## Cleaned up all the code that was sweeping readout amplitudes for clarity.  We can put that back in later, it's
## Just another for loop

# experiment_names = ['resonator_spectroscopy']
# experiment_names = ['ff_resonator_spectroscopy']
#
# for i in [0,1,2,3,4,5,6]:
#     setup = lattice_cfg["qubit"]["setup"][i]
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
#
#     for experiment_name in experiment_names:
#
#         if experiment_name=='ff_resonator_spectroscopy':
#             experiment_cfg[experiment_name]['ff_vec'] = V_list[i]
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         if experiment_name == 'ff_resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, ff=True)
# #
#
#         ##################################################################################
#         #################### Res Spec Pi and crosstalk###################
#         ##################################################################################
#         #pi, and pi pi!
#         # attenlist2 = [5,10,15,20]
#         # amplist = [0.7]
#     #     # #for kk,amp in enumerate(amplist):
#
# experiment_names = ['resonator_spectroscopy_pi_2setups']
# templist = []
# # templist.append((1, [0,1]))
# for ii in range(7):
#     templist.append((3, [ii]))
#     templist.append((3,[3,ii]))
#     # templist.append((i, [i]))
# # templist= [(0, [0,1])]
#
# # assumptions: at least one of pi qubits is on readout qb's setup
# for (rd_qb, pi_qbs) in templist:
#     experiment_cfg[experiment_names[0]]["pi_qb"] = pi_qbs
#     experiment_cfg[experiment_names[0]]["rd_qb"] = rd_qb
#     setup = lattice_cfg["qubit"]["setup"][rd_qb]
#
#     if len(pi_qbs) > 1:
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#             lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0],
#             lattice_cfg["qubit"]["setup"][pi_qbs[1]]: pi_qbs[1]})
#     elif pi_qbs[0] == rd_qb:
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#             lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0]})
#     else:
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#             lattice_cfg["qubit"]["setup"][rd_qb]: rd_qb,
#             lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0]
#             })
#       #quantum_device_cfg['readout']['A']['amp'] = [0.7,1,0.7,0.9,0.562,1,1,0.666][5] * readoutampscalefactor[jj]
#     for experiment_name in experiment_names:
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"][setup]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = \
#         lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#         quantum_device_cfg["readout"][setup]["window"] = lattice_cfg["readout"][setup]["window"][rd_qb]
#
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True, ff=False, two_setups=True)

# experiment_names = ['resonator_spectroscopy_pi']
# templist = []
# # templist.append((0, [0,1]))
# for ii in range(7):
#     templist.append((ii, [ii]))
#     templist.append((ii,[ii,ii]))
#
# for (rd_qb, pi_qbs) in ([(1, [1])]):
#     experiment_cfg[experiment_names[0]]["pi_qb"] = pi_qbs
#     setup = lattice_cfg["qubit"]["setup"][rd_qb]
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#     #quantum_device_cfg['readout']['A']['amp'] = [0.7,1,0.7,0.9,0.562,1,1,0.666][5] * readoutampscalefactor[jj]
#     for experiment_name in experiment_names:
#         #quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = \
#         lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         # quantum_device_cfg['powers'][setup]["readout_drive_digital_attenuation"] = atten
#         #quantum_device_cfg['readout'][setup]["amp"] = amp
#         # if experiment_name == 'ff_resonator_spectroscopy_pi':
#         #     experiment_cfg[experiment_name]['ff_vec'] = V_list[rd_qb]
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True)
#         if experiment_name == 'ff_resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True, ff=True)

# #
# ##################################################################################
# #################### Pulse Probe IQ ###################
# ##################################################################################
import time
#PPIQ
#
# experiment_names = ['pulse_probe_iq']
# # ppiq_avgs = [200,200,200,200,200,200,200,200]
# ppiq_avgs = [150,150,150,150,150,150,150,150]
# # qphaselist = np.arange(-1.6,1.6,.1)
# # for reps in range(50):
# for i in [0]:
# # for j in np.linspace(-0.0005,0.0005,10):
#     setup = lattice_cfg["qubit"]["setup"][i]
#     ppiq_amps = lattice_cfg["pulse_info"][setup]["pi_amp"]
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#
#     # quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i] + 0.2*s
#     # quantum_device_cfg['pulse_info'][setup]['Q_phase'] = qphase
#     # quantum_device_cfg['readout'][setup]['amp'] = amp
#     #quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i]
#     # quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq'][i] + j
#     experiment_cfg['pulse_probe_iq']['acquisition_num'] = ppiq_avgs[i]
#     experiment_cfg['pulse_probe_iq']['amp'] = ppiq_amps[i]
#     experiment_cfg['pulse_probe_iq']['pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#     experiment_cfg['pulse_probe_iq']['amp'] = quantum_device_cfg["pulse_info"][setup]["pi_amp"]
#
#     # if i == 5 or i == 6:
#     #     experiment_cfg['pulse_probe_iq']['pulse_length'] = 180
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# ##################################################################################
# #################### Pulse Probe IQ -ef ###################
# ##################################################################################
# experiment_names = ['pulse_probe_ef_iq']
# # amplist = np.linspace(0.5,1.0,6)
# # for jj,amp in enumerate(amplist):
# ppiq_ef_avgs = [4000,500,500,500,500,500,500]
# for i in [0]:
#
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#
#     experiment_cfg['pulse_probe_ef_iq']['acquisition_num'] = ppiq_ef_avgs[i]
#     experiment_cfg['pulse_probe_ef_iq']['pulse_length'] = 5000#quantum_device_cfg["pulse_info"][setup]["pi_len"]
#
#     experiment_cfg['pulse_probe_ef_iq']['start'] = -0.020 + quantum_device_cfg['qubit'][setup]['anharmonicity']
#     experiment_cfg['pulse_probe_ef_iq']['stop'] = 0.020 + quantum_device_cfg['qubit'][setup]['anharmonicity']
#     experiment_cfg['pulse_probe_ef_iq']['step'] = 0.0005
#
#     quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq_ef'][i]
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
# # ##################################################################################
# # #################### Rabi ###################
# # ##################################################################################

# experiment_names = ['rabi']
# # # gauss wave
# rabi_start_array = [0,0,0,0,0,0,0,0]
# rabi_stop_array = [800]*8
# rabi_step_array = [80]*8
# # rabi_amp_factor = [0.2,0.4,0.6,0.8,1]
# # rabi_avgs = [500,500,500,500,500,500,500,2000]
# rabi_avgs = [200,200,200,200,200,200,200,1000]
# # for jj,rabi_amp in enumerate(rabi_amp_factor):
# # for reps in range(50):
# for i in [0]:
#     # for j in np.linspace(-0.0005,0.0005,10):
#     setup = lattice_cfg["qubit"]["setup"][i]
#     rabi_amp_array  = lattice_cfg["pulse_info"][setup]["pi_amp"]
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][i]
#     experiment_cfg['rabi']['start'] = rabi_start_array[i]
#     experiment_cfg['rabi']['stop'] = rabi_stop_array[i]
#     experiment_cfg['rabi']['step'] = rabi_step_array[i]
#     experiment_cfg['rabi']['acquisition_num'] = rabi_avgs[i]
#     experiment_cfg['rabi']['amp'] = rabi_amp_array[i]
#     # experiment_cfg['rabi']['amp'] = rabi_amp_array[i]*rabi_amp_factor[jj]
#     # if i==0:
#         # experiment_cfg['rabi']['amp'] = 0.2
#     quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i]
#     # quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq'][i] + j
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

#
# ##################################################################################
# #################### Rabi Crosstalk ###################
# ##################################################################################
# # #
# experiment_names = ['ff_rabi']
# for ii in range(5):
# experiment_names = ['rabi']
# # # # # # # gauss
# # #     # # rabi_start_array = [0,0,0,0,0,0,0,0]
# # #     # # rabi_stop_array = [600]*8
# # #     # # rabi_step_array = [6]*8
# # #     # #
# rabi_start_array = [0,0,0,0,0,0,0,0]
# rabi_stop_array = [800]*8
# rabi_step_array = [8]*8
# rabi_amp_array = [1,1,1,0.71,0.71,1,0.71,1]
# #
# # # square wave
# # # rabi_start_array = [0,0,0,0,0,0,0,0]
# # # rabi_stop_array = [1000]*8
# # # rabi_step_array = [10]*8
# #
# rabi_step_avgs = [250,250,250,250,250,250,250,1000]
# for rd_qb in [2]:
#     for pi_qb in [0]:
#         # if (rd_qb == 4 and pi_qb >= 6) or (rd_qb > 4):
#         # if rd_qb == pi_qb:
#         for experiment_name in experiment_names:
#             # experiment_cfg[experiment_name]['ff_vec'] = V_list[rd_qb]
#             setup = lattice_cfg["qubit"]["setup"][i]
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:pi_qb})
#             quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#             quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#             experiment_cfg[experiment_name]['start'] = rabi_start_array[pi_qb]
#             experiment_cfg[experiment_name]['stop'] = rabi_stop_array[pi_qb]
#             experiment_cfg[experiment_name]['step'] = rabi_step_array[pi_qb]
#             experiment_cfg[experiment_name]['acquisition_num'] = rabi_step_avgs[rd_qb]
#             experiment_cfg['rabi']['amp'] = rabi_amp_array[pi_qb]
#
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
#
# ##################################################################################
# #################### T1 ###################
# ##################################################################################
# # #
# experiment_names = ['t1']
# t1_start_array = [0,0,0,0,0,0,0,0]
# t1_stop_array = [250000,450000,350000,250000,450000,450000,450000,10000]
# t1_step_array = [20000,40000,30000,20000,40000,40000,40000,500]
# t1_avgs = [1000,1000,1000,1000,1000,1000,1000,5000]
# for i in [0, 1, 2, 3, 4, 5, 6]:
#     setup = lattice_cfg["qubit"]["setup"][i]
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg['t1']['start'] = t1_start_array[i]
#     experiment_cfg['t1']['stop'] = t1_stop_array[i]
#     experiment_cfg['t1']['step'] = t1_step_array[i]
#     experiment_cfg['t1']['acquisition_num'] = t1_avgs[i]
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# ##################################################################################
# #################### T1 vs time ###################
# ##################################################################################
# # #
# experiment_names = ['t1']
# t1_start_array = [0,0,0,0,0,0,0,0]
# t1_stop_array = [250000,450000,350000,250000,450000,450000,450000,10000]
# t1_step_array = [20000,40000,30000,20000,40000,40000,40000,500]
# t1_avgs = [1000,1000,1000,1000,1000,1000,1000,5000]
#
# for reps in range(30):
#     for i in [6]:
#         setup = lattice_cfg["qubit"]["setup"][i]
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#         experiment_cfg['t1']['start'] = t1_start_array[i]
#         experiment_cfg['t1']['stop'] = t1_stop_array[i]
#         experiment_cfg['t1']['step'] = t1_step_array[i]
#         experiment_cfg['t1']['acquisition_num'] = t1_avgs[i]
#         for experiment_name in experiment_names:
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
# #
# ##################################################################################
# #################### Ramsey###################
# ##################################################################################
# experiment_names = ['ramsey']
# ramsey_start_array = [0,0,0,0,0,0,0,0]
# ramsey_stop_array = [10000,1000,4000,4000,4000,4000,4000,1000]
# ramsey_step_array = [100,5,20,20,20,20,20,10]
# ramsey_freq_array = [0.010,0.010,0.0025,0.0025,0.0025,0.0025,0.0025,0.005,]
# ramsey_avgs = [1000,1000,500,500,500,500,500,2000]
# # # # while True:
# for reps in range(20):
#     for i in [0]:
#         setup = lattice_cfg["qubit"]["setup"][i]
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#         experiment_cfg['ramsey']['start'] = ramsey_start_array[i]
#         experiment_cfg['ramsey']['stop'] = ramsey_stop_array[i]
#         experiment_cfg['ramsey']['step'] = ramsey_step_array[i]
#         experiment_cfg['ramsey']['ramsey_freq'] = ramsey_freq_array[i]
#         experiment_cfg['ramsey']['acquisition_num'] = ramsey_avgs[i]
#         for experiment_name in experiment_names:
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
# ##################################################################################
# #################### Echo ###################
# ##################################################################################
# experiment_names = ['echo']
# echo_start_array = [0,0,0,0,0,0,0,0]
# echo_stop_array = [10000,2000,2000,2000,2000,2000,2000,1000]
# echo_step_array = [100,20,20,20,20,20,20,10]
# echo_freq_array = [0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.005,]
# echo_avgs = [1000,1000,1000,1000,1000,1000,1000,8000]
# for reps in range(1):
#     for i in [0]:
#         setup = lattice_cfg["qubit"]["setup"][i]
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#         experiment_cfg['echo']['start'] = echo_start_array[i]
#         experiment_cfg['echo']['stop'] = echo_stop_array[i]
#         experiment_cfg['echo']['step'] = echo_step_array[i]
#         experiment_cfg['echo']['ramsey_freq'] = echo_freq_array[i]
#         experiment_cfg['echo']['acquisition_num'] = echo_avgs[i]
#         for experiment_name in experiment_names:
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
# ##################################################################################
# #################### histogram ###################
# ##################################################################################
# experiment_names = ['ff_histogram']
experiment_names = ['histogram']
### for window in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]:
for i in [1]:
    #for atten in atten_list:
    setup = lattice_cfg["qubit"]["setup"][i]
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
    # quantum_device_cfg['readout'][setup]['window'] = [500, window]
    for experiment_name in experiment_names:
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
        if experiment_name == 'resonator_spectroscopy':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
        if experiment_name == 'resonator_spectroscopy_pi':
            exp.run_experiment_pxi_resspec_pi(sequences, path, experiment_name, expt_num=0, check_sync=False)
        else:
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

##################################################################################
# #################### histogram vs readout attenuator ###################
# ##################################################################################
# experiment_names = ['ff_histogram']
# experiment_names = ['histogram']
# #
# read_atten_list = np.arange(0, 15, 0.5).tolist()
# for i in [0]:
#     for atten in read_atten_list:
#         setup = lattice_cfg["qubit"]["setup"][i]
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
#         quantum_device_cfg['powers'][setup]['readout_drive_digital_attenuation'] = atten
#         for experiment_name in experiment_names:
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             if experiment_name == 'resonator_spectroscopy_pi':
#                 exp.run_experiment_pxi_resspec_pi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

##################################################################################
#################### histogram crosstalk ###################
##################################################################################
# update_lattice = copy.deepcopy(lattice_cfg)
# lower = -.0025
# upper = .001
# # for rd in list(np.arange(6.8302939, 6.8308439, 0.00005)):
# # for rd in list(np.arange(6.1655-0.0025,6.831194+0.001, 0.00005)):
# # for detune in list(np.arange(0+lower,0+upper,0.00005)):
# for rd in [6.16425]:
# for i in [1]:
#     RD_QB = i
#     experiment_names = ['histogram_crosstalk']
#     for cross_qb in [0, 1, 2, 3, 4, 5, 6]:
#         rd_setup = lattice_cfg["qubit"]["setup"][RD_QB]
#         cross_setup = lattice_cfg["qubit"]["setup"][cross_qb]
#         for experiment_name in experiment_names:
#             expt_cfg = experiment_cfg[experiment_name]
#             expt_cfg['cross_qb'] = cross_qb
#             expt_cfg['rd_qb'] = RD_QB
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#                     rd_setup:RD_QB,
#                     cross_setup: cross_qb})
#
#             #quantum_device_cfg["readout"][rd_setup]["freq"] = detune+quantum_device_cfg['readout'][rd_setup]['freq'] # for sweeping readout frequency
#             quantum_device_cfg["readout"][rd_setup]["freq"] = lattice_cfg["readout"][rd_setup]["freq"][expt_cfg['rd_qb']] #+ detune
#             quantum_device_cfg["readout"][rd_setup]["window"] = lattice_cfg["readout"][rd_setup]["window"][
#                 expt_cfg['rd_qb']]
#             quantum_device_cfg["powers"][rd_setup]["readout_drive_digital_attenuation"] = \
#                 lattice_cfg["powers"][rd_setup]["readout_drive_digital_attenuation"][expt_cfg['rd_qb']]
#
#             if rd_setup!=cross_setup:
#                 quantum_device_cfg["readout"][cross_setup]["freq"] = 8
#                 quantum_device_cfg["powers"][cross_setup]["readout_drive_digital_attenuation"] = 99
#
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# ##################################################################################
# #################### histogram 2setups ###################
# ##################################################################################
# RD_QB = 1
# templist = []
# for ii in range(7):
#     templist.append((RD_QB, [ii]))
#     templist.append((RD_QB,[RD_QB,ii]))
#
# experiment_names = ['histogram_2setups']
# for rd_qb, pi_qbs in templist:
#     setup = lattice_cfg["qubit"]["setup"][rd_qb]
#     for experiment_name in experiment_names:
#
#         expt_cfg = experiment_cfg[experiment_name]
#         expt_cfg['pi_qbs'] = pi_qbs
#         expt_cfg['rd_qb'] = rd_qb
#         #pi_qbs = expt_cfg['pi_qbs']
#         #rd_qb = expt_cfg['rd_qb']
#
#         #ASSUMPTIONS: 1) if 2piqbs, at lesat one is rd_qb setup. 2) max 2 piqbs. 3) all qbs on same setup have same rd atten etc
#
#         if len(pi_qbs) > 1:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#                 lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0],
#                 lattice_cfg["qubit"]["setup"][pi_qbs[1]]: pi_qbs[1]})
#         elif pi_qbs[0]==rd_qb:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#                 lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0]})
#         else:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={
#                 lattice_cfg["qubit"]["setup"][rd_qb]:rd_qb,
#                 lattice_cfg["qubit"]["setup"][pi_qbs[0]]: pi_qbs[0]})
#
#
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"][setup]["freq"][expt_cfg['rd_qb']]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = \
#             lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][expt_cfg['rd_qb']]
#
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         if experiment_name == 'resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec_pi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# NOTE: THIS CODE IS SUPER HACKY AND WILL ONLY WORK IF DRIVE SETUP AND READOUT SETUP ARE DIFFERENT
# experiment_names = ['pulse_probe_iq_2setups']
# ppiq_avgs = [200,100,200,200,200,200,200,200]
# #ppiq_avgs = [250,250,250,250,250,250,250,1000]
# scan_range = [0]
# for i in [1]:
#     # for j in np.linspace(-0.0005,0.0005,10):
#     experiment_cfg['pulse_probe_iq_2setups']["rd_setup"] = "B"
#     experiment_cfg['pulse_probe_iq_2setups']["pi_qb"] = i
#
#     readout_setup = experiment_cfg['pulse_probe_iq_2setups']["rd_setup"]
#     drive_setup = lattice_cfg["qubit"]["setup"][i]
#
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={drive_setup:i, readout_setup:i})
#     quantum_device_cfg["powers"][drive_setup]["readout_drive_digital_attenuation"] = 30
#     quantum_device_cfg["readout"][drive_setup]["freq"] = 8
#     quantum_device_cfg["powers"][readout_setup]["drive_digital_attenuation"] = 30
#     quantum_device_cfg["qubit"][readout_setup]["freq"] = 8
#
#     experiment_cfg['pulse_probe_iq_2setups']['acquisition_num'] = ppiq_avgs[i]
#     experiment_cfg['pulse_probe_iq_2setups']['pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


# # ##################################################################################
# # #################### FF CTM measurement SMALL LOCAL WIGGLE FIRST  ######################
# # ##################################################################################
# # HERE
# # Flux/qubit tests/
# # this is for wiggle local slope measurement
#
# V_list = [[0.02, 0., 0., 0., 0., 0., 0.],
#           [0., 0.02, 0., 0., 0., 0., 0.],
#           [0., 0., 0.02, 0., 0., 0., 0.],
#           [0., 0., 0., 0.02, 0., 0., 0.],
#           [0., 0., 0., 0., 0.02, 0., 0.],
#           [0., 0., 0., 0., 0., 0.02, 0.],
#           [0., 0., 0., 0., 0., 0., 0.02]]

# # # #
# # # # # #
#
# jumplist = [ 0.05347021, -0.04639491,  0.12233141, -0.09793431,  0.08551732, -0.07265191,
#   0.06945336, -1.247     ]

# jumplist = [0] * 8
#
# experiment_name = "ff_pulse_probe_iq"
#
# for flux_line in [0, 1, 2, 3, 4, 5, 6]:
#     for cross_qb in [0, 1, 2, 3, 4, 5, 6]:
#         if cross_qb == flux_line:
#             # for flux in np.linspace(-V_list[flux_line], V_list[flux_line], 2):
#             #     flux_vec = [0]*8
#             # flux_vec[flux_line] = flux
#             # if (cross_qb == flux_line - 1 or  cross_qb == flux_line or cross_qb == flux_line  + 1 ):
#             freq_shift = 0
#             if (cross_qb == flux_line):
#                 freq_shift = jumplist[cross_qb]
#
#             for j, center in enumerate([0, freq_shift]):
#                 # for j, center in enumerate([freq_shift]):
#                 # if j!=0:
#                 setup = lattice_cfg["qubit"]["setup"][cross_qb]
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name,
#                                                                              on_qubits={setup: cross_qb})
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb == flux_line:
#                     quantum_device_cfg["qubit"][setup]["freq"] = quantum_device_cfg["qubit"][setup]["freq"] + center
#
#                 experiment_cfg[experiment_name]['start'] = -0.030
#                 experiment_cfg[experiment_name]['stop'] = 0.030
#                 experiment_cfg[experiment_name]['step'] = .0005
#                 experiment_cfg[experiment_name]['acquisition_num'] = 250
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#                 experiment_cfg[experiment_name]['qb_amp'] = [1,1,1,1,1,0.5,1,1][cross_qb]
#
#                 # Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb == 7:
#                     experiment_cfg[experiment_name]['step'] = .001
#                     experiment_cfg[experiment_name]['acquisition_num'] = 2000
#                 if cross_qb == 6:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 200
#                 if cross_qb == 1:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 200
#
#                 ## Switch between no flux, and flux
#                 if (j == 0):
#                     experiment_cfg[experiment_name]['ff_vec'] = [0, 0, 0, 0, 0, 0, 0, 0]
#                 else:
#                     experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
#
#                     # else:
#                     # quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg['readout']['freq'][cross_qb]
#
#                 # experiment_cfg[experiment_name]['qb_pulse_length'] =  experiment_cfg[experiment_name]['qb_pulse_length']*2
#                 # quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] =quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] + 6
#                 # experiment_cfg[experiment_name]["qb_amp"]=0.5
#                 experiment_cfg[experiment_name]['ff_pulse_type'] = "square"
#                 print(experiment_cfg[experiment_name]['ff_vec'])
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences(experiment_name)
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)



##################################################################################
#################### FF CTM measurement ###################
##################################################################################
# HERE
# Flux/qubit tests/

# this is for CTM measurement
# V_list =[[ 0.65001781,  0.02934143,  0.01255597,  0.00930305,  0.00830787,
#          0.00659039,  0.00443601],
#        [ 0.00506227, -0.30820868, -0.01566092, -0.01060703, -0.00507664,
#         -0.0055305 , -0.00378356],
#        [ 0.01422392, -0.00961284,  0.9150532 ,  0.0677343 ,  0.03278028,
#          0.03282445,  0.02497602],
#        [ 0.01806297,  0.05540103,  0.04086814,  0.67736874,  0.04603815,
#          0.04271691,  0.03411778],
#        [ 0.01988727,  0.03793593,  0.03760065,  0.04612469, -0.34479041,
#         -0.03802976, -0.03514452],
#        [-0.01625366, -0.02849765, -0.03094502, -0.03840405, -0.04601012,
#          0.7571359 , -0.04308529],
#        [ 0.00390484,  0.00986016,  0.00872847,  0.01206763,  0.00811583,
#          0.01975255, -0.48733125]]

# # #
# # # # #
# jumplist = [ 0.107139, -0.055122,  0.137144, -0.102   ,  0.048722, -0.132894, 0.077477]
# jumplist = [0]*8

# res_freq = [6.16485,  6.39145, 6.61393, 6.83014, 6.95255, 6.7282, 6.50173, 6.27853]
# lattice_readout_freq = [6.16455,6.39125, 6.613775, 6.82979, 6.9526, 6.728, 6.50168,6.27853]

# piLen_lattice = [59.58428948336989,
#  30.806186656367757,
#  42.71307477137261,
#  56.950733554042635/2,
#  69.52654149538475/2,
#  22.26893232006009,
#  43.2114415351338,
#         37.25]


# experiment_name = "ff_pulse_probe_iq"
#
# for flux_line in [0,1,2,3,4,5,6]:
#     for cross_qb in [0,1,2,3,4,5,6]:
#         # if cross_qb == flux_line:
#         # for flux in np.linspace(-V_list[flux_line], V_list[flux_line], 2):
#         #     flux_vec = [0]*8
#         #     flux_vec[flux_line] = flux
#         if (cross_qb == flux_line - 1 or  cross_qb == flux_line or cross_qb == flux_line  + 1 ):
#             freq_shift = 0
#             if (cross_qb == flux_line):
#                 freq_shift = jumplist[cross_qb]
#
#             for j, center in enumerate([0, freq_shift]):
#             # for j, center in enumerate([freq_shift]):
#                 # if j!=0:
#                 setup = lattice_cfg["qubit"]["setup"][cross_qb]
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb==flux_line:
#                     quantum_device_cfg["qubit"][setup]["freq"] = quantum_device_cfg["qubit"][setup]["freq"] + center
#
#                 experiment_cfg[experiment_name]['start'] = -0.030
#                 experiment_cfg[experiment_name]['stop'] = 0.030
#                 experiment_cfg[experiment_name]['step'] = .0005
#                 experiment_cfg[experiment_name]['acquisition_num'] = 250
#                 # if j==0:
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]/1.5
#                 if j==5:
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"] / 1.1
#                 if j==0:
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"] / 1.25
#                 # if j==1 and (cross_qb == flux_line):
#                     # experiment_cfg[experiment_name]['qb_pulse_length'] = piLen_lattice[cross_qb]
#                 #Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['step'] = .001
#                     experiment_cfg[experiment_name]['acquisition_num'] = 2000
#                 if cross_qb == 2:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 500
#                 if cross_qb==6:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 500
#
#
#                 ## Switch between no flux, and flux
#                 if (j==0):
#                     experiment_cfg[experiment_name]['ff_vec'] = [0,0,0,0,0,0,0,0]
#                 else:
#                     experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
#
#                     # if cross_qb == flux_line:
#                     #     # need to shift readout for better signal
#                     #     quantum_device_cfg["readout"][setup]["freq"] = lattice_readout_freq[cross_qb]
#                     #
#                     # else:
#                     #     quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg['readout']['freq'][cross_qb]
#                 quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg['readout'][setup]['freq'][cross_qb]
#
#                 # experiment_cfg[experiment_name]['qb_pulse_length'] =  experiment_cfg[experiment_name]['qb_pulse_length']*2
#                 # quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] =quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] + 6
#                 # experiment_cfg[experiment_name]["qb_amp"]=0.5
#                 experiment_cfg[experiment_name]['ff_pulse_type'] = "square"
#                 print(experiment_cfg[experiment_name]['ff_vec'])
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences(experiment_name)
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


#
#
# ##################################################################################
# #################### UNFINISHED Pairwise Lattice Calibration ###################
# ##################################################################################
#
# # Pair-wise  lattice measurements
# from slab.gerb_fit.gerb_fit_210519 import *
# from tuning_functions import Tuning
# filenames = {
#         "energylistarray_name": "210804_sol_energylistarray.npy",
#         "flxquantaarray_name": "08251204_flxquantaarray.npy",
#         "reslistarray_name": "210802_reslistarray.npy",
#         "qb_reslistarray_name": "210802_qbreslistarray.npy",
#         "FF_SWCTM_name": "210826_FF_SWCTM.npy",
#         "DC_CTM_name": "210803_DC_CrossTalkMatrix.npy",
#         "FF_LocalSlopes_name": "210826_FF_LocalSlopes.npy",
#         "FF_dVdphi_name": "210707_FF_dVdphi.npy",
#         "DC_dVdphi_name": "210802_DC_dVdphi.npy"
#     }
# t = Tuning(filenames)
#
# ## Generate a 2D array of jumplists, V_lists to sweep Q_i,Q_i+1 jumps about lattice
#
#
# V_list =[[ 0.63527146,  0.03041066,  0.01572852,  0.00915228,  0.00793424,
#          0.00818977,  0.00760864],
#        [ 0.00341973, -0.27365483, -0.01976808, -0.0086374 , -0.00685179,
#         -0.00668994, -0.00533627],
#        [-0.00460462, -0.00423405, -0.35363072, -0.02239987, -0.01443018,
#         -0.0122814 , -0.01010023],
#        [ 0.01690322,  0.05129255,  0.03292043,  0.63327636,  0.05836578,
#          0.04646321,  0.03642394],
#        [ 0.04427291,  0.07825429,  0.07262398,  0.08669728, -0.65614138,
#         -0.06712824, -0.06459105],
#        [-0.00820212, -0.01512552, -0.01509731, -0.02021261, -0.02781525,
#          0.39781323, -0.02477274],
#        [ 0.00549871,  0.00760583,  0.00982781,  0.013257  ,  0.01479856,
#          0.02139228, -0.45316229]]
#
#
#
# # # # # # #
# jumplist = [ 0.10206704, -0.04904661,  0.05419289, -0.1039731, 0.08674705, -0.06887273, 0.07128758]
# # jumplist = [0]*8
# #
# #res_freq = [6.16485,  6.39145, 6.61393, 6.83014, 6.95255, 6.7282, 6.50173, 6.27853]
# lattice_readout_freq = [6.16455,6.39125, 6.613775, 6.82979, 6.9526, 6.728, 6.50168,6.27853]
#
# piLen_lattice = [59.58428948336989,
#  30.806186656367757,
#  42.71307477137261,
#  56.950733554042635/2,
#  69.52654149538475/2,
#  22.26893232006009,
#  43.2114415351338,
#         37.25]
#
# experiment_name = "ff_pulse_probe_iq"
#
# for flux_line in [0, 1]:
#     for cross_qb in [0, 1]:
#     # for flux in np.linspace(-V_list[flux_line], V_list[flux_line], 2):
#     #     flux_vec = [0]*8
#         # flux_vec[flux_line] = flux
#         if (cross_qb == flux_line - 1 or  cross_qb == flux_line or cross_qb == flux_line  + 1 ):
#             freq_shift = 0
#             if (cross_qb == flux_line):
#                 freq_shift = jumplist[cross_qb]
#
#             for j, center in enumerate([0, freq_shift]):
#             # for j, center in enumerate([freq_shift]):
#                 # if j!=0:
#                 setup = lattice_cfg["qubit"]["setup"][i]
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb==flux_line:
#                     quantum_device_cfg["qubit"][setup]["freq"] = quantum_device_cfg["qubit"][setup]["freq"] + center
#
#                 experiment_cfg[experiment_name]['start'] = -0.030
#                 experiment_cfg[experiment_name]['stop'] = 0.030
#                 experiment_cfg[experiment_name]['step'] = .0005
#                 experiment_cfg[experiment_name]['acquisition_num'] = 250
#                 if j==0:
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#                 if j==1 and (cross_qb == flux_line):
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = piLen_lattice[cross_qb]
#                 #Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['step'] = .001
#                     experiment_cfg[experiment_name]['acquisition_num'] = 2000
#                 if cross_qb == 6:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 800
#
#                 ## Switch between no flux, and flux
#                 if (j==0):
#                     experiment_cfg[experiment_name]['ff_vec'] = [0,0,0,0,0,0,0,0]
#                 else:
#                     experiment_cfg[experiment_name]['ff_vec'] = [0.619907, -0.326585, -0.0472154, 0.0120497, -0.00971174, 0.0167149, \
# -0.0133336]#V_list[flux_line]
#
#                     if cross_qb == flux_line:
#                     #     need to shift readout for better signal
#                         quantum_device_cfg["readout"][setup]["freq"] = lattice_readout_freq[cross_qb]
#
#                     else:
#                         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg['readout']['freq'][cross_qb]
#                 # experiment_cfg[experiment_name]['qb_pulse_length'] =  experiment_cfg[experiment_name]['qb_pulse_length']*2
#                 # quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] =quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] + 6
#                 # experiment_cfg[experiment_name]["qb_amp"]=0.5
#
#                 print(experiment_cfg[experiment_name]['ff_vec'])
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences(experiment_name)
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
