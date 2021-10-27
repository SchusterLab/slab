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
setup = 'A'
lattice_cfg_name = '210705_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

V_list = [[ 0.32556449,  0.01342028,  0.00622671,  0.00484516,  0.00445977,
         0.0031316 ,  0.00363651],
       [ 0.00689562, -0.26285993, -0.01777962, -0.00815524, -0.00630689,
        -0.0036616 , -0.00711628],
       [ 0.62481905,  0.13474743, -0.5573459 ,  0.09098651, -0.97391766,
        -0.12019276, -0.10893818],
       [ 0.01173371, -0.46314119, -0.03037233,  0.53852987, -0.00200902,
         0.67389202, -0.02071839],
       [ 0.05282506,  0.09389752,  0.08884115,  0.10929427, -0.63019219,
        -0.03666445, -0.76151887],
       [-0.01033089, -0.01603059, -0.01957736, -0.02252642, -0.02839919,
         0.41154625, -0.02712875],
       [ 0.00523075,  0.00900302,  0.00906376,  0.01344745,  0.01380037,
         0.02056385, -0.44417019]]
#
#
experiment_names = ['ff_resonator_spectroscopy']
amplist = np.linspace(0.4,0.8,5)
amplist= [0.666]
attenlist2 = [0,5,10,15,20,25]
#experiment_names = ['resonator_spectroscopy']
amplist = [0.9,1.0]
#for kk,amp in enumerate(amplist):
for i in [0]:
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: i})
    for experiment_name in experiment_names:

        #quantum_device_cfg['readout'][setup]['amp'] = amp
        # print(atten)
        # quantum_device_cfg['powers'][setup]['readout_drive_digital_attenuation'] = atten

        if experiment_name=='ff_resonator_spectroscopy':
            experiment_cfg[experiment_name]['ff_vec'] = V_list[i]

        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        if experiment_name == 'resonator_spectroscopy':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
        if experiment_name == 'ff_resonator_spectroscopy':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, ff=True)



# experiment_names = ['ff_resonator_spectroscopy_pi']
# experiment_names = ['resonator_spectroscopy_pi']
# for (rd_qb, pi_qb) in [(6, 6)]:
#     experiment_cfg['resonator_spectroscopy_pi']["pi_qb"] = []
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: pi_qb})
#     for experiment_name in experiment_names:
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#
#
#
#         # if experiment_name == 'ff_resonator_spectroscopy_pi':
#         #     experiment_cfg[experiment_name]['ff_vec'] = V_list[rd_qb]
#
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True)
#         if experiment_name == 'ff_resonator_spectroscopy_pi':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True, ff=True)
#

# pi pi!
attenlist2 = [5,10,15,20]
amplist = [0.7]
#for kk,amp in enumerate(amplist):
templist = []
for ii in range(7):
    templist.append((0,[ii]))
    templist.append((0,[0,ii]))
experiment_names = ['ff_resonator_spectroscopy_pi']
#experiment_names = ['resonator_spectroscopy_pi']
for (rd_qb, pi_qbs) in templist:
    experiment_cfg['resonator_spectroscopy_pi']["pi_qb"] = pi_qbs
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
    for experiment_name in experiment_names:
        quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
        quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = \
        lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]

        # quantum_device_cfg['powers'][setup]["readout_drive_digital_attenuation"] = atten
        #quantum_device_cfg['readout'][setup]["amp"] = amp
        if experiment_name == 'ff_resonator_spectroscopy_pi':
            experiment_cfg[experiment_name]['ff_vec'] = V_list[rd_qb]
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
        if experiment_name == 'resonator_spectroscopy_pi':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True)
        if experiment_name == 'ff_resonator_spectroscopy_pi':
            exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False, pi=True, ff=True)

# #
# experiment_names = ['pulse_probe_iq']
# # amplist = np.linspace(0.5,1.0,6)
# # for jj,amp in enumerate(amplist):
# # ppiq_avgs = [100,100,100,100,100,100,100,1000]
# ppiq_avgs = [250,250,250,250,250,250,300,1000]
# for i in [0,1,2,3,4,5,6]:
#     # for j in np.linspace(-0.0005,0.0005,10):
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     # quantum_device_cfg['readout'][setup]['amp'] = amp
#     #quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i]
#     # quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq'][i] + j
#     experiment_cfg['pulse_probe_iq']['acquisition_num'] = ppiq_avgs[i]
#     experiment_cfg['pulse_probe_iq']['pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#
#
#     if i == 7:
#         experiment_cfg['pulse_probe_iq']['start'] = -0.120
#         experiment_cfg['pulse_probe_iq']['stop'] = 0.120
#         experiment_cfg['pulse_probe_iq']['step'] = 0.001
#         #experiment_cfg['pulse_probe_iq']['pulse_length'] = 100
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

# #
# experiment_names = ['rabi']
# # # gauss wave
# rabi_start_array = [0,0,0,0,0,0,0,0]
# rabi_stop_array = [800]*8
# rabi_step_array = [8]*8
# rabi_amp_array = [1,1,1,0.71,0.71,1,0.71,1]
#
# rabi_avgs = [500,500,500,500,500,500,500,2000]
# rabi_avgs = [250,250,250,250,250,250,250,1000]
# for i in [0,1,2,3,4,5,6]:
#     # for j in np.linspace(-0.0005,0.0005,10):
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][i]
#     experiment_cfg['rabi']['start'] = rabi_start_array[i]
#     experiment_cfg['rabi']['stop'] = rabi_stop_array[i]
#     experiment_cfg['rabi']['step'] = rabi_step_array[i]
#     experiment_cfg['rabi']['acquisition_num'] = rabi_avgs[i]
#     experiment_cfg['rabi']['amp'] = rabi_amp_array[i]
#     # if i==0:
#         # experiment_cfg['rabi']['amp'] = 0.2
#     quantum_device_cfg['qubit'][setup]['freq'] = lattice_cfg['qubit']['freq'][i]
#     # quantum_device_cfg['readout'][setup]['freq'] = lattice_cfg['readout']['freq'][i] + j
#
#     for experiment_name in experiment_names:
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         if experiment_name == 'resonator_spectroscopy':
#             exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#         else:
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#

# # #
# experiment_names = ['ff_rabi']
# for ii in range(5):
# experiment_names = ['rabi']
# # # # # # gauss
# #     # # rabi_start_array = [0,0,0,0,0,0,0,0]
# #     # # rabi_stop_array = [600]*8
# #     # # rabi_step_array = [6]*8
# #     # #
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
# for rd_qb in [0, 1, 2, 3, 4, 5, 6]:
#     for pi_qb in [0, 1, 2, 3, 4, 5, 6]:
#         # if (rd_qb == 4 and pi_qb >= 6) or (rd_qb > 4):
#         # if rd_qb == pi_qb:
#         for experiment_name in experiment_names:
#             # experiment_cfg[experiment_name]['ff_vec'] = V_list[rd_qb]
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
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#             if experiment_name == 'resonator_spectroscopy':
#                 exp.run_experiment_pxi_resspec(sequences, path, experiment_name, expt_num=0, check_sync=False)
#             else:
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#

# # #
# experiment_names = ['t1']
# t1_start_array = [0,0,0,0,0,0,0,0]
# t1_stop_array = [250000,450000,350000,250000,450000,450000,450000,10000]
# t1_step_array = [20000,40000,30000,20000,40000,40000,40000,500]
# t1_avgs = [1000,1000,1000,1000,1000,1000,1000,5000]
# for i in [0, 1, 2, 3, 4, 5, 6]:
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
# #
# experiment_names = ['ramsey']
# ramsey_start_array = [0,0,0,0,0,0,0,0]
# ramsey_stop_array = [1000,1000,4000,4000,4000,4000,4000,1000]
# ramsey_step_array = [5,5,20,20,20,20,20,10]
# ramsey_freq_array = [0.010,0.010,0.0025,0.0025,0.0025,0.0025,0.0025,0.005,]
# ramsey_avgs = [1000,1000,500,500,500,500,500,2000]
# # while True:
# for i in [0,1,2,3,4,5,6]:
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

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# V_list = [[0.5990336681020711,
#   0.026864992852580552,
#   0.011365024102616743,
#   0.008903455705563,
#   0.007129532479532182,
#   0.007222366357991043,
#   0.0055873525550230235],
#  [0.005117828322312159,
#   -0.2800076155604174,
#   -0.020962522739977694,
#   -0.007880347252124024,
#   -0.008335873828849365,
#   -0.006465541471946751,
#   -0.007674895668861125],
#  [-0.005420164243462385,
#   -0.003962256348477053,
#   -0.32905707969288267,
#   -0.020346691216266182,
#   -0.014204044159143904,
#   -0.011799998143979668,
#   -0.01027120531613697],
#  [0.01570147795023769,
#   0.04903434710751812,
#   0.026860568537171028,
#   0.618755467389919,
#   0.05680220698536698,
#   0.04318875495898233,
#   0.03727282555169338],
#  [0.03951186521786921,
#   0.06805144231601183,
#   0.06619997146914139,
#   0.07671722020632377,
#   -0.5916744295360528,
#   -0.06286924576128675,
#   -0.060551068169962814],
#  [-0.0099324236876094,
#   -0.016195663125753854,
#   -0.01799485087544414,
#   -0.022564185984658917,
#   -0.032652259934095845,
#   0.4299566723988896,
#   -0.02938452930898622],
#  [0.00467158572365611,
#   0.006858581603927008,
#   0.010508038008114971,
#   0.010794018807919805,
#   0.013838086827231774,
#   0.019215364060556506,
#   -0.4138634971709679]]
# # #
# for ii in range(7):
#     experiment_names = ['melting_single_readout_full_ramp']
#     for experiment_name in experiment_names:
#         # for ii,time in evolution([])
#         expt_cfg = experiment_cfg[experiment_name]
#         #expt_cfg['wait_post_flux'] = ii
#
#         #expt_cfg["ff_vec"] = V_list[ii]
#         expt_cfg["rd_qb"] = ii
#         rd_qb = expt_cfg["rd_qb"]
#         expt_cfg["Mott_qbs"] = [3] #expt_cfg["Mott_qbs"]
#
#         qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in expt_cfg["Mott_qbs"]]
#         lo_qb_temp_ind = np.argmax(qb_freq_list)
#         lo_qb = expt_cfg["Mott_qbs"][lo_qb_temp_ind]
#
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         data_path = os.path.join(path, 'data/')
#         melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#         #run melting
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False, seq_data_file=None)
#
#         #run pi_cal
#         pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#         if expt_cfg["pi_calibration"]:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#             experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp"]
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences("pi_cal")
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#             exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                    seq_data_file=None)
#
#             slab_file = SlabFile(melt_file)
#             with slab_file as f:
#                 f.attrs["pi_cal_fname"] = pi_file

#



# Pairwise tunneling
# FF vectors for pairwise tunneling expts - (Q0Q1), (Q1Q2), ... (Q5Q6)

# tunnelingtuningarray =[[0.6043382863096016,
#   -0.25046677490396035,
#   -0.007691141260722036,
#   0.0002886357583266521,
#   0.00040600446502927524,
#   0.0013677486476665751,
#   0.001422597022918243],
#  [-0.002797576759139791,
#   -0.28223280910064347,
#   -0.3430444042984379,
#   -0.027874523385000394,
#   -0.021679391631326643,
#   -0.017169765418033923,
#   -0.016076632010272082],
#  [0.010610735163242502,
#   0.045798568770114,
#   -0.2919873694652839,
#   0.6067637201326526,
#   0.04100117101417293,
#   0.031896935243841594,
#   0.027752679824877398],
#  [0.05633863556594235,
#   0.11756488429155298,
#   0.09482014934414314,
#   0.7017681837530778,
#   -0.5297840206050002,
#   -0.01761123442918173,
#   -0.020638396963140494],
#  [0.0295629125421897,
#   0.05101141011825303,
#   0.04609176713055803,
#   0.05266945373648069,
#   -0.6185260027157709,
#   0.37380020454226304,
#   -0.08729763548538327],
#  [-0.006126276893592233,
#   -0.009470110103434173,
#   -0.011473681944644735,
#   -0.012247608285815831,
#   -0.019940664950618574,
#   0.45355572988773324,
#   -0.4337127646164075]]
#
# experiment_names = ['melting_single_readout_full_ramp']
#
#
# for pi_qb in range(6):
#     for rd_qb in [pi_qb, pi_qb+1]:
#
#         for experiment_name in experiment_names:
#             expt_cfg = experiment_cfg[experiment_name]
#
#             ## Modification for tunneling Expt
#             expt_cfg['ff_vec'] = tunnelingtuningarray[pi_qb]
#             expt_cfg['rd_qb'] = rd_qb
#             expt_cfg['Mott_qbs'] = [pi_qb]
#             # End tunneling expt interjection
#
#             rd_qb = expt_cfg["rd_qb"]
#             Mott_qbs = expt_cfg["Mott_qbs"]
#
#             qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
#             lo_qb_temp_ind = np.argmax(qb_freq_list)
#             lo_qb = Mott_qbs[lo_qb_temp_ind]
#
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#             quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#             quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#
#             data_path = os.path.join(path, 'data/')
#             melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#             # run melting
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                    seq_data_file=None)
#
#             #run pi_cal
#             pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#             if expt_cfg["pi_calibration"]:
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#                 experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp"]
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences("pi_cal")
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#                 exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                        seq_data_file=None)
#
#                 slab_file = SlabFile(melt_file)
#                 with slab_file as f:
#                     f.attrs["pi_cal_fname"] = pi_file


#Quantum random walk + Melting
# #
# experiment_names = ['melting_single_readout_full_ramp']
# for ii in [1,3,5]:
#     for experiment_name in experiment_names:
#         expt_cfg = experiment_cfg[experiment_name]
#
#         ## Modification for random walk
#         expt_cfg['rd_qb'] = ii
#         expt_cfg['Mott_qbs'] = [ii]
#
#         rd_qb = expt_cfg["rd_qb"]
#         Mott_qbs = expt_cfg["Mott_qbs"]
#
#         qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
#         lo_qb_temp_ind = np.argmax(qb_freq_list)
#         lo_qb = Mott_qbs[lo_qb_temp_ind]
#
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         data_path = os.path.join(path, 'data/')
#         melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#         # run melting
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                seq_data_file=None)
#
#         #run pi_cal
#         pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#         if expt_cfg["pi_calibration"]:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#
#             experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp"]
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences("pi_cal")
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#             exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                    seq_data_file=None)
#
#             slab_file = SlabFile(melt_file)
#             with slab_file as f:
#                 f.attrs["pi_cal_fname"] = pi_file



# experiment_names = ['melting_single_readout_full_ramp_Q3']
# # # # # pilenchangelist = np.linspace(-10,10,6)
# # # # # for jj,delta in enumerate(pilenchangelist):
# # # # # for reps in range(10):
# for ii in [0,1,2,3,4,5,6]:
#     for experiment_name in experiment_names:
#         expt_cfg = experiment_cfg[experiment_name]
#
#         ## Modification for random walk
#         expt_cfg['rd_qb'] = ii
#         expt_cfg['Mott_qbs'] = [1,3,5]
#
#         rd_qb = expt_cfg["rd_qb"]
#         Mott_qbs = expt_cfg["Mott_qbs"]
#
#         lo_qb = 3 # modify to always refer to Q3
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         # lattice_cfg['pulse_info'][setup]['pi_len_melt'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'] + np.ones(8)*delta
#
#         data_path = os.path.join(path, 'data/')
#         melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#         # run melting
#         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#         sequences = ps.get_experiment_sequences(experiment_name)
#         print("Sequences generated")
#         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#         exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                seq_data_file=None)
#
#         #run pi_cal
#         pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#         if expt_cfg["pi_calibration"]:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#
#             # quantum_device_cfg['pulse_info'][setup]['pi_len'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'][ii] - (np.ones(8)*delta)[ii]
#
#             experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp_Q3"]
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences("pi_cal")
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#             exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                    seq_data_file=None)
#
#             slab_file = SlabFile(melt_file)
#             with slab_file as f:
#                 f.attrs["pi_cal_fname"] = pi_file

############################################################
############# TESTING MULTI-READOUT RAMP CODE ##############
############################################################
# experiment_names = ['melting_multi_readout_full_ramp_Q3']
# # # pilenchangelist = np.linspace(-10,10,6)
# # # for jj,delta in enumerate(pilenchangelist):
# for experiment_name in experiment_names:
#     expt_cfg = experiment_cfg[experiment_name]
#
#     ## Modification for random walk
#     # expt_cfg['rd_qb'] = ii
#     expt_cfg['Mott_qbs'] = [1,3,5]
#
#     rd_qb = expt_cfg["rd_qb"]
#     Mott_qbs = expt_cfg["Mott_qbs"]
#
#     lo_qb = 3 # modify to always refer to Q3
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#     # THIS SHOULD BE MOVED INTO THE RUN_EXPT_PXI SCRIPT?
#     # quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#     # quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#     # lattice_cfg['pulse_info'][setup]['pi_len_melt'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'] + np.ones(8)*delta
#
#     data_path = os.path.join(path, 'data/')
#     melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#     # run melting
#     ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#     sequences = ps.get_experiment_sequences(experiment_name)
#     print("Sequences generated")
#     exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name,lattice_cfg)
#     exp.run_experiment_pxi_melt(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                            seq_data_file=None)



############################################################
############ T1 EFFECTIVE OVER RAMP EXPERIMENT #############
############################################################
## Tune each qubit independently - avoid NNN collisions

#
# Vlist = [[ 0.32870921,  0.01573542,  0.00813843,  0.00473567,  0.00410542,
#          0.00423764,  0.00393695],
#        [ 0.00329826, -0.26393504, -0.01906595, -0.00833061, -0.00660842,
#         -0.00645232, -0.00514674],
#        [ 0.63332699,  0.13344196, -0.55161402,  0.09189483, -0.97461612,
#         -0.11352074, -0.10621645],
#        [ 0.00851155, -0.46386104, -0.02903834,  0.54286124, -0.00300122,
#          0.67916629, -0.01583193],
#        [ 0.0524714 ,  0.08948378,  0.08731939,  0.1065936 , -0.62903438,
#         -0.03367583, -0.76355934],
#        [-0.00859071, -0.0158421 , -0.01581256, -0.0211702 , -0.02913302,
#          0.41665998, -0.02594637],
#        [ 0.00540107,  0.00747078,  0.0096533 ,  0.0130216 ,  0.01453579,
#          0.02101243, -0.44511561]]

# ramplenlist = [1.00000000e+00, 1.24407162e+00, 1.54771421e+00, 1.92546733e+00,
#        2.39541927e+00, 2.98007314e+00, 3.70742444e+00, 4.61230155e+00,
#        5.73803348e+00, 7.13852463e+00, 8.88083594e+00, 1.10483960e+01,
#        1.37449960e+01, 1.70997595e+01, 2.12733255e+01, 2.64655407e+01,
#        3.29250282e+01, 4.09610933e+01, 5.09585339e+01, 6.33960661e+01,
#        7.88692470e+01, 9.81189923e+01, 1.22067054e+02, 1.51860158e+02,
#        1.88924914e+02, 2.35036125e+02, 2.92401774e+02, 3.63768750e+02,
#        4.52554380e+02, 5.63010063e+02, 7.00424844e+02, 8.71378673e+02,
#        1.08405748e+03, 1.34864515e+03, 1.67781117e+03, 2.08731727e+03,
#        2.59677218e+03, 3.23057059e+03, 4.01906120e+03, 5.00000000e+03]

# Loop over different shapes as well!
# Can only go for like 30us before epxonential functions fail, adjust pulse sequence generation for this

# experiment_names = ['melting_single_readout_full_ramp_Q3']
# # for kk,len in enumerate(ramplenlist):
# for jj,ff_vec in enumerate(Vlist):
#     for ii in [0,1,2,3,4,5,6]:
#         if jj == ii:
#             for experiment_name in experiment_names:
#                 expt_cfg = experiment_cfg[experiment_name]
#                 expt_cfg['evolution_t_start'] = 0
#                 expt_cfg['evolution_t_step'] = 500#round(500/len)
#                 expt_cfg['evolution_t_stop'] = 30000
#
#                 # TODO: Import latticecfg, modify exponential ramp parameters
#
#
#                 ## Modification for random walk
#                 expt_cfg['rd_qb'] = ii
#                 expt_cfg['Mott_qbs'] = [ii]
#
#                 rd_qb = expt_cfg["rd_qb"]
#                 Mott_qbs = expt_cfg["Mott_qbs"]
#
#                 lo_qb = 3 # modify to always refer to Q3
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#                 quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#                 quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#                 # lattice_cfg['pulse_info'][setup]['pi_len_melt'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'] + np.ones(8)*delta
#                 expt_cfg['ff_vec'] = ff_vec
#
#                 data_path = os.path.join(path, 'data/')
#                 melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#                 # lattice_cfg['ff_info']['ff_exp_ramp_len'] = [len,len,len,len,len,len,len,len]
#                 # lattice_cfg['ff_info']['ff_exp_ramp_tau'] = [len*(2/5),len*(2/5),len*(2/5),len*(2/5),len*(2/5),len*(2/5),len*(2/5),len*(2/5)]
#
#                 # run melting
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences(experiment_name)
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                        seq_data_file=None)
#
#                 #run pi_cal
#                 pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#                 if expt_cfg["pi_calibration"]:
#                     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#
#                     # quantum_device_cfg['pulse_info'][setup]['pi_len'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'][ii] - (np.ones(8)*delta)[ii]
#
#                     experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp_Q3"]
#                     ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                     sequences = ps.get_experiment_sequences("pi_cal")
#                     print("Sequences generated")
#                     exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#                     exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                            seq_data_file=None)
#
#                     slab_file = SlabFile(melt_file)
#                     with slab_file as f:
#                         f.attrs["pi_cal_fname"] = pi_file


############################################################
############ T1 SIDEBAND #############
############################################################
# experiment_names = ['sideband_t1']
# sb_freq_list = 10**(np.linspace(0, 2, 15))/1000
# for sb_freq in sb_freq_list:
#     for sb_qb in [0]:
#         for rd_qb in [0]:
#             for experiment_name in experiment_names:
#                 expt_cfg = experiment_cfg[experiment_name]
#
#                 expt_cfg['rd_qb'] = rd_qb
#                 expt_cfg['sb_qb'] = sb_qb
#                 expt_cfg["sb_freq"]=sb_freq
#
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: sb_qb})
#
#                 quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][expt_cfg["rd_qb"]]
#                 quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][expt_cfg["rd_qb"]]
#
#                 data_path = os.path.join(path, 'data/')
#                 melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#
#                 # run sideband
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences(experiment_name)
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                        seq_data_file=None)
#
#                 #run pi_cal
#                 pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#                 if expt_cfg["pi_calibration"]:
#                     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: expt_cfg["rd_qb"]})
#
#                     experiment_cfg["pi_cal"] = experiment_cfg[experiment_name]
#                     ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                     sequences = ps.get_experiment_sequences("pi_cal")
#                     print("Sequences generated")
#                     exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#                     exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                            seq_data_file=None)
#
#                     slab_file = SlabFile(melt_file)
#                     with slab_file as f:
#                         f.attrs["pi_cal_fname"] = pi_file

# ##################################################################################
# #################### ADIABATICITY AND REVERSABILITY EXPERIMENT ###################
# ##################################################################################

# expramplenlist = [1.00000000e+00, 1.24407162e+00, 1.54771421e+00, 1.92546733e+00,
#        2.39541927e+00, 2.98007314e+00, 3.70742444e+00, 4.61230155e+00,
#        5.73803348e+00, 7.13852463e+00, 8.88083594e+00, 1.10483960e+01,
#        1.37449960e+01, 1.70997595e+01, 2.12733255e+01, 2.64655407e+01,
#        3.29250282e+01, 4.09610933e+01, 5.09585339e+01, 6.33960661e+01,
#        7.88692470e+01, 9.81189923e+01, 1.22067054e+02, 1.51860158e+02,
#        1.88924914e+02, 2.35036125e+02, 2.92401774e+02, 3.63768750e+02,
#        4.52554380e+02, 5.63010063e+02, 7.00424844e+02, 8.71378673e+02,
#        1.08405748e+03, 1.34864515e+03, 1.67781117e+03, 2.08731727e+03,
#        2.59677218e+03, 3.23057059e+03, 4.01906120e+03, 5.00000000e+03]
#
# experiment_names = ['melting_single_readout_full_ramp_Q3']
# # pilenchangelist = np.linspace(-10,10,6)
# for jj,expramplen in enumerate(expramplenlist):
#     for ii in [0, 1, 2, 3, 4, 5, 6]:
#         for experiment_name in experiment_names:
#             expt_cfg = experiment_cfg[experiment_name]
#
#             expt_cfg['rd_qb'] = ii
#             expt_cfg['Mott_qbs'] = [1,3,5]
#             expt_cfg['ramp_type'] = "rexp"
#             rd_qb = expt_cfg["rd_qb"]
#             Mott_qbs = expt_cfg["Mott_qbs"]
#             lo_qb = 3 # modify to always refer to Q3
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#             lattice_cfg['ff_info']['ff_exp_ramp_len'] = np.ones(7)*expramplen
#             lattice_cfg['ff_info']['ff_exp_ramp_tau'] = np.ones(7)*expramplen*(2/5)
#             quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#             quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#             data_path = os.path.join(path, 'data/')
#             melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#
#             # run melting
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
#                                    seq_data_file=None)
#
#             #run pi_cal
#             pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
#             if expt_cfg["pi_calibration"]:
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})
#
#                 # quantum_device_cfg['pulse_info'][setup]['pi_len'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'][ii] - (np.ones(8)*delta)[ii]
#
#                 experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp_Q3"]
#                 ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#                 sequences = ps.get_experiment_sequences("pi_cal")
#                 print("Sequences generated")
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal')
#                 exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
#                                        seq_data_file=None)
#
#                 slab_file = SlabFile(melt_file)
#                 with slab_file as f:
#                     f.attrs["pi_cal_fname"] = pi_file

# HERE
# Flux/qubit tests/
# # this is for wiggle local slope measurement
# V_list = [[0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.  , 0.02, 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.  , 0.  , 0.02, 0.  , 0.  , 0.  , 0.  ],
#        [0.  , 0.  , 0.  , 0.02, 0.  , 0.  , 0.  ],
#        [0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.  ],
#        [0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  ],
#        [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.02]]


# # this is for CTM measurement
# V_list = [[ 0.32786727,  0.        ,  0.        ,  0.        ,  0.        ,
#          0.        ,  0.        ],
#        [ 0.        , -0.26548862,  0.        ,  0.        ,  0.        ,
#          0.        ,  0.        ],
#        [ 0.        ,  0.        , -0.67336828,  0.        ,  0.        ,
#          0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.59550323,  0.        ,
#          0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        , -0.67169369,
#          0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#          0.41975652,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#          0.        , -0.44608034]]
#
# # # # #
# # # # # # #
# jumplist = [ 0.05262813, -0.04755308,  0.10311684, -0.09683284,  0.08624327, -0.07188669,  0.06991989, -1.247     ]
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
# for flux_line in [0, 1, 2, 3, 4, 5, 6]:
#     for cross_qb in [0, 1, 2, 3, 4, 5, 6]:
#         # if cross_qb == flux_line:
#     # for flux in np.linspace(-V_list[flux_line], V_list[flux_line], 2):
#     #     flux_vec = [0]*8
#         # flux_vec[flux_line] = flux
#         #if (cross_qb == flux_line - 1 or  cross_qb == flux_line or cross_qb == flux_line  + 1 ):
#         freq_shift = 0
#         if (cross_qb == flux_line):
#             freq_shift = jumplist[cross_qb]
#
#         for j, center in enumerate([0, freq_shift]):
#         # for j, center in enumerate([freq_shift]):
#             # if j!=0:
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#             ## truncate frequency domain to qubit center, or tuned frequency center + window
#             if cross_qb==flux_line:
#                 quantum_device_cfg["qubit"][setup]["freq"] = quantum_device_cfg["qubit"][setup]["freq"] + center
#
#             experiment_cfg[experiment_name]['start'] = -0.030
#             experiment_cfg[experiment_name]['stop'] = 0.030
#             experiment_cfg[experiment_name]['step'] = .0005
#             experiment_cfg[experiment_name]['acquisition_num'] = 250
#             if j==0:
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#             if j==1 and (cross_qb == flux_line):
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = piLen_lattice[cross_qb]
#             #Q7 needs twice as much averaging and a shorter pi-time
#             if cross_qb ==7:
#                 experiment_cfg[experiment_name]['step'] = .001
#                 experiment_cfg[experiment_name]['acquisition_num'] = 2000
#             if cross_qb == 6:
#                 experiment_cfg[experiment_name]['acquisition_num'] = 800
#
#             ## Switch between no flux, and flux
#             if (j==0):
#                 experiment_cfg[experiment_name]['ff_vec'] = [0,0,0,0,0,0,0,0]
#             else:
#                 experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
#
#                 if cross_qb == flux_line:
#                 #     need to shift readout for better signal
#                     quantum_device_cfg["readout"][setup]["freq"] = lattice_readout_freq[cross_qb]
#
#                 else:
#                     quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg['readout']['freq'][cross_qb]
#             # experiment_cfg[experiment_name]['qb_pulse_length'] =  experiment_cfg[experiment_name]['qb_pulse_length']*2
#             # quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] =quantum_device_cfg["powers"][setup]["drive_digital_attenuation"] + 6
#             # experiment_cfg[experiment_name]["qb_amp"]=0.5
#             experiment_cfg[experiment_name]['ff_pulse_type'] = "square"
#             print(experiment_cfg[experiment_name]['ff_vec'])
#             ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
#             sequences = ps.get_experiment_sequences(experiment_name)
#             print("Sequences generated")
#             exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#             exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


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
#                 exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
#                 exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
#
