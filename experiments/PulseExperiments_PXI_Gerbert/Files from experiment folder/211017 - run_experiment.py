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
lattice_cfg_name = '211008_2qb_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)


# ##################################################################################
# #################### Pair-Wise Tunneling ###################
# ##################################################################################
# FF vectors for pairwise tunneling expts - (Q0Q1), (Q1Q2), ... (Q5Q6)

tunnelingtuningarray = [0.0021537070237979076, 0.042709406303825075, -0.6445610650169042, 0.5441308722777675, 0.026913406030430572, 0.01586734077476971, 0.013037553438887461]

# experiment_names = ['melting_single_readout_full_ramp']
# for ii in [2]:
#     for experiment_name in experiment_names:
#         expt_cfg = experiment_cfg[experiment_name]
#
#         ## Modification for random walk
#         expt_cfg['rd_qb'] = ii
#         expt_cfg['Mott_qbs'] = [3]
#
#         rd_qb = expt_cfg["rd_qb"]
#         Mott_qbs = expt_cfg["Mott_qbs"]
#
#         qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
#         lo_qb_temp_ind = np.argmax(qb_freq_list)
#         lo_qb = 3#Mott_qbs[lo_qb_temp_ind]
#
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#         quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         data_path = os.path.join(path, 'data/')
#         melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#         expt_cfg['ff_vec'] = tunnelingtuningarray
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
# tunnelingtuningarray = [0.0021537070237979076, 0.042709406303825075, -0.6445610650169042, 0.5441308722777675, 0.026913406030430572, 0.01586734077476971, 0.013037553438887461]
#
# experiment_names = ['pop_swap_for_debugging']
# for ii in [2]:
#     for experiment_name in experiment_names:
#         expt_cfg = experiment_cfg[experiment_name]
#
#         ## Modification for random walk
#
#         # qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
#         # lo_qb_temp_ind = np.argmax(qb_freq_list)
#         lo_qb = 0#Mott_qbs[lo_qb_temp_ind]
#
#         quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#         # quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#         # quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#         data_path = os.path.join(path, 'data/')
#         melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))
#         expt_cfg['ff_vec'] = tunnelingtuningarray
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
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
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

# ##################################################################################
# #################### Normal Melting Code (Ref Q3 LO) ###################
# ##################################################################################
experiment_names = ['melting_single_readout_full_ramp_Q3']
# # # # pilenchangelist = np.linspace(-10,10,6)
# # # # for jj,delta in enumerate(pilenchangelist):
# # # # for reps in range(10):
avgs_list = [2000]*8
avgs_list[1]= 2000
for ii in [0,1,2,3,4,5,6]:
    for experiment_name in experiment_names:
        expt_cfg = experiment_cfg[experiment_name]

        ## Modification for random walk
        expt_cfg['rd_qb'] = ii
        expt_cfg['Mott_qbs'] = [5]
        expt_cfg['acquisition_num'] = avgs_list[ii]

        rd_qb = expt_cfg["rd_qb"]
        Mott_qbs = expt_cfg["Mott_qbs"]

        lo_qb = 5 # modify to always refer to Q3
        quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})

        quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"][setup]["freq"][rd_qb]
        quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]

        # lattice_cfg['pulse_info'][setup]['pi_len_melt'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'] + np.ones(8)*delta

        data_path = os.path.join(path, 'data/')
        melt_file = os.path.join(data_path, get_next_filename(data_path, experiment_name, suffix='.h5'))

        # run melting
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
        sequences = ps.get_experiment_sequences(experiment_name)
        print("Sequences generated")
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
        exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
                               seq_data_file=None)

        #run pi_cal
        pi_file = os.path.join(data_path, get_next_filename(data_path, "pi_cal", suffix='.h5'))
        if expt_cfg["pi_calibration"]:
            quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: rd_qb})

            # quantum_device_cfg['pulse_info'][setup]['pi_len'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'][ii] - (np.ones(8)*delta)[ii]

            experiment_cfg["pi_cal"] = experiment_cfg["melting_single_readout_full_ramp_Q3"]
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences("pi_cal")
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, 'pi_cal', lattice_cfg=lattice_cfg)
            exp.run_experiment_pxi(sequences, path, 'pi_cal', expt_num=0, check_sync=False,
                                   seq_data_file=None)

            slab_file = SlabFile(melt_file)
            with slab_file as f:
                f.attrs["pi_cal_fname"] = pi_file

# ##################################################################################
# #################### Normal Melting Code (Ref Q3 LO) ###################
# #################### with detuning factor for transversing the Phase Transition ####################
# ##################################################################################


# experiment_names = ['melting_single_readout_full_ramp_Q3']
#
# ## ff vector for going to the SF
# expt_cfg = experiment_cfg['melting_single_readout_full_ramp_Q3']
# FFvec = expt_cfg['ff_vec']
#
# # detuning factor
# detfactor = np.array([0.        , 0.21524003, 0.38415179, 0.51670698, 0.62073098,
#        0.70236486, 0.76642785, 0.81670193, 0.85615501, 0.88711621,
#        0.91141332, 0.93048072, 0.94544405, 0.95718668, 0.96640182,
#        0.97363349, 0.97930862, 0.98376223, 0.98725725, 0.99      ,
#        1.        ])
#
# avgs_list = [4000]*8
# avgs_list[1]= 4000
#
# for jj in range(len(detfactor)):
#     for ii in [0,1,2,3,4,5,6]:
#         for experiment_name in experiment_names:
#             expt_cfg = experiment_cfg[experiment_name]
#
#             # update FF vector
#             expt_cfg['ff_vec'] = list(np.array(FFvec) * detfactor[jj])
#
#             ## Modification for random walk
#             expt_cfg['rd_qb'] = ii
#             expt_cfg['Mott_qbs'] = [3]
#             expt_cfg['acquisition_num'] = avgs_list[ii]
#
#             rd_qb = expt_cfg["rd_qb"]
#             Mott_qbs = expt_cfg["Mott_qbs"]
#
#             lo_qb = 3 # modify to always refer to Q3
#             quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#
#             quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#             quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#             # lattice_cfg['pulse_info'][setup]['pi_len_melt'] = lattice_cfg['pulse_info'][setup]['pi_len_melt'] + np.ones(8)*delta
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

##################################################################################
#################### Rabi ###################
##################################################################################
#
# experiment_names = ['rabi_pi']
# # # gauss wave
# rabi_start_array = [0,0,0,0,0,0,0,0]
# rabi_stop_array = [800]*8
# rabi_step_array = [8]*8
# rabi_amp_array = [1,1,1,0.71,0.71,1,0.71,1]
#
# rabi_avgs = [500,500,500,500,500,500,500,2000]
# rabi_avgs = [250,4000,250,250,250,250,250,1000]
# for i in [5]:
#     # for j in np.linspace(-0.0005,0.0005,10):
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][i]
#     experiment_cfg['rabi_pi']['start'] = rabi_start_array[i]
#     experiment_cfg['rabi_pi']['stop'] = rabi_stop_array[i]
#     experiment_cfg['rabi_pi']['step'] = rabi_step_array[i]
#     experiment_cfg['rabi_pi']['acquisition_num'] = rabi_avgs[i]
#     experiment_cfg['rabi_pi']['amp'] = rabi_amp_array[i]
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
#         exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


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
