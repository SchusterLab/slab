import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI_Gerbert.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import generate_quantum_device_from_lattice_v3
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import melting_update

from slab.instruments import *


from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile

import copy
import json
import numpy as np
import os
import time
import pickle
import sys

from tuning_functions import Tuning

path = os.getcwd()

path = "C:\\210801 - PHMIV3_56 - BF4 cooldown 4"

show = 'I'

lattice_cfg_name = '211008_2qb_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)


data_path = os.path.join(path, 'data\\')

# /////////////////////////////////////////////////////////////////////////////////////////////////////////
# setup cont data file
# /////////////////////////////////////////////////////////////////////////////////////////////////////////

dac = AD5780_serial()

qb_list = [0, 1, 2, 3, 4, 5, 6]
Q7_FREQ = 4.35
cont_name = ["00005_cont_100Mhz_Q%s"%qb for qb in qb_list]
cont_data_file = [os.path.join(data_path, cont_name[qb] +".h5") for qb in qb_list]

slab_file = [SlabFile(file) for file in cont_data_file]

ABORT_DAC_SHIFT = 0.1
ABORT_TOTAL_DAC_SHIFT = 1

ABORT_PPIQ_SHIFT = 0.015
ACCEPTABLE_PPIQ_SHIFT = 0.001

ABORT_RABI_PI_RATIO = 0.5

intended_freq_list_start = [5.105583,5.269932,5.07874,5.313972,5.163985,5.34145,5.136182,4.35]
flux_list =[-0.3924474, 1.02115336, 1.20782875, 0.18441662, -0.30690919, 0.0035382, 0.10224945, -0.1158568]
flux_list_start =[-0.3924474, 1.02115336, 1.20782875, 0.18441662, -0.30690919, 0.0035382, 0.10224945, -0.1158568]
filenames = {
        "energylistarray_name": "211004_energylistarray.npy",
        "flxquantaarray_name": "1020012_flxquantaarray.npy",
        "reslistarray_name": "211004_reslistarray.npy",
        "qb_reslistarray_name": "211004_qbreslistarray.npy",
        "FF_SWCTM_name": "211015_FF_SWCTM.npy",
        "DC_CTM_name": "210803_DC_CrossTalkMatrix.npy",
        "FF_LocalSlopes_name": '211015_FF_LocalSlopesWiggle.npy',
        "FF_dVdphi_name": "210707_FF_dVdphi.npy",
        "DC_dVdphi_name": "211004_DC_dVdphi.npy"
    }
flx_name = filenames['flxquantaarray_name']

show = 'I'


# large loop: incrimentally update the frequency of the whole lattice
for i in np.arange(0,100,3):

    intended_freq_list = [freq + i/1000 for freq in intended_freq_list_start] # update lattice frequency
    intended_freq_list[7] = Q7_FREQ
    lattice_cfg["qubit"]["freq"] = intended_freq_list

# /////////////////////////////////////////////////////////////////////////////////////////////////////////
# DAQ Update
# /////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Enter dac tuning/ppiq feedback look

    hit_freqs = False #flag: have we tuned correctly to the intended frequencies?
    loop = 0 # keep track of how many times go through loop
    meta_list = [[] for i in qb_list]
    p_list = [[] for i in qb_list] # keep track of qubit fits every time we go through loop

    while not hit_freqs:

        # while not hit_freqs, tune dac to intended freq list
        tuna = Tuning(filenames, 8)
        flux_list_new = tuna.omega_to_V_thru_CTM("DC", intended_freq_list)

        if np.any(np.abs(np.array(flux_list_new)- np.array(flux_list))>ABORT_DAC_SHIFT):
            print("HUGE DAC SHIFT! exiting program")
            sys.exit()

        if np.any(np.abs(np.array(flux_list_start)- np.array(flux_list_new))>ABORT_TOTAL_DAC_SHIFT):
            print("TOTAL DAC SHIFT EXCEEND 1V! exiting program")
            sys.exit()

        flux_list = copy.copy(flux_list_new)

        dac.parallelramp(flux_list,1,1)

# # /////////////////////////////////////////////////////////////////////////////////////////////////////////
# # PPIQ
# # /////////////////////////////////////////////////////////////////////////////////////////////////////////

        # after tuning, run ppiq on all qubits
        experiment_names = ['pulse_probe_iq']

        freqlist = []
        ppiq_avgs = [150,150,150,150,150,150,150,150]
        ppiq_amps = [1,1,1,1,1,0.5,1,1]
        for i in qb_list:
            print("Start a CONT experiment! Cont_name is: " + cont_data_file[i])
            setup = lattice_cfg["qubit"]["setup"][i]
            quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i}, lattice_cfg_dict=lattice_cfg)

            experiment_cfg['pulse_probe_iq']['acquisition_num'] = ppiq_avgs[i]
            experiment_cfg['pulse_probe_iq']['amp'] = ppiq_amps[i]
            experiment_cfg['pulse_probe_iq']['pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
            experiment_cfg['pulse_probe_iq']['amp'] = quantum_device_cfg["pulse_info"][setup]["pi_amp"]

            for experiment_name in experiment_names:
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("Sequences generated")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
                exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
                print("about to begin post-analysis")
                PA = exp.post_analysisandsave(path, experiment_name, cont_name[i], P=show, phi=quantum_device_cfg[
                    "readout"][
                    setup]["phi"]*180/np.pi, cont_data_file=cont_data_file[i], save=False, obj=True)
                freqlist.append(PA.p[2])

                #append ppiq loop data to cont data file
                exp_nb = PA.current_file_index(prefix=PA.exptname)
                pulse_probe_meta = [exp_nb, time.time()]
                meta_list[i].append(pulse_probe_meta)
                p_list[i].append(PA.p)
                with SlabFile(PA.cont_data_file) as file:
                    file.append_line('ppiq_loop_meta', pulse_probe_meta)
                    file.append_line('ppiq_loop_fit', PA.p)
                    file.append_line('dac_flux', flux_list)
                    print("appended line correctly")

            print("End a CONT experiment! Cont_name is: " + cont_data_file[i])

        freqlist.append(Q7_FREQ)
        if np.any(np.abs(np.array(freqlist)- np.array(intended_freq_list))>ABORT_PPIQ_SHIFT):
            # if we are off from intended frequencies by a catastraphic amount, abort
            print("HUGE SHIFT! exiting program")
            sys.exit()

        if np.any(np.abs(np.array(freqlist)- np.array(intended_freq_list))>ACCEPTABLE_PPIQ_SHIFT):
            # if we are off from intended frequencies by a small but non-catastrophic amount, go through loop again
            print("didn't hit freqs the first time")

            og_dir = os.getcwd()
            os.chdir("C:\\210801 - PHMIV3_56 - BF4 cooldown 4\\ipython notebook")
            flx_name = tuna.correct_flux_offsets(intended_freq= intended_freq_list, measured_freq=freqlist);
            os.chdir(og_dir)
            filenames['flxquantaarray_name'] = flx_name
            loop = loop + 1
            if loop>4:
                print("DAC can't converge on ppiq freq. ABORT!")
                sys.exit()

        else:
            #if we hit frequencies we want, exit loop!
            hit_freqs = True
            lattice_cfg["qubit"]["freq"] = freqlist
            IQlist = []
            for ii in [0, 1, 2, 3, 4, 5, 6]:
                if ii == 1 or ii == 3 or ii == 5:
                    IQlist.append(-0.230 + (freqlist[ii] - freqlist[3]))
                if ii == 0 or ii == 2 or ii == 4 or ii == 6:
                    IQlist.append(-0.187 + (freqlist[ii] - freqlist[4]))
            IQlist.append(-0.185)
            lattice_cfg['pulse_info']['A']['iq_freq'] = IQlist
            lattice_cfg['pulse_info']['B']['iq_freq'] = IQlist

            #update cont data file with finalized qubit frequencies
            for i in qb_list:
                with SlabFile(cont_data_file[i]) as file:
                    file.append_line('ppiq_loop_nb', [loop])
                    file.append_line('ppiq_meta', meta_list[i][-1])
                    file.append_line('ppiq_fit', p_list[i][-1])
                    print("appended line correctly")

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # rabi
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    #run rabi

    experiment_names = ['rabi']
    rabi_start_array = [0,0,0,0,0,0,0,0]
    rabi_stop_array = [800]*8
    rabi_step_array = [8]*8
    rabi_amp_array = [1,1,1,1,1,0.5,1,1]
    rabi_avgs = [200,200,200,200,200,200,200,1000]

    for i in qb_list:
        setup = lattice_cfg["qubit"]["setup"][i]
        quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i}, lattice_cfg_dict=lattice_cfg)
        quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][i]
        experiment_cfg['rabi']['start'] = rabi_start_array[i]
        experiment_cfg['rabi']['stop'] = rabi_stop_array[i]
        experiment_cfg['rabi']['step'] = rabi_step_array[i]
        experiment_cfg['rabi']['acquisition_num'] = rabi_avgs[i]
        experiment_cfg['rabi']['amp'] = rabi_amp_array[i]

        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
            PA = exp.post_analysisandsave(path, experiment_name, cont_name[i], P=show, phi=quantum_device_cfg[
                "readout"][setup]["phi"]*180/np.pi, cont_data_file=cont_data_file[i], save=False, obj=True)

            #append rabi results to cont data file
            exp_nb = PA.current_file_index(prefix=PA.exptname)
            rabi_meta = [exp_nb, time.time()]
            with SlabFile(PA.cont_data_file) as file:
                file.append_line('rabi_meta', rabi_meta)
                file.append_line('rabi_fit', PA.p)
                print("appended line correctly")

            t_pi = 1 / (2 * PA.p[1])
            if lattice_cfg["pulse_info"]["pulse_type"][i] == 'gauss':
                t_pi = t_pi / 4

            # check if weirdly large change in rabi time
            pi_len_og = lattice_cfg["pulse_info"][setup]["pi_len"][i]
            if t_pi > (ABORT_RABI_PI_RATIO + 1)* pi_len_og or t_pi < (ABORT_RABI_PI_RATIO)* pi_len_og:
                print("rabi time changed by more that {} percent!".format(ABORT_RABI_PI_RATIO*100))
                t_pi = pi_len_og

            #if not, append and continue!
            else:
                lattice_cfg["pulse_info"]["A"]["pi_len"][i] = t_pi
                lattice_cfg["pulse_info"]["B"]["pi_len"][i] = t_pi

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # histogram
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # get new phi
    # experiment_names = ['histogram']
    # for i in qb_list:
    #     setup = lattice_cfg["qubit"]["setup"][i]
    #     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i}, lattice_cfg_dict=lattice_cfg)
    #     for experiment_name in experiment_names:
    #         ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
    #         sequences = ps.get_experiment_sequences(experiment_name)
    #         print("Sequences generated")
    #         exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
    #         exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
    #         PA = exp.post_analysisandsave(path, experiment_name, cont_name[i], P=show, phi=quantum_device_cfg[
    #             "readout"][setup]["phi"]*180/np.pi, cont_data_file=cont_data_file[i], save=False, obj=True)
    #
    #     # append t1 results to cont data file
    #     exp_nb = PA.current_file_index(prefix=PA.exptname)
    #     hist_meta = [exp_nb, time.time()]
    #     with SlabFile(PA.cont_data_file) as file:
    #         file.append_line('hist_meta', hist_meta)
    #         file.append_line('hist_fit', PA.p)
    #         print("appended line correctly")
    #
    #     lattice_cfg["readout"]["A"]["phi"][i] = PA.p[1]
    #     lattice_cfg["readout"]["B"]["phi"][i] = PA.p[1]

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # t1
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # get t1
    experiment_names = ['t1']
    t1_start_array = [0,0,0,0,0,0,0,0]
    t1_stop_array = [250000,450000,350000,250000,450000,450000,450000,10000]
    t1_step_array = [20000,40000,30000,20000,40000,40000,40000,500]
    t1_avgs = [1000,1000,1000,1000,1000,1000,1000,5000]
    for i in qb_list:
        setup = lattice_cfg["qubit"]["setup"][i]
        quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i}, lattice_cfg_dict=lattice_cfg)
        experiment_cfg['t1']['start'] = t1_start_array[i]
        experiment_cfg['t1']['stop'] = t1_stop_array[i]
        experiment_cfg['t1']['step'] = t1_step_array[i]
        experiment_cfg['t1']['acquisition_num'] = t1_avgs[i]
        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, lattice_cfg=lattice_cfg)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)
            PA = exp.post_analysisandsave(path, experiment_name, cont_name[i], P=show, phi=quantum_device_cfg[
                "readout"][setup]["phi"]*180/np.pi, cont_data_file=cont_data_file[i], save=False, obj=True)

        # append t1 results to cont data file
        exp_nb = PA.current_file_index(prefix=PA.exptname)
        t1_meta = [exp_nb, time.time()]
        with SlabFile(PA.cont_data_file) as file:
            file.append_line('t1_meta', t1_meta)
            file.append_line('t1_fit', PA.p)
            print("appended line correctly")