from slab.experiments.PulseExperiments_PXI_Gerbert.sequential_experiment_pxi import SequentialExperiment
from slab.experiments.PulseExperiments_PXI_Gerbert.pulse_experiment import generate_quantum_device_from_lattice_v3
import json
from loadNsave_lattice import load_lattice_to_quantum_device
from loadNsave_lattice import load_quantum_device_to_lattice
from loadNsave_lattice import generate_quantum_device_from_lattice
import numpy as np
import os
import cProfile
path = os.getcwd()
path = "C:\\210801 - PHMIV3_56 - BF4 cooldown 4"

lattice_cfg_name = '211004_2qb_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

setup = 'A'


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
experiment_name = 'rabi_chevron'
#for atten in [15,20,25]:
for i in [0,1,2,3,4,5,6] :
    setup = lattice_cfg["qubit"]["setup"][i]
    quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
    #quantum_device_cfg['powers'][setup]['drive_digital_attenuation'] = atten
    SequentialExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, experiment_name, path,
                         analyze=False, show=False)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
#

# experiment_name = 'ff_ramp_cal_ppiq'
# # for len in [30, 45, 60, 75, 90, 105, 120]
# for i in [0,1,2,3,4,5,6]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     fluxvec =   [ 0.00449093,  0.00720628,  0.00833802,  0.01046896,  0.01268877,
#         0.01870046, -0.40597256]
#     # quantum_device_cfg["readout"][setup]["freq"] = 6.95274431
#     # quantum_device_cfg["qubit"][setup]["freq"] = 4.9
#     experiment_cfg[experiment_name]['ff_vec'] = fluxvec
#     experiment_cfg[experiment_name]['qb_pulse_length'] = 20#quantum_device_cfg["pulse_info"][setup]["pi_len"]
#     if i == 7:
#         experiment_cfg[experiment_name]['start'] = -0.080
#         experiment_cfg[experiment_name]['stop'] = 0.080
#         experiment_cfg[experiment_name]['step'] = 0.002
#         experiment_cfg[experiment_name]['acquisition_num'] = 1000
#     SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
# for i in [0]:
#     experiment_name = 'ff_track_traj'
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#     SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# FF_sweep_J
# # FF vectors for pairwise tunneling expts - (Q0Q1), (Q1Q2), ... (Q6Q7)
# tunnelingtuningarray = [[0.6111356561464186,
#   -0.2409600465624277,
#   -0.009096762534661785,
#   -0.0001603163141109757,
#   0.001986750019511134,
#   0.0016582130820045607,
#   0.004321265852499337],
#  [-0.0026592946276772634,
#   -0.27404228353397614,
#   -0.3313054512081607,
#   -0.028381893351336288,
#   -0.018583320844547477,
#   -0.016616870814918708,
#   -0.01553000885834576],
#  [0.010417972180164046,
#   0.04861520958943927,
#   -0.28181735352311194,
#   0.6066331636539565,
#   0.04368875337927353,
#   0.03279840615597376,
#   0.0271399911570401],
#  [0.05637023947280395,
#   0.12092043735018243,
#   0.09486722387202276,
#   0.7035317959345865,
#   -0.5251592783818917,
#   -0.018194960929093763,
#   -0.023052974828639057],
#  [0.02991577551526566,
#   0.051619240955039926,
#   0.04731334344431247,
#   0.054875499420508754,
#   -0.6131983621202085,
#   0.3742576742643604,
#   -0.08722315426571213],
#  [-0.006075205216874307,
#   -0.009249795018051489,
#   -0.010077843832960686,
#   -0.01150052739694312,
#   -0.018835963941975932,
#   0.45525716664889243,
#   -0.4324411926216422]]
#
# for ii in [4, 5]:
#     experiment_name = 'ff_sweep_j'
#     expt_cfg = experiment_cfg[experiment_name]
#
#     expt_cfg = experiment_cfg[experiment_name]
#
#     ## Modification for tunneling Expt
#     expt_cfg['ff_vec'] = tunnelingtuningarray[ii]
#     expt_cfg['rd_qb'] = ii
#     expt_cfg['Mott_qbs'] = [ii]
#     # End tunneling expt interjection
#
#     rd_qb = expt_cfg["rd_qb"]
#     Mott_qbs = expt_cfg["Mott_qbs"]
#
#     qb_freq_list = [lattice_cfg["qubit"]["freq"][i] for i in Mott_qbs]
#     lo_qb_temp_ind = np.argmax(qb_freq_list)
#     lo_qb = Mott_qbs[lo_qb_temp_ind]
#
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: lo_qb})
#     quantum_device_cfg["readout"][setup]["freq"] = lattice_cfg["readout"]["freq"][rd_qb]
#     quantum_device_cfg["powers"][setup]["readout_drive_digital_attenuation"] = \
#     lattice_cfg["powers"][setup]["readout_drive_digital_attenuation"][rd_qb]
#
#     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#     SequentialExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, experiment_name, path,
#                          analyze=False, show=False)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////