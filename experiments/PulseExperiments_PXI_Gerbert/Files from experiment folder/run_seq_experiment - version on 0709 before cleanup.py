from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice_v3
import json
from loadNsave_lattice import load_lattice_to_quantum_device
from loadNsave_lattice import load_quantum_device_to_lattice
from loadNsave_lattice import generate_quantum_device_from_lattice
import numpy as np
import os
import cProfile
path = os.getcwd()
path = "C:\\210701 - PHMIV3_56 - BF4 cooldown 3"

lattice_cfg_name = '210705_sawtooth_lattice_device_config.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

setup = 'A'
# experiment_name = 'rabi_chevron'
# for i in [4] :
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={'A':i})
#     SequentialExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, experiment_name, path,
#                          analyze=False, show=False)

#
# experiment_name = 'ff_ramp_cal_ppiq'
# tuning_V = 0.050
# # for len in [30, 45, 60, 75, 90, 105, 120]
# for i in [0, 1, 2, 3, 4, 5, 6, 7]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     fluxvec = [0]*8
#     fluxvec[i] = tuning_V
#     experiment_cfg[experiment_name]['ff_vec'] = fluxvec
#     experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#     SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)
#

# experiment_name = 'ff_ramp_cal_ppiq'
# slopearray = [0.15882328551508448,
#  0.1821419485249365,
#  0.15720193542662636,
#  -0.16165676236004214,
#  -0.1292428802162228,
#  -0.1764891070349379,
#  -0.15164379767398176,
#  0.14394962674217154] # should be GHz / V * 1.5 because of AWG scaling
#
# freq_shift = 0.080 #in GHz
#
# # fluxvec = [-0.59810378, -0.03356398, -0.01415546, -0.00825048, -0.00687183,-0.00631984, -0.00547   , -0.0034931 ]
#
# for flux_line in [0,1,2,3,4,5,6,7]:
#
#     ## move qubit frequency to +-freq_shift.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#     windowcenterarray = [-freq_shift, 0.0, freq_shift]
#     if flux_line%2 == 0:
#         windowcenterarray = windowcenterarray[0:2]
#     else:
#         windowcenterarray = windowcenterarray[1:3]
#     if flux_line%2 == 0:
#         trajectorysign = -1
#     else:
#         trajectorysign = 1
#
#     for cross_qb in [0,1,2,3,4,5,6,7]:
#         for j,center in enumerate(windowcenterarray):
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#                 ## Truncate PPIQ time domain based direction of flux tuning
#                 if ((flux_line%2 == 0) and (j==1)) or ((flux_line%2==1) and (j==0)) :
#                     experiment_cfg[experiment_name]['dt_start'] = -1500
#                     experiment_cfg[experiment_name]['dt_stop'] = -549
#                 else:
#                     experiment_cfg[experiment_name]['dt_start'] = 1000
#                     experiment_cfg[experiment_name]['dt_stop'] = 1751
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb==flux_line:
#                     experiment_cfg[experiment_name]['start'] = center -0.02
#                     experiment_cfg[experiment_name]['stop'] = center +0.02
#                 else:
#                     experiment_cfg[experiment_name]['start'] =  -0.02
#                     experiment_cfg[experiment_name]['stop'] =  0.02
#
#                 #Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 4000
#                 else:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 300
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = 200
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#
#                 ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#                 #experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
#                 experiment_cfg[experiment_name]['ff_vec'] = [0]*8
#                 print("Tuning Fast Flux line %s"%(flux_line) + ' To %s'%(trajectorysign * freq_shift / slopearray[flux_line]))
#                 experiment_cfg[experiment_name]['ff_vec'][flux_line] = trajectorysign * freq_shift / slopearray[flux_line]
#                 SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)
# #

# experiment_name = 'ff_ramp_cal_ppiq'
# slopearray = [ 0.16704477,  0.17774266, -0.15928814, -0.15937513, -0.13656257,-0.1745105 , -0.160629  ,  0.12958199] # should be GHz / V * 1.5 because of AWG scaling
#
# freq_shift = 0.080 #in GHz
#
# # fluxvec = [-0.59810378, -0.03356398, -0.01415546, -0.00825048, -0.00687183,-0.00631984, -0.00547   , -0.0034931 ]
#
# for flux_line in [0,1,2,3,4,5,6,7]:
#
#     ## move qubit frequency to +-freq_shift.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#     windowcenterarray = [-freq_shift, 0.0, freq_shift]
#     if flux_line%2 == 0:
#         windowcenterarray = windowcenterarray[0:2]
#     else:
#         windowcenterarray = windowcenterarray[1:3]
#     if flux_line%2 == 0:
#         trajectorysign = -1
#     else:
#         trajectorysign = 1
#
#     for cross_qb in [0,1,2,3,4,5,6,7]:
#         for j,center in enumerate(windowcenterarray):
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#                 ## Truncate PPIQ time domain based direction of flux tuning
#                 if ((flux_line%2 == 0) and (j==1)) or ((flux_line%2==1) and (j==0)) :
#                     experiment_cfg[experiment_name]['dt_start'] = -1500
#                     experiment_cfg[experiment_name]['dt_stop'] = -549
#                 else:
#                     experiment_cfg[experiment_name]['dt_start'] = 1000
#                     experiment_cfg[experiment_name]['dt_stop'] = 1751
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb==flux_line:
#                     experiment_cfg[experiment_name]['start'] = center -0.03
#                     experiment_cfg[experiment_name]['stop'] = center +0.03
#                 else:
#                     experiment_cfg[experiment_name]['start'] =  -0.03
#                     experiment_cfg[experiment_name]['stop'] =  0.03
#
#                 #Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 4000
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = 60
#                 elif cross_qb ==6:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 150
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = 150
#                 else:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 300
#                     experiment_cfg[experiment_name]['qb_pulse_length'] = 200
#
#
#                 ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#                 #experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
#                 experiment_cfg[experiment_name]['ff_vec'] = [0]*8
#                 print("Tuning Fast Flux line %s"%(flux_line) + ' To %s'%(trajectorysign * freq_shift / slopearray[flux_line]))
#                 experiment_cfg[experiment_name]['ff_vec'][flux_line] = trajectorysign * freq_shift / slopearray[flux_line]
#                 # experiment_cfg[experiment_name]['ff_vec'] = fluxvec
#                 SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)



# experiment_name = 'ff_ramp_cal_ppiq'
#
# V_list = [[0.23483078, 0.01284155, 0.00515893, 0.003004  , 0.00311569,
#        0.00257362, 0.00259529, 0.00256007], [ 0.00302902, -0.34445746, -0.01885907, -0.01016972, -0.00758939,
#        -0.00622871, -0.00590171, -0.00379736], [ 0.005653  , -0.00244087,  0.36089365,  0.02559683,  0.01517352,
#         0.01261154,  0.01070689,  0.0084521 ], [0.00703008, 0.02070509, 0.01595513, 0.25099971, 0.02226905,
#        0.01723681, 0.01468481, 0.01057084], [ 0.01543991,  0.02574118,  0.02470395,  0.02898914, -0.21921029,
#        -0.02356032, -0.02298587, -0.0118993 ], [-0.00961156, -0.01417453, -0.01637297, -0.01865268, -0.026335  ,
#         0.39632695, -0.0225863 , -0.04051239], [ 0.0034967 ,  0.00654562,  0.00686027,  0.00933327,  0.01151545,
#         0.01317258, -0.32594937, -0.01817888], [0., 0., 0., 0., 0., 0., 0., 0.]]
#
# # V_list = [-0.00995246,  0.0042973 , -0.63537615, -0.04506484, -0.02671395,
# #        -0.02220342, -0.01885015, -0.01488046]
# jumplist = [ 0.03733, -0.06283,  0.0568 , -0.0409 ,  0.0292 , -0.0706 ,0.0497 ,  0.     ]
#
# for flux_line in [0,1,2,3,4,5,6,7]:
#     for cross_qb in [0,1,2,3,4,5,6,7]:
#         if ( cross_qb == (flux_line - 1) or cross_qb == flux_line or cross_qb == (flux_line+1) ):
#             freq_shift = jumplist[cross_qb]
#             for j, center in enumerate([0, freq_shift * (-1)**flux_line]):
#                 quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})
#
#                 ## Truncate PPIQ time domain based direction of flux tuning
#                 if (j==0):
#                     experiment_cfg[experiment_name]['dt_start'] = -1500
#                     experiment_cfg[experiment_name]['dt_stop'] = -1499
#                 else:
#                     experiment_cfg[experiment_name]['dt_start'] = 1000
#                     experiment_cfg[experiment_name]['dt_stop'] = 1499
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#
#                 if cross_qb==flux_line:
#                     experiment_cfg[experiment_name]['start'] = center - 0.02
#                     experiment_cfg[experiment_name]['stop'] = center + 0.02
#
#                 else:
#                     experiment_cfg[experiment_name]['start'] =  -0.02
#                     experiment_cfg[experiment_name]['stop'] =  0.02
#
#                 #Q7 needs twice as much averaging and a shorter pi-time
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 4000
#                 else:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 300
#                 experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]
#
#
#                 ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#                 #experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
#                 experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
#                 print(experiment_cfg[experiment_name]['ff_vec'])
#
#                 SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)



## Testing ramp from staggered lattice to the final on-resonance lattice condition
experiment_name = 'ff_ramp_cal_ppiq'

## With CTM
# V_list = [[0.22657339, 0.01352542, 0.00609166, 0.00329662, 0.00290219,
#        0.00311061, 0.0019908 , 0.0015436 ], [ 0.00495107, -0.35874478, -0.02076211, -0.01180641, -0.00734337,
#        -0.00575619, -0.00732297, -0.00098393], [ 0.0059029 , -0.00267301,  0.37140529,  0.02589965,  0.01626236,
#         0.01305199,  0.0123199 ,  0.00743531], [0.00637232, 0.02117982, 0.01697612, 0.2624225 , 0.02351743,
#        0.01990844, 0.01422417, 0.00899521], [ 0.01433459,  0.0257779 ,  0.02431668,  0.02849693, -0.21468905,
#        -0.02281037, -0.02184248, -0.01340088], [-0.00974062, -0.01570241, -0.01586462, -0.02066014, -0.02925874,
#         0.41526071, -0.0234367 , -0.04746055], [ 0.0034137 ,  0.00649158,  0.00733163,  0.00715506,  0.0099164 ,
#         0.01318231, -0.31318849, -0.01682033], [0., 0., 0., 0., 0., 0., 0., 0.]]

## Without CTM
V_list = [[0.2264264975059633,
  0.012321900009107803,
  0.004825306975249993,
  0.003905758149464604,
  0.0026779109936013626,
  0.0014666037766942106,
  0.0017156715748820118,
  0],
 [0.011965929247545836,
  -0.35820804808242174,
  -0.015236099732522242,
  -0.009072932730948556,
  -0.006750836791521886,
  -0.007046161563234375,
  -0.005329823472464158,
  0],
 [0.006914365797166769,
  -0.005086265259106407,
  0.3723303743536399,
  0.031957037468498393,
  0.013820470916331835,
  0.013814652857867621,
  0.011816675673113028,
  0],
 [0.007304999491257838,
  0.024140288801107867,
  0.020493115140302703,
  0.26447631607352606,
  0.010834473806496402,
  0.016575665252493717,
  0.014298679435754202,
  0],
 [0.015007780921877068,
  0.02483577837606683,
  0.02418286134970009,
  0.03307651352484442,
  -0.2158072153682244,
  -0.022000176921444535,
  -0.02100510742613357,
  0],
 [-0.008778216009486261,
  -0.014758011139852338,
  -0.01589549547316927,
  -0.017861292818182564,
  -0.03621812676705754,
  0.4144078768141624,
  -0.02992504124521431,
  0],
 [0.004197160701179531,
  0.005710779238524829,
  0.006748633101961223,
  0.00818575282175494,
  0.008928791541453502,
  0.015234653682910514,
  -0.3139204676719589,
  0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]]


jumplist = [ 0.03733, -0.06283,  0.0568 , -0.0409 ,  0.0292 , -0.0706 ,0.0497 ,  0]

for flux_line in [0,1,2,3,4,5,6,7]:
    for cross_qb in [0,1,2,3,4,5,6,7]:
        #if (cross_qb == flux_line - 1 or  cross_qb == flux_line or cross_qb == flux_line  + 1 ):
        freq_shift = jumplist[cross_qb]
        for j, center in enumerate([0, freq_shift]):
            quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup: cross_qb})

            ## Truncate PPIQ time domain based direction of flux tuning
            if (j==0):
                experiment_cfg[experiment_name]['dt_start'] = -1500
                experiment_cfg[experiment_name]['dt_stop'] = -1499
            else:
                experiment_cfg[experiment_name]['dt_start'] = 1000
                experiment_cfg[experiment_name]['dt_stop'] = 1499

            ## truncate frequency domain to qubit center, or tuned frequency center + window

            if cross_qb==flux_line:
                experiment_cfg[experiment_name]['start'] = center - 0.02
                experiment_cfg[experiment_name]['stop'] = center + 0.02

            else:
                experiment_cfg[experiment_name]['start'] =  -0.02
                experiment_cfg[experiment_name]['stop'] =  0.02

            #Q7 needs twice as much averaging and a shorter pi-time
            if cross_qb ==7:
                experiment_cfg[experiment_name]['acquisition_num'] = 4000
            else:
                experiment_cfg[experiment_name]['acquisition_num'] = 200
            experiment_cfg[experiment_name]['qb_pulse_length'] = quantum_device_cfg["pulse_info"][setup]["pi_len"]


            ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
            #experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
            experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
            print(experiment_cfg[experiment_name]['ff_vec'])

            SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)