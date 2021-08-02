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
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

lattice_cfg_name = '210526_sawtooth_lattice_device_config.json'
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
# # tuning_V = -0.2724424465331699
# tuning_V = 0.30
# # for len in [30, 45, 60, 75, 90, 105, 120]
# for i in [0]:
#     quantum_device_cfg = generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits={setup:i})
#     experiment_cfg[experiment_name]['ff_vec'] = [0]*8
#     print("Tuning Fast Flux line %s"%(i) + ' To %s'%(tuning_V))
#     experiment_cfg[experiment_name]['ff_vec'][i] = tuning_V
#     SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)
#

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

experiment_name = 'ff_ramp_cal_ppiq'
V_list = [[-0.59803692, -0.03281403, -0.01352511, -0.00901331, -0.00725197,
        -0.00675848, -0.00585488, -0.00300657],
 [-0.00833183,  0.56181223,  0.03412918,  0.01664799,  0.0129929 ,
         0.01079088,  0.01122221, -0.00081603],
 [1.10925915e-02, 2.26603137e-04, 6.26638566e-01, 4.02085207e-02,
        2.50865936e-02, 2.13568446e-02, 1.96832404e-02, 9.94222598e-03],
 [-0.01699203, -0.05036118, -0.03125702, -0.62197699, -0.05545865,
        -0.04160449, -0.03788166, -0.01605463],
 [-0.04847151, -0.08294779, -0.07930629, -0.09185497,  0.71090097,
         0.07649574,  0.06984086,  0.0429169 ],
 [ 0.01271953,  0.02020309,  0.02227524,  0.02734896,  0.03873521,
        -0.56760954,  0.02927935,  0.05638923],
 [-0.00666383, -0.01061193, -0.0134053 , -0.01431529, -0.02047026,
        -0.02543666,  0.61949683,  0.0243462 ],
 [-0.00403606, -0.0066451 , -0.00835878, -0.00859257, -0.01146417,
        -0.0138364 , -0.01644578,  0.77122239]]



for flux_line in [0, 1, 2, 3, 4, 5, 6, 7]:
    freq_shift = 0.100 * (-1)**(flux_line+1)  # in GHz
    for cross_qb in [0, 1, 2,3 , 4, 5, 6, 7]:
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
                experiment_cfg[experiment_name]['start'] = center -0.02
                experiment_cfg[experiment_name]['stop'] = center +0.02
            else:
                experiment_cfg[experiment_name]['start'] =  -0.02
                experiment_cfg[experiment_name]['stop'] =  0.02

            #Q7 needs twice as much averaging and a shorter pi-time
            if cross_qb ==7:
                experiment_cfg[experiment_name]['acquisition_num'] = 4000
                experiment_cfg[experiment_name]['qb_pulse_length'] = 60
            elif cross_qb ==6:
                experiment_cfg[experiment_name]['acquisition_num'] = 150
                experiment_cfg[experiment_name]['qb_pulse_length'] = 150
            else:
                experiment_cfg[experiment_name]['acquisition_num'] = 150
                experiment_cfg[experiment_name]['qb_pulse_length'] = 200


            ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
            #experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
            if j==0:
                experiment_cfg[experiment_name]['ff_vec'] = [0]*8
            else:
                experiment_cfg[experiment_name]['ff_vec'] = V_list[flux_line]
            print(experiment_cfg[experiment_name]['ff_vec'])

            SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)

