from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice_v2
import json
from loadNsave_lattice import load_lattice_to_quantum_device
from loadNsave_lattice import load_quantum_device_to_lattice
from loadNsave_lattice import generate_quantum_device_from_lattice
import numpy as np
import os
import cProfile
path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

lattice_cfg_name = '210510_sawtooth_lattice_device_config_wff.json'
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)


experiment_name = 'rabi_chevron'
for i in [0] :
    quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[i], setups=["A", "B"])
    SequentialExperiment(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, experiment_name, path,
                         analyze=False, show=False)



# experiment_name = 'ff_ramp_cal_ppiq'
# slopearray = np.array([1.0933,1.1105,-0.9836,-0.97072,-0.93937,-1.21225,-1.02141,0.91383])/10 # should be MHz / mV
# for i in [0]:
#     # windowcenterarray = [-0.0500, 0.0, 0.0500]
#     windowcenterarray = [-0.100, 0.0, 0.100]
#     if i%2 == 0:
#         windowcenterarray = windowcenterarray[0:2]
#     else:
#         windowcenterarray = windowcenterarray[1:3]
#     if i%2 == 0:
#         trajectorysign = -1
#     else:
#         trajectorysign = 1
#
#     for cross_qb in [0,1,2,3,4,5,6,7]:
#
#         for j,center in enumerate(windowcenterarray):
#                 quantum_device_cfg = generate_quantum_device_from_lattice(lattice_cfg_name, qb_ids=[cross_qb], setups=["A","B"])
#
#                 ## Truncate PPIQ time domain based on qubit index, direction of flux tuning
#                 if ((i%2 == 0) and (j==1)) or ((i%2==1) and (j==0)) :
#                     experiment_cfg[experiment_name]['dt_start'] = -1500
#                     experiment_cfg[experiment_name]['dt_stop'] = -549
#                 else:
#                     experiment_cfg[experiment_name]['dt_start'] = 1000
#                     experiment_cfg[experiment_name]['dt_stop'] = 1751
#
#                 ## truncate frequency domain to qubit center, or tuned frequency center + window
#                 if cross_qb==i:
#                     experiment_cfg[experiment_name]['start'] = center -0.01
#                     experiment_cfg[experiment_name]['stop'] = center +0.01
#                 else:
#                     experiment_cfg[experiment_name]['start'] =  -0.01
#                     experiment_cfg[experiment_name]['stop'] =  0.01
#
#                 #Q7 needs twice as much averaging
#                 if cross_qb ==7:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 5000
#                 else:
#                     experiment_cfg[experiment_name]['acquisition_num'] = 250
#
#
#                 ## move qubit frequency to +-100MHz.  If bottom of sawtooth, go down.  If top, go up to avoid collisions. (trajsign)
#                 experiment_cfg[experiment_name]['ff_vec'] = (np.array([-0.09410226, -0.00528077, -0.00222714, -0.00129808, -0.00108117,-0.00099433, -0.00086062, -0.00054959])*10).tolist()
#                 # print("Tuning Fast Flux line %s"%(i) + ' To %s'%(trajectorysign * 0.0500 / slopearray[i]))
#                 # experiment_cfg[experiment_name]['ff_vec'][i] = trajectorysign * 0.0500 / slopearray[i]
#                 SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,lattice_cfg,experiment_name,path,analyze=False,show=False)

