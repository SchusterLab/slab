import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json

import os
path = os.getcwd()

'''
General Experiments:
=======================
resonator_spectroscopy
pulse_probe_iq
rabi
t1
ramsey
echo
pulse_probe_ef_iq
ef_rabi
ef_ramsey
ef_t1
ef_echo
pulse_probe_fh_iq
fh_rabi
fh_ramsey
histogram
=======================

Charge sideband Experiments
=======================
sideband_rabi
sideband_rabi_freq_scan
sideband_rabi_two_tone
sideband_t1
sideband_ramsey
sideband_pi_pi_offset
sideband_rabi_two_tone_freq_scan
sideband_rabi_two_tone
sideband_histogram
sideband_pulse_probe_iq
sideband_chi_ge_calibration
sideband_chi_ef_calibration
sideband_chi_gf_calibration
sideband_transmon_reset
sideband_parity_measurement
sideband_repetitive_parity_measurement
sideband_cavity_photon_number
=======================
'''
## Remember to run python -m visdom.server
## Remember to start instrumentmanager with .cfg file configured correctly

### Need to form an array of cfg files for each qubit

# with open('lattice_config.json','r') as f:
#     lattice_config_cfg = json.load(f)d
#     for i in range(8):
#         quantum_device_cfg_temp = {}
#         quantum_device_cfg_temp['qubit'] = []
#         quantum_device_cfg_temp.append({"1":})
#         with open('quantum_device_cfg%s'%i+'.txt','w') as outfile:
#             json.dump(quantum_device_cfg_temp,outfile)
#
# from DACInterface import AD5780_serial
#
# dac = AD5780_serial()
# # dac.init()
# dac.ramp3(fluxInd+1,pt[fluxInd],step=step,speed=rampspeed)

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
experiment_names = ['t1']

# quantum_device_cfg['readout']['window'] = [150,420]
# quantum_device_cfg['pulse_info']['1']['iq_freq'] = -0.02
# hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']= 500
# experiment_cfg['pulse_probe_iq']['acquisition_num'] = 1000


show = 'I'

for experiment_name in experiment_names:
    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=False)
    sequences = ps.get_experiment_sequences(experiment_name)
    print("Sequences generated")
    exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,sequences, experiment_name)
    exp.run_experiment_pxi(sequences, path, experiment_name,expt_num=0,check_sync=True)
    # exp.run_experiment_pxi(sequences, path, experiment_name, check_sync=False)
    #p = exp.post_analysis(experiment_name, P=show)
    #print(p)





