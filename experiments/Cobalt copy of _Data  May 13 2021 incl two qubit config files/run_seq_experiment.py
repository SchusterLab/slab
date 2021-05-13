from slab.experiments.PulseExperiments_PXI.sequential_experiment_pxi import SequentialExperiment
import json
import numpy as np

'''
General Sequential Experiments:
=======================
resonator_spectroscopy
qubit_temperature
histogram_sweep
histogram_amp_and_freq_sweep
sequential_pulse_probe_ef_iq
=======================
'''

'''
Sequential Charge sideband Experiments:
=======================
sideband_rabi_sweep
sideband_rabi_freq_scan_length_sweep
sideband_rabi_freq_scan_amp_sweepnote
sequential_sideband_ramsey
sideband_rabi_two_tone_detuning_sweep
sideband_reset_qubit_temperature
sideband_reset_qubit_temperature_wait_sweep
cavity_drive_pulse_probe_iq_amp_sweep
wigner_tomography_test_phase_sweep
wigner_tomography_sideband_only_phase_sweep
wigner_tomography_sideband_one_pulse_phase_sweep
wigner_tomography_alltek2_phase_sweep
wigner_tomography_2d_offset_sweep
=======================
'''



import os
path = os.getcwd()

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)


#experiment_name = 'cavity_drive_rabi_freq_scan_vstime'
#experiment_name = 'amplitude_time_rabi'
#experiment_name = 'cavity_drive_rabi_freq_scan_vstime'
#experiment_name = 'cavity_drive_histogram_amp_and_freq_sweep_mixedtones'


# Find the resonator - freq sweep, fixed power
# experiment_name = 'resonator_spectroscopy'
# fcenter = 10.570; fspan = 5e-3
# fstart = fcenter - fspan/2; fstop = fcenter + fspan/2
# experiment_cfg[experiment_name]['start'] = fstart
# experiment_cfg[experiment_name]['stop'] = fstop
# experiment_cfg[experiment_name]['step'] = 0.025e-3
# experiment_cfg[experiment_name]['acquisition_num'] = 1000
# quantum_device_cfg['readout']['dig_atten'] = -35.0
# sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=True,show=True)


# Find the resonator - freq sweep, fixed power
experiment_name = "resonator_spectroscopy"
fcenter = 10.5701; fspan = 0.5e-2
fstart = fcenter - fspan/2; fstop = fcenter + fspan/2
experiment_cfg[experiment_name]["start"] = fstart
experiment_cfg[experiment_name]["stop"] = fstop
experiment_cfg[experiment_name]["step"] = 2.5e-5
#experiment_cfg[experiment_name]["step"] = 0.05e-3
experiment_cfg[experiment_name]["acquisition_num"] = 1000
experiment_cfg[experiment_name]['pi_qubit'] = False
experiment_cfg[experiment_name]['pi_ef_qubit'] = False
experiment_cfg[experiment_name]['pi_calibration'] = False
quantum_device_cfg["readout"]["dig_atten"] = -16.0
quantum_device_cfg["readout"]["rotate_iq_dig"] = True
sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=False,show=True)

# experiment_name = 'histogram_amp_and_freq_sweep'
# fcenter = 10.5702; fspan = 1.0e-3
# fstart = fcenter - fspan/2; fstop = fcenter + fspan/2
# experiment_cfg[experiment_name]["atten_start"] = -30.0
# experiment_cfg[experiment_name]["atten_stop"] = -10.0
# experiment_cfg[experiment_name]["freq_start"] = fstart
# experiment_cfg[experiment_name]["freq_stop"] = fstop
# experiment_cfg[experiment_name]["freq_step"] = 0.0001
# sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=False,show=True)


# Find the resonator - freq sweep, sweep power (digital attenuator)
# experiment_name = 'resonator_spectroscopy_power_sweep'
# fcenter = 10.5705; fspan = 5e-3
# fstart = fcenter - fspan/2; fstop = fcenter + fspan/2
# experiment_cfg[experiment_name]['start'] = fstart
# experiment_cfg[experiment_name]['stop'] = fstop
# experiment_cfg[experiment_name]['step'] = 0.05e-3
# experiment_cfg[experiment_name]['acquisition_num'] = 2000
# experiment_cfg[experiment_name]['pwr_start'] = -30.0
# experiment_cfg[experiment_name]['pwr_stop'] = -4.0
# experiment_cfg[experiment_name]['pwr_step'] = 5.0
# quantum_device_cfg['readout']['rotate_iq_dig'] = True
# sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path,analyze=True,show=True)
