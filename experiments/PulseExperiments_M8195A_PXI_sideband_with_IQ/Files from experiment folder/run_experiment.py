
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json

import os
path = os.getcwd()


'''
=================================================================================================
General qubit experiments with PXI alone:
=================================================================================================
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
=================================================================================================
Charge sideband experiments PXI and TEK2
NOTE: 
1) The keyword "sideband" is necessary anywhere in the expt_name to trigger TEK2
2) The keyword "cavity_drive" is necessary anywhere in the expt_name to trigger cavity drive LO 
and generate cavity drive pulses
=================================================================================================
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
cavity_drive_direct_spectroscopy
direct_cavity_sideband_spectroscopy
cavity_drive_pulse_probe_iq
wigner_tomography_test_cavity_drive_sideband
wigner_tomography_test_sideband_only
wigner_tomography_test_sideband_one_pulse
==================================================================================================
Qubit experiments all using TEK2 alone & using the PXI for readout
==================================================================================================
sideband_transmon_ge_rabi
sideband_transmon_ge_ramsey
sideband_transmon_pulse_probe_ge
sideband_transmon_pulse_probe_ef
sideband_transmon_ef_rabi
sideband_f0g1rabi_freq_scan
sideband_f0g1rabi
sideband_f0g1ramsey
sideband_f0g1_pi_pi_offset
sideband_chi_ge_calibration_alltek2
wigner_tomography_test_sideband_alltek2
wigner_tomography_2d_sideband_alltek2
sideband_number_splitting_pulse_probe_ge_tek2
sideband_photon_number_distribution_tek2
sideband_cavity_spectroscopy_alltek2
==================================================================================================
'''



with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)


experiment_names = ['sideband_f0g1rabi','sideband_f0g1ramsey','sideband_f0g1_pi_pi_offset','sideband_chi_ge_calibration_alltek2']

show = 'Q'


for experiment_name in experiment_names:

    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=True)
    sequences = ps.get_experiment_sequences(experiment_name)
    # sequences = ps.get_experiment_sequences(experiment_name,save=True,filename = 'sideband_transmon_ef_rabi')
    print("Sequences generated")
    exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,sequences, experiment_name)
    exp.run_experiment_pxi(sequences, path, experiment_name,expt_num=0,check_sync=False)
    exp.post_analysis(experiment_name, P=show, show=False)
