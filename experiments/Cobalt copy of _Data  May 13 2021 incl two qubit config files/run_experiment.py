
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
gf_ramsey
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
Sideband drive with IQ mixer and seperate LO
==================================================================================================
cavity_sideband_rabi_freq_scan
'''



with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

#experiment_names = ['cavity_drive_direct_spectroscopy']
#experiment_names = ['cavity_drive_pulse_probe_iq_signalcore']
#experiment_names = ['cavity_drive_ef_ramsey_mixedtones']
#experiment_names = ['cavity_drive_rabi_freq_scan_mixedtones']
#experiment_names = ['cavity_drive_ef_rabi_testing_mixedtones']
#experiment_names = ['cavity_drive_histogram_mixedtones']
#experiment_names = ['cavity_drive_rabi_freq_scan_mixedtones']
#experiment_names = ['rabi']
# using wrong LO as a drive so i can test if bnc used for readout normally still works
# to do this, change which is drive los in hardware config
# also need to... switch trigger??
# have done that remember to turn back. did it
#hardware_cfg['drive_los'] = "RF6"

show = 'I'
# will need to start picking IA or IB etc
# could edit post experiment analysis, stay tuned

# #Readout properties
# quantum_device_cfg['readout']['freq'] = 10.5705
# quantum_device_cfg['readout']['dig_atten'] = -20

# Qubit pulse properties
# quantum_device_cfg['pulse_info']['1']['pi_amp'] = 0.99099 # X180 amp
# quantum_device_cfg['pulse_info']['1']['pi_len'] = 19.0 # X180 sig
# quantum_device_cfg['pulse_info']['1']['half_pi_amp'] = 0.94144 # X90 amp
# quantum_device_cfg['pulse_info']['1']['half_pi_len'] = 10.0 # X90 sig


# experiment_names = ['pulse_probe_iq']
# experiment_name = 'pulse_probe_iq'
# # experiment_cfg[experiment_name]['cavity_pulse_len'] = 100
# # experiment_cfg[experiment_name]['qubit_pulse_len'] = 100
# quantum_device_cfg['qubit']['1']['freq'] = 7.0
# experiment_cfg[experiment_name]['on_qubits'] = ["1", "2"]
# experiment_cfg[experiment_name]['start'] = -4e-3
# experiment_cfg[experiment_name]['stop'] = 4e-3
# experiment_cfg[experiment_name]['step'] = 0.1e-3
# experiment_cfg[experiment_name]['amp'] = 1.0
# # experiment_cfg[experiment_name]['pulse_length'] = 100
# experiment_cfg[experiment_name]['acquisition_num'] = 50
#

experiment_names = ['cavity_drive_pulse_probe_iq']
experiment_name = 'cavity_drive_pulse_probe_iq'
experiment_cfg[experiment_name]['cavity_pulse_len'] = 400
experiment_cfg[experiment_name]['qubit_pulse_len'] = 100
quantum_device_cfg['qubit']['1']['freq'] = 7.0
experiment_cfg[experiment_name]['on_qubits'] = ["2"]
experiment_cfg[experiment_name]['start'] = -4e-3
experiment_cfg[experiment_name]['stop'] = 4e-3
experiment_cfg[experiment_name]['step'] = 0.1e-3
experiment_cfg[experiment_name]['amp'] = 1.0
# experiment_cfg[experiment_name]['pulse_length'] = 100
experiment_cfg[experiment_name]['acquisition_num'] = 50
# #
# experiment_names = ['rabi']
# experiment_name = 'rabi'
# experiment_cfg[experiment_name]['amp'] = 1.0
# experiment_cfg[experiment_name]['start'] = 20.0
# experiment_cfg[experiment_name]['stop'] = 100.0
# experiment_cfg[experiment_name]['step'] = 2.0
# experiment_cfg[experiment_name]['acquisition_num'] =8000
# experiment_cfg[experiment_name]['pi_calibration']= False
# # # #
# experiment_names = ['histogram']
# #
# experiment_names = ['ramsey']
# experiment_name = 'ramsey'
# experiment_cfg[experiment_name]['start'] = 0.0
# experiment_cfg[experiment_name]['stop'] = 2000.0
# experiment_cfg[experiment_name]['step'] = 20.0
# experiment_cfg[experiment_name]['ramsey_freq'] = 5.0e-3
# experiment_cfg[experiment_name]['acquisition_num'] = 10000
# experiment_cfg[experiment_name]['pi_calibration'] = False

# experiment_names = ['t1']
# experiment_name = 't1'

# experiment_names = ['pulse_probe_ef_iq']
# experiment_name = 'pulse_probe_ef_iq'
#
# experiment_names = ['ef_rabi']
# experiment_name = 'ef_rabi'
# experiment_cfg[experiment_name]['amp'] = 1.0
# experiment_cfg[experiment_name]['start'] = 0.0
# experiment_cfg[experiment_name]['stop'] = 300.0
# experiment_cfg[experiment_name]['step'] = 3.0
# experiment_cfg[experiment_name]['acquisition_num'] = 4000
# experiment_cfg[experiment_name]['ge_pi'] = False
# experiment_cfg[experiment_name]['pi_calibration'] = True
# experiment_cfg[experiment_name]['ef_pi_for_cal'] = True

# experiment_names = ['ef_ramsey']
# experiment_name = 'ef_ramsey'

# experiment_names = ['ef_t1']

for experiment_name in experiment_names:

    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg, plot_visdom=True)
    sequences = ps.get_experiment_sequences(experiment_name)
    # sequences = ps.get_experiment_sequences(experiment_name,save=True,filename = 'sideband_transmon_ef_rabi')
    exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg, sequences, experiment_name)

    exp.run_experiment_pxi(sequences, path, experiment_name,expt_num=0,check_sync=False)
    exp.post_analysis(experiment_name, P=show, show=True)

