
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
=======================
'''



with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)


experiment_names = ['sideband_rabi_two_tone_freq_scan']

for experiment_name in experiment_names:

    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=True)
    sequences = ps.get_experiment_sequences(experiment_name)
    exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,sequences, experiment_name)
    exp.run_experiment_pxi(sequences, path, experiment_name,expt_num=0,check_sync=False)
    exp.post_analysis(experiment_name,P = 'Q',show=True)
