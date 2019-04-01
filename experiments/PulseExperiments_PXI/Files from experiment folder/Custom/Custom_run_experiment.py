
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


if experiment_names == 'Custom':
    #input customizeable circuit code
    Operations = "           "
    Operations = Operations.replace(" ", "")

    GS = []
    GE = []

    # GS for Gate Start, GE for Gate End

    for i in range(len(Operations) - len('circ.')):
        A = Operations.find('circ.', i, i + 5)
        if A > -1:
            GS.append(A)

    for i in range(len(Operations) - len('(q[')):
        B = Operations.find('(q[', i, i + 3)
        if B > -1:
            GE.append(B)

    GS = np.array(GS) + len('circ.')
    Tot_Gates = np.array([GS, np.array(GE)])

    Gates = []
    a = 0

    for j in range(len(np.transpose(Tot_Gates))):
        Gates.append(Operations[Tot_Gates[0, a]: Tot_Gates[1, a]])
        a = a + 1

    Gates1 = []
    Gates2 = []
    Gates3 = []
    GS1 = []
    GE1 = []
    GS2 = []
    GE2 = []
    GS3 = []
    GE3 = []

    a = 0
    index = 0
    # GI for Gate Interactions, which qubits are involved in each Gate Operation

    for i in Gates:
        if i in ['h', 'x', 'y', 'z', 't', 's', 'iden']:
            Gates1.append(i)
            GE1.append(GE[index])
            GS1.append(GS[index])
            index = index + 1
        elif i in ['swap', 'cx', 'ch']:
            Gates2.append(i)
            GE2.append(GE[index])
            GS2.append(GS[index])
            index = index + 1
        else:
            Gates3.append(i)
            GE3.append(GE[index])
            GS3.append(GS[index])
            index = index + 1

    Bit_Op1 = []
    Bit_Op2 = []
    Bit_Op3 = []
    index = 0

    for i in range(len(GE1)):
        Bit_Op1.append(Operations[GE1[index] + 3])
        index = index + 1

    index = 0
    for i in range(len(GE2)):
        Bit_Op2.append(Operations[GE2[index] + 3])
        Bit_Op2.append(Operations[GE2[index] + 5])
        index = index + 1

    index = 0
    for i in range(len(GE3)):
        Bit_Op3.append(Operations[GE3[index] + 3])
        Bit_Op3.append(Operations[GE3[index] + 5])
        Bit_Op3.append(Operations[GE3[index] + 7])
        index = index + 1


for experiment_name in experiment_names:

    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=True)
    sequences = ps.get_experiment_sequences(experiment_name)
    exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,sequences, experiment_name)
    exp.run_experiment_pxi(sequences, path, experiment_name,expt_num=0,check_sync=False)
    exp.post_analysis(experiment_name,P = 'Q',show=True)
