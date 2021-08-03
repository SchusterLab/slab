import json
from numpy import*
import matplotlib.pyplot as plt
from slab.dsfit import*
import os
path = os.getcwd()

from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.simulate_pulse_experiment import Simulate_Multimode_Experiment

multimode_params = {"Ej":21,"Ec":0.13,"N":100,"nus":[8.07,6.015],"gs":[0.177,0.067],"T1s":[50e3,300,1e6],"nths":[0.1,0.01,0.1],"truncation":[3,3,3],"amp_cal":[0.1,1.0]}

with open('quantum_device_sim_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('simulation_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

dtsim = 0.05
hardware_cfg['awg_info']['keysight_pxi']['dt'] = dtsim
hardware_cfg['awg_info']['tek70001a']['dt'] = dtsim
quantum_device_cfg['qubit']['1']['frequency'] = 4.527556074037375
quantum_device_cfg['pulse_info']['1']['iq_freq'] = 0.2
quantum_device_cfg['pulse_info']['1']['pulse_type'] = 'gauss'


experiment_names = ['rabi']
seq_list = [100]

for experiment_name in experiment_names:
    ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg,plot_visdom=False)
    sequences = ps.get_experiment_sequences(experiment_name)
    s = Simulate_Multimode_Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg,multimode_params,True,dtsim,sequences, experiment_name)

    tlist, cp = s.get_charge_pulse(experiment_name, sequences)
    output = s.psb_mesolve(experiment_name, sequences, seq_list=seq_list)

    P = []
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131, title=experiment_name)
    for ii, o in enumerate(output):
        P.append(o.expect[0])
        ax.plot(tlist, P[ii], 'o-')
        fits = fitdecaysin(tlist, P[ii][:], showfit=True)
        print("Rabi frequency = %s MHz " % (2 * fits[1] * 1e3))
        print("Rabi decay time = %s mus" % (fits[3] / 1e3))
        ax.set_ylabel('Occupation')
        ax.set_xlabel('Time (ns)')

    ax = fig.add_subplot(132, title='pulse')
    ax.plot(tlist, cp[-3])
    ax.set_ylabel('Amp')
    ax.set_xlabel('Time (ns)')

    ax = fig.add_subplot(133, title='pulse FFT')
    f = fft.fft(cp[-1])
    freq = fft.fftfreq(tlist.shape[-1])

    ax.plot(freq[:int(len(f) / 2)] * 1 / dtsim, abs(f)[:int(len(f) / 2)])
    print(freq[argmax(abs(f))] * 1 / dtsim)
    print(s.find_freq(1, 0, 0) - s.find_freq(0, 0, 0))
    ax.axvline((s.find_freq(1, 0, 0) - s.find_freq(0, 0, 0)), color='k', linestyle='dashed')

    ax.set_ylabel('Amp FFT')
    ax.set_xlabel('Freq (GHz)')
    fig.tight_layout()

    plt.show()