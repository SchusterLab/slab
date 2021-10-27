import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
import json
import numpy as np
import os
from DACInterface import AD5780_serial
import time

path = os.getcwd()

with open('210301_quantum_device_config_wff.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('210301_experiment_config_wff.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('210226_hardware_config_wff.json', 'r') as f:
    hardware_cfg = json.load(f)

dac = AD5780_serial()
Mott_freq = 4.85

Mott_flux_vecs = [[-0.0533204045389686,-0.3291024670554185,0.4618740926447375,0.4952553928851188,1.5130523903476178,
                   -0.7702480512672444,1.0807651561430962,0.02455033185379301],
                  [-0.5355349176628919,0.09989503284452496,0.4762689969941896,0.5004867366697248,1.516348433010241,
                   -0.7687891095530588,1.0832691261921845,0.02456323583643092],
                  [-0.5324475880031694,-0.3683409768486023,0.973366815850341,0.524980882178294,1.5309341091035904,
                   -0.7572379010661903,1.0973037211147094,0.034507227896383594],
                  [-0.5878608935146372,-0.3813358487305377,0.4306071675084669,0.9615491763551496,1.5211980929987057,
                   -0.7738475130113344,1.079068308375672,0.018718758457770095],
                  [-0.5670415915080468,-0.4149692959992617,0.38266709512629754,0.4140761879175279,1.4216524780191926,
                   -0.8529620081746109,0.9732169001025622,-0.059472201334626336],
                  [-0.38086272454569814,-0.15610636065983932,0.6443168728339257,0.6945013086017903,
                   1.0947230682578466,-1.1439212643515813,1.106044126881091,0.07909462356974822],
                  [-0.46359196245505896,-0.2780947040410806,0.49645796576969303,0.5214943204918907,
                   0.8877491243746013,-0.9148854854299793,1.3899740901567588,-0.0975478845268747],
                  [-0.4598872411782834,-0.30773154940808833,0.4982305341447243,0.5253240535624477,0.8945644432435799,
                   -0.9090251844589246,0.8766285084899854,0.399320622348767]
                  ]

qb_rd_freqs = [6.162, 6.3860, 6.608, 6.82329, 6.95277, 6.72277,6.4975,6.27205]
ppiq_expect = [Mott_freq]*8
#ppiq_meas1 = [0]*8
#ppiq_meas1 = [0]*8

for V_ramp in [0.125]:
    for qb_ind in range(0,1):
        #ramp qb so that it is at 4.85GHz, and rest are at 5.5GHz
        for ii in range(0,8):
            dac.ramp3(ii+1,Mott_flux_vecs[qb_ind][ii],1,7)

        time.sleep(2)
        #run pulse probe iq
        quantum_device_cfg['qubit']['1']['freq'] = ppiq_expect[qb_ind]
        quantum_device_cfg['readout']['freq'] = qb_rd_freqs[qb_ind]
        experiment_names = ['pulse_probe_iq']

        for experiment_name in experiment_names:
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            # ramp qb so that it is at V_ramp
            dac.ramp3(1, V_ramp + Mott_flux_vecs[0][0], 1, 7)
            time.sleep(2)
            dac.ramp3(1, -V_ramp + Mott_flux_vecs[0][0], 1, 7)
            time.sleep(2)
            # ramp qb back so that it is at 4.85GHz, and rest are at 5.5GHz
            dac.ramp3(1, Mott_flux_vecs[0][0], 1, 7)
            time.sleep(2)
            # run pulse probe iq
            quantum_device_cfg['qubit']['1']['freq'] = ppiq_expect[qb_ind]
            quantum_device_cfg['readout']['freq'] = qb_rd_freqs[qb_ind]
            experiment_names = ['pulse_probe_iq']
            for experiment_name in experiment_names:
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("Sequences generated")
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)


