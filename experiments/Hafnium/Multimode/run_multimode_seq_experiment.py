__author__ = 'Nelson'

from slab.experiments.ExpLib.SequentialExperiment import *
from slab.experiments.General.run_seq_experiment import *
from slab import *
import os
import json


datapath = os.getcwd() + '\data'
config_file = os.path.join(datapath, "..\\config" + ".json")
with open(config_file, 'r') as fid:
        cfg_str = fid.read()

cfg = AttrDict(json.loads(cfg_str))

def get_data_filename(prefix):
    return  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))


def multimode_pulse_calibration(seq_exp, mode):
    pulse_calibration(seq_exp)
    seq_exp.run(('multimode_calibrate_offset',{'exp':'multimode_rabi','dc_offset_guess':0,'mode':mode,'update_config':True}))

def multimode_pi_pi_phase_calibration(seq_exp,mode):
    seq_exp.run(('multimode_pi_pi_experiment',{'mode':mode,'update_config':True}))



def run_multimode_seq_experiment(expt_name,lp_enable=True,**kwargs):
    seq_exp = SequentialExperiment(lp_enable)

    if expt_name.lower() == 'multimode_pulse_calibration':
        multimode_pulse_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_pi_pi_phase_calibration':
        multimode_pi_pi_phase_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'sequential_single_mode_rb':
        prefix = expt_name.lower()
        data_file = get_data_filename(prefix)

        with SlabFile(data_file) as f:
            f.append_pt('mode', kwargs['mode'])
            f.close()
        for i in arange(32):
            multimode_pulse_calibration(seq_exp,kwargs['mode'])
            multimode_pi_pi_phase_calibration(seq_exp,kwargs['mode'])
            seq_exp.run(('single_mode_rb',{'mode':kwargs['mode'],"data_file":data_file}))



