__author__ = 'Nelson'

from slab.experiments.Alex.ExpLib.SequentialExperiment import *
from slab import *
import os
import json
import numpy as np


datapath = os.getcwd() + '\data'
config_file = os.path.join(datapath, "..\\config" + ".json")
with open(config_file, 'r') as fid:
        cfg_str = fid.read()

cfg = AttrDict(json.loads(cfg_str))

def get_data_filename(prefix):
    return  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))




def frequency_stabilization(seq_exp):
    seq_exp.run(('Ramsey',{}))
    if (abs(seq_exp.expt.offset_freq) < 50e3):
        print "Frequency is within expected value. No further calibration required."
        pass
    else:
        print seq_exp.expt.flux
        flux_offset = -seq_exp.expt.offset_freq/(seq_exp.expt.freq_flux_slope)
        print flux_offset
        if (abs(flux_offset) < 0.000002):
            flux2 = seq_exp.expt.flux + flux_offset
            print flux2
            seq_exp.run(('Ramsey',{'flux':flux2}))
            offset_freq2 = seq_exp.expt.offset_freq
            flux_offset2 = -seq_exp.expt.offset_freq/(seq_exp.expt.freq_flux_slope)
            flux3 = flux2 + flux_offset2
            if (abs(offset_freq2) < 50e3):
                print "Great success! Frequency calibrated"
                seq_exp.expt.save_config()
            else:
                if (abs(flux_offset2) < 0.000002):
                    seq_exp.run(('Ramsey',{'flux':flux3}))
                    if (abs(seq_exp.expt.offset_freq) < 100e3):
                        print "Frequency calibrated"
                        seq_exp.expt.save_config()
                    else:
                        print "Try again: not converged after 2 tries"
                else:
                    print "Large change in flux is required; please do so manually"
                    pass
        else:
            print "Large change in flux is required; please do so manually"
            pass

def pulse_calibration(seq_exp,phase_exp=True):
    frequency_stabilization(seq_exp)
    seq_exp.run(('Rabi',{'update_config':True}))
    print "ge pi and pi/2 pulses recalibrated"

    if phase_exp:
        seq_exp.run(('HalfPiYPhaseOptimization',{'update_config':True}))
        print "Offset phase recalibrated"
    pass


def run_seq_experiment(expt_name,lp_enable=True):
    seq_exp = SequentialExperiment(lp_enable)

    if expt_name.lower() == 'testing':
        prefix = 'Testing'
        data_file = get_data_filename(prefix)

        def testing(self):
            print "testing: " + str(self.expt.offset_freq)

        seq_exp.run(('Ramsey',{'seq_post_run':testing}))
        seq_exp.run(('Rabi',{"trigger_period":0.0003,"data_file":data_file}))
        seq_exp.run(('Rabi',{"data_file":data_file}))
        seq_exp.run(('Rabi',{"data_file":data_file}))
        seq_exp.run(('Ramsey',{}))
        seq_exp.run(('T1',{}))

    if expt_name.lower() == 'frequency_stabilization':
        frequency_stabilization(seq_exp)

    if expt_name.lower() == 'pulse_calibration':
        pulse_calibration(seq_exp)

    if expt_name.lower() == 'rabi_ramsey_t1':
        # ef_pulse_calibration(seq_exp)
        prefix = 'rabi_ramsey_t1'
        data_file = get_data_filename(prefix)
        flux_pts = np.load(r'S:\_Data\160711 - Nb Tunable Coupler\data\flux_total.npy')
        drive_pts = np.load(r'S:\_Data\160711 - Nb Tunable Coupler\data\probe_frequencies_total.npy')
        readout_pts = np.load(r'S:\_Data\160711 - Nb Tunable Coupler\data\read_frequencies_7_28.npy')


        seq_exp.run(('rabi_ramsey_t1_flux_sweep', {'exp':'rabi','flux':0.0055,'drive_freq':5.071e9,'update_config':False,"data_file":data_file}))
        # print seq_exp.expt.offset_freq


