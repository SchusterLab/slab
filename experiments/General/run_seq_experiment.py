__author__ = 'Nelson'

from slab.experiments.ExpLib.SequentialExperiment import *
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


def frequency_stabilization(seq_exp):
    seq_exp.run('Ramsey',{})
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
            seq_exp.run('Ramsey',{'flux':flux2})
            offset_freq2 = seq_exp.expt.offset_freq
            flux_offset2 = -seq_exp.expt.offset_freq/(seq_exp.expt.freq_flux_slope)
            flux3 = flux2 + flux_offset2
            if (abs(offset_freq2) < 50e3):
                print "Great success! Frequency calibrated"
                seq_exp.expt.save_config()
            else:
                if (abs(flux_offset2) < 0.000002):
                    seq_exp.run('Ramsey',{'flux':flux3})
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
    seq_exp.run('Rabi',{'update_config':True})
    print "ge pi and pi/2 pulses recalibrated"

    if phase_exp:
        seq_exp.run('HalfPiYPhaseOptimization',{'update_config':True})
        print "Offset phase recalibrated"
    pass

def ef_pulse_calibration(seq_exp):
    seq_exp.run('ef_Rabi',{'update_config':True})
    print "ef pi and pi/2 pulses recalibrated"


def ef_frequency_calibration(seq_exp):
    seq_exp.run('ef_Rabi',{'update_config':True})
    print "ef pi and pi/2 pulses recalibrated"

    seq_exp.run('ef_Ramsey',{})
    if abs(seq_exp.expt.offset_freq) < 50e6:
        print "Anharmonicity well calibrated: no change!"

    elif  abs(seq_exp.expt.offset_freq) > 50e6 and abs(seq_exp.expt.offset_freq) < 500e6:
        seq_exp.expt.save_config()
        print "Alpha changed by + %s Hz"%(seq.exp.expt.offset_freq)

    else:
        print "Anharmonicity suggested change > 250 kHz: Rerunnig EF Ramsey"
        seq_exp.run('ef_Ramsey',{})
        if abs(seq_exp.expt.offset_freq) > 500e6:
            print "Large anharmonicity change suggested again: check manually"
        else:
            seq_exp.expt.save_config()
            print "Something wierd about previous ef Ramsey: new anharmonicity saved to config"
            print "Alpha changed by +  %s Hz"%(seq.exp.expt.offset_freq)



def run_seq_experiment(expt_name,lp_enable=True):
    seq_exp = SequentialExperiment(lp_enable)

    if expt_name.lower() == 'frequency_stabilization':
        frequency_stabilization(seq_exp)

    if expt_name.lower() == 'pulse_calibration':
        pulse_calibration(seq_exp)

    if expt_name.lower() == 'ef_pulse_calibration':
        ef_pulse_calibration(seq_exp)

