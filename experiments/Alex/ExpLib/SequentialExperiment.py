__author__ = 'Nelson'

from slab.experiments.Alex.General.run_experiment import *
from slab.experiments.Alex.Multimode.run_multimode_experiment import *

import gc

class SequentialExperiment():
    def __init__(self, lp_enable = True):
        self.expt = None
        self.lp_enable = lp_enable


    def run(self,exp):

        if self.expt is not None:
            del self.expt
            gc.collect()

        expt_name = exp[0]
        expt_kwargs = exp[1]

        if 'seq_pre_run' in expt_kwargs:
            expt_kwargs['seq_pre_run'](self)

        self.expt = run_experiment(expt_name,self.lp_enable,**expt_kwargs)
        if self.expt is None:
            self.expt = run_multimode_experiment(expt_name,self.lp_enable,**expt_kwargs)

        if 'seq_post_run' in expt_kwargs:
            expt_kwargs['seq_post_run'](self)

        if 'update_config' in expt_kwargs:
            if expt_kwargs['update_config']:
                self.save_config()

    def save_config(self):
        self.expt.save_config()
        print "config saved!"





