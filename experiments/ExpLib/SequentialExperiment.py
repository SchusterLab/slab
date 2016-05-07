__author__ = 'Nelson'

from slab.experiments.General.run_experiment import *
from slab.experiments.Multimode.run_multimode_experiment import *
import numbers
from slab import *

import gc

class SequentialExperiment():
    def __init__(self, lp_enable = True):
        self.expt = None
        self.lp_enable = lp_enable


    def run(self,expt_name, expt_kwargs = {}):

        if self.expt is not None:
            del self.expt
            gc.collect()

        ## automatically save kwargs to data_file
        if 'data_file' in expt_kwargs:
            data_file = expt_kwargs['data_file']
            for key in expt_kwargs:
                if isinstance(expt_kwargs[key],numbers.Number):
                    with SlabFile(data_file) as f:
                        f.append_pt(key, expt_kwargs[key])
                        f.close()

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





