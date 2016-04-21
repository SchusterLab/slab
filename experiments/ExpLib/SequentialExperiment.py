__author__ = 'Nelson'

from slab.experiments.Multimode.MultimodePulseSequenceExperiment import *
from slab.experiments.General.SingleQubitPulseSequenceExperiment import *
from slab.experiments.General.run_experiment import *
import gc

class SequentialExperiment():
    def __init__(self):
        pass


    def run(self,exp):

        expt_name = exp[0]
        expt_kwargs = exp[1]

        if 'seq_pre_run' in expt_kwargs:
            expt_kwargs['seq_pre_run'](self)

        self.expt = run_experiment(expt_name,**expt_kwargs)

        if 'seq_post_run' in expt_kwargs:
            expt_kwargs['seq_post_run'](self)

        del self.expt
        gc.collect()




