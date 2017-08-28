__author__ = 'AlexMa'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.ExpLib.QubitPulseSequenceExperiment import *
from numpy import mean, arange
import numpy as np
from tqdm import tqdm

class HistogramHeteroExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='HistogramHetero', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix='Histogram_Hetero', config_file=config_file,
                                              PulseSequence=HistogramHeteroSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass


