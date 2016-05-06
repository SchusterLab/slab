__author__ = 'Nitrogen'

from slab.experiments.Multimode.MultimodePulseSequenceExperiment import *
from slab.experiments.General.SingleQubitPulseSequenceExperiment import *

def foo(**kwargs):
    for key, value in kwargs.iteritems():
            print key
            print value

class SequentialExperiment():
    def __init__(self,exp_list,kwargs_list=None,adc=None):
        for ii in arange(len(exp_list)):
            print exp_list[ii]
            foo(**kwargs_list[ii])


exp_l = []
kwargs_l = []

exp_l.append('1')
exp_l.append('2')
exp_l.append('3')

kwargs_l.append({ 'foo' : 123, 'bar' : 456 })
kwargs_l.append({})
kwargs_l.append({ 'foo2' : 222, 'bar2' : 333 })


seq_exp = SequentialExperiment(exp_l,kwargs_l)
