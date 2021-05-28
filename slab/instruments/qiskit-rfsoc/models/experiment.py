"""
experiment.py
"""

class PulseExperiment(object):
    def __init__(self, qobj_dict, expt_dict, backend):
        self.qobj_dict = qobj_dict
        self.expt_dict = expt_dict
        self.backend = backend
    #ENDDEF
#ENDDEF

