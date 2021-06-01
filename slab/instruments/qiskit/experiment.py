"""
experiment.py
"""

import copy

import numpy as np

from qiskit.qobj.utils import MeasReturnType, MeasLevel

class PulseExperiment(object):
    """
    Base class for executing a qobj experiment representation.
    """
    def __init__(self, qobj, qexpt, backend, log_extra):
        """
        args:
        backend :: SLabBackend
        """
        self.qobj = qobj
        self.qexpt = qexpt
        # config uses job-level config info
        self.config = copy.copy(qobj.config)
        # config prioritizes experiment-level config info
        if qexpt.config is not None:
            self.config.__dict__.update(qexpt.config.__dict__)
        #ENDIF
        self.backend = backend
        self.shots_per_set = getattr(self.config, "shots_per_set",
                                     self.backend.default_shots_per_set)
        self.shots = self.config.shots
        self.sets = int(np.ceil(self.shots / self.shots_per_set))
        self.shots_completed = 0
        self.empty_dict = dict()
        self.exhausted = False
        self.memory_count = 0
    #ENDDEF
    
    def run_next_set(self, prev_result):
        # run the next set if all sets have not been run
        if self.exhausted:
            result = None
        else:
            if self.shots_completed + self.shots_per_set > self.shots:
                shots = self.shots - self.shots_completed
            else:
                shots = self.shots_per_set
            #ENDIF
            result = self._run(shots)
            self.shots_completed += shots
            if self.shots_completed == self.shots:
                self.exhausted = True
            #ENDIF
        #ENDIF

        # concatenate this result with previous result
        if result is None:
            result = prev_result
        else:
            shots = (prev_result.shots[0], result.shots[1])
            result.shots = shots
            # memory is uninitialized for the first prev_result
            if prev_result.data.memory is not None:
                if (self.config.meas_level == MeasLevel.KERNELED
                and self.config.meas_return == MeasReturnType.AVERAGE):
                    prev_count = prev_result.shots[1] - prev_result.shots[0]
                    prev_data = prev_result.data.memory
                    this_count = result["shots"][1] - result["shots"][0]
                    this_data = prev_result["data"]["memory"]
                    data = ((prev_data * prev_count + this_data * this_count)
                            / (prev_count + this_count))
                else:
                    raise NotImplementedError("Only MeasLevel.KERNELED and MeasReturn.AVERAGE "
                                              "are currently supported.")
                #ENDIF
                result.data.memory = data
            #ENDIF
        #ENDIF
        
        return result
    #ENDDEF

    def _run(self):
        raise NotImplementedError()
    #ENDDEF
#ENDDEF
