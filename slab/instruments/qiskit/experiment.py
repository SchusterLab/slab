"""
experiment.py

References:
[0] https://github.com/Qiskit/qiskit-terra/blob/2eee56616d50a9e26756f855ef4aa0135920ad78/qiskit/result/models.py#L99
"""

import copy

import numpy as np
from qiskit.providers import JobStatus
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.qobj import PulseQobjConfig
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
        self.config = PulseQobjConfig.from_dict(copy.copy(qobj.config.__dict__))
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
            memory = None
        else:
            if self.shots_completed + self.shots_per_set > self.shots:
                shots = self.shots - self.shots_completed
            else:
                shots = self.shots_per_set
            #ENDIF
            memory = self._run(shots)
            self.shots_completed += shots
            if self.shots_completed == self.shots:
                self.exhausted = True
            #ENDIF
        #ENDIF

        # concatenate this result with previous result
        if memory is None:
            result = prev_result
        else:
            # memory is not set for the first `prev_result`
            if hasattr(prev_result.data, "memory"):
                if (self.config.meas_level == MeasLevel.KERNELED
                    and self.config.meas_return == MeasReturnType.AVERAGE):
                    prev_count = prev_result.shots[1] - prev_result.shots[0]
                    prev_memory = prev_result.data.memory
                    this_count = self.shots_completed - shots
                    memory = ((prev_memory * prev_count + memory * this_count)
                            / (prev_count + this_count))
                else:
                    raise NotImplementedError("Only MeasLevel.KERNELED and MeasReturn.AVERAGE "
                                              "are currently supported.")
                #ENDIF
            #ENDIF
            # see [0]
            success = self.exhausted
            status = JobStatus.DONE if self.exhausted else JobStatus.RUNNING
            result = ExperimentResult(
                shots=(prev_result.shots[0], self.shots_completed),
                success=success,
                data=ExperimentResultData(
                    memory=memory,
                ),
                meas_level=self.config.meas_level,
                status=status,
                meas_return=self.config.meas_return,
                header=self.qexpt.header,
            )
        #ENDIF
        
        return result
    #ENDDEF

    def _run(self):
        raise NotImplementedError()
    #ENDDEF
#ENDDEF
