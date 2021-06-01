"""
experiment.py
"""

class PulseExperiment(object):
    """
    Base class for executing a qobj experiment representation.
    """
    def __init__(self, qobj_dict, expt_dict, backend):
        """
        args:
        backend :: SLabBackend
        """
        self.qobj_dict = qobj_dict
        self.expt_dict = expt_dict
        # generate a config that uses qobj_dict["config"] information, and overrides it
        # when expt_dict["config"] information is present
        self.config = copy(qobj_dict)
        if "config" in expt_dict:
            expt_config = expt_dict["config"]
            for key in qobj_dict.keys():
                if key in expt_config:
                    self.config[key] = expt_config[key]
                #ENDIF
            #ENDFOR
            # add all keys in expt_dict["config"] to self.config that were not
            # present in qobj_dict["config"]
            qobj_config = qobj_dict["config"]
            qobj_config_keys = qobj_config.keys()
            for key in expt_config.keys():
                if not key in qobj_config_keys:
                    self.config[key] = expt_config[key]
                #ENDIF
            #ENDFOR
        #ENDIF
        self.backend = backend
        self.shots_per_set = self.config.get("shots_per_set",
                                             self.backend.default_shots_per_set)
        self.shots = self.config["shots"]
        self.sets = int(np.ceil(shots / shots_per_set))
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
        elif prev_result is not None:
            shots = (prev_result["shots"][0], result["shots"][1])
            if (self.config["meas_level"] == MeasLevel.KERNELED.value
                and self.config["meas_return"] == MeasReturn.AVERAGE.value):
                prev_count = prev_result["shots"][1] - prev_result["shots"][0]
                prev_data = prev_result["data"]["memory"]
                this_count = result["shots"][1] - result["shots"][0]
                this_data = prev_result["data"]["memory"]
                data = (prev_data * prev_count + this_data * this_count) / (prev_count + this_count)
            else:
                raise NotImplementedError("Only MeasLevel.KERNELED and MeasReturn.AVERAGE "
                                          "are currently supported.")
            #ENDIF
            result["shots"] = shots
            result["data"]["memory"] = data
        #ENDIF
        
        return result
    #ENDDEF

    def _run(self):
        raise NotImplementedError()
    #ENDDEF
#ENDDEF

