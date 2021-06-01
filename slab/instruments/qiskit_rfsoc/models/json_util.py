"""
json_util.py
"""

import enum
import json

import numpy as np

class PulseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            ret = obj.tolist()
        elif isinstance(obj, complex):
            ret = (obj.real, obj.imag)
        elif isinstance(obj, enum.Enum):
            ret = obj.value
        else:
            ret = json.JSONEncoder.default(self, obj)
        #ENDELSE
        
        return ret
    #ENDDEF
#ENDCLASS
