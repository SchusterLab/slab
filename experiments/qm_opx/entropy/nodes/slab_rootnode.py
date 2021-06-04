from quaentropy.api.execution import EntropyContext
from quaentropy.graph_experiment import pynode
from entropyext_cal import *
import numpy as np
from configuration import config

@pynode("root", output_vars={'config'})
def root(context: EntropyContext):
    qpu_db = context.get_resource('qpu_db1')
    
    for con in config["controllers"].keys():
        qpu_db.set(con, 'adc1_offset', config["controllers"][con]["analog_inputs"][1]["offset"])
        qpu_db.set(con, 'adc2_offset', config["controllers"][con]["analog_inputs"][2]["offset"])

    qpu_db.commit()
    return {'config': QuaConfig(config)}
