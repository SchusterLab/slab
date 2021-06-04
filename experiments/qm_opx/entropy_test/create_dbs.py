import os
import qm
from entropylab import *
import numpy as np
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from entropylab_qpudb import create_new_qpu_database, CalState, QpuDatabaseConnection

db_file='docs_cache/tutorial.db'
if os.path.exists(db_file):
    os.remove(db_file)
db = SqlAlchemyDB(db_file)
lab = LabResources(db)
lab.register_resource('qmm',QuantumMachinesManager)

initial_dict = {
    'qubit': {'ge_freq': 4.74*1e9, 't1_ge': 100, 't2_ge': 100, 'ge_if': 100*1e6, 'anharmonicity': -140*1e6,  'q_lo': 4.64*1e9,
              'ge_fast_pi_len': 40e-3 , 'ge_fast_pi_amp': 0.45,'ge_fast_pi2_len': 40e-3 , 'ge_fast_pi2_amp': 0.225,
              'ge_slow_pi_len': 3000e-3 , 'ge_slow_pi_amp':0.005, 'ramsey_freq': 100e3,

              'ef_freq': 4.60*1e9, 't1_ef': 50, 't2_ef': 50, 'ef_if': -40*1e6,
              'ef_fast_pi_len': 40e-3 , 'ef_fast_pi_amp':0.45, 'ef_slow_pi_len': 3000e-3 , 'ef_slow_pi_amp':0.005,
              },

    'storage': {'mode_freq': 6.011*1e9, 't1_mode': 681, 't2_mode': 1362,'s_if': 100*1e6, 'self_kerr': -3*1e3, 's_lo': 6.11*1e9,
                },

    'readout': {'rr_freq': 8.0816*1e9, 't1_rr':200*1e-3, 'rr_if':100e6, 'rr_amp':0.012, 'rr_len':3,
                'integration_weights': {}, 'tof':540e-3, 'smearing' :0,  'adc1_offset': 0.1, 'adc2_offset': 0.1,
                },
    'jpa_pump': {'jpa_freq': 8.0816*1e9, 'jpa_if':100e6, 'jpa_amp':0.06, 'pump_len':3, 'flux_bias': 0.439*1e-3, #YOKO

                 }
}

create_new_qpu_database('qpu_db', initial_dict, force_create=True)
lab.register_resource('qpu_db', QpuDatabaseConnection, ['qpu_db'])