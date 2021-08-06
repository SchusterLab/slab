import os
import qm
from entropylab import *
import numpy as np
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from entropylab_qpudb import create_new_qpu_database, CalState, QpuDatabaseConnection
from entropylab import Graph
from entropylab_qpudb import create_new_qpu_database, CalState, QpuDatabaseConnection
from entropylab.instruments.lab_topology import LabResources, ExperimentResources
from entropylab.results_backend.sqlalchemy.db import SqlAlchemyDB

# import nodes #
from root_configuration import root
from nodes.slab_adc_offsets_node import AdcOffsetsNode
from nodes.slab_rr_spec_node import ResSpec
from nodes.slab_qubit_spec_node import QubitSpec
from nodes.slab_ge_power_rabi_node import PowerRabi
from nodes.slab_ge_ramsey_phase_node import RamseyPhase

db_file='docs_cache/tutorial.db'
db = SqlAlchemyDB(db_file)
experiment_resources = ExperimentResources(db)
experiment_resources.import_lab_resource('qmm')

adc_offsets_node1 = AdcOffsetsNode(dependency=root)
adc_offsets_node2 = AdcOffsetsNode(dependency=adc_offsets_node1)
res_spec = ResSpec(95e6, 105e6, 0.05e6, 3000, 0.1, 1000, [2.0] * int(3000/4), dependency=adc_offsets_node2)
qubit_spec = QubitSpec(95e6, 105e6, 0.1e6, 100e3, 0.001, 1000, dependency=res_spec)
power_rabi = PowerRabi(0.0, 1.0, 0.01, 40, 1000, dependency=qubit_spec)
ramsey_phase = RamseyPhase(0.0, 120e3, 1000, 100e3, 1000, dependency=power_rabi)

experiment_resources.import_lab_resource('qpu_db')

run_set = set().union([root, adc_offsets_node1, adc_offsets_node2, res_spec, qubit_spec, power_rabi, ramsey_phase])

calibration_experiment = Graph(experiment_resources, run_set, 'my_calib_graph')

calibration_experiment.run()
# calibration_experiment.run_to_node(res_spec)

qpu_db = experiment_resources.get_resource('qpu_db')

qm_id = experiment_resources.get_resource('qmm').list_open_quantum_machines()[0]
config = experiment_resources.get_resource('qmm').get_qm(qm_id).get_config()

# config_node = PyNode("config_node", get_config, output_vars={'config'})
# qua_node = PyNode("QUA_node", AdcOffsetsNode(False),input_vars={'config':config_node.outputs['config']})
# experiment_with_QUA = Graph(experiment_resources, {config_node,qua_node}, "a QUA run")
# handle_with_QUA = experiment_with_QUA.run()


# config_node = PyNode("config_node", get_config, output_vars={'config'})
# qua_node = PyNode("QUA_node", QUA_node_action,input_vars={'config':config_node.outputs['config']},output_vars={'samples'})
# experiment_with_QUA = Graph(experiment_resources, {config_node,qua_node}, "a QUA run")
# handle_with_QUA = experiment_with_QUA.run()
#
# plt.figure()
# plt.plot(handle_with_QUA.results.get_results()[1].data)
# plt.show()
#
# qpu_db = QpuDatabaseConnection('qpu_db')


#
# qpu_db.set('q1', 'f01', 5.36e9, CalState.COARSE)
#
# qpu_db.commit('a test commit')
#
# qpu_db.close()
# del qpu_db
# qpu_db = QpuDatabaseConnection('qpu_db')
# qpu_db.print()
#
# print(qpu_db.get_history())
#
# qpu_db.add_attribute('q1', 'anharmonicity')
#
# print(qpu_db.q(1).anharmonicity)
#
# qpu_db.update_q(1, 'anharmonicity', -300e6, new_cal_state=CalState.FINE)
#
# print(qpu_db.q(1).anharmonicity)
#
# qpu_db.restore_from_history(0)
# print(qpu_db.q(1).f01)

