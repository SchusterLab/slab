import os
import qm
from quaentropy.api.graph import Graph
from quaentropy.graph_experiment import GraphExperiment
from quaentropy.instruments.lab_topology import ExperimentResources
from quaentropy.results_backend.sqlalchemy.db import SqlAlchemyDB
import numpy as np
import matplotlib.pyplot as plt
from scope import my_scope

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration_10q import config
from qpu_resolver import resolve
from rootnode import root
import configuration_10q
from nodes.tof_QUA_cal_node import TofCalib
from nodes.res_spec_QUA_node import ResSpec
from nodes.q_spec_QUA_node import QubitSpec
from nodes.PowerRabi_QUA_node import PowerRabiNode
from nodes.t1_QUA_node import T1Node
from nodes.t2Ramsey_QUA_node import T2RamseyNode
from nodes.t2Ramsey_QUA_virtual_node import T2RamseyVirtualNode
from nodes.t2Ramsey_chevron_QUA_virtual_node import T2RamseyChevronNode
from nodes.t2Ramsey_QUA_virtual2D_node import T2RamseyVirtual2DNode
from nodes.res_spec_pi_nopi import ResSpecPiNopi
from nodes.res_scatter_pi_nopi import IQblobs
from nodes.nn_multi_readout_node import NNTrainBenchmark
# from nodes.single_qubit_active_reset_switch_case import SingleQubitActiveReset
from nodes.single_qubit_TwoStateDiscriminator_node import TwoStateDiscriminatorNode
from nodes.multi_readout_benchmark_node import MultiReadoutBenchmark
from entropyext_cal import *
from nodes.multi_readout_benchmark_w_active_reset_node import ActiveResetBenchmark

#############################
# connect to the entropy db #
#############################
db_file='entropy_db1.db'
entropy_db = SqlAlchemyDB(db_file)


##############################################
# define resources needed for the experiment #
##############################################
experiment_resources = ExperimentResources(entropy_db)
experiment_resources.import_lab_resource('qmm')
experiment_resources.import_lab_resource('qpu_db1')

# config_node = PyNode("config_node", get_config, output_vars={'config'})

tof_cals = [0]*2
tof_cals[0] = TofCalib(1, dependency=root)
tof_cals[1] = TofCalib(6, dependency=root)

# resonator_spec_cals = [0]*10
# f_min = []
# f_max = []
# for i in range(10):
#     f_min.append(-400e6+i*80e6)
#     f_max.append(-400e6+(i+1)*80e6)
    
# for i in range(10):
#     if i<5:
#         resonator_spec_cals[i] = ResSpec(i+1, f_min[i], f_max[i], 1e6, dependency=tof_cals[0])
#     else:
#         resonator_spec_cals[i] = ResSpec(i+1, f_min[i], f_max[i], 1e6, dependency=tof_cals[1])
qpu = experiment_resources.get_resource('qpu_db1')
ro_LO_freq = configuration_10q.ro_LO_freq
expected_rr_f = np.array([6.11587e9, 5.612e9, 5.925e9, 6.244e9]) - ro_LO_freq

resonator_spec_cals = [
    ResSpec(2, expected_rr_f[0] - 50e6, expected_rr_f[0] + 50e6, 1e6, dependency=tof_cals[0]),
    ResSpec(4, expected_rr_f[1] - 50e6, expected_rr_f[1] + 50e6, 1e6, dependency=tof_cals[0]),
    ResSpec(6, expected_rr_f[2] - 50e6, expected_rr_f[2] + 50e6, 1e6, dependency=tof_cals[1]),
    ResSpec(8, expected_rr_f[3] - 50e6, expected_rr_f[3] + 50e6, 1e6, dependency=tof_cals[1])]

qubit_spec_cals = [
    QubitSpec(2, -400e6, 400e6, 1e6, dependency=resonator_spec_cals[0]),
    QubitSpec(4, -400e6, 400e6, 1e6, dependency=resonator_spec_cals[1]),
    QubitSpec(6, -400e6, 400e6, 1e6, dependency=resonator_spec_cals[2]),
    QubitSpec(8, -400e6, 400e6, 1e6, dependency=resonator_spec_cals[3])
    ]

power_rabi_cals = [
    PowerRabiNode(2, 1.1, 0.01, dependency = qubit_spec_cals[0]),
    PowerRabiNode(4, 1.1, 0.01, dependency = qubit_spec_cals[1]),
    PowerRabiNode(6, 1.1, 0.01, dependency = qubit_spec_cals[2]),
    PowerRabiNode(8, 1.1, 0.01, dependency = qubit_spec_cals[3])]

t1_cals = [
    T1Node(2, 4, 30000, 200, dependency=power_rabi_cals[0]),
    T1Node(4, 4, 30000, 200, dependency=power_rabi_cals[1]),
    T1Node(6, 4, 30000, 200, dependency=power_rabi_cals[2]),
    T1Node(8, 4, 30000, 200, dependency=power_rabi_cals[3])
    ]

# t2Ramsey_cals = [
#     T2RamseyNode(2, 4, 1000, 20, 1e6, dependency=power_rabi_cals[0]),
#     T2RamseyNode(4, 4, 1000, 20, 1e6, dependency=power_rabi_cals[1]),
#     T2RamseyNode(6, 4, 2000, 20, 1e6, dependency=power_rabi_cals[2]),
#     T2RamseyNode(8, 4, 2000, 20, 1e6, dependency=power_rabi_cals[3])
#     ]

t2Ramsey_virtual_cals = [
    T2RamseyVirtualNode(2, 4, 1000, 4, 4e6, dependency=power_rabi_cals[0]),
    T2RamseyVirtualNode(4, 4, 1000, 4, 4e6, dependency=power_rabi_cals[1]),
    T2RamseyVirtualNode(6, 4, 1000, 4, 4e6, dependency=power_rabi_cals[2]),
    T2RamseyVirtualNode(8, 4, 1000, 4, 4e6, dependency=power_rabi_cals[3])
    ]

power_rabi_cals2 = [
    PowerRabiNode(2, 1.1, 0.02, dependency=t2Ramsey_virtual_cals[0]),
    PowerRabiNode(4, 1.1, 0.02, dependency=t2Ramsey_virtual_cals[1]),
    PowerRabiNode(6, 1.1, 0.02, dependency=t2Ramsey_virtual_cals[2]),
    PowerRabiNode(8, 1.1, 0.02, dependency=t2Ramsey_virtual_cals[3])]

# t2RamseyChevron_cals = [
#     T2RamseyChevronNode(2, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[0]),
#     T2RamseyChevronNode(2, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[1]),
#     T2RamseyChevronNode(2, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[2]),
#     T2RamseyChevronNode(2, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[3])]

t2RamseyChevron_cals = [
    T2RamseyVirtual2DNode(2, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[0]),
    T2RamseyVirtual2DNode(4, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[1]),
    T2RamseyVirtual2DNode(6, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[2]),
    T2RamseyVirtual2DNode(8, 4, 1000, 20, -1/5, 1/5, 1/20, dependency=power_rabi_cals[3])]

ro_pinopi_cals = [
    ResSpecPiNopi(2, 50e6, 1e6, dependency=power_rabi_cals2[0]),
    ResSpecPiNopi(4, 50e6, 1e6, dependency=power_rabi_cals2[1]),
    ResSpecPiNopi(6, 50e6, 1e6, dependency=power_rabi_cals2[2]),
    ResSpecPiNopi(8, 50e6, 1e6, dependency=power_rabi_cals2[3])
    ]

ro_blobs_cals = [
    IQblobs(2, 10000, dependency=ro_pinopi_cals[0]),
    IQblobs(4, 10000, dependency=ro_pinopi_cals[1]),
    IQblobs(6, 10000, dependency=ro_pinopi_cals[2]),
    IQblobs(8, 10000, dependency=ro_pinopi_cals[3])
    ]

single_qubit_twostate_discriminator = [
    TwoStateDiscriminatorNode(2, 1000, 200000, True, dependency=ro_pinopi_cals[0]),
    TwoStateDiscriminatorNode(4, 1000, 200000, True, dependency=ro_pinopi_cals[1]),
    TwoStateDiscriminatorNode(6, 1000, 200000, True, dependency=ro_pinopi_cals[2]),
    TwoStateDiscriminatorNode(8, 1000, 200000, True, dependency=ro_pinopi_cals[3])]
# nn_calib = [
#     NNTrainBenchmark(2, 3000, False, dependency=ro_pinopi_cals[0]),
#     NNTrainBenchmark(4, 500, False, dependency=ro_pinopi_cals[1]),
#     NNTrainBenchmark(6, 3000, False, dependency=ro_pinopi_cals[2]),
#     NNTrainBenchmark(8, 3000, False, dependency=ro_pinopi_cals[3])
#     ]

nn_calib = [NNTrainBenchmark([2, 4], 400, True, dependency=ro_pinopi_cals)]

multi_qubit_readout_benchmark = [
    MultiReadoutBenchmark([2, 4, 6, 8], 1000, 200000, dependency=single_qubit_twostate_discriminator)]

active_reset_benchmark = [
    ActiveResetBenchmark([2], 100000, 1000, dependency=single_qubit_twostate_discriminator)]
# single_qubit_active_reset = [
#     SingleQubitActiveReset(4, dependency=nn_calib[0])]

# nodes = set().union([root], [tof_cals[0]], [resonator_spec_cals[1]], 
#                     [qubit_spec_cals[1]], [power_rabi_cals[1]])
nodes = set().union([root], tof_cals, resonator_spec_cals, 
                    qubit_spec_cals, power_rabi_cals, t2Ramsey_virtual_cals, t1_cals, 
                    power_rabi_cals2, ro_pinopi_cals, single_qubit_twostate_discriminator,
                    multi_qubit_readout_benchmark, active_reset_benchmark)
                    # t2RamseyChevron_cals, ro_pinopi_cals,
                    # ro_blobs_cals, , single_qubit_active_reset)


graph = Graph(nodes, "calibration_graph")
experiment = GraphExperiment(experiment_resources, graph, "calibration_graph")
# handle = experiment.run()
handle = experiment.run_to_node(active_reset_benchmark[0], strategy=AncestorRunStrategy.RunOnlyLast)


# handle = experiment.run_to_node(t1_cals[1], strategy=AncestorRunStrategy.RunOnlyLast)
