import os
import qm
from quaentropy.api.graph import Graph
from quaentropy.graph_experiment import GraphExperiment
from quaentropy.instruments.lab_topology import ExperimentResources
from quaentropy.results_backend.sqlalchemy.db import SqlAlchemyDB
import numpy as np
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from nodes.slab_rootnode import config
from nodes.slab_adc_offsets_node import AdcOffsetsNode
from entropyext_cal import *
from entropylab import *


#############################
# connect to the entropy db #
#############################
db_name='C:\\_Lib\\python\\slab\\experiments\\qm_opx\\entropy\\slab_entropy.db'
# if os.path.exists(db_name):
#     os.remove(db_name)
slab_entropy_db = SqlAlchemyDB(db_name)
lab = LabResources(slab_entropy_db)


# entropy_db = SqlAlchemyDB(db_file)

##############################################
# define resources needed for the experiment #
##############################################
# experiment_resources = ExperimentResources(entropy_db)
# experiment_resources.import_lab_resource('qmm')
# experiment_resources.import_lab_resource('slab_qpu_db')
# qpu = experiment_resources.get_resource('qpu_db')
#
# #########
# # nodes #
# #########
# adc_offsets = [AdcOffsetsNode]
# nodes = set().union([root], adc_offsets)
#
# #########
# # graph #
# #########
# graph = Graph(nodes, "calibration_graph")
# experiment = GraphExperiment(experiment_resources, graph, "calibration_graph")
# # handle = experiment.run()
# handle = experiment.run_to_node(active_reset_benchmark[0], strategy=AncestorRunStrategy.RunOnlyLast)


# handle = experiment.run_to_node(t1_cals[1], strategy=AncestorRunStrategy.RunOnlyLast)
