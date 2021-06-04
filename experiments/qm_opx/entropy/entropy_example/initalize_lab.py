# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:08:16 2021

@author: paint
"""
import os
import qm
from entropylab import *
import numpy as np
import matplotlib.pyplot as plt
from scope import my_scope
from entropylab_qpudb import create_new_qpu_database, QpuDatabaseConnection
from qm.QuantumMachinesManager import QuantumMachinesManager
from caltech_qpu_db import qpu_base_data
from qpu_resolver import resolve

entropy_db_file='entropy_db.db'
if os.path.exists(entropy_db_file):
  os.remove(entropy_db_file)
entropy_db = SqlAlchemyDB(entropy_db_file)

qpu_db_file='qpu_db'
create_new_qpu_database(qpu_db_file, qpu_base_data, force_create=False)

qpudb_name='qpu_db'
lab = LabResources(entropy_db)
lab.register_resource(name="my_scope", resource_class=my_scope, kwargs={'name': 'scope1', 'ip':0})
lab.register_resource('qmm',QuantumMachinesManager)
lab.register_resource(qpudb_name, QpuDatabaseConnection, [qpudb_name, resolve])