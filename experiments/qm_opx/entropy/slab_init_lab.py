import os
import qm
from entropylab import *
from entropylab_qpudb import create_new_qpu_database, QpuDatabaseConnection
import numpy as np
import matplotlib.pyplot as plt
from slab_qpu_db import qpu_base_data
# from entropylab_qpudb import create_new_qpu_database, QpuDatabaseConnection
from qm.QuantumMachinesManager import QuantumMachinesManager

db_name='C:\\_Lib\\python\\slab\\experiments\\qm_opx\\entropy\\slab_entropy.db'
if os.path.exists(db_name):
    os.remove(db_name)
slab_entropy_db = SqlAlchemyDB(db_name)

qpu_db_file = 'C:\\_Lib\\python\\slab\\experiments\\qm_opx\\entropy\\slab_qpu.db'

create_new_qpu_database(qpu_db_file, qpu_base_data, force_create=True)

qpudb_name='qpu_db'
lab = LabResources(slab_entropy_db)
lab.register_resource('qmm', QuantumMachinesManager)
lab.register_resource_if_not_exist(qpudb_name, QpuDatabaseConnection, [qpu_db_file])

