import qctrlqua
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, storage_cal_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import scipy
import os
from slab.dataanalysis import get_next_filename

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
qm = qmm.open_qm(config)
i_wf = [0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]
q_wf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]

qctrlqua.add_pulse_to_config('cw', 'qubit_mode0', i_wf, q_wf, config)
