from configuration_IQ import config, biased_th_g, disc_file_opt, pi_len_resolved
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

reset_time = int(7.5e6)
avgs = 1000

a_min= 0.00
a_max = 1.0
da = 0.01
a_vec = np.arange(a_min, a_max + da/2, da)

# a_min= 0
# a_max = 0.03
#
# da = (a_max - a_min)/100
# a_vec = np.arange(a_min, a_max + da/2, da)

l_min = 5
l_max = 25
dl = 1

l_vec = np.arange(l_min, l_max + dl/2, dl)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

with program() as power_rabi:

    n = declare(int)
    a = declare(fixed)

    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    l = declare(int)

    Q_st = declare_stream()
    I_st = declare_stream()

    with for_(n, 0, n < avgs, n + 1):

        with for_(l, l_min, l < l_max + dl/2, l+dl):

            with for_(a, a_min, a< a_max + da/2, a + da):

                wait(reset_time//4, "storage_mode1")
                play("CW"*amp(1.0), "storage_mode1", duration=l)
                align("qubit_mode0", "storage_mode1")
                play('gaussian'*amp(a), 'qubit_mode0')
                align('qubit_mode0', 'rr')
                measure("clear", "rr", None,
                        demod.full("clear_integW1", I1, 'out1'),
                        demod.full("clear_integW2", Q1, 'out1'),
                        demod.full("clear_integW1", I2, 'out2'),
                        demod.full("clear_integW2", Q2, 'out2'))

                assign(I, I1-Q2)
                assign(Q, I2+Q1)

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        Q_st.buffer(len(l_vec), len(a_vec)).average().save('Q')
        I_st.buffer(len(l_vec), len(a_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()

res = result_handles.get('Q').fetch_all()
I = result_handles.get('I').fetch_all()

plt.figure()
plt.pcolormesh(a_vec, 4*l_vec, res, cmap='RdBu', shading='auto')
plt.colorbar()
plt.xlabel('Amplitude')
plt.ylabel('Storage coherent len (ns)')
plt.show()

job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'power_rabi_alpha', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Q", data=res)
    f.create_dataset("I", data=I)
    f.create_dataset("amps", data=a_vec)
    f.create_dataset("len", data=4*l_vec)
