from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, biased_th_g, disc_file_opt, pi_amp_resolved, pi_len_resolved
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

reset_time = 500000
N = 2000

a_min= 0.95*pi_amp_resolved
a_max = 1.05*pi_amp_resolved

da = (a_max - a_min)/100
a_vec = np.arange(a_min, a_max + da/2, da)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
n_pi_pulses = 12

with program() as power_rabi:

    n = declare(int)
    i = declare(int)
    a = declare(fixed)
    res = declare(bool)
    res_reset = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            wait(reset_time//10, 'qubit_mode0')
            align('qubit_mode0', 'rr')
            play('pi2', 'qubit_mode0')
            with for_(i, 0, i < n_pi_pulses, i+1):
                play('gaussian'*amp(a), 'qubit_mode0', duration=pi_len_resolved//4)
            play('pi2', 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(a_vec)).average().save('res')
        I_st.buffer(len(a_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

plt.figure()
plt.plot(a_vec, I, '.')
plt.show()
plt.figure()
plt.plot(a_vec, res, '.')
plt.show()

print('Suggest amp: %f'%a_vec[np.argmin(res)])

job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'weak_rabi', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Q", data=res)
    f.create_dataset("I", data=I)
    f.create_dataset("amps", data=a_vec)
    f.create_dataset("pi", data=pi)