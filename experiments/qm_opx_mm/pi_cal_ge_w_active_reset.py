from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config,  disc_file_opt, pi_amp, half_pi_amp, disc_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
readout = 'readout' #'clear'
if readout=='readout':
    disc = disc_file
else:
    disc = disc_file_opt

reset_time = 500000
N = 1000

n_pi_pulses = 12
a_pi_min= 0.95*pi_amp
a_pi_max = 1.05*pi_amp

n_pi_2_pulses = 24
a_pi_2_min= 0.95*half_pi_amp
a_pi_2_max = 1.05*half_pi_amp

da_pi = (a_pi_max - a_pi_min)/100
a_pi_vec = np.arange(a_pi_min, a_pi_max + da_pi/2, da_pi)

da_pi_2 = (a_pi_2_max - a_pi_2_min)/100
a_pi_2_vec = np.arange(a_pi_2_min, a_pi_2_max + da_pi_2/2, da_pi_2)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc, lsb=True)

with program() as power_rabi:

    n = declare(int)
    i = declare(int)
    a = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    I_pi_st = declare_stream()
    I_pi_2_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_pi_min, a < a_pi_max + da_pi/2, a + da_pi):

            discriminator.measure_state(readout, "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            align('qubit_mode0', 'rr')
            wait(reset_time//50, 'qubit_mode0')
            play('pi2', 'qubit_mode0')
            with for_(i, 0, i < n_pi_pulses, i+1):
                play('gaussian'*amp(a), 'qubit_mode0')
            play('pi2', 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state(readout, "out1", "out2", res, I=I)

            save(res, I_pi_st)

        with for_(a, a_pi_2_min, a < a_pi_2_max + da_pi_2/2, a + da_pi_2):

            discriminator.measure_state(readout, "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            align('qubit_mode0', 'rr')
            wait(reset_time//50, 'qubit_mode0')
            play('pi2', 'qubit_mode0')
            with for_(i, 0, i < n_pi_2_pulses, i+1):
                play('gaussian'*amp(a), 'qubit_mode0')
            play('pi2', 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state(readout, "out1", "out2", res, I=I)

            save(res, I_pi_2_st)

    with stream_processing():
        I_pi_st.boolean_to_int().buffer(len(a_pi_vec)).average().save('I_pi')
        I_pi_2_st.boolean_to_int().buffer(len(a_pi_2_vec)).average().save('I_pi_2')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
I_pi = result_handles.get('I_pi').fetch_all()
I_pi_2 = result_handles.get('I_pi_2').fetch_all()

plt.figure()
plt.plot(a_pi_vec, I_pi, '.')
plt.show()
plt.figure()
plt.plot(a_pi_2_vec, I_pi_2, '.')
plt.show()

job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'power_rabi', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("I_pi", data=I_pi)
    f.create_dataset("I_pi_2", data=I_pi_2)
    f.create_dataset("amps_pi", data=a_pi_vec)
    f.create_dataset("amps_pi_2", data=a_pi_2_vec)