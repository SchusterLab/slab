from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, biased_th_g_jpa
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

wait_time = 500000
N = 2000

a_min= 0
a_max = 1.0
da = 0.01
a_vec = np.arange(a_min, a_max + da/2, da)

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(5000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as power_rabi:

    n = declare(int)
    a = declare(fixed)
    res = declare(bool)
    res_reset = declare(bool)
    I = declare(fixed)
    counter1 = declare(int)
    counter2 = declare(int)

    res_st = declare_stream()
    I_st = declare_stream()
    counter1_st = declare_stream()
    counter2_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_min, a< a_max + da/2, a + da):
            active_reset(biased_th_g_jpa)
            align('qubit', 'rr')
            play('gaussian'*amp(a), 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)
            save(counter1, counter1_st)
            save(counter2, counter2_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(a_vec)).average().save('res')
        I_st.buffer(len(a_vec)).average().save('I')
        counter1_st.save_all('counter1')
        counter2_st.save_all('counter2')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()
counter1 = result_handles.get('counter1').fetch_all()
counter2 = result_handles.get('counter2').fetch_all()
# plt.plot(I, '.')

plt.figure()
plt.plot(res, '.')

job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'power_rabi', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Q", data=res)
    f.create_dataset("I", data=I)
    f.create_dataset("amps", data=a_vec)