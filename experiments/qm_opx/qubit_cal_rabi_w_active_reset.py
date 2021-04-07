from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

wait_time = 500000
N = 1000

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

biased_th_g = 0.0014
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)


a_min = 0.0001
a_max = 0.001
da = 0.00005
a_vec = np.arange(a_min, a_max + da/2, da)

t_min = 0
t_max = 32000
dt = 320

t_vec = np.arange(t_min, t_max + dt/2, dt)
dshift = 0

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    wait(5000//4, 'rr')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as power_rabi:

    n = declare(int)
    a = declare(fixed)
    t = declare(int)

    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()
    shift = declare(int, value=0)

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_min, a< a_max + da/2, a + da):

            assign(shift, shift+dshift)

            with for_(t, t_min + shift, t < t_max + dt/2 + shift, t + dt):
                active_reset(biased_th_g)
                align('qubit', 'rr')
                play('CW'*amp(a), 'qubit', duration=t)
                align('qubit', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)


    with stream_processing():
        res_st.boolean_to_int().buffer(len(a_vec), len(t_vec)).average().save('res')
        I_st.buffer(len(a_vec), len(t_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles

res_handle = result_handles.get('res')
res_handle.wait_for_values(1)

t_vec = 4*t_vec/1e3

while(res_handle.is_processing()):
    res = res_handle.fetch_all()
    plt.pcolor(res, cmap="RdBu")
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'AWG amp. (a.u.)')
    plt.pause(5)

# result_handles.wait_for_all_values()
# res = result_handles.get('res').fetch_all()
# I = result_handles.get('I').fetch_all()
# data_path = "S:\_Data\\210326 - QM_OPX\\data"
# seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qubit_square_cal', suffix='.h5'))
#
#
# print(seq_data_file)
# with File(seq_data_file, 'w') as f:
#     f.create_dataset("Q", data=res)
#     f.create_dataset("I", data=I)
#     f.create_dataset("amps", data=a_vec)
#     f.create_dataset("times", data=t_vec)