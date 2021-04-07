from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, biased_th_g
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
import json

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)

with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)

expt_name = 'power_rabi'
expt_cfg = experiment_cfg[expt_name]

opx_config = config

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]

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

def time_rabi(expt_cfg, opx_config):

    a = expt_cfg['amp']
    avgs = expt_cfg['avgs']
    t_min = expt_cfg['t_min']
    t_max = expt_cfg['t_max']
    dt = expt_cfg['dt']
    pulse = expt_cfg['pulse_type']

    if pulse =='square':
        amp_vec  = a*[np.ones(len(t_vec))]

    t_vec = np.arange(t_min, t_max + dt/2, dt)

    n = declare(int)
    t = declare(int)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    with program() as exp:

        with for_(n, 0, n < avgs, n + 1):

            with for_(t, t_min, t < t_max + dt/2, t + dt):

                active_reset(biased_th_g)
                # wait(reset_time//4, 'qubit')
                align('qubit', 'rr')
                play('CW'*amp(a), 'qubit', duration=t)
                align('qubit', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)

        with stream_processing():
            res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
            I_st.buffer(len(t_vec)).average().save('I')

    qm = qmm.open_qm(opx_config)
    job = qm.execute(exp, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    return I, res, opx_config



with program() as time_rabi:

    n = declare(int)
    t = declare(int)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(t, t_min, t < t_max + dt/2, t + dt):

            active_reset(biased_th_g)
            # wait(reset_time//4, 'qubit')
            align('qubit', 'rr')
            play('CW'*amp(pi_amp), 'qubit', duration=t)
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
        I_st.buffer(len(t_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(time_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

plt.plot(4*t_vec, I, '.')
plt.figure()
plt.plot(4*t_vec, res, '.')

# path = os.getcwd()
# data_path = os.path.join(path, "data/")
# seq_data_file = os.path.join(data_path,
#                              get_next_filename(data_path, 'time_rabi', suffix='.h5'))
# print(seq_data_file)
#
# t_vec = 4*t_vec
#
# with File(seq_data_file, 'w') as f:
#     f.create_dataset("Q", data=res)
#     f.create_dataset("I", data=I)
#     f.create_dataset("times", data=t_vec)