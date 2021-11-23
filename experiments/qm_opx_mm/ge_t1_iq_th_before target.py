from configuration_IQ import config, qubit_freq, rr_freq, qubit_LO, rr_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
from h5py import File
import os
from slab.dsfit import*
from slab.dataanalysis import get_next_filename

LO_q = im['RF5']
LO_r = im['RF8']

##################
# t1:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

dt = 1000
T_max = 100000
T_min = 25
times = np.arange(T_min, T_max + dt/2, dt)

avgs = 1000
reset_time = 500000
simulation = 0 #1 to simulate the pulses

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

biased_th_g = 0.0012
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)

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


with program() as ge_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    Q = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, T_min, t < T_max + dt/2, t + dt):
            # wait(reset_time//4, "qubit")
            active_reset(biased_th_g)
            align("qubit", 'rr')
            play("pi", "qubit")
            wait(t, "qubit")
            align("qubit", "rr")
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        I_st.buffer(len(times)).average().save('I')

qmm = QuantumMachinesManager()


i, q = power_rabi(expt_cfg, expt_name ='rabi')



if simulation:
    qm = qmm.open_qm(config)
    job = qm.simulate(ge_t1, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:

    while t1<50e6:
        qm = qmm.open_qm(config)
        job = qm.execute(ge_t1, duration_limit=0, data_limit=0)
        print ("Execution done")

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        res = result_handles.get('res').fetch_all()
        I = result_handles.get('I').fetch_all()
        job.halt()
        print ("Data collection done")

        x = 4*times/1e3
        y = res

        p = fitexp(x, y, showfit=False)
        print("T1: %.3f μs"%(p[3]))
        t1 = p[3]

    i, q, config = power_rabi()




    # send me e-mail



        # path = os.getcwd()
        # data_path = os.path.join(path, "data/")
        # seq_data_file = os.path.join(data_path,
        #                              get_next_filename(data_path, 't1', suffix='.h5'))
        # print(seq_data_file)
        #
        # times = 4*times #actual clock time
        # with File(seq_data_file, 'w') as f:
        #     f.create_dataset("I", data=I)
        #     f.create_dataset("res", data=res)
        #     f.create_dataset("time", data=times)