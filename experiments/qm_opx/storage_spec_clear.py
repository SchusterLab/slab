from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator import TwoStateDiscriminator
from configuration_IQ import config, qubit_freq, rr_LO, qubit_LO, ge_IF, storage_IF, storage_freq, storage_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*

import time
# im = InstrumentManager()
# LO_q = im['RF5']
# LO_r = im['RF8']
# LO_s  = im['sccav']
# LO_sb = im['scsb']

# LO_s.set_frequency(storage_LO)
# LO_s.set_power(12.0)
simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)


qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz')
###############
# qubit_spec_prog:
###############
f_min = -50e3
f_max = 50e3
df = 5e3

f_vec = np.arange(f_min, f_max + df/2, df)

# LO_q.set_frequency(qubit_LO)
# LO_q.set_ext_pulse(mod=False)
# LO_q.set_power(18)
# LO_r.set_frequency(rr_LO)
# LO_r.set_ext_pulse(mod=False)
# LO_r.set_power(18)

avgs = 1000
reset_time = 5000000
simulation = 0
with program() as storage_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies

    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, storage_IF + f_min, f < storage_IF + f_max + df/2, f + df):

            update_frequency("storage", f)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            play("saturation"*amp(0.005), "storage", duration=50000)
            align("storage", "qubit")
            play("res_pi", "qubit")
            align("qubit", "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1 - Q2)
            assign(Q, Q1 + I2)

            save(I, I_st)
            save(Q, Q_st)
    with stream_processing():

        I_st.buffer(len(f_vec)).average().save('I')
        Q_st.buffer(len(f_vec)).average().save('Q')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(storage_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(storage_spec, duration_limit=0, data_limit=0)
    print("Experiment done")
    start_time = time.time()
    res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    # print("Data collection done")
    #
    # stop_time = time.time()
    # print(f"Time taken: {stop_time-start_time}")
    #
    # job.halt()
    #
    # plt.plot(amps, I, '.-')
    # plt.plot(amps, Q, '.-')

    # plt.figure()
    # while(result_handles.is_processing()):
    #     I = statistic_handle.fetch_all()
    #     plt.plot(np.array(I)**2)
    #     plt.pause(5)
    #     plt.clf()


