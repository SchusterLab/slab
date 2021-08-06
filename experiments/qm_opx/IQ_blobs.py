from configuration_IQ import config, qubit_freq, rr_freq, qubit_LO, rr_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator

im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
##################
# power_rabi_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

avgs = 3000
reset_time = 500000
simulation = 0
qmm = QuantumMachinesManager()

discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)


with program() as IQ_blobs:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    a = declare(fixed)      # Amplitudes
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    res = declare(bool)
    I=declare(fixed)
    res_st = declare_stream()
    I_st = declare_stream()


    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        wait(reset_time//4, "qubit")
        # play("saturation"*amp(0.01), "qubit", duration=125000)
        align('qubit', 'rr')
        # measure("clear", "rr", None,
        #         demod.full("clear_integW1", I1, 'out1'),
        #         demod.full("clear_integW2", Q1, 'out1'),
        #         demod.full("clear_integW1", I2, 'out2'),
        #         demod.full("clear_integW2", Q2, 'out2'))
        # assign(I, I1-Q2)
        # assign(Q, I2+Q1)
        # save(I, I_st)
        # save(Q, Q_st)
        discriminator.measure_state("clear", "out1", "out2", res, I=I)
        save(res, res_st)
        save(I, I_st)

    with stream_processing():
        # I_st.save_all('I')
        # Q_st.save_all('Q')
        res_st.boolean_to_int().save_all('res')
        I_st.save_all('I')

qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(IQ_blobs, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(IQ_blobs, duration_limit=0, data_limit=0)
    print("Waiting for the data")
    start_time = time.time()

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("res")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    # print("Data collection done")
    #
    # stop_time = time.time()
    # print(f"Time taken: {stop_time-start_time}")
    #
    plt.plot(I,'.')
    plt.axis('equal')