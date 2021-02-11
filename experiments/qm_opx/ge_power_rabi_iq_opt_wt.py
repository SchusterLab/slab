from configuration_IQ import config, qubit_freq, rr_freq, qubit_LO, rr_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from state_disc.TwoStateDiscriminator import TwoStateDiscriminator

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

path = './state_disc/qubit_ge_disc_params.npz'
tsd = TwoStateDiscriminator(qmm, config, rr_qe='rr', path= 'qubit_ge_disc_params.npz')

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
LO_r.set_power(13)

a_min = 0.0
a_max = 1.0
da = 0.01
amps = np.arange(a_min, a_max + da/2, da)
avgs = 1000
reset_time = 500000
simulation = 0

with program() as ge_rabi:

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
    res_stream = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            wait(reset_time//4, "qubit")
            play("gaussian"*amp(a), "qubit")
            align("qubit", "rr")
            tsd.measure_state("long_readout", 'out1', 'out2', res)
            save(res, res_stream)


    with stream_processing():
        res_stream.buffer(len(amps)).save_all('res')

if simulation:
    job = qm.simulate(ge_rabi, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(ge_rabi, duration_limit=0, data_limit=0)
    print("Waiting for the data")
    start_time = time.time()

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    print("Data collection done")

    stop_time = time.time()
    print(f"Time taken: {stop_time-start_time}")

    data = pd.DataFrame(res_handles.get('res').fetch_all()['value'])




    # z = 2
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].plot(amps[z:len(I)], I[z:],'bo')
    # p = fitdecaysin(amps[z:len(I)], I[z:], showfit=False)
    # print("fits :", p)
    # print("a_pi", 1/2/p[1])
    # axs[0].axvline(1/2/p[1])
    # axs[0].plot(amps[z:len(I)], decaysin(np.append(p,0), amps[z:len(I)]), 'b-')
    # axs[0].set_xlabel('Amps')
    # axs[0].set_ylabel('I')
    #
    # z = 2
    # axs[1].plot(amps[z:len(I)], Q[z:],'ro')
    # p = fitdecaysin(amps[z:len(I)], Q[z:], showfit=False)
    # axs[1].plot(amps[z:len(I)], decaysin(np.append(p,0), amps[z:len(I)]), 'r-')
    # print("fits :", p)
    # print("a_pi", 1/2/p[1])
    # axs[1].axvline(1/2/p[1])
    # axs[1].set_xlabel('Amps')
    # axs[1].set_ylabel('Q')
    # plt.tight_layout()
    # fig.show()