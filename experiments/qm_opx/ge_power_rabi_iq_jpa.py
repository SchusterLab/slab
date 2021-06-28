from configuration_IQ import config, qubit_freq, rr_freq, qubit_LO, rr_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.dsfit import*
##################
# power_rabi_prog:
##################
a_min = 0.0
a_max = 1.0
da = 0.01
amps = np.arange(a_min, a_max + da/2, da)
avgs = 2000
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

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            wait(reset_time//4, "qubit")
            play("gaussian"*amp(a), "qubit")
            align("qubit", "rr")
            align("rr", "jpa_pump")
            play('pump_square', 'jpa_pump')
            measure("long_readout", "rr", None,
                    demod.full("long_integW1", I1, 'out1'),
                    demod.full("long_integW2", Q1, 'out1'),
                    demod.full("long_integW1", I2, 'out2'),
                    demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1 - Q2)
            assign(Q, Q1 + I2)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(amps)).average().save('I')
        Q_st.buffer(len(amps)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

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
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done")

    stop_time = time.time()
    print(f"Time taken: {stop_time-start_time}")

    with program() as stop_playing:
        pass
    job = qm.execute(stop_playing, duration_limit=0, data_limit=0)

    z = 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(amps[z:len(I)], I[z:],'bo')
    p = fitdecaysin(amps[z:len(I)], I[z:], showfit=False)
    print("fits :", p)
    print("a_pi", 1/2/p[1])
    axs[0].axvline(1/2/p[1])
    axs[0].plot(amps[z:len(I)], decaysin(np.append(p,0), amps[z:len(I)]), 'b-')
    axs[0].set_xlabel('Amps')
    axs[0].set_ylabel('I')

    z = 2
    axs[1].plot(amps[z:len(I)], Q[z:],'ro')
    p = fitdecaysin(amps[z:len(I)], Q[z:], showfit=False)
    axs[1].plot(amps[z:len(I)], decaysin(np.append(p,0), amps[z:len(I)]), 'r-')
    print("fits :", p)
    print("a_pi", 1/2/p[1])
    axs[1].axvline(1/2/p[1])
    axs[1].set_xlabel('Amps')
    axs[1].set_ylabel('Q')
    plt.tight_layout()
    fig.show()