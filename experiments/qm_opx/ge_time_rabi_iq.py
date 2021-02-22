from configuration_IQ import config, qubit_freq, rr_freq, qubit_LO, rr_LO, pi_amp
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
from slab.dsfit import*

def roundint(value, base=4):
    return int(value) - int(value) % int(base)


##################
# time_rabi_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(13)

t_min = 4
t_max = 40
dt = 1
times = np.arange(t_min, t_max + dt/2, dt)
avgs = 1000
reset_time = 500000
simulation = 0
with program() as ge_rabi:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # times
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

        with for_(t, t_min, t <= t_max, t + dt):

            wait(reset_time//4, "qubit")
            play("gaussian_16"*amp(pi_amp) , "qubit", duration=t)
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(times)).average().save('I')
        Q_st.buffer(len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(ge_rabi, SimulationConfig(5000))
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

    times = 4*times
    z = 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(times[z:len(I)], I[z:],'bo')
    p = fitdecaysin(times[z:len(I)], I[z:], showfit=False)
    axs[0].plot(times[z:len(I)], decaysin(np.append(p,0), times[z:len(I)]), 'b-')
    axs[0].set_xlabel('Time (ns)')
    axs[0].set_ylabel('I')

    z = 2
    axs[1].plot(times[z:len(I)], Q[z:],'ro')
    p = fitdecaysin(times[z:len(I)], Q[z:], showfit=False)
    axs[1].plot(times[z:len(I)], decaysin(np.append(p,0), times[z:len(I)]), 'r-')
    axs[1].set_xlabel('Time (ns)')
    axs[1].set_ylabel('Q')
    t_pi = 1/(2*p[1])
    t_half_pi = 1/(4*p[1])
    axs[1].axvline(t_pi, color='k', linestyle='dashed')
    axs[1].axvline(t_half_pi, color='k', linestyle='dashed')
    plt.tight_layout()
    fig.show()

    print("Half pi length =", t_half_pi, "ns")
    print("pi length =", t_pi, "ns")
    print ("Rabi decay time = ", p[3], "ns")
    print("suggested_pi_length = ", roundint(t_pi, 4), "suggested_pi_amp = ", pi_amp*(t_pi)/float(roundint(t_pi, 4)))
    print("suggested_half_pi_length = ", roundint(t_half_pi, 4), "suggested_piby2_amp = ", pi_amp*(t_half_pi)/float(roundint(t_half_pi, 4)))