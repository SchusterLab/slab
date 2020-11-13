from configuration_IQ import config
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

##################
# T1:
##################
qubit_freq = 4.748488058227563e9
ge_IF = 100e6
qubit_LO = qubit_freq - ge_IF
rr_IF = 100e6
rr_LO = 8.0518e9 - rr_IF

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod = False)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod = True)
LO_r.set_power(18)

dt = 1000
T_max = 100000
T_min = 1000
times = np.arange(T_min, T_max + dt/2, dt)

avgs = 1000
reset_time = 500000
simulation = 0 # 1 to simulate the pulses

with program() as ge_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
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
    update_frequency("qubit", ge_IF)
    update_frequency("rr", rr_IF)

    with for_(n, 0, n < avgs, n + 1):
        with for_(t, T_min, t < T_max + dt/2, t + dt):
            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            wait(t, "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1+Q2)
            assign(Q, I2-Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(times)).average().save('I')
        Q_st.buffer(len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(ge_t1, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(ge_t1, duration_limit=0, data_limit=0)
    print ("Execution done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print ("Data collection done")
    #
    times = 4*times #actual clock time

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(times[:len(I)]/1e3, I,'bo')
    p = fitexp(times[:len(I)], I, showfit=False)
    print ("fits :", p)
    print("T1: %.3f"%(p[3]/1e3))
    axs[0].plot(times[:len(I)]/1e3, expfunc(p, times[:len(I)]), 'b-',
                label='T1: %.3f $\mu$s'%(p[3]/1e3))
    axs[0].set_xlabel('Time ($\mu$s)')
    axs[0].set_ylabel('I')
    axs[0].legend()

    axs[1].plot(times[:len(I)]/1e3, Q,'ro')
    p = fitexp(times[:len(I)], Q, showfit=False)
    axs[1].plot(times[:len(I)]/1e3, expfunc(p, times[:len(I)]), 'r-')
    print ("fits :", p)
    print("T1: %.3f"%(p[3]/1e3))
    axs[1].set_xlabel('Time ($\mu$s)')
    axs[1].set_ylabel('Q')
    plt.tight_layout()
    fig.show()