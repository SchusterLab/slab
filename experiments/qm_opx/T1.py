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
# power_rabi_prog:
##################
avgs = 1000
qubit_freq = 4.744401713421724e9
iq_freq = 200e6
qubit_LO = qubit_freq + iq_freq #had to change to + sign as the power in the first peak was lower (phase got mixed up somewhere)
rr_LO = 8.0517e9
reset_time = 500000

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod = True)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod = True)
LO_r.set_power(16)

dt = 4000
T_max = 400000
T_min = 4000
times = T_min + np.arange(0, T_max, dt)
t_num = 100

with program() as t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############
    # with for_(j, 0, j < t_num, j+1):
    #     assign(t[j], dt*(j+1))

    with for_(n, 0, n < avgs, n + 1):

        with for_(i, 1, i < t_num, i+1):
            assign(t, i*1000)
            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            wait(t, "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(t_num).average().save('I')
        Q_st.buffer(t_num).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
# job = qm.simulate(t1, SimulationConfig(50000))
# samples = job.get_simulated_samples()
# samples.con1.plot()

job = qm.execute(t1, duration_limit=0, data_limit=0)
print ("Done")

res_handles = job.result_handles
res_handles.wait_for_all_values()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
print ("Done 2")
# #

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(times[:len(I)]/1e3,- I,'bo')
p = fitexp(times[:len(I)], -I,showfit=False)
print ("fits :",p)
print("T1: %.3f"%p[3])
axs[0].plot(times[:len(I)]/1e3,expfunc(p, times[:len(I)]),'b-', label='T1: %.3f $\mu$s'%(p[3]/1e3))
axs[0].set_xlabel('Time ($\mu$s)')
axs[0].set_ylabel('I')
axs[0].legend()

axs[1].plot(times[:len(I)]/1e3, Q,'ro')
p = fitexp(times[:len(I)], Q,showfit=False)
axs[1].plot(times[:len(I)]/1e3,expfunc(p,times[:len(I)]),'r-')
print ("fits :",p)
print("T1: %.3f"%p[3])
axs[1].set_xlabel('Time ($\mu$s)')
axs[1].set_ylabel('Q')
plt.tight_layout()
fig.show()