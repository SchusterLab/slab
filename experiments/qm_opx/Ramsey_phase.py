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
# ramsey_prog:
##################
avgs = 2000
qubit_freq = 4.748488058822229e9
iq_freq = 200e6
qubit_LO = qubit_freq + iq_freq #had to change to + sign as the power in the first peak was lower (phase got mixed up somewhere)
rr_LO = 8.0517e9
reset_time = 500000

ramsey_freq = 4e3
detune_freq = iq_freq - ramsey_freq

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod = True)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod = True)
LO_r.set_power(16)

dt = 25
T_max = 2500
T_min = 25
times = np.arange(T_min, T_max + T_min, dt)
t_num = int((T_max)//dt) # need to be calc

with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int, size=t_num) #array of time delays
    omega = declare(fixed)
    phi = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    assign(omega, -2*pi*ramsey_freq*4)
    ###############
    # the sequence:
    ###############
    with for_(i, 0, i<t_num, i+1):
        assign(t[i], dt*(i+1))

    with for_(n, 0, n < avgs, n + 1):

        with for_(i, 0, i < t_num, i+1):
            wait(reset_time//4, "qubit")
            update_frequency("qubit", detune_freq)
            play("pi2", "qubit")
            wait(t[i], "qubit")
            assign(phi, omega*t[i])
            z_rotation(phi, "qubit") #shift the phase of the qubit oscillator
            play("pi2", "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(t_num).average().save('I')
        Q_st.buffer(t_num).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

# """To simulate the pulse sequence"""
# job = qm.simulate(ramsey, SimulationConfig(150000))
# samples = job.get_simulated_samples()
# samples.con1.plot()

"""To run the actual experiment"""
job = qm.execute(ramsey, duration_limit=0, data_limit=0)
print("Done")

res_handles = job.result_handles
res_handles.wait_for_all_values()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
print("Done 2")

times = 4*times/1e3
z = 2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(times[z:len(I)], I[z:],'bo')
p = fitdecaysin(times[z:len(I)], I[z:],showfit=False)
print ("fits :",p)
print ("a_pi",1/2/p[1])
axs[0].axvline(1/2/p[1])
axs[0].plot(times[z:len(I)],decaysin(np.append(p,0),times[z:len(I)]),'b-')
axs[0].set_xlabel('Time ($\mu$s)')
axs[0].set_ylabel('I')

z = 1
axs[1].plot(times[z:len(I)], Q[z:],'ro')
p = fitdecaysin(times[z:len(I)], Q[z:],showfit=False)
axs[1].plot(times[z:len(I)],decaysin(np.append(p,0),times[z:len(I)]),'r-')
print ("fits :",p)
print ("a_pi",1/2/p[1])
axs[1].axvline(1/2/p[1])
axs[1].set_xlabel('Time ($\mu$s)')
axs[1].set_ylabel('Q')
plt.tight_layout()
fig.show()

offset = ramsey_freq/1e9 - p[1]
nu_q_new = qubit_freq/1e9 + offset/1e9

print("Original qubit frequency choice =", qubit_freq/1e9, "GHz")
print("Offset freq =", offset * 1e3, "kHz")
print("Suggested qubit frequency choice =", nu_q_new, "GHz")
print("T2* =", p[3], "ns")