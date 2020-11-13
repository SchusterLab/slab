from configuration_IQ import config
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
###############
# rr_spec_prog:
###############

f_min = 37e6
f_max = 43e6
df = 0.1e6
freq_num = int((f_max-f_min)//df) # need to be calc
avgs  =  100

qubit_LO = 4.7457e9

freqs = qubit_LO + np.linspace(f_min, f_max, freq_num)
LO.set_frequency(qubit_LO)

with program() as qubit_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, f_min, f < f_max, f + df):

            wait(500000 // 4, "qubit")
            update_frequency("qubit", f)
            play("saturation", "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():

        I_st.buffer(freq_num).average().save('I')
        Q_st.buffer(freq_num).average().save('Q')


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
# job = qm.simulate(qubit_spec, SimulationConfig(50000))
# samples = job.get_simulated_samples()
# samples.con1.plot()

job = qm.execute(qubit_spec, duration_limit=0, data_limit=0)


res_handles = job.result_handles
res_handles.wait_for_all_values()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()



fig,axs  = plt.subplots(1,2,figsize = (10,5))
axs[0].plot(freqs/1e9,I)
axs[0].plot(freqs/1e9,Q)
axs[0].set_xlabel('Freq (GHz)')
axs[0].set_ylabel('I/Q')
amps =  np.sqrt(np.array(I)**2 + np.array(Q)**2)
ph = np.arctan2(np.array(Q),np.array(I))
axs[1].plot(freqs/1e9,amps,'b-')
p = fitlor(freqs/1e9,amps,showfit = False)
axs[1].plot(freqs/1e9,lorfunc(p,freqs/1e9))
print ("fits = ",p)
ax2  = axs[1].twinx()
ax2.plot(freqs/1e9,ph,'r-')
axs[1].set_xlabel('Freq (GHz)')
axs[1].set_ylabel('amp')
ax2.set_ylabel('$\\varphi$')
fig.show()