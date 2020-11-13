from configuration_IQ import config
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
im = InstrumentManager()
LO = im['RF8']

nu_r = 8.0517
dnus = np.linspace(-3, 3, 300)*(1e-3)

freqs =(nu_r + dnus)*1e9
avgs = 100
LO.set_frequency(freqs[0])
LO.set_power(18)
LO.set_ext_pulse(mod=True)

with program() as resonator_spectroscopy:

    n = declare(int)
    i = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    update_frequency("rr", 0)
    with for_(n, 0, n < len(freqs), n + 1):

        with for_(i, 0, i < avgs, i + 1):
            wait(10000//4, "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)
        pause()

    with stream_processing():
        I_st.buffer(avgs).map(FUNCTIONS.average()).save_all("I")
        Q_st.buffer(avgs).map(FUNCTIONS.average()).save_all("Q")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

for freq in (freqs):
    # change LO to freqs[i]
    LO.set_frequency(freq)
    # print(LO.get_frequency())
    time.sleep(0.1)
    job.resume()

print("Experiment over")

res_handles = job.result_handles
res_handles.wait_for_all_values()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()

print("Data collection done")

fig, axs  = plt.subplots(1, 2, figsize = (10, 5))
axs[0].plot(freqs/1e9, I)
axs[0].plot(freqs/1e9, Q)
axs[0].set_xlabel('Freq (GHz)')
axs[0].set_ylabel('I/Q')
amps = np.sqrt(np.array(I)['value']**2 + 1*np.array(Q)['value']**2)
ph = np.arctan2(np.array(Q)['value'], np.array(I)['value'])
axs[1].plot(freqs/1e9, amps,'b-')
p = fitlor(freqs/1e9, amps,showfit=False)
x = np.array(freqs)/1e9
axs[1].plot(freqs/1e9, lorfunc(p, freqs/1e9), label=r'$\nu_{r}$ = %.4f GHz'% x[np.argmax(amps)])
print ("fits = ", p)
ax2  = axs[1].twinx()
ax2.plot(freqs/1e9, ph, 'r-')
axs[1].set_xlabel('Freq (GHz)')
axs[1].set_ylabel('amp')
ax2.set_ylabel('$\\varphi$')
axs[1].legend(loc='best')
fig.show()