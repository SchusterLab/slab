from configuration_IQ import config, qubit_LO, rr_LO, rr_IF, rr_freq
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
# from numpy import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

##################
# histogram_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=True)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

atten.set_attenuator(13.0)
time.sleep(1)

reset_time = 500000
avgs = 2000
simulation = 0

f_min = -450e3
f_max = 50e3
df = 20e3
f_vec = np.arange(f_min, f_max + df/2, df)

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    f = declare(int)
    i= declare(int)
    
    n_st = declare_stream()

    Ig = declare(fixed)
    Qg = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    Ie = declare(fixed)
    Qe = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(n, 0, n < avgs, n + 1):
        save(n, n_st)
        with for_(f, rr_IF + f_min, f < rr_IF + f_max + df/2, f + df):

            update_frequency("rr", f)

            """Just readout without playing anything"""
            wait(reset_time//4, "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),
                    demod.full("long_integW2", Q1, 'out1'),
                    demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

            assign(Ig, I1 + Q2)
            assign(Qg, I2 - Q1)
            save(Ig, Ig_st)
            save(Qg, Qg_st)

            align("qubit", "rr")

            """Play a ge pi pulse and then readout"""
            wait(reset_time // 4, "qubit")
            play("pi", "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),
                    demod.full("long_integW2", Q1, 'out1'),
                    demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

            assign(Ie, I1 + Q2)
            assign(Qe, I2 - Q1)
            save(Ie, Ie_st)
            save(Qe, Qe_st)

    with stream_processing():
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

        n_st.save('n')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(histogram, duration_limit=0, data_limit=0)
print("Waiting for the data")

job.result_handles.wait_for_all_values()
Ig = job.result_handles.Ig.fetch_all()['value']
Ie = job.result_handles.Ie.fetch_all()['value']
Qg = job.result_handles.Qg.fetch_all()['value']
Qe = job.result_handles.Qe.fetch_all()['value']
print("Data fetched")

fid_f = []
for jj in range(len(f_vec)):
    ig = Ig[jj::len(f_vec)]
    qg = Qg[jj::len(f_vec)]
    ie = Ie[jj::len(f_vec)]
    qe = Qe[jj::len(f_vec)]
    p = [ig, qg, ie, qe]
    f = hist(p)[0]
    fid_f.append(f)

"""Plotting the fidelity data as a function of amp and freq"""
f_vec = (rr_freq + f_vec)/1e9

ind = np.argmax(fid_f) #index for maximum fidelity
#
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(f_vec, fid_f, 'bo')
ax.axvline(x=f_vec[ind], color='k', linestyle='--')
# ax.axhline(y=amp_vec[ind[0]], color='k', linestyle='--')
# ax.axvline(x=8.051744, color='r', linestyle='--')
# ax.axvline(x=8.051406, color='b', linestyle='--')
ax.set_title('F = %.2f at readout frequency = %.4f GHz'%(fid_f[ind], f_vec[ind]))
#
# print("#############################################################################################")
# print('Optimal fidelity of %f at readout power = %f (V) and readout frequency = %f GHz'%(fid_max, amp_vec[ind[0]],f_vec[ind[1]]))
# print("#############################################################################################")
ax.set_xlim(np.min(f_vec), np.max(f_vec))
ax.set_xlabel('Readout frequency (GHz)')
ax.set_ylabel('F ')
plt.show()
