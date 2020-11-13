from configuration_IQ import config
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
# from numpy import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

from slab.dsfit import*

def doublegauss(bins, *p):
    a1, sigma1, mu1 = p[0], p[1], p[2]
    a2, sigma2, mu2 = p[3], p[4], p[5]

    y1 = a1 * ((1 / (np.sqrt(2 * np.pi) * sigma1)) *
               np.exp(-0.5 * (1 / sigma1 * (bins - mu1)) ** 2))
    y2 = a2 * ((1 / (np.sqrt(2 * np.pi) * sigma2)) *
               np.exp(-0.5 * (1 / sigma2 * (bins - mu2)) ** 2))
    y = y1 + y2

    return y
##################
# histogram_prog:
##################
qubit_freq = 4.748488058227563e9
ge_IF = 100e6
qubit_LO = qubit_freq + ge_IF
rr_IF = 100e6
rr_LO = 8.05181538e9 - rr_IF

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

reset_time = 500000
avgs = 1000
simulation = 0

f_min = -0.5e6
f_max = 0.5e6
df = 0.1e6
f_vec = np.arange(f_min, f_max + df/2, df)

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    f = declare(int)

    Ig = declare(fixed)
    Qg = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    Ie = declare(fixed)
    Qe = declare(fixed)
    # If = declare(fixed)
    # Qf = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, rr_IF + f_min, f < rr_IF + f_max + df/2, f + df):

            update_frequency("rr", f)

            """Just readout without playing anything"""
            wait(reset_time//4, "rr")

            measure("long_readout"*amp(1.30), "rr", None, demod.full("long_integW1", I1, 'out1'),
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
            measure("long_readout"*amp(1.30), "rr", None, demod.full("long_integW1", I1, 'out1'),
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

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(histogram, duration_limit=0, data_limit=0)

job.result_handles.wait_for_all_values()

Ig = job.result_handles.Ig.fetch_all()['value']
Ie = job.result_handles.Ie.fetch_all()['value']
Qg = job.result_handles.Qg.fetch_all()['value']
Qe = job.result_handles.Qe.fetch_all()['value']
fid_f = []
for ii in range(len(f_vec)):
    ig_opt = Ig[ii::len(f_vec)]
    ie_opt = Ie[ii::len(f_vec)]
    qg_opt = Qg[ii::len(f_vec)]
    qe_opt = Qe[ii::len(f_vec)]

    ran = 1
    numbins = 200

    xg, yg = np.median(ig_opt), np.median(qg_opt)
    xe, ye = np.median(ie_opt), np.median(qe_opt)

    xlims = [xg-ran/5, xg+ran/5]
    ylims = [yg-ran/5, yg+ran/5]

    theta = -np.arctan((ye-yg)/(xe-xg))
    print("Rotation angle:   %.3f"%theta)

    ig_new, qg_new = ig_opt*cos(theta)- qg_opt*sin(theta), ig_opt*sin(theta)+ qg_opt*cos(theta)
    ie_new, qe_new = ie_opt*cos(theta)- qe_opt*sin(theta), ie_opt*sin(theta)+ qe_opt*cos(theta)

    xg, yg = np.median(ig_new), np.median(qg_new)
    xe, ye = np.median(ie_new), np.median(qe_new)

    xlims = [xg-ran/5, xg+ran/5]
    ylims = [yg-ran/5, yg+ran/5]

    print("here")
    ng, binsg = np.histogram(ig_new, bins=numbins, range = xlims)
    ne, binse = np.histogram(ie_new, bins=numbins, range = xlims)
    fid = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / ng.sum())).max()
    fid_f.append(fid)
    ng, binsg = np.histogram(qg_new, bins=numbins,range =ylims)
    ne, binse = np.histogram(qe_new, bins=numbins,range =ylims)
    fid = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / ng.sum())).max()

f = (f_vec+rr_LO+rr_IF)/1e9
plt.figure()
plt.plot(f, fid_f, 'bo', label=r"$F_{max} = %.4f$"%np.max(fid_f))
plt.axvline(x=f[np.argmax(fid_f)], linestyle='--')
plt.xlabel('Readout Freq. (GHz)')
plt.ylabel('Fidelity')
plt.legend()
plt.show()

print("Maximum fidelity of %.4f at RR Freq %.4f"%(np.max(fid_f), f[np.argmax(fid_f)]))