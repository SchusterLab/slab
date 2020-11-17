from configuration_IQ import config, qubit_LO, rr_LO, rr_IF
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
atten = im["atten"]

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
# rr_freq = 8.0516e9
# rr_LO = rr_freq - rr_IF

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

reset_time = 500000
avgs = 2000
simulation = 0
with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging

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

    # If_st = declare_stream()
    # Qf_st = declare_stream()


    ###############
    # the sequence:
    ###############
    # update_frequency("qubit", ge_IF)
    # update_frequency("qubit_ef", ef_IF)
    # update_frequency("rr", rr_IF)

    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "rr")
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

        # align("qubit_ef", "rr")
        # """Play a ge pi pulse followed by a ef pi pulse and then readout"""
        # wait(reset_time//4, "qubit")
        # play("pi", "qubit")
        # align("qubit", "qubit_ef")
        # play("pi", "qubit_ef")
        # align("qubit_ef", "rr")
        # measure("long_readout", "rr", None, demod.full("long_integW1", If, 'out1'),demod.full("long_integW1", Qf, 'out2'))
        # save(If, If_st)
        # save(Qf, Qf_st)

    with stream_processing():
        Ig_st.buffer(avgs).save_all('Ig')
        Qg_st.buffer(avgs).save_all('Qg')

        Ie_st.buffer(avgs).save_all('Ie')
        Qe_st.buffer(avgs).save_all('Qe')

        # If_st.buffer(avgs).save_all('If')
        # Qf_st.buffer(avgs).save_all('Qf')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(histogram, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(histogram, duration_limit=0, data_limit=0)
    print("Done")

res_handles = job.result_handles
res_handles.wait_for_all_values()

Ig_handle = res_handles.get("Ig")
Qg_handle = res_handles.get("Qg")

Ie_handle = res_handles.get("Ie")
Qe_handle = res_handles.get("Qe")

# If_handle = res_handles.get("If")
# Qf_handle = res_handles.get("Qf")

I_g = np.array(Ig_handle.fetch_all())
Q_g = np.array(Qg_handle.fetch_all())

I_e = np.array(Ie_handle.fetch_all())
Q_e = np.array(Qe_handle.fetch_all())

# I_f = np.array(If_handle.fetch_all())
# Q_f = np.array(Qf_handle.fetch_all())

ig_opt, qg_opt = I_g[0][0], Q_g[0][0]
ie_opt, qe_opt = I_e[0][0], Q_e[0][0]

ran = 1
numbins = 200

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
fig.tight_layout()

ax = axs[0, 0]
ax.set_title('Unrotated')
ax.scatter(ig_opt, qg_opt, label='g', alpha=0.25)
ax.scatter(ie_opt, qe_opt, label='e', alpha=0.25)
xg, yg = np.median(ig_opt), np.median(qg_opt)
xe, ye = np.median(ie_opt), np.median(qe_opt)

xlims = [xg-ran/5, xg+ran/5]
ax.set_xlim(xlims[0], xlims[1])
ylims = [yg-ran/5, yg+ran/5]
ax.set_ylim(ylims[0], ylims[1])
ax.errorbar(xg, yg, xerr=np.std(ig_opt), yerr=np.std(qg_opt), fmt='o', capthick=4,
            markerfacecolor='#003f5c', markeredgecolor='k', ecolor='#003f5c')
ax.errorbar(xe, ye, xerr=np.std(ie_opt), yerr=np.std(qe_opt), fmt='o', capthick=4,
            markerfacecolor='#003f5c', markeredgecolor='k', ecolor='#003f5c')
ax.set_ylabel('Q(V)')
ax.legend(loc='best', fontsize=16)

theta = -np.arctan((ye-yg)/(xe-xg))
print("Rotation angle:   %.3f"%theta)

ax = axs[0, 1]
ax.set_title('Unrotated')
ax.scatter(xg, yg, label='g', color='b', alpha=0.25)
ax.scatter(xe, ye, label='e', color='r', alpha=0.25)
ax.errorbar(xg, yg, xerr=np.std(ig_opt), yerr=np.std(qg_opt), fmt='o', capthick=4,
            markerfacecolor='b', markeredgecolor='k', ecolor='#003f5c', alpha=0.25)
ax.errorbar(xe, ye, xerr=np.std(ie_opt), yerr=np.std(qe_opt), fmt='o', capthick=4,
            markerfacecolor='r', markeredgecolor='k', ecolor='#003f5c', alpha=0.25)
ax.legend(loc='best')
ax.set_ylim(ylims[0], ylims[1])
ax.set_xlim(xlims[0], xlims[1])

ax = axs[1, 0]
ax.set_title('Rotated')
ig_new, qg_new = ig_opt*cos(theta)- qg_opt*sin(theta), ig_opt*sin(theta)+ qg_opt*cos(theta)
ie_new, qe_new = ie_opt*cos(theta)- qe_opt*sin(theta), ie_opt*sin(theta)+ qe_opt*cos(theta)

ax.scatter(ig_new, qg_new, label='g', alpha=0.25)
ax.scatter(ie_new, qe_new, label='e', alpha=0.25)

xg, yg = np.median(ig_new), np.median(qg_new)
xe, ye = np.median(ie_new), np.median(qe_new)

ax.errorbar(xg, yg, xerr=np.std(ig_new), yerr=np.std(qg_new), fmt='o', capthick=4,
            markerfacecolor='#003f5c', markeredgecolor='k', ecolor='#003f5c')
ax.errorbar(xe, ye, xerr=np.std(ie_new), yerr=np.std(qe_new), fmt='o', capthick=4,
            markerfacecolor='#003f5c', markeredgecolor='k', ecolor='#003f5c')
ax.set_ylim(ylims[0], ylims[1])

ax.set_xlabel('I(V)')
ax.set_ylabel('Q(V)')
ax.legend(loc='best', fontsize=16)
xlims = [xg-ran/5, xg+ran/5]
ax.set_xlim(xlims[0], xlims[1])
ylims = [yg-ran/5, yg+ran/5]
ax.set_ylim(ylims[0], ylims[1])

ax = axs[1, 1]
ax.set_title('Rotated')
ax.scatter(xg, yg, label='g', color='b', alpha=0.25)
ax.scatter(xe, ye, label='e', color='r', alpha=0.25)
ax.errorbar(xg, yg, xerr=np.std(ig_opt), yerr=np.std(qg_opt), fmt='o', capthick=4,
            markerfacecolor='b', markeredgecolor='k', ecolor='#003f5c', alpha=0.25)
ax.errorbar(xe, ye, xerr=np.std(ie_opt), yerr=np.std(qe_opt), fmt='o', capthick=4,
            markerfacecolor='r', markeredgecolor='k', ecolor='#003f5c', alpha=0.25)
ax.set_xlabel('I(V)')
ax.legend(loc='best')
ax.set_xlim(xlims[0], xlims[1])
ax.set_ylim(ylims[0], ylims[1])
fig.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.tight_layout()
ax = axs[0, 0]
ax.set_title('I')
xlims = [xg-ran/5, xg+ran/5]
ng, binsg, p = ax.hist(ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
# popt, pcov = curve_fit(doublegauss, xdata=binsg[:-1], ydata=ng, p0=[1, 0.01, 0.02, 1, 0.01, 0.0])
# ax.plot(binsg, doublegauss(binsg, *popt), 'k--', linewidth=2 )
# ax.text(xg, 50, "$\mu_{g}$ = %.4f \n $\sigma_{g}$ = %.5f"%(popt[1], popt[2]))
ax.set_ylabel('# of counts')

ne, binse, p = ax.hist(ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
# popt, pcov = curve_fit(doublegauss, xdata=binse[:-1], ydata=ne, p0=[1, 0.01, -0.02, 1, 0.01, -0.01])
# ax.plot(binse, doublegauss(binse, *popt), 'k--', linewidth=2 )
# ax.text(xe, 150, "$\mu_{e}$ = %.4f \n $\sigma_{e}$ = %.5f"%(popt[4], popt[5]))
fid = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / ng.sum())).max()
ax.legend(loc='best')
ax = axs[1, 0]
ax.plot(binse[:-1], np.cumsum(ng)/ng.sum(), color='r')
ax.plot(binse[:-1], np.cumsum(ne)/ng.sum(), color='b')
ax.plot(binse[:-1], np.abs(np.cumsum(ne)-np.cumsum(ng))/ng.sum(), color='k')
ax.set_xlabel('I(V)')
ax.set_ylabel('F')
print("##################################################################################")
print ("Single shot readout fidility from channel ", "I", " after rotation = ", fid)
print ("Optimal angle =", theta)
print("##################################################################################")

ax = axs[0, 1]
ax.set_title('Q')
ylims  = [yg-ran/5, yg+ran/5]
ng, binsg, p = ax.hist(qg_new, bins=numbins,range =ylims, color='b', label='g', alpha=0.5)
ne, binse, p = ax.hist(qe_new, bins=numbins,range =ylims, color='r', label='e', alpha=0.5)
fid = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / ng.sum())).max()
ax.legend(loc='best')
ax.set_xlabel('Q(V)')

print ("Single shot readout fidility from channel ","Q"," after rotation = ",fid)
ax = axs[1, 1]
ax.plot(binse[:-1],np.cumsum(ng)/ng.sum(),color='r')
ax.plot(binse[:-1],np.cumsum(ne)/ng.sum(),color='b')
ax.plot(binse[:-1],np.abs(np.cumsum(ne)-np.cumsum(ng))/ng.sum(),color='k')
ax.set_xlabel('Q(V)')
ax.set_ylabel('F')

plt.show()