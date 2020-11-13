from configuration_IQ import config
from state_disc.TwoStateDiscriminator import TwoStateDiscriminator
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

###############
# qubit_spec_prog:
###############
qubit_freq = 4.748488058227563e9
ge_IF = 100e6
qubit_LO = qubit_freq + ge_IF
rr_IF = 0e6
rr_LO = 8.05181538e9 - rr_IF
reset_time = 500000
simulate = 0

f_min = -20e6
f_max = 20e6
df = 1e6
avgs = 500

freqs = np.arange(f_min, f_max + df/2, df)
freqs = freqs + qubit_freq

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

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
    update_frequency("rr", rr_IF)
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ge_IF + f_min, f < ge_IF + f_max + df/2, f + df):

            wait(reset_time// 4, "qubit") # wait for the qubit to relax, several T1s
            update_frequency("qubit", f)
            play("saturation"*amp(0.03), "qubit", duration=125000)
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),
                    demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():

        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulate:
    job = qm.simulate(qubit_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(qubit_spec, duration_limit=0, data_limit=0)
    print ("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print ("Done 2")

    fig,axs  = plt.subplots(1, 2, figsize = (10,5))
    axs[0].plot(freqs/1e9, I)
    axs[0].plot(freqs/1e9, Q)
    axs[0].set_xlabel('Freq (GHz)')
    axs[0].set_ylabel('I/Q')
    amps =  np.sqrt(np.array(I)**2 + np.array(Q)**2)
    ph = np.arctan2(np.array(Q), np.array(I))
    axs[1].plot(freqs/1e9, amps, 'b-')
    p = fitlor(freqs/1e9, -amps, showfit = False)
    axs[1].plot(freqs/1e9, -lorfunc(p, freqs/1e9), label=r'$\nu$ = %.3f GHz, $\Delta \nu$ = %.3f MHz'%(p[2], p[3]*1e3))
    print ("fits = ", p)
    print ("center freq", p[2], "GHz")
    print ("linewidth", p[3]*1e3, "MHz")
    ax2  = axs[1].twinx()
    ax2.plot(freqs/1e9, ph, 'r-')
    axs[1].set_xlabel('Freq (GHz)')
    axs[1].set_ylabel('amp')
    ax2.set_ylabel('$\\varphi$')
    axs[1].legend()
    fig.show()