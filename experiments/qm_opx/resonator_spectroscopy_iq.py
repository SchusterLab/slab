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
LO_r = im['RF8']
# atten = im["atten"]

rr_IF = 100e6
rr_freq = 8.0518e9
rr_LO = rr_freq - rr_IF

f_min = -20e6
f_max = 20e6
df = 40e3

f_vec = rr_freq + np.arange(f_min, f_max + df/2, df)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)
LO_r.set_output(True)
# atten.set_attenuator(0.0)
time.sleep(1)

avgs = 1000
reset_time = 10000
simulation = 0
with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f < f_max + rr_IF + df / 2, f + df):
            update_frequency("rr", f)
            wait(reset_time//4, "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1+Q2)
            assign(Q, I2-Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(f_vec)).average().save('I')
        Q_st.buffer(len(f_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(f_vec/1e9, I)
    axs[0].plot(f_vec/1e9, Q)
    axs[0].set_xlabel('Freq (GHz)')
    axs[0].set_ylabel('I/Q')
    amps = np.sqrt(np.array(I)**2 + 1*np.array(Q)**2)
    ph = np.arctan2(np.array(Q), np.array(I))
    ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    m = (ph[-1]-ph[0])/(f_vec[-1] - f_vec[0])
    ph = ph - m*f_vec*0.95
    ph = ph -np.mean(ph)
    axs[1].plot(f_vec/1e9, amps, 'b-')
    p = fitlor(f_vec/1e9, amps, showfit=False)
    x = np.array(f_vec)/1e9
    axs[1].plot(f_vec/1e9, lorfunc(p, f_vec/1e9), label=r'$\nu_{r}$ = %.4f GHz'% x[np.argmax(amps)])
    print ("fits = ", p)
    ax2  = axs[1].twinx()
    ax2.plot(f_vec/1e9, ph, 'r-')
    axs[1].set_xlabel('Freq (GHz)')
    axs[1].set_ylabel('amp')
    ax2.set_ylabel('$\\varphi$')
    axs[1].legend(loc='best')
    fig.show()