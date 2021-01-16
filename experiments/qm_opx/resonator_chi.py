from configuration_IQ import config, rr_LO, rr_freq, rr_IF, qubit_LO
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
atten = im['atten']
from slab.dsfit import*
##################
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)
atten.set_attenuator(12.0)
time.sleep(1)

f_min = -5e6
f_max = 5e6
df = 10e3
f_vec = rr_freq + np.arange(f_min, f_max + df/2, df)
reset_time = 500000
avgs = 1000
simulation = 0
with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    Ig = declare(fixed)
    Qg = declare(fixed)
    Ie = declare(fixed)
    Qe = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()


    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f < f_max + rr_IF + df / 2, f + df):
            update_frequency("rr", f)
            wait(10000//4, "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),demod.full("long_integW2", Q1, 'out1'),
                    demod.full("long_integW1", I2, 'out2'),demod.full("long_integW2", Q2, 'out2'))

            assign(Ig, I1+Q2)
            assign(Qg, I2-Q1)

            save(Ig, Ig_st)
            save(Qg, Qg_st)

            align("rr", 'qubit')

            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            align("qubit", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),demod.full("long_integW2", Q2, 'out2'))

            assign(Ie, I1+Q2)
            assign(Qe, I2-Q1)

            save(Ie, Ie_st)
            save(Qe, Qe_st)

    with stream_processing():

        Ig_st.buffer(len(f_vec)).average().save('Ig')
        Qg_st.buffer(len(f_vec)).average().save('Qg')
        Ie_st.buffer(len(f_vec)).average().save('Ie')
        Qe_st.buffer(len(f_vec)).average().save('Qe')


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)
    print("Starting the experiment")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig_handle = res_handles.get("Ig")
    Qg_handle = res_handles.get("Qg")

    Ie_handle = res_handles.get("Ie")
    Qe_handle = res_handles.get("Qe")

    I_g = np.array(Ig_handle.fetch_all())
    Q_g = np.array(Qg_handle.fetch_all())

    I_e = np.array(Ie_handle.fetch_all())
    Q_e = np.array(Qe_handle.fetch_all())


    print("Data collection done")

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    amps = np.sqrt(np.array(I_g) ** 2 + 1 * np.array(Q_g) ** 2)
    ph = np.arctan2(np.array(Q_g), np.array(I_g))
    ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    m = (ph[-1] - ph[0]) / (f_vec[-1] - f_vec[0])
    ph = ph - m * f_vec * 0.95
    ph_g = ph - np.mean(ph)

    # pg = fitlor(f_vec / 1e9, amps, showfit=False)

    axs.plot(f_vec/1e9, ph_g, 'r', label='g')
    amps = np.sqrt(np.array(I_e) ** 2 + 1 * np.array(Q_e) ** 2)
    pe = fitlor(f_vec / 1e9, amps, showfit=False)
    ph = np.arctan2(np.array(Q_e), np.array(I_e))
    ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    m = (ph[-1] - ph[0]) / (f_vec[-1] - f_vec[0])
    ph = ph - m * f_vec * 0.95
    ph_e = ph - np.mean(ph)

    axs.plot(f_vec/1e9, ph_e, 'b', label='e')
    axs.legend(loc='best')

    # axs[0].plot(f_vec / 1e9, I)
    # axs[0].plot(f_vec / 1e9, Q)
    # axs[0].set_xlabel('Freq (GHz)')
    # axs[0].set_ylabel('I/Q')
    # amps = np.sqrt(np.array(I) ** 2 + 1 * np.array(Q) ** 2)
    # ph = np.arctan2(np.array(Q), np.array(I))
    # ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    # m = (ph[-1] - ph[0]) / (f_vec[-1] - f_vec[0])
    # ph = ph - m * f_vec * 0.95
    # ph = ph - np.mean(ph)
    # axs[1].plot(f_vec / 1e9, amps, 'b-')
    # p = fitlor(f_vec / 1e9, amps, showfit=False)
    # x = np.array(f_vec) / 1e9
    # axs[1].plot(f_vec / 1e9, lorfunc(p, f_vec / 1e9), label=r'$\nu_{r}$ = %.4f GHz' % x[np.argmax(amps)])
    # print("fits = ", p)
    # ax2 = axs[1].twinx()
    # ax2.plot(f_vec / 1e9, ph, 'r-')
    # axs[1].set_xlabel('Freq (GHz)')
    # axs[1].set_ylabel('amp')
    # ax2.set_ylabel('$\\varphi$')
    # axs[1].legend(loc='best')
    # fig.show()