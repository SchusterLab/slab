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
qubit_freq = 4.748488058227563e9
ge_IF = 100e6
qubit_LO = qubit_freq + ge_IF
rr_IF = 100e6
rr_LO = 8.05157848e9 + rr_IF

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

dt = 500
T_max = 25000
times = np.arange(0, T_max + dt/2, dt)

f_min = -10e3
f_max = 10e3
df = 2e3
f_vec = np.arange(f_min, f_max + df/2, df)
reset_time = 500000
avgs = 1000
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, f_min + ge_IF, f < f_max + ge_IF + df/2, f + df):

            update_frequency("qubit", f)

            with for_(t, 0, t < T_max + dt/2, t + dt):
                wait(reset_time//4, "qubit")
                play("pi2", "qubit")
                wait(t, "qubit")
                play("pi2", "qubit")
                align("qubit", "rr")
                measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),
                        demod.full("long_integW2", Q1, 'out1'),
                        demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

                assign(I, I1 - Q2)
                assign(Q, I2 + Q1)

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(f_vec), len(times)).average().save('I')
        Q_st.buffer(len(f_vec), len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")

    I_handle.wait_for_values(1)
    Q_handle.wait_for_values(1)
    #
    x = 4*np.arange(0, T_max + dt/2, dt)/1000
    y = (f_vec)/1e3

    while(res_handles.is_processing()):
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        sig = I + 1j*Q
        power = np.abs(sig)
        plt.pcolor(x, y, power)
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel(r'$\Delta \nu$ (kHz)')
        plt.pause(5)

    # I_handle = res_handles.get("I")
    # Q_handle = res_handles.get("Q")
    #
    # I_handle.wait_for_all_values()
    # Q_handle.wait_for_all_values()
    #
    # I = I_handle.fetch_all()
    # Q = Q_handle.fetch_all()
    #
    # print("Data collection done")
    #
    # sig = I + 1j*Q
    # power = np.abs(sig)
    #
    # plt.figure(figsize=(8, 6))
    # plt.pcolor(x, y, power)
    # plt.xlabel(r'Time ($\mu$s)')
    # plt.ylabel(r'$\Delta \nu$ (kHz)')
    # plt.show()