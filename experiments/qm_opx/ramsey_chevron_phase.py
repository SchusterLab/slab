from configuration_IQ import config, qubit_LO, rr_LO, ge_IF
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
atten = im['atten']
from slab.dsfit import*

##################
# ramsey_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)
atten.set_attenuator(15.5)

dt = 15
T_min = 4
T_max = 6000
times = np.arange(T_min, T_max + dt/2, dt)

dphi_min = -0.05
dphi_max = 0.05
ddphi = 0.001
dphi_vec = np.arange(dphi_min, dphi_max + ddphi/2, ddphi)
reset_time = 500000
avgs = 500
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    dphi = declare(fixed)
    phi = declare(fixed)
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
        with for_(dphi, dphi_min, dphi < dphi_max + ddphi/2, dphi + ddphi):
            assign(phi, 0)
            with for_(t, 0, t < T_max + dt/2, t + dt):
                wait(reset_time//4, "qubit")
                play("pi2", "qubit")
                wait(t, "qubit")
                frame_rotation_2pi(phi, "qubit")
                play("pi2", "qubit")
                align("qubit", "rr")
                measure("long_readout", "rr", None, demod.full("long_integW1", I1, 'out1'),
                        demod.full("long_integW2", Q1, 'out1'),
                        demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

                assign(I, I1 + Q2)
                assign(Q, I2 - Q1)
                assign(phi, phi + dphi)

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(dphi_vec), len(times)).average().save('I')
        Q_st.buffer(len(dphi_vec), len(times)).average().save('Q')

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
    x = 4*times/1000
    y = (dphi_vec)/1e3

    while(res_handles.is_processing()):
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        sig = I + 1j*Q
        power = np.abs(sig)
        plt.pcolor(x, y, power, cmap="RdBu")
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