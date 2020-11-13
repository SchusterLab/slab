from configuration_IQ import config
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

##################
# ramsey_prog:
##################
qubit_freq = 4.748488058822229e9 #g-e
qubit_ef_freq = 4.608488058632734e9

ge_IF = 100e6 # g-e IF
ef_IF = int(ge_IF - (qubit_freq-qubit_ef_freq))
qubit_LO = qubit_freq - ge_IF
rr_IF = 100e6
rr_LO = 8.0518e9 - rr_IF


ramsey_freq = 500e3
detune_freq = ef_IF - ramsey_freq

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=True)
LO_r.set_power(18)

dt = 100
T_max = 12500
times = np.arange(0, T_max + dt/2, dt)
reset_time = 500000
simulation = 0
avgs = 1000
with program() as ef_ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    t = declare(int)      # Wait time
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############
    # update_frequency("qubit", ge_IF)
    # update_frequency("rr", rr_IF)
    update_frequency("qubit_ef", detune_freq)

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, 0, t < T_max + dt/2, t + dt):
            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            align("qubit", "qubit_ef")
            play("pi2", "qubit_ef")
            wait(t, "qubit_ef")
            play("pi2", "qubit_ef")
            align("qubit_ef", "rr")
            measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(times)).average().save('I')
        Q_st.buffer(len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ef_ramsey, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(ef_ramsey, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Done 2")

    times = 4*times/1e3
    z = 1
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(times[z:len(I)], I[z:], 'bo')
    p = fitdecaysin(times[z:len(I)], I[z:], showfit=False)
    print("fits :", p)
    axs[0].plot(times[z:len(I)], decaysin(np.append(p, 0), times[z:len(I)]), 'b-',
                label=r'$T_{2}^{*}$ = %.2f $\mu$s' % p[3])
    axs[0].set_xlabel('Time ($\mu$s)')
    axs[0].set_ylabel('I')
    axs[0].legend()
    offset = ramsey_freq / 1e9 - p[1]
    nu_q_new = qubit_ef_freq / 1e9 + offset / 1e9

    print("Original qubit frequency choice =", qubit_ef_freq / 1e9, "GHz")
    print("Offset freq =", offset, "Hz")
    print("Suggested qubit frequency choice =", nu_q_new, "GHz")
    print("T2* =", p[3], "us")

    z = 1
    axs[1].plot(times[z:len(I)], Q[z:], 'ro')
    p = fitdecaysin(times[z:len(I)], Q[z:], showfit=False)
    axs[1].plot(times[z:len(I)], decaysin(np.append(p, 0), times[z:len(I)]), 'r-',
                label=r'$T_{2}^{*}$ = %.2f $\mu$s' % p[3])
    print("fits :", p)
    axs[1].set_xlabel('Time ($\mu$s)')
    axs[1].set_ylabel('Q')
    axs[1].legend()
    plt.tight_layout()
    fig.show()

    offset = ramsey_freq / 1e9 - p[1]
    nu_q_new = qubit_ef_freq / 1e9 + offset / 1e9

    print("Original qubit frequency choice =", qubit_ef_freq / 1e9, "GHz")
    print("Offset freq =", offset, "Hz")
    print("Suggested qubit frequency choice =", nu_q_new, "GHz")
    print("T2* =", p[3], "us")