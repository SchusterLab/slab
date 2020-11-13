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
points = 1000
qubit_freq = 4.748488058822229e9
iq_freq = 200e6
qubit_LO = qubit_freq + iq_freq #had to change to + sign as the power in the first peak was lower (phase got mixed up somewhere)
rr_LO = 8.0517e9
reset_time = 500000

ramsey_freq = 4e3
detune_freq = iq_freq - ramsey_freq

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod = True)
LO_q.set_power(16)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod = True)
LO_r.set_power(16)


with program() as IQ_blobs:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # points
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############
    update_frequency("qubit", detune_freq)

    with for_(n, 0, n < points, n + 1):

        wait(reset_time//4, "qubit")
        align("qubit", "rr")
        measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
        save(I, I_st)
        save(Q, Q_st)

        align("qubit", "rr")
        wait(reset_time//4, "qubit")
        play("pi", "qubit")
        align("qubit", "rr")
        measure("long_readout", "rr", None, demod.full("long_integW1", I, 'out1'),demod.full("long_integW1", Q, 'out2'))
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

"""To simulate the pulse sequence"""
# job = qm.simulate(IQ_blobs, SimulationConfig(150000))
# samples = job.get_simulated_samples()
# samples.con1.plot()

"""To run the actual experiment"""
job = qm.execute(IQ_blobs, duration_limit=0, data_limit=0)
print("Done")

res_handles = job.result_handles
res_handles.wait_for_all_values()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()

plt.plot(I, Q, '.')
print("Done 2")

