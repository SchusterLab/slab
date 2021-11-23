from configuration_IQ import config, rr_LO, rr_freq, rr_IF
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dsfit import*
from slab.dataanalysis import get_next_filename
im = InstrumentManager()
LO_r = im['RF8']

f_min = -1e6
f_max = 1e6
df = 50e3
f_vec = rr_freq - np.arange(f_min, f_max + df/2, df)

a_min = 0.4
a_max = 1.0
da = 0.05
a_vec = np.arange(a_min, a_max + da/2, da)

LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

avgs = 5000
reset_time = 50000
simulation = 0

with program() as resonator_spectroscopy_vs_power:

    i = declare(int)
    f = declare(int)
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f <= f_max + rr_IF, f + df):

            update_frequency("rr", f)

            with for_(a, a_min, a < a_max + da/2, a + da):

                wait(reset_time//4, "rr")
                measure("clear"*amp(a), "rr", None,
                        demod.full("clear_integW1", I1, 'out1'),
                        demod.full("clear_integW2", Q1, 'out1'),
                        demod.full("clear_integW1", I2, 'out2'),
                        demod.full("clear_integW2", Q2, 'out2'))

                assign(I, I1-Q2)
                assign(Q, I2+Q1)

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(f_vec), len(a_vec)).average().save('I')
        Q_st.buffer(len(f_vec), len(a_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy_vs_power, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy_vs_power, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print ("Data collection done")

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'resonator_spec_opt_power', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=Q)
        f.create_dataset("freqs", data=f_vec)
        f.create_dataset("amps", data=a_vec)