from configuration_IQ import config, rr_LO, rr_freq, rr_IF, storage_cal_file
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

"""readout resonator spectroscopy as a function of power with a square readout shape
"""
f_min = -2.5e6
f_max = 0.5e6
df = 6e3
f_vec = np.arange(f_min, f_max + df/2, df)

a_min = 0.0
a_max = 1.0
da = 0.01

a_vec = np.arange(a_min, a_max + da/2, da)

avgs = 1000
reset_time = int(7.5e3)
simulation = 0
with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    a = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(a, a_min, a < a_max + da/2, a+da):

            with for_(f, f_min + rr_IF, f <= f_max + rr_IF, f + df):

                update_frequency("rr", f)
                wait(reset_time, 'rr')
                measure("readout"*amp(a), "rr", None,
                        dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                        dual_demod.full('sin', 'out1', 'cos', 'out2', Q))

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(a_vec), len(f_vec)).average().save('I')
        Q_st.buffer(len(a_vec), len(f_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    # I_handle = res_handles.get("I")
    # Q_handle = res_handles.get("Q")
    # I = I_handle.fetch_all()
    # Q = Q_handle.fetch_all()
    #
    # plt.figure()
    # plt.pcolormesh(f_vec, a_vec, I**2+Q**2, cmap='RdBu', shading='auto')
    # plt.colorbar()
    # plt.xlabel('IF Freq (MHz)')
    # plt.ylabel('Readout amp (arb. unit)')
    # plt.show()
    #
    # job.halt()
    #
    path = os.getcwd()
    data_path = os.path.join(path, "data/thesis/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'resonator_spec_punchout', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=Q)
        f.create_dataset("freqs", data=f_vec)
        f.create_dataset("amp", data=a_vec)