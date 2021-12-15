from configuration_IQ import config, rr_LO, rr_freq, rr_IF, qubit_LO
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from slab import*
import os
from slab.dataanalysis import get_next_filename

f_min = -2.5e6
f_max = 2.5e6
df = 25e3
f_vec = rr_freq - np.arange(f_min, f_max + df/2, df)
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
            wait(reset_time//4, "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))
            assign(Ig, I1 - Q2)
            assign(Qg, I2 + Q1)

            save(Ig, Ig_st)
            save(Qg, Qg_st)

            align("rr", 'qubit')

            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            align("qubit", "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))
            assign(Ie, I1 - Q2)
            assign(Qe, I2 + Q1)

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

    Ig = np.array(Ig_handle.fetch_all())
    Qg = np.array(Qg_handle.fetch_all())

    Ie = np.array(Ie_handle.fetch_all())
    Qe = np.array(Qe_handle.fetch_all())

    print ("Data collection done")

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'resonator_chi_opt', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("Ig", data=Ig)
        f.create_dataset("Qg", data=Qg)
        f.create_dataset("Ie", data=Ie)
        f.create_dataset("Qe", data=Qe)
        f.create_dataset("freqs", data=f_vec)