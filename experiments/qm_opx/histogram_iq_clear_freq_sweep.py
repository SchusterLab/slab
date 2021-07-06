from configuration_IQ import config, rr_IF, rr_freq
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from slab import*
from h5py import File
##################
# histogram_prog:
##################
reset_time = 500000
avgs = 5000
simulation = 0

f_min = -100e3
f_max = 100e3
df = 10e3
f_vec = np.arange(f_min, f_max + df/2, df)

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    f = declare(int)
    a = declare(fixed)

    Ig = declare(fixed)
    Qg = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    Ie = declare(fixed)
    Qe = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()


    with for_(f, rr_IF + f_min, f < rr_IF + f_max + df/2, f + df):

        update_frequency("rr", f)

        with for_(n, 0, n < avgs, n + 1):

            reset_frame("rr", "qubit")

            """Just readout without playing anything"""
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

            align("qubit", "rr")

            """Play a ge pi pulse and then readout"""
            wait(reset_time // 4, "qubit")
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
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(histogram, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(histogram, duration_limit=0, data_limit=0)

    print("Waiting for the data")

    job.result_handles.wait_for_all_values()

    Ig = job.result_handles.Ig.fetch_all()['value']
    Ie = job.result_handles.Ie.fetch_all()['value']
    Qg = job.result_handles.Qg.fetch_all()['value']
    Qe = job.result_handles.Qe.fetch_all()['value']

    print("Data fetched")

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'histogram_clear_freq_sweep', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        dset = f.create_dataset("ig", data=Ig)
        dset = f.create_dataset("qg", data=Qg)
        dset = f.create_dataset("ie", data=Ie)
        dset = f.create_dataset("qe", data=Qe)
        dset = f.create_dataset("freq", data=f_vec)