from configuration_IQ import config, qubit_LO, rr_LO, rr_IF, rr_freq
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
# from numpy import *
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from slab import*
from slab.instruments import instrumentmanager
from h5py import File
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

##################
# histogram_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(13)

reset_time = 500000
avgs = 3000
simulation = 0

a_min = 0.1
a_max = 0.5
da = 0.01
amp_vec = np.arange(a_min, a_max + da/2, da)
f_min = -0.4e6
f_max = 0.4e6
df = 20e3
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


    with for_(a, a_min, a < a_max + da/2, a + da):

        with for_(f, rr_IF + f_min, f < rr_IF + f_max + df/2, f + df):
            update_frequency("rr", f)

            with for_(n, 0, n < avgs, n + 1):

                """Just readout without playing anything"""
                wait(reset_time//4, "rr")
                measure("long_readout"*amp(a), "rr", None, demod.full("long_integW1", I1, 'out1'),
                        demod.full("long_integW2", Q1, 'out1'),
                        demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

                assign(Ig, I1 + Q2)
                assign(Qg, I2 - Q1)
                save(Ig, Ig_st)
                save(Qg, Qg_st)

                align("qubit", "rr")

                """Play a ge pi pulse and then readout"""
                wait(reset_time // 4, "qubit")
                play("pi", "qubit")
                align("qubit", "rr")
                measure("long_readout"*amp(a), "rr", None, demod.full("long_integW1", I1, 'out1'),
                        demod.full("long_integW2", Q1, 'out1'),
                        demod.full("long_integW1", I2, 'out2'), demod.full("long_integW2", Q2, 'out2'))

                assign(Ie, I1 + Q2)
                assign(Qe, I2 - Q1)
                save(Ie, Ie_st)
                save(Qe, Qe_st)

    with stream_processing():
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')
        # Ig_st.buffer(len(amp_vec), len(f_vec)).save_all('Ig')
        # Qg_st.buffer(len(amp_vec), len(f_vec)).save_all('Qg')
        #
        # Ie_st.buffer(len(amp_vec), len(f_vec)).save_all('Ie')
        # Qe_st.buffer(len(amp_vec), len(f_vec)).save_all('Qe')

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
    print("Done")
    start_time = time.time()
    # for att in tqdm(amp_vec):
    #     while not job.is_paused():
    #         time.sleep(2.5)
    #     atten.set_attenuator(att)
    #     print(att)
    #     time.sleep(1.0)
    #     job.resume()

    print("Waiting for the data")

    job.result_handles.wait_for_all_values()

    Ig = job.result_handles.Ig.fetch_all()['value']
    Ie = job.result_handles.Ie.fetch_all()['value']
    Qg = job.result_handles.Qg.fetch_all()['value']
    Qe = job.result_handles.Qe.fetch_all()['value']
    print("Data fetched")
    stop_time = time.time()

    print(f"Time taken: {stop_time - start_time}")

    path = "C:\\_Lib\python\\slab\\experiments\\qm_opx\\data\\"
    filename = path + "histogram_amp_freq_sweep_100MHz_3us_3.h5"
    with File(filename, 'w') as f:
        dset = f.create_dataset("ig", data=Ig)
        dset = f.create_dataset("qg", data=Qg)
        dset = f.create_dataset("ie", data=Ie)
        dset = f.create_dataset("qe", data=Qe)
        dset = f.create_dataset("att", data=amp_vec)
        dset = f.create_dataset("freq", data=f_vec)