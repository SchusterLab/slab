from configuration_IQ import config, qubit_LO, rr_LO, rr_IF, opt_len
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from h5py import File
from slab import*
import os
from slab.dataanalysis import get_next_filename

##################
# histogram_prog:
##################
reset_time = 500000
avgs = 10000
simulation = 0

w_plus = [(1.0, opt_len)]
w_minus = [(-1.0, opt_len)]
w_zero = [(0.0, opt_len)]

b = (30.0/180)*np.pi
w_plus_cos = [(np.cos(b), opt_len)]
w_minus_cos = [(-np.cos(b), opt_len)]
w_plus_sin = [(np.sin(b), opt_len)]
w_minus_sin = [(-np.sin(b), opt_len)]
config['integration_weights']['clear_integW1']['cosine'] = w_plus_cos
config['integration_weights']['clear_integW1']['sine'] = w_minus_sin
config['integration_weights']['clear_integW2']['cosine'] = w_plus_sin
config['integration_weights']['clear_integW2']['sine'] = w_plus_cos
config['integration_weights']['clear_integW3']['cosine'] = w_minus_sin
config['integration_weights']['clear_integW3']['sine'] = w_minus_cos

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging

    I = declare(fixed)
    Q = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()

    # update_frequency('rr', rr_IF+5e3)

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "rr")
        measure("clear", "rr", None,
                dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))

        save(I, Ig_st)
        save(Q, Qg_st)

        align("qubit_mode0", "rr")

        """Play a ge pi pulse and then readout"""
        wait(reset_time // 4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        measure("clear", "rr", None,
                dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))

        save(I, Ie_st)
        save(Q, Qe_st)

    with stream_processing():
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(histogram, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(histogram, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig_handle = res_handles.get("Ig")
    Qg_handle = res_handles.get("Qg")

    Ie_handle = res_handles.get("Ie")
    Qe_handle = res_handles.get("Qe")

    Ig = np.array(Ig_handle.fetch_all()['value'])
    Qg = np.array(Qg_handle.fetch_all()['value'])

    Ie = np.array(Ie_handle.fetch_all()['value'])
    Qe = np.array(Qe_handle.fetch_all()['value'])

    plt.figure()
    plt.plot(Ig, Qg,'.')
    plt.plot(Ie, Qe,'.')
    # plt.plot(If, Qf,'.')

    plt.axis('equal')

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/stim_em_oct_rp_nodisc")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'histogram_clear', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        dset = f.create_dataset("ig", data=Ig)
        dset = f.create_dataset("qg", data=Qg)
        dset = f.create_dataset("ie", data=Ie)
        dset = f.create_dataset("qe", data=Qe)