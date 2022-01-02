from configuration_IQ import config, rr_LO, readout_len
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from slab import*
import os
from slab.dataanalysis import get_next_filename
from slab import*

##################
# histogram_prog:
##################
pulse_len = readout_len

w_plus = [(1.0, pulse_len)]
w_minus = [(-1.0, pulse_len)]
w_zero = [(0.0, pulse_len)]

b = (30.0/180)*np.pi
w_plus_cos = [(np.cos(b), pulse_len)]
w_minus_cos = [(-np.cos(b), pulse_len)]
w_plus_sin = [(np.sin(b), pulse_len)]
w_minus_sin = [(-np.sin(b), pulse_len)]

config['integration_weights']['cos']['cosine'] = w_plus_cos
config['integration_weights']['cos']['sine'] = w_minus_sin
config['integration_weights']['sin']['cosine'] = w_plus_sin
config['integration_weights']['sin']['sine'] = w_plus_cos
config['integration_weights']['minus_sin']['cosine'] = w_minus_sin
config['integration_weights']['minus_sin']['sine'] = w_minus_cos

reset_time = 500000
avgs = 5000
simulation = 0
with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    I = declare(fixed)
    Q = declare(fixed)
    # If = declare(fixed)
    # Qf = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()

    # If_st = declare_stream()
    # Qf_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "rr")
        measure("readout", "rr", None,
                dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                dual_demod.full('sin', 'out1', 'cos', 'out2', Q))

        save(I, Ig_st)
        save(Q, Qg_st)

        align("qubit_mode0", "rr")

        """Play a ge pi pulse and then readout"""
        wait(reset_time // 4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        measure("readout", "rr", None,
                dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                dual_demod.full('sin', 'out1', 'cos', 'out2', Q))

        save(I, Ie_st)
        save(Q, Qe_st)

        # align("qubit", "rr")
        #
        # """Play a ge pi pulse and then an ef pi pulse and then readout"""
        # wait(reset_time // 4, "qubit")
        # play("pi", "qubit")
        # align("qubit", "qubit_ef")
        # play("pi", "qubit_ef")
        # align("qubit_ef", "rr")
        # measure("long_readout", "rr", None,
        #         demod.full("long_integW1", I1, 'out1'),
        #         demod.full("long_integW2", Q1, 'out1'),
        #         demod.full("long_integW1", I2, 'out2'),
        #         demod.full("long_integW2", Q2, 'out2'))
        #
        # assign(If, I1 - Q2)
        # assign(Qf, I2 + Q1)
        #
        # save(If, If_st)
        # save(Qf, Qf_st)

    with stream_processing():
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

        # If_st.save_all('If')
        # Qf_st.save_all('Qf')

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

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig_handle = res_handles.get("Ig")
    Qg_handle = res_handles.get("Qg")

    Ie_handle = res_handles.get("Ie")
    Qe_handle = res_handles.get("Qe")

    # If_handle = res_handles.get("If")
    # Qf_handle = res_handles.get("Qf")

    Ig = np.array(Ig_handle.fetch_all()['value'])
    Qg = np.array(Qg_handle.fetch_all()['value'])

    Ie = np.array(Ie_handle.fetch_all()['value'])
    Qe = np.array(Qe_handle.fetch_all()['value'])

    # If = np.array(If_handle.fetch_all()['value'])
    # Qf = np.array(Qf_handle.fetch_all()['value'])


    plt.figure()
    plt.plot(Ig, Qg,'.')
    plt.plot(Ie, Qe,'.')
    # plt.plot(If, Qf,'.')

    plt.axis('equal')

    job.halt()

    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'histogram', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     dset = f.create_dataset("ig", data=Ig)
    #     dset = f.create_dataset("qg", data=Qg)
    #     dset = f.create_dataset("ie", data=Ie)
    #     dset = f.create_dataset("qe", data=Qe)