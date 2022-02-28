from configuration_IQ import config, rr_IF, readout_len
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

def readout_fidelity(readout_len, delta_f):
    pulse_len = readout_len
    b = ((30.0+90)/180)*np.pi
    w_plus_cos = [(np.cos(b), pulse_len)]
    w_minus_cos = [(-np.cos(b), pulse_len)]
    w_plus_sin = [(np.sin(b), pulse_len)]
    w_minus_sin = [(-np.sin(b), pulse_len)]
    config['integration_weights']['r_cos']['cosine'] = w_plus_cos
    config['integration_weights']['r_cos']['sine'] = w_minus_sin
    config['integration_weights']['r_sin']['cosine'] = w_plus_sin
    config['integration_weights']['r_sin']['sine'] = w_plus_cos
    config['integration_weights']['r_minus_sin']['cosine'] = w_minus_sin
    config['integration_weights']['r_minus_sin']['sine'] = w_minus_cos
    config['pulses']['readout_pulse']['length'] = pulse_len
    reset_time = 500000
    avgs = 2000
    simulation = False

    with program() as readout_pulse_opt:

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

        If_st = declare_stream()
        Qf_st = declare_stream()

        update_frequency('rr', rr_IF-delta_f)

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            """Just readout without playing anything"""
            wait(reset_time // 4, "rr")
            measure("readout", "rr", None,
                    dual_demod.full('r_cos', 'out1', 'r_minus_sin', 'out2', I),
                    dual_demod.full('r_sin', 'out1', 'r_cos', 'out2', Q))

            save(I, Ig_st)
            save(Q, Qg_st)

            align("qubit_mode0", "rr")

            """Play a ge pi pulse and then readout"""
            wait(reset_time // 4, "qubit_mode0")
            play("pi", "qubit_mode0")
            align("qubit_mode0", "rr")
            measure("readout", "rr", None,
                    dual_demod.full('r_cos', 'out1', 'r_minus_sin', 'out2', I),
                    dual_demod.full('r_sin', 'out1', 'r_cos', 'out2', Q))

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
        job = qm.simulate(histogram, SimulationConfig(150000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        """To run the actual experiment"""
        job = qm.execute(readout_pulse_opt, duration_limit=0, data_limit=0)
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

        Ig_avg = np.mean(Ig); Ig_std = np.std(Ig)
        Qg_avg = np.mean(Qg); Qg_std = np.std(Qg)
        Ie_avg = np.mean(Ie); Ie_std = np.std(Ie)
        Qe_avg = np.mean(Qe); Qe_std = np.std(Qe)
        F = np.sqrt((Ig_avg-Ie_avg)**2+(Qg_avg-Qe_avg)**2)/(np.mean(Ig_std +Qg_std +Ie_std +Qe_std ))
        print(F)

        # plt.figure()
        # plt.plot(Ig, Qg,'.')
        # plt.plot(Ie, Qe,'.')
        # plt.axis('equal')
        # plt.errorbar(x=Ig_avg, y=Qg_avg, xerr=Ig_std, yerr=Qg_std, linewidth=14)
        # plt.errorbar(x=Ie_avg, y=Qe_avg, xerr=Ie_std, yerr=Qe_std, linewidth=14)
        # plt.axis('equal')

        return(F)


lens = (np.arange(2000, 3200, 50)//4)*4
dfs = np.arange(-0.1e6, 0.1e6, 0.025e6)
F = []
for len_ in lens:
    for df_ in dfs:
        F_ = readout_fidelity(len_, df_)
        F.append(F_)

F= np.array(F).reshape(len(lens),len(dfs))
plt.figure()
plt.pcolormesh(dfs, lens, F, shading='auto')
plt.colorbar()
plt.show()
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