from configuration_IQ import config
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
a_min = 0.00
a_max = 0.10
da = 0.005
amp_vec = np.arange(a_min, a_max + da/2, da)

phi_min = -0.5
phi_max = 0.5
dphi = 0.5/20

phi_vec = np.arange(phi_min, phi_max + dphi/2, dphi)

reset_time = 500000
avgs = 5000
simulation = 0

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    phi = declare(fixed)
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

        with for_(phi, phi_min, phi < phi_max + dphi/2, phi + dphi):

            with for_(n, 0, n < avgs, n + 1):

                """Just readout without playing anything"""
                wait(reset_time//4, "rr")
                reset_frame('jpa_pump')
                reset_frame('rr')
                frame_rotation_2pi(phi, 'jpa_pump')
                align("rr", "jpa_pump")
                play('pump_square'*amp(a), 'jpa_pump')
                measure("clear", "rr", None,
                        demod.full("clear_integW1", I1, 'out1'),
                        demod.full("clear_integW2", Q1, 'out1'),
                        demod.full("clear_integW1", I2, 'out2'),
                        demod.full("clear_integW2", Q2, 'out2'))

                assign(Ig, I1 - Q2)
                assign(Qg, I2 + Q1)
                save(Ig, Ig_st)
                save(Qg, Qg_st)

                align("qubit", "rr", "jpa_pump")

                """Play a ge pi pulse and then readout"""
                wait(reset_time // 4, "qubit")
                reset_frame('jpa_pump')
                reset_frame('rr')
                frame_rotation_2pi(phi, 'jpa_pump')
                play("pi", "qubit")
                align("qubit", "rr")
                align("rr", "jpa_pump")
                play('pump_square'*amp(a), 'jpa_pump')
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
                             get_next_filename(data_path, 'histogram_jpa_amp_phase_sweep', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        dset = f.create_dataset("ig", data=Ig)
        dset = f.create_dataset("qg", data=Qg)
        dset = f.create_dataset("ie", data=Ie)
        dset = f.create_dataset("qe", data=Qe)
        dset = f.create_dataset("amp", data=amp_vec)
        dset = f.create_dataset("phase", data=phi_vec)
