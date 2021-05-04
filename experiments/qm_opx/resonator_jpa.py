from configuration_IQ import config, rr_LO, rr_freq, rr_IF, qubit_LO, long_redout_len
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from slab import*
import os
from slab.dataanalysis import get_next_filename
##################
##################

f_min = -2.5e6
f_max = 2.5e6
df = 25e3
f_vec = rr_freq + np.arange(f_min, f_max + df/2, df)
reset_time = 50000
avgs = 1000
simulation = 0

with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    I = declare(fixed)
    Q = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f < f_max + rr_IF + df / 2, f + df):

            update_frequency("rr", f)
            reset_frame('jpa_pump')
            reset_frame('rr')
            frame_rotation_2pi(0.01, 'jpa_pump')
            wait(reset_time//4, 'rr')
            align("rr", "jpa_pump")
            play('pump_square'*amp(0.0045), 'jpa_pump')
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I_st)
            save(Q, Q_st)


    with stream_processing():

        I_st.buffer(len(f_vec)).average().save('I')
        Q_st.buffer(len(f_vec)).average().save('Q')

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

    I = res_handles.get("I").fetch_all()
    Q = res_handles.get("Q").fetch_all()

    print ("Data collection done")

    job.halt()

    plt.plot(I**2 + Q**2)

    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'resonator_jpa', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("Ig", data=Ig)
    #     f.create_dataset("Qg", data=Qg)
    #     f.create_dataset("Ie", data=Ie)
    #     f.create_dataset("Qe", data=Qe)
    #     f.create_dataset("freqs", data=f_vec)