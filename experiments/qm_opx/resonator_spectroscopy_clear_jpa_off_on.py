from configuration_IQ import config, rr_freq, rr_IF
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

"""readout resonator spectroscopy, varying the IF frequency"""
f_min = -5e6
f_max = 5e6
df = 50e3
f_vec = rr_freq - np.arange(f_min, f_max + df/2, df)

avgs = 500
reset_time = 50000
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

    I1_st = declare_stream()
    Q1_st = declare_stream()

    I2_st = declare_stream()
    Q2_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f <= f_max + rr_IF, f + df):
            update_frequency("rr", f)
            wait(reset_time//4, "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I1_st)
            save(Q, Q1_st)

            wait(reset_time//4, 'rr')

            align('rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I2_st)
            save(Q, Q2_st)

    with stream_processing():
        I1_st.buffer(len(f_vec)).average().save('I1')
        Q1_st.buffer(len(f_vec)).average().save('Q1')
        I2_st.buffer(len(f_vec)).average().save('I2')
        Q2_st.buffer(len(f_vec)).average().save('Q2')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I1 = res_handles.get("I1").fetch_all()
    Q1 = res_handles.get("Q1").fetch_all()

    I2 = res_handles.get("I2").fetch_all()
    Q2 = res_handles.get("Q2").fetch_all()

    print ("Data collection done")

    job.halt()

    x = f_vec

    plt.figure()
    pow_1 = 10*np.log10(I1**2 + Q1**2)
    pow_2 = 10*np.log10(I2**2 + Q2**2)

    print(np.max(pow_2)-np.max(pow_1))

    plt.plot(x/1e9, pow_1, 'b.', label='JPA Off')
    plt.plot(x/1e9, pow_2, 'r.', label='JPA On')
    plt.xlabel(r' Frequency (GHz)')
    plt.ylabel('Arb. units')
    plt.legend()
    plt.show()

    #
    # plt.figure(dpi=300)
    # pow_1 = 10*np.log10(I1**2 + Q1**2)
    # pow_2 = 10*np.log10(I2**2 + Q2**2)
    # plt.plot(x/1e9, pow_1, 'b.', label='JPA Off')
    # plt.plot(x/1e9, pow_2, 'r.', label='JPA On')
    # plt.xlabel(r' Frequency (GHz)')
    # plt.ylabel('Arb. units')
    # plt.legend()
    # plt.show()


    # plt.figure()
    # ph = np.arctan2(np.array(Q1), np.array(I1))
    # ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    # m = (ph[-1]-ph[0])/(x[-1] - x[0])
    # ph = ph - m*x*0.95
    # ph = ph -np.mean(ph)
    # plt.plot(ph)
    # ph = np.arctan2(np.array(Q2), np.array(I2))
    # ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    # m = (ph[-1]-ph[0])/(x[-1] - x[0])
    # ph = ph - m*x*0.95
    # ph = ph -np.mean(ph)
    # plt.plot(ph)

    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'resonator_spec', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=Q)
    #     f.create_dataset("freqs", data=f_vec)