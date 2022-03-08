from configuration_IQ import config, qubit_LO, ef_IF, disc_file, readout_len
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

###############
# qubit_spec_prog:
###############
f_min = -5e6
f_max = 5e6
df = 50e3
f_vec = np.arange(f_min, f_max + df/2, df)
f_vec = f_vec + qubit_LO + ef_IF

avgs = 100
reset_time = 500000
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

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


with program() as qubit_ef_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    I = declare(fixed)
    Q = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ef_IF + f_min, f < ef_IF + f_max + df/2, f + df):

            update_frequency("qubit_ef", f)
            wait(reset_time//4, 'qubit_mode0')
            play("pi", "qubit_mode0")
            align("qubit_mode0", "qubit_ef")
            play("saturation"*amp(0.2), "qubit_ef")
            align("qubit_ef", "rr")
            measure("readout", "rr", None,
                    dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                    dual_demod.full('sin', 'out1', 'cos', 'out2', Q))

            save(I, I_st)
            save(Q, Q_st)


    with stream_processing():

        I_st.buffer(len(f_vec)).average().save('I')
        Q_st.buffer(len(f_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(qubit_ef_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(qubit_ef_spec, duration_limit=0, data_limit=0)
    print("Experiment done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done!")

    job.halt()

    plt.figure()
    plt.plot(f_vec/1e9,  Q, '.-')
    plt.plot(f_vec/1e9, I, '.-')
    plt.show()

    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'ef_spectroscopy', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=Q)
    #     f.create_dataset("freqs", data=f_vec)