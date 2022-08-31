from configuration_IQ import config, qubit_LO, disc_file_opt, disc_file, readout_len
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from h5py import File
import os
from slab.dataanalysis import get_next_filename

"""ef Rabi"""

a_min = 0.2
a_max = 0.4
da = 0.002
amps = np.arange(a_min, a_max + da/2, da)
avgs = 1000
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

with program() as ef_rabi_IQ:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    a = declare(fixed)      # Amplitudes
    I = declare(fixed)
    res = declare(bool)

    I_st = declare_stream()
    res_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            wait(reset_time//4, "qubit_mode0")
            play("pi", "qubit_mode0")
            align("qubit_mode0", "qubit_ef")
            play("gaussian"*amp(a), "qubit_ef")
            align("qubit_mode0", "qubit_ef")
            play("pi", "qubit_mode0")
            align("qubit_mode0", "rr")
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(amps)).average().save('res')
        I_st.buffer(len(amps)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

"""To simulate the pulse sequence"""
if simulation:
    job = qm.simulate(ef_rabi_IQ, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    job = qm.execute(ef_rabi_IQ, duration_limit=0, data_limit=0)
    print("Experiment done")

    result_handles = job.result_handles
    # res_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    #
    # plt.figure()
    # plt.plot(amps, res, '.-')
    # plt.show()
    #
    # job.halt()
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'ef_power_rabi', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("amps", data=amps)