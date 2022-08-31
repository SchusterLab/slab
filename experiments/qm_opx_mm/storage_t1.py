from configuration_IQ import config, ge_IF, disc_file_opt, two_chi
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
from slab.dsfit import*
from slab.dataanalysis import get_next_filename

"""Storage cavity t1 experiment"""

dt = 12500
T_max = int(1.5e6)
T_min = 250
t_vec = np.arange(T_min, T_max + dt/2, dt)

cav_len = 10
cav_amp = 1.0

avgs = 1000
reset_time = int(7.5e6)
simulation = 0 #1 to simulate the pulses

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

with program() as storage_mode1_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    Q = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            wait(reset_time//4, "storage_mode1")
            play("CW"*amp(cav_amp), "storage_mode1", duration=cav_len)
            align("storage_mode1", "qubit_mode0")
            wait(t, "qubit_mode0")
            play("res_pi", "qubit_mode0")
            align("qubit_mode0", "rr")
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
        I_st.buffer(len(t_vec)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(storage_mode1_t1, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(storage_mode1_t1, duration_limit=0, data_limit=0)
    print ("Execution done")

    result_handles = job.result_handles

    # res_handle = result_handles.get("res")
    # res_handle.wait_for_values(1)
    # plt.figure()
    # while(result_handles.is_processing()):
    #     res = res_handle.fetch_all()
    #     plt.plot(4*t_vec, res, '.-')
    #     # plt.xlabel(r'Time ($\mu$s)')
    #     # plt.ylabel(r'$\Delta \nu$ (kHz)')
    #     plt.pause(5)
    #     plt.clf()


    # result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    plt.figure()
    plt.plot(4*t_vec/1e3, res, '.-')
    plt.show()

    # print ("Data collection done")
    #
    # job.halt()
    #
    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_mode_t1', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("res", data=res)
        f.create_dataset("time", data=4*t_vec)
        # f.create_dataset("cav_len", data=4*cav_len)
        # f.create_dataset("cav_amp", data=cav_amp)