from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, rr_LO, qubit_LO, storage_LO, sb_LO, sb_IF
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

import time
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
LO_s  = im['sccav']
LO_sb  = im['scsb']

LO_s.set_frequency(storage_LO)
LO_s.set_power(12.0)
LO_sb.set_frequency(sb_LO)
LO_sb.set_power(12.0)
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)


qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)
biased_th = 0.0014

###############
# qubit_spec_prog:
###############
f_min = -10e6
f_max = 10e6
df = 100e3
f_vec = np.arange(f_min, f_max + df/2, df)

t_min = 250
t_max = 10000
dt = 500

t_vec = np.arange(t_min, t_max+dt/2, dt)

avgs = 3000
reset_time = 5000000
simulation = 0
with program() as sideband:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    a = declare(fixed)
    t = declare(int)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, sb_IF + f_min, f < sb_IF + f_max + df/2, f + df):

            update_frequency("sideband", f)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            align("storage", "qubit")
            play("pi", "qubit")
            align("qubit", "qubit_ef")
            play("pi", "qubit_ef")
            align("qubit_ef", "sideband")
            play("saturation"*amp(1.0), "sideband", duration=8000)
            align("sideband", "rr")
            play("pi", "qubit_ef")
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
        I_st.buffer(len(f_vec)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(sideband, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(sideband, duration_limit=0, data_limit=0)
    print("Experiment done")
    start_time = time.time()

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    # statistic_handle.wait_for_values(1)
    # plt.figure()
    # while(result_handles.is_processing()):
    #     I = statistic_handle.fetch_all()
    #     power = I**2
    #     plt.pcolor(power, cmap="RdBu")
    #     # plt.xlabel(r'Time ($\mu$s)')
    #     # plt.ylabel(r'$\Delta \nu$ (kHz)')
    #     plt.pause(5)
    #     # plt.clf()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'alpha_cal', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("res", data=res)
    #     f.create_dataset("amps", data=a_vec)
    #     f.create_dataset("times", data=t_vec)
    #     f.create_dataset("freq", data=f_vec)