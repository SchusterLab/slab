from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from tqdm import tqdm
from h5py import File
import os
from slab.dataanalysis import get_next_filename

im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
##################
# ramsey_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

ramsey_freq = 100e3
omega = 2*np.pi*ramsey_freq

dt = 250

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 0
T_max = 30000
times = np.arange(T_min, T_max + dt/2, dt)
avgs = 1000
reset_time = 500000
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):
        assign(phi, 0)
        with for_(t, T_min, t < T_max + dt/2, t + dt):
            wait(reset_time//4, "qubit")
            play("pi2", "qubit")
            wait(t, "qubit")
            frame_rotation_2pi(phi, "qubit") #2pi is already multiplied to the phase
            play("pi2", "qubit")
            align("qubit", "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))
            # measure("long_readout", "rr", None,
            #         demod.full("long_integW1", I1, 'out1'),
            #         demod.full("long_integW2", Q1, 'out1'),
            #         demod.full("long_integW1", I2, 'out2'),
            #         demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)
            assign(phi, phi + dphi)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(times)).average().save('I')
        Q_st.buffer(len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done")

    job.halt()

    times = 4*times/1e3

    plt.plot(I)
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'ramsey_phase', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=Q)
    #     f.create_dataset("time", data=times)
    #     f.create_dataset("ramsey_freq", data=ramsey_freq)
    #     f.create_dataset("qubit_freq", data=qubit_freq)
