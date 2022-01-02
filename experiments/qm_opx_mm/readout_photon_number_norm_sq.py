from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq, disc_file
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
##################
# ramsey_prog:
##################
""""Calibrating the readout power (OPX amplitude) by performing a qubit Ramsey experiment 
which results in the Stark shift, delta omega = 2*chi*n, where n=1. The sequence is as follows:
1. Play a CW tone on the readout with a variable amplitude
2. Play a Ramsey sequence on the qubit
3. Measure the qubit with a readout pulse
4. Fit the resultan traces with a Ramsey fit to extract the oscillation frequency equalt to 2*chi  
"""
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)


ramsey_freq = 2000e3
omega = 2*np.pi*ramsey_freq

dt = 4

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 4
T_max = 250
times = np.arange(T_min, T_max + dt/2, dt)

#Varying the readout pulse power
a_min = 0.0
a_max = 0.2
da = 0.01

a_vec = np.arange(a_min, a_max+da/2, da)

#fixing the wait time after the readout pulse is played
wait_time = 60 #240ns fixed
t_buffer = 125

avgs = 2000
reset_time = int(5e5)
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    a = declare(fixed)      # wait_times after CLEAR
    t = declare(int)        #array of time delays
    I = declare(fixed)
    res = declare(bool)
    phi = declare(fixed)

    I_st = declare_stream()
    res_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2 , a + da):

            assign(phi, 0)

            with for_(t, T_min, t < T_max + dt/2, t + dt):

                wait(reset_time//4, "rr")
                align("rr", "qubit_mode0")
                play('readout'*amp(a), 'rr', duration=(t + 40)) #add the two pi/2 pulses
                play("pi2", "qubit_mode0")
                wait(t, "qubit_mode0")
                frame_rotation_2pi(phi, "qubit_mode0") #2pi is already multiplied to the phase
                play("pi2", "qubit_mode0")
                align("qubit_mode0", "rr")
                # wait(t_buffer, "rr")
                discriminator.measure_state("readout", "out1", "out2", res, I=I)
                assign(phi, phi + dphi)

                save(I, I_st)
                save(res, res_st)

    with stream_processing():
        I_st.buffer(len(a_vec), len(times)).average().save('I')
        res_st.boolean_to_int().buffer(len(a_vec), len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    start_time = time.time()
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    #
    # I = res_handles.get('I').fetch_all()
    # Q = res_handles.get('Q').fetch_all()
    # plt.figure()
    # plt.pcolormesh(4*times/1e3, a_vec, Q, shading='auto')
    # plt.colorbar()
    # plt.xlabel('Ramsey time (μs)')
    # plt.ylabel('OPx amp (a.u.)')
    #
    # plt.figure()
    # plt.plot(4*times/1e3, Q[1])
    # plt.show()
    #
    # print("Data collection done")
    #
    # """Stop the output from OPX,heats up the fridge"""
    # job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'readout_photon_number_norm_square', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     dset = f.create_dataset("I", data=I)
    #     dset = f.create_dataset("Q", data=Q)
    #     dset = f.create_dataset("wait_time", data=4*wait_time)
    #     dset = f.create_dataset("ramsey_times", data=4*times)
    #     dset = f.create_dataset("ramsey_freq", data=ramsey_freq)
    #     dset = f.create_dataset("amps", data=a_vec)