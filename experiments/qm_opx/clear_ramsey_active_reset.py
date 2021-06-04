from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, disc_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
##################
simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(1000//4, "rr")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')
    # save(I, "check")

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

##################
# ramsey_prog:
##################
ramsey_freq = 1000e3
omega = 2*np.pi*ramsey_freq

dt = 25

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 0
T_max = 750
times = np.arange(T_min, T_max + dt/2, dt)

wait_tmin = 25
wait_tmax = 1000
wait_dt = 25
wait_tvec = np.arange(wait_tmin, wait_tmax + wait_dt/2, wait_dt)
t_buffer = 250

avgs = 200
reset_time = 500000
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # wait_times after CLEAR
    t = declare(int)        #array of time delays

    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(i, wait_tmin, i < wait_tmax + wait_dt/2, i + wait_dt):
            assign(phi, 0)
            with for_(t, T_min, t < T_max + dt/2, t + dt):
                # active_reset(biased_th_g)
                reset_frame("qubit", "rr")
                wait(reset_time//4, 'rr')
                play('clear', 'rr')
                align("rr", "qubit")
                wait(i, "qubit")
                play("pi2", "qubit")
                wait(t, "qubit")
                frame_rotation_2pi(phi, "qubit") #2pi is already multiplied to the phase
                play("pi2", "qubit")
                align("qubit", "rr")
                wait(t_buffer, "rr")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)
                assign(phi, phi + dphi)
                # wait(reset_time//40, 'rr')

                save(res, res_st)
                save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(wait_tvec), len(times)).average().save('res')
        I_st.buffer(len(wait_tvec), len(times)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    res = res_handles.get('res').fetch_all()
    I = res_handles.get('I').fetch_all()


    plt.figure()
    plt.pcolormesh(4*times*1e6, 4*wait_tvec*1e6, res, shading='auto')
    plt.xlabel('Ramsey time (μs)')
    plt.ylabel('Wait time (μs)')
    """Stop the output from OPX,heats up the fridge"""
    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'ramsey_clear', suffix='.h5'))
    print(seq_data_file)

    wait_tvec = 4*wait_tvec
    times = 4*times
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Q", data=res)
        f.create_dataset("I", data=I)
        f.create_dataset("wait_time", data=wait_tvec)
        f.create_dataset("ramsey_times", data=times)