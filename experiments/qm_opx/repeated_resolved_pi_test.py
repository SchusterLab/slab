from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
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
# repeated resolved pi pulse program for single photon counting:
##################
# LO_q.set_frequency(qubit_LO)
# LO_q.set_ext_pulse(mod=False)
# LO_q.set_power(18)
# LO_r.set_frequency(rr_LO)
# LO_r.set_ext_pulse(mod=False)
# LO_r.set_power(18)

avgs = 10000
reset_time = 2500000
simulation = 0
two_chi = 1.13e6

num_pi_pulses = 30

cav_amp = 0.75
cav_len = 10

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

biased_th_g = 0.0014
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    wait(5000//4, 'rr')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as photon_counting:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############


    with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage')
            # align('storage', 'rr')
            # active_reset(biased_th_g)
            # align('storage', 'rr')
            update_frequency('qubit', ge_IF-two_chi)
            play('CW'*amp(cav_amp), 'storage', duration=cav_len)
            align('storage', 'qubit')
            with for_(i, 0, i < num_pi_pulses, i+1):
                play("res_pi", "qubit")
                align("qubit", "rr")
                discriminator.measure_state("clear", "out1", "out2", res, I=I)
                save(res, res_st)
                save(I, I_st)
                wait(250, 'rr')
                align('qubit', 'rr')

    with stream_processing():
        res_st.boolean_to_int().buffer(num_pi_pulses).save_all('res')
        I_st.buffer(num_pi_pulses).save_all('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(photon_counting, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(photon_counting, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    job.halt()
    #
    # data_path = "S:\\_Data\\210326 - QM_OPX\\data\\"
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'repeated_resolved_pi_pulse', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("cav_time", data=4*cav_len)
    #     f.create_dataset("cav_amp", data=cav_amp)
