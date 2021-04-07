from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
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

a_min = 0.0
a_max = 1.0
da = 0.01
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

with program() as qubit_temp:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    a = declare(fixed) #array of time delays
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()


    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            # active_reset(biased_th_g)
            # align("qubit", 'rr')
            wait(reset_time//4, "qubit")
            play("pi", "qubit")
            align("qubit", "qubit_ef")
            play("gaussian"*amp(a), "qubit_ef")
            align("qubit_ef", "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1+Q2)
            assign(Q, I2-Q1)

            save(I, Ig_st)
            save(Q, Qg_st)

            align("qubit_ef", 'rr')
            wait(reset_time//4, "qubit_ef")
            play("gaussian"*amp(a), "qubit_ef")
            align("qubit_ef", "rr")
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1+Q2)
            assign(Q, I2-Q1)

            save(I, Ie_st)
            save(Q, Qe_st)

    with stream_processing():
        Ig_st.buffer(len(amps)).average().save('Ig')
        Qg_st.buffer(len(amps)).average().save('Qg')
        Ie_st.buffer(len(amps)).average().save('Ie')
        Qe_st.buffer(len(amps)).average().save('Qe')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(qubit_temp, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(qubit_temp, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    Ig = res_handles.get('Ig').fetch_all()
    Qg = res_handles.get('Qg').fetch_all()
    Ie = res_handles.get('Ie').fetch_all()
    Qe = res_handles.get('Qe').fetch_all()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'qubit_temp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Ig", data=Ig)
        f.create_dataset("Qg", data=Qg)
        f.create_dataset("Ie", data=Ie)
        f.create_dataset("Qe", data=Qe)
        f.create_dataset("amps", data=amps)
