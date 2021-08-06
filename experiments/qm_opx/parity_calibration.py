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
# ramsey_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)


avgs = 1000
reset_time = 5000000
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

chi_mem_qubit = 1.13e6
t_min = int((np.pi/chi_mem_qubit)*1e9/4 * 0.0)
t_max = int((np.pi/chi_mem_qubit)*1e9/4 * 1.5)
dt = 10
t_vec = np.arange(t_min, t_max, dt)

a_min = 0.05
a_max = 0.18
da = 0.03
a_vec = np.arange(a_min, a_max+da/2, da)

with program() as parity:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    a = declare(fixed)
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res_minus = declare(bool)
    res_plus = declare(bool)
    Iplus = declare(fixed)
    Iminus = declare(fixed)
    I = declare(fixed)
    t = declare(int)

    res_minus_st = declare_stream()
    res_plus_st = declare_stream()
    I_plus_st = declare_stream()
    I_minus_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a<a_max + da/2, a+da):

            with for_(t, t_min, t < t_max, t + dt):

                wait(reset_time//4, 'storage')
                # align('rr', 'storage')
                # active_reset(biased_th_g)
                # align('storage', 'rr')
                play("saturation"*amp(a), "storage", duration=1000)
                align('qubit', 'storage')
                play("pi2", "qubit")
                wait(t, "qubit")
                play("minus_pi2", "qubit")
                align("qubit", "rr")
                discriminator.measure_state("clear", "out1", "out2", res_minus, I=Iminus)

                align('rr', 'storage')
                wait(reset_time//4, 'storage')
                # align('rr', 'storage')
                # active_reset(biased_th_g)
                # align('storage', 'rr')
                play("saturation"*amp(a), "storage", duration=1000)
                align('qubit', 'storage')
                play("pi2", "qubit")
                wait(t, "qubit")
                play("pi2", "qubit")
                align("qubit", "rr")
                discriminator.measure_state("clear", "out1", "out2", res_plus, I=Iplus)

                save(res_minus, res_minus_st)
                save(res_plus, res_plus_st)
                save(Iminus, I_minus_st)
                save(Iplus, I_plus_st)


    with stream_processing():

        (res_minus_st.boolean_to_int()-res_plus_st.boolean_to_int()).buffer(len(a_vec), len(t_vec)).average().save('S12')
        res_minus_st.boolean_to_int().buffer(len(a_vec), len(t_vec)).average().save('res_minus')
        res_plus_st.boolean_to_int().buffer(len(a_vec), len(t_vec)).average().save('res_plus')
        I_plus_st.buffer(len(a_vec), len(t_vec)).average().save('I_plus')
        I_minus_st.buffer(len(a_vec), len(t_vec)).average().save('I_minus')


qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(parity, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(parity, duration_limit=0, data_limit=0)


result_handles = job.result_handles
S12_handle = result_handles.get("S12")
S12_handle.wait_for_values(1)
plt.figure()
while(result_handles.is_processing()):
    S12 = S12_handle.fetch_all()
    plt.pcolor(S12)
    plt.pause(5)

result_handles.wait_for_all_values()
# res = result_handles.get('res').fetch_all()

res_plus_handle = result_handles.get("res_plus").fetch_all()
res_minus_handle = result_handles.get("res_minus").fetch_all()
I_plus_handle = result_handles.get("I_plus").fetch_all()
I_minus_handle = result_handles.get("I_minus").fetch_all()



times = 4*t_vec /1e3
path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'parity_calibration', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Qplus", data=res_plus_handle)
    f.create_dataset("Iplus", data=I_plus_handle)
    f.create_dataset("Qminus", data=res_minus_handle)
    f.create_dataset("Iminus", data=I_minus_handle)
    f.create_dataset("time", data=times)
    f.create_dataset("amps", data=a_vec)
