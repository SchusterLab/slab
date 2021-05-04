from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from tqdm import tqdm
from h5py import File
import os
from slab.dataanalysis import get_next_filename
"""Binary decomposition with qubit resolved spectroscopy"""

t_chi = int(0.5*1e9/1.118e6) #qubit rotates by pi in this time
cav_len = 600
cav_amp = 0.5 # 0.08

f_min = -9e6
f_max = 1e6
df = 100e3
f_vec = np.arange(f_min, f_max + df/2, df)

avgs = 1000
reset_time = int(5e6)
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(5000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

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

with program() as binary_decomposition:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    bit1 = declare(bool)
    bit2 = declare(bool)
    bit3 = declare(bool)

    I = declare(fixed)

    bit1_st = declare_stream()
    bit2_st = declare_stream()
    bit3_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):
        wait(reset_time//4, 'storage')
        # align('storage', 'rr', 'jpa_pump', 'qubit')
        # active_reset(biased_th_g_jpa)
        # align('storage', 'rr', 'jpa_pump', 'qubit')
        play('CW'*amp(cav_amp), 'storage', duration=cav_len)
        align('storage', 'qubit')

        play("pi2", "qubit") # unconditional
        wait(t_chi//4, "qubit")
        frame_rotation(np.pi, 'qubit') #
        play("pi2", "qubit")

        align('qubit', 'rr', 'jpa_pump')
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", bit1, I=I)
        save(bit1, bit1_st)

        reset_frame("qubit")
        wait(1000//4, "rr")
        align("qubit", "rr", 'jpa_pump')

        play("pi2", "qubit") # unconditional
        wait(t_chi//4//2, "qubit")
        with if_(bit1==0):
            frame_rotation(np.pi, 'qubit')
            play("pi2", "qubit")
        with else_():
            frame_rotation(3/2*np.pi, 'qubit')
            play("pi2", "qubit")
        align('qubit', 'rr', 'jpa_pump')
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", bit2, I=I)
        save(bit2, bit2_st)

        reset_frame("qubit")
        wait(1000//4, "rr")
        align("qubit", "rr", 'jpa_pump')

        play("pi2", "qubit") # unconditional
        wait(t_chi//4//4, "qubit")

        """How to incorporate higher order chi correction?"""

        with if_(bit1==0 and bit2==0):
            frame_rotation(np.pi, 'qubit')
            play("pi2", "qubit")
        with if_(bit1==0 and bit2==1):
            frame_rotation(np.pi/2, 'qubit')
            play("pi2", "qubit")
        with if_(bit1==1 and bit2==0):
            frame_rotation(np.pi + np.pi/2 + np.pi/4, 'qubit')
            play("pi2", "qubit")
        with if_(bit1==1 and bit2==1):
            frame_rotation(np.pi*2 + np.pi/4, 'qubit')
            play("pi2", "qubit")
        align('qubit', 'rr', 'jpa_pump')
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
        save(bit3, bit3_st)

    with stream_processing():
        bit1_st.boolean_to_int().save_all('bit1')
        bit2_st.boolean_to_int().save_all('bit2')
        bit3_st.boolean_to_int().save_all('bit3')

qm = qmm.open_qm(config)

job = qm.execute(binary_decomposition, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
bit1 = result_handles.get('bit1').fetch_all()['value']
bit2 = result_handles.get('bit2').fetch_all()['value']
bit3 = result_handles.get('bit3').fetch_all()['value']

num = bit1 + 2*bit2 + 4*bit3

p_cav = []

for i in range(2**3):
    p_cav.append(np.sum(num==i)*100/avgs)
    print(" n = {} => {}".format(i, np.sum(num==i)*100/avgs))

job.halt()

with program() as storage_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ge_IF + f_min, f < ge_IF + f_max + df/2, f + df):

            update_frequency("qubit", f)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            play("CW"*amp(cav_amp), "storage", duration=cav_len)
            align("storage", "qubit")
            play("res_pi", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
        I_st.buffer(len(f_vec)).average().save('I')

job = qm.execute(storage_spec, duration_limit=0, data_limit=0)
result_handles = job.result_handles
result_handles.wait_for_all_values()

res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'binary_decomp_3bits_test', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("p_cav", data=p_cav)
    f.create_dataset("amp", data=cav_amp)
    f.create_dataset("time", data=cav_len*4)
    f.create_dataset("I", data=I)
    f.create_dataset("Q", data=res)
    f.create_dataset("freq", data=f_vec)
