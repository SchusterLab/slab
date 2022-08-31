from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, st_self_kerr, storage_IF, two_chi_2, pi_amp_resolved, pi_len_resolved
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq, oct_to_opx_amp_test

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
N = 1000

f_state = 3
reset_time = int((f_state+0.5)*7.5e6)
simulation = 0

pulse_filename = './oct_pulses/g'+str(f_state)+'.h5'

if f_state==0 or f_state==1 or f_state==2:
    pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)
else:
    pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)//2

a_min= 0.5*pi_amp_resolved
a_max = 1.5*pi_amp_resolved

da = (a_max - a_min)/100
a_vec = np.arange(a_min, a_max + da/2, da)

f_min = -3.9e6
f_max = -3.6e6
df = 10e3
f_vec = np.arange(f_min, f_max + df/2, df)
n_pi_pulses = 1
with program() as power_rabi:

    n = declare(int)
    bit = declare(bool)
    I = declare(fixed)
    a = declare(fixed)
    i = declare(int)
    f = declare(int)

    res_st = declare_stream()
    I_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            with for_(f, f_min + ge_IF[0], f < f_max + ge_IF[0] + df/2, f + df):

                wait(reset_time//4, 'storage_mode1')
                update_frequency('qubit_mode0', ge_IF[0])
                update_frequency('storage_mode1', storage_IF[1])
                align('storage_mode1', 'qubit_mode0')
                ########################
                # snap_seq(fock_state=f_state)
                # """OCT pulses to create Fock states"""
                play("soct", 'storage_mode1', duration=pulse_len)
                play("qoct", 'qubit_mode0', duration=pulse_len)
                align('storage_mode1', 'qubit_mode0')
                # update_frequency('qubit_mode0', ge_IF[0] + (f_state)*two_chi[1]+ (f_state)*(f_state-1)*two_chi_2)
                update_frequency('qubit_mode0', f)
                with for_(i, 0, i < n_pi_pulses, i+1):
                    play('gaussian'*amp(a), 'qubit_mode0', duration=pi_len_resolved//4)
                # play('gaussian'*amp(a), 'qubit_mode0', duration=pi_len_resolved//4)
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)

                save(bit, res_st)
                save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(a_vec), len(f_vec)).average().save('res')
        I_st.buffer(len(a_vec), len(f_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
# result_handles.wait_for_all_values()
# res = result_handles.get('res').fetch_all()
# I = result_handles.get('I').fetch_all()
#
# plt.figure()
# plt.plot(a_vec, res, '.-')
# plt.show()
#
# job.halt()
#
#
path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'weak_rabi_chevron_fock', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Q", data=res)
    f.create_dataset("I", data=I)
    f.create_dataset("amps", data=a_vec)
    f.create_dataset("freqs", data=f_vec )
    f.create_dataset("two_chi", data=two_chi[1])
    f.create_dataset("chi2", data=two_chi_2)
    f.create_dataset("fock_num", data=f_state)

res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

plt.figure()
plt.pcolormesh(f_vec, a_vec, res, shading='nearest', cmap='RdBu')
plt.axhline(y=0.0082, linestyle='--', color='y')
plt.axvline(x=f_state*two_chi[1], linestyle='--', color='y')
plt.colorbar()
plt.show()
