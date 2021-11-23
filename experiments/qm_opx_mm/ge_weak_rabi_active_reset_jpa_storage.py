from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
from configuration_IQ import config, ge_IF, two_chi, biased_th_g_jpa, pi_len_resolved
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename


simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)
def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_05_20_cavity_square.h5'

    with File(fn_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = omegas
    a_s = amps

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(a_s, o_s)

    omega_desired = transfer_fn(cav_amp)

    pulse_length = (alpha/omega_desired)
    """Returns time in units of 4ns for FPGA"""
    return abs(pulse_length)//4+1

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(1000//4, "jpa_pump")
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
reset_time = int(3.5e6)
N = 1000

t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time

a_min= 0.0
a_max = 0.02
da = 0.0002
a_vec = np.arange(a_min, a_max + da/2, da)

with program() as power_rabi:

    n = declare(int)
    a = declare(fixed)
    res = declare(bool)
    res_reset = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        with for_(a, a_min, a< a_max + da/2, a + da):

            update_frequency("qubit", ge_IF)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(1.143))
            align("storage", "qubit")
            play("res_pi"*amp(2.0), "qubit")
            align("storage", "qubit")
            play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(-0.58)) #249

            update_frequency("qubit", ge_IF+two_chi)
            align("storage", "qubit")
            play('gaussian'*amp(a), 'qubit', duration=pi_len_resolved//4)
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(a_vec)).average().save('res')
        I_st.buffer(len(a_vec)).average().save('I')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
# result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

plt.figure()
plt.plot(a_vec, res, '.-')

job.halt()


path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'weak_rabi_storage', suffix='.h5'))
print(seq_data_file)
with File(seq_data_file, 'w') as f:
    f.create_dataset("Q", data=res)
    f.create_dataset("I", data=I)
    f.create_dataset("amps", data=a_vec)