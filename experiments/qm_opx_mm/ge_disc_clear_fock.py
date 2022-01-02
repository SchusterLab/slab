from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from h5py import File
import os
from slab.dataanalysis import get_next_filename
from fock_state_prep import oct_to_opx_amp

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

N = 5000
reset_time = int(7.5e6)
jpa_amp = 1.0

phi = 0.0

lsb = True
qmm = QuantumMachinesManager()

discriminator = TwoStateDiscriminator(qmm=qmm,
                                      config=config,
                                      update_tof=False,
                                      rr_qe='rr',
                                      path='ge_disc_params_fock.npz',
                                      lsb=lsb)

f_target=1

pulse_len = oct_to_opx_amp(opx_config=config, fock_state=f_target)//2

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed, value=0)
    res = declare(bool)

    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < N, n + 1):

        wait(reset_time// 4, 'storage_mode1')# wait for the storage to relax, several T1s
        align('storage_mode1', 'qubit_mode0')
        #########################
        play("soct", 'storage_mode1', duration=pulse_len)
        play("qoct", 'qubit_mode0', duration=pulse_len)
        #########################
        align("rr", "storage_mode1")
        measure("clear", "rr", adc_st,
                dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))

        save(I, I_st)
        save(Q, Q_st)

        align('storage_mode1', 'qubit_mode0', 'rr')

        # wait(reset_time//4, 'storage_mode1')
        # #########################
        # play("soct", 'storage_mode1', duration=pulse_len)
        # play("qoct", 'qubit_mode0', duration=pulse_len)
        # #########################
        # align('storage_mode1', 'qubit_mode0')
        wait(1000, 'qubit_mode0')
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        measure("clear", "rr", adc_st,
                dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')
        adc_st.input1().with_timestamps().save_all("adc1")
        adc_st.input2().save_all("adc2")

# training + testing to get fidelity:
discriminator.train(program=training_program, plot=True, dry_run=False, use_hann_filter=True, correction_method='robust')

with program() as benchmark_readout:

    n = declare(int)
    res = declare(bool)
    I = declare(fixed)
    Q = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        wait(reset_time//4, "rr")
        discriminator.measure_state("clear", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        align("qubit_mode0", "rr")

        wait(reset_time//4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        discriminator.measure_state("clear", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        seq0 = [0, 1] * N

    with stream_processing():
        res_st.save_all('res')
        I_st.save_all('I')
        Q_st.save_all('Q')

qm = qmm.open_qm(config)
job = qm.execute(benchmark_readout, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']
I = result_handles.get('I').fetch_all()['value']
Q = result_handles.get('Q').fetch_all()['value']

plt.figure()
plt.hist(I[np.array(seq0) == 0], 50)
plt.hist(I[np.array(seq0) == 1], 50)
plt.plot([discriminator.get_threshold()] * 2, [0, 60], 'g')
plt.show()

plt.figure()
plt.plot(I, Q, '.')

p_s = np.zeros(shape=(2, 2))
for i in range(2):
    res_i = res[np.array(seq0) == i]
    p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

labels = ['g', 'e']
plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')

ax.set_xlabel('Predicted labels')
ax.set_ylabel('Prepared labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

plt.show()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'histogram_disc', suffix='.h5'))
print(seq_data_file)

# with File(seq_data_file, 'w') as f:
#     f.create_dataset("I", data=I)
#     f.create_dataset("Q", data=Q)
#     f.create_dataset("res", data=res)
#     f.create_dataset("seq0", data=seq0)
#     f.create_dataset("avgs", data=N)

# """Extracting the qubit thermal population from Gaussian fitting of the histograms"""
# def gaus(x, a0, x0, sigma, a1, x1):
#     return a0*np.exp(-(x-x0)**2/(2*sigma**2)) + a1*np.exp(-(x-x1)**2/(2*sigma**2))
# from scipy.optimize import curve_fit
# y, x = np.histogram(I[np.array(seq0) == 0], 50)
# popt, pcov = curve_fit(gaus, x[:-1], y, p0=[1, 0.003, 0.001, 0, -0.002])