from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, disc_file_gauss
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from h5py import File
import os
from slab.dataanalysis import get_next_filename

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

N = 10000
wait_time = 500000

lsb = True
qmm = QuantumMachinesManager()

discriminator = TwoStateDiscriminator(qmm=qmm,
                                      config=config,
                                      update_tof=False,
                                      rr_qe='rr',
                                      path=disc_file_gauss,
                                      lsb=lsb)

use_opt_weights = False

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed, value=0)
    res = declare(bool)

    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < N, n + 1):

        wait(wait_time//4, "rr")
        measure("gaussian", "rr", adc_st,
                dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                dual_demod.full('sin', 'out1', 'cos', 'out2', Q))
        save(I, I_st)
        save(Q, Q_st)

        align("qubit_mode0", "rr")

        wait(wait_time//4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        measure("gaussian", "rr", adc_st,
                dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                dual_demod.full('sin', 'out1', 'cos', 'out2', Q))
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

        wait(wait_time//4, "rr")
        discriminator.measure_state("gaussian", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        align("qubit_mode0", "rr")

        wait(wait_time//4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        discriminator.measure_state("gaussian", "out1", "out2", res, I=I, Q=Q)
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
                             get_next_filename(data_path, 'histogram_disc_gauss', suffix='.h5'))
print(seq_data_file)

with File(seq_data_file, 'w') as f:
    f.create_dataset("I", data=I)
    f.create_dataset("Q", data=Q)
    f.create_dataset("res", data=res)
    f.create_dataset("seq0", data=seq0)
    f.create_dataset("avgs", data=N)
