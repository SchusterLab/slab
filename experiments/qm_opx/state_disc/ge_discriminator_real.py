from qm import SimulationConfig, LoopbackInterface

from TwoStateDiscriminator import TwoStateDiscriminator
from configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=180000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.09**2
    )
)

Ng = 300
Ne = 300
g2e = 1000*0
e2g = 1000*0
wait_time = 10

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    with for_(n, 0, n < Ng - g2e, n + 1):

        measure("readout_pulse_g", "rr1a", "adc", demod.full("integW_cos", I1, "out1"),
                                                    demod.full("integW_sin", Q1, "out1"),
                                                    demod.full("integW_cos", I2, "out2"),
                                                    demod.full("integW_sin", Q2, "out2"))
        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(wait_time, "rr1a")

    with for_(n, 0, n < g2e, n + 1):

        measure("readout_pulse_e", "rr1a", "adc", demod.full("integW_cos", I1, "out1"),
                                                    demod.full("integW_sin", Q1, "out1"),
                                                    demod.full("integW_cos", I2, "out2"),
                                                    demod.full("integW_sin", Q2, "out2"))
        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(wait_time, "rr1a")

    with for_(n, 0, n < Ne - e2g, n + 1):

        measure("readout_pulse_e", "rr1a", "adc", demod.full("integW_cos", I1, "out1"),
                                                    demod.full("integW_sin", Q1, "out1"),
                                                    demod.full("integW_cos", I2, "out2"),
                                                    demod.full("integW_sin", Q2, "out2"))
        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(wait_time, "rr1a")

    with for_(n, 0, n < e2g, n + 1):

        measure("readout_pulse_g", "rr1a", "adc", demod.full("integW_cos", I1, "out1"),
                                                    demod.full("integW_sin", Q1, "out1"),
                                                    demod.full("integW_cos", I2, "out2"),
                                                    demod.full("integW_sin", Q2, "out2"))
        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(wait_time, "rr1a")


qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, 'rr1a', 'ge_disc_params.npz')
discriminator.train(program=training_program, plot=True, dry_run=True, simulate=simulation_config)

with program() as test_program:

    n = declare(int)
    res = declare(bool)
    statistic = declare(fixed)

    with for_(n, 0, n < Ng, n + 1):

        seq0 = [0] * int(Ng)
        discriminator.measure_state("readout_pulse_g", "out1", "out2", res, statistic=statistic)

        save(res, 'res')
        save(statistic, 'statistic')
        wait(wait_time, "rr1a")

    with for_(n, 0, n < Ne, n + 1):

        seq0 = seq0 + [1] * int(Ne)
        discriminator.measure_state("readout_pulse_e", "out1", "out2", res, statistic=statistic)

        save(res, 'res')
        save(statistic, 'statistic')
        wait(wait_time, "rr1a")


qm = qmm.open_qm(config)
job = qm.simulate(test_program, simulate=simulation_config)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']
statistic = result_handles.get('statistic').fetch_all()['value']

plt.figure()
plt.hist(statistic[np.array(seq0) == 0], 50)
plt.hist(statistic[np.array(seq0) == 1], 50)
plt.plot([discriminator.get_threshold()]*2, [0, 60], 'g')
plt.show()

p_s = np.zeros(shape=(2, 2))
for i in range(2):
    res_i = res[np.array(seq0) == i]
    p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

labels = ['g', 'e']
fig = plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
