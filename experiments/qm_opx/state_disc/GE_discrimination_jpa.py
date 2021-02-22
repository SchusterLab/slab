from qm import SimulationConfig, LoopbackInterface
from state_disc.TwoStateDiscriminator import TwoStateDiscriminator
from configuration_IQ import config, qubit_LO, rr_LO, rr_IF
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()

LO_q = im['RF5']
LO_r = im['RF8']

LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(13)

reset_time = 500000
avgs = 3000
wait_time = 100
N = 3000

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    with for_(n, 0, n < N, n + 1):
        align("rr", "jpa_pump")
        play('CW'*amp(0.04), 'jpa_pump')
        measure("long_readout", "rr", "adc", demod.full("long_integW1", I1, "out1"),
                demod.full("long_integW2", Q1, "out1"),
                demod.full("long_integW1", I2, "out2"),
                demod.full("long_integW2", Q2, "out2"))

        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(reset_time//4, "rr")

        align("qubit", "rr")

    with for_(n, 0, n < N, n + 1):
        play("pi", "qubit")
        align("qubit", "rr")
        align("rr", "jpa_pump")
        play('CW'*amp(0.04), 'jpa_pump')

        measure("long_readout", "rr", "adc", demod.full("long_integW1", I1, "out1"),
                demod.full("long_integW2", Q1, "out1"),
                demod.full("long_integW1", I2, "out2"),
                demod.full("long_integW2", Q2, "out2"))

        assign(I, I1 + Q2)
        assign(Q, -Q1 + I2)
        save(I, 'I')
        save(Q, 'Q')
        wait(reset_time//4, "rr")

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, 'rr', 'qubit_ge_disc_params.npz')
discriminator.train(program=training_program, plot=True, correction_method='gmm')


with program() as ge_discrimination:

    n = declare(int)
    res = declare(bool)
    statistic = declare(fixed)

    with for_(n, 0, n < N, n + 1):

        seq0 = [0] * int(N)
        align("rr", "jpa_pump")
        play('CW'*amp(0.04), 'jpa_pump')

        discriminator.measure_state("long_readout", "out1", "out2", res, statistic=statistic)

        save(res, 'res')
        save(statistic, 'statistic')
        wait(reset_time//4, 'rr')

    with for_(n, 0, n < N, n + 1):

        seq0 = seq0 + [1] * int(N)
        play("pi", "qubit")
        align("qubit", "rr")
        align("rr", "jpa_pump")
        play('CW'*amp(0.04), 'jpa_pump')
        discriminator.measure_state("long_readout", "out1", "out2", res, statistic=statistic)

        save(res, 'res')
        save(statistic, 'statistic')
        wait(reset_time//4, "rr")


##############
# simulation #
##############
# qm = qmm.open_qm(config)
# job = qm.simulate(ge_discrimination, SimulationConfig(280000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)])))


#############
# execution #
#############
qm = qmm.open_qm(config)
job = qm.execute(ge_discrimination, duration_limit=0, data_limit=0)

# show diagrams:
result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']
statistic = result_handles.get('statistic').fetch_all()['value']

plt.figure()
plt.hist(statistic[np.array(seq0) == 0], 50)
plt.hist(statistic[np.array(seq0) == 1], 50)
plt.plot([discriminator.get_threshold()]*2, [0, 60], 'g')
plt.show()

# show confusion matrix:
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
