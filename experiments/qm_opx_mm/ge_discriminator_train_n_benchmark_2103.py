# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# this file works with version 0.7.411 & gateway configuration of a single controller #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_2103 import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 2), ("con1", 2, "con1", 1)], latency=230, noisePower=0.07**2
    )
)

N = 100
wait_time = 10
lsb = True

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm=qmm,
                                      config=config,
                                      update_tof=False,
                                      rr_qe='rr1a',
                                      path='ge_disc_params.npz',
                                      lsb=lsb)
use_opt_weights = True

def training_measurement(readout_pulse, use_opt_weights):

    if use_opt_weights:
        discriminator.measure_state(readout_pulse, "out1", "out2", res, I=I, adc=adc_st)
    else:
        measure(readout_pulse, "rr1a", adc_st, demod.full("integW_cos", I1, "out1"),
                demod.full("integW_sin", Q1, "out1"),
                demod.full("integW_cos", I2, "out2"),
                demod.full("integW_sin", Q2, "out2"))

        if lsb == False:
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
        else:
            assign(I, I1 - Q2)
            assign(Q, Q1 + I2)

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed, value=0)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    res = declare(bool)

    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < N, n + 1):

        wait(wait_time, "rr1a")
        training_measurement("readout_pulse_g", use_opt_weights=use_opt_weights)
        save(I, I_st)
        save(Q, Q_st)

        wait(wait_time, "rr1a")
        training_measurement("readout_pulse_e", use_opt_weights=use_opt_weights)
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')
        adc_st.input1().with_timestamps().save_all("adc1")
        adc_st.input2().save_all("adc2")

discriminator.train(program=training_program, plot=True, dry_run=True, simulate=simulation_config, correction_method='robust')

with program() as benchmark_readout:

    n = declare(int)
    res = declare(bool)
    I = declare(fixed)
    Q = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        wait(wait_time, "rr1a")
        discriminator.measure_state("readout_pulse_g", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        wait(wait_time, "rr1a")
        discriminator.measure_state("readout_pulse_e", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        seq0 = [0, 1] * N

    with stream_processing():
        res_st.save_all('res')
        I_st.save_all('I')
        Q_st.save_all('Q')

qm = qmm.open_qm(config)
job = qm.simulate(benchmark_readout, simulate=simulation_config)

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
