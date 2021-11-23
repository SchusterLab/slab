from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 2), ("con1", 2, "con1", 1)], latency=230, noisePower=0.07 ** 2
    )
)

N = 1500
wait_time = 600
lsb = False
rr_qe = 'rr'
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm=qmm,
                                      config=config,
                                      update_tof=False,
                                      rr_qe='rr',
                                      path=f'ge_disc_params_{rr_qe}_new.npz',
                                      lsb=lsb)


def training_measurement(readout_pulse, use_opt_weights):
    if use_opt_weights:
        discriminator.measure_state(readout_pulse, "out1", "out2", res, I=I, adc=adc_st)
    else:
        measure(readout_pulse, 'rr', adc_st,
                demod.full("clear_integW1", I1, 'out1'),
                demod.full("clear_integW2", Q1, 'out1'),
                demod.full("clear_integW1", I2, 'out2'),
                demod.full("clear_integW2", Q2, 'out2')
                )

        if not lsb:
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
        else:
            assign(I, I1 - Q2)
            assign(Q, Q1 + I2)

use_opt_weights = False

with program() as training_program:
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed, value=0)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < N, n + 1):
        wait(wait_time//4, rr_qe)
        align("rr", "jpa_pump")
        play('pump_square', 'jpa_pump')
        training_measurement("clear", use_opt_weights=use_opt_weights)
        save(I, I_st)
        save(Q, Q_st)

        align("qubit", "rr", 'jpa_pump')

        wait(wait_time//4, "qubit")
        play("pi", "qubit")
        align("qubit", "rr", 'jpa_pump')
        play('pump_square', 'jpa_pump')
        training_measurement("clear", use_opt_weights=use_opt_weights)
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')
        adc_st.input1().with_timestamps().save_all("adc1")
        adc_st.input2().save_all("adc2")

    # discriminator.train(program=training_program, plot=True, dry_run=True, simulate=simulation_config,
#                     correction_method='robust')

discriminator.train(program=training_program, plot=False, correction_method='robust')

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
        align("rr", "jpa_pump")
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)

        align("qubit", "rr", "jpa_pump")

        wait(wait_time//4, "qubit")
        play("pi", "qubit")
        align("qubit", "rr", "jpa_pump")
        play('pump_square', 'jpa_pump')
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
#
# plt.show()

th_num = 5
# ths_vec = [500, 700, 980, 990, 997]
reps = 30000
th0 = (discriminator.mu[0][0] + discriminator.mu[1][0]) / 2
sigma = (discriminator.sigma[0] + discriminator.sigma[1]) / 2
mu_g = discriminator.mu[0][0]
mu_e = discriminator.mu[1][0]

# ################
# # active reset #
# ################
# with program() as get_thresholds:
#     k = declare(int)
#     n = declare(int)
#     idx = declare(int)
#     count_g = declare(int)
#     count_e = declare(int)
#     I = declare(fixed)
#     I_st = declare_stream()
#     ths = declare(fixed, size=th_num)
#     ths_st = declare_stream()
#     count_g_st = declare_stream()
#     count_e_st = declare_stream()
#     res = declare(bool)  # todo
#     m = declare(int)  # todo
#     a = Random()
#     b = declare(int)
#
#     assign(ths[0], th0)
#
#     with for_(k, 0, k < th_num, k + 1):
#         assign(count_g, 0)
#         assign(count_e, 0)
#
#         with for_(n, 0, n < reps, n + 1):
#
#             measure("readout", rr_qe, None)
#             align(rr_qe, "qubit")
#             play("pi2", "qubit")
#
#             with for_(idx, 0, idx < k, idx + 1):
#                 align(rr_qe, "qubit")
#                 discriminator.measure_state("readout", "out1", "out2", res, I=I)
#                 play("pi", "qubit", condition=(I > ths[idx]))
#
#             align(rr_qe, "qubit")
#             discriminator.measure_state("readout", "out1", "out2", res, I=I)
#             assign(count_e, Util.cond(I > ths[idx], count_e, count_e + 1))
#             assign(count_g, Util.cond(I <= ths[idx], count_g, count_g + 1))
#             save(I, I_st)
#
#         save(count_g, count_g_st)
#         save(count_e, count_e_st)
#         assign(ths[k], ths[0] + ((sigma ** 2) / 2 / (mu_g - mu_e)) * Math.ln(Math.div(count_e, count_g)))
#
#     with for_(m, 0, m < ths.length(), m + 1):
#         save(ths[m], ths_st)
#
#     with stream_processing():
#         ths_st.save_all('ths')
#         I_st.buffer(reps).save_all('I')
#         count_g_st.save_all('count_g')
#         count_e_st.save_all('count_e')
#
# job = qmm.simulate(config, get_thresholds, simulation_config)
# # qm = qmm.open_qm(config)
# # job = qm.execute(get_thresholds, duration_limit=0, data_limit=0)
# job.result_handles.wait_for_all_values()
# ths = job.result_handles.ths.fetch_all()
# I = job.result_handles.I.fetch_all()['value']
# for arg in range(th_num):
#     plt.figure()
#     plt.hist(I[arg], 100)
#     plt.axvline(x=ths[arg][0])
# plt.figure()
# plt.plot(ths)

