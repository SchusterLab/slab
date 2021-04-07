# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# this file works with version 0.7.411 & gateway configuration of a single controller #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator import TwoStateDiscriminator
from configuration_IQ import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from h5py import File


simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.07**2
    )
)

N = 3000
wait_time = 500000

opt_for_train = False

with program() as training_program:

    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < N, n + 1):

        wait(wait_time, "rr")

        measure("clear", "rr", adc_st,
                demod.full("clear_integW1", I1, 'out1'),
                demod.full("clear_integW2", Q1, 'out1'),
                demod.full("clear_integW1", I2, 'out2'),
                demod.full("clear_integW2", Q2, 'out2'))

        assign(I, I1 - Q2)
        assign(Q, Q1 + I2)
        save(I, I_st)
        save(Q, Q_st)

        align("qubit", "rr")

        wait(wait_time, "qubit")
        play("pi", "qubit")
        align("qubit", "rr")
        measure("clear", "rr", adc_st,
                demod.full("clear_integW1", I1, 'out1'),
                demod.full("clear_integW2", Q1, 'out1'),
                demod.full("clear_integW1", I2, 'out2'),
                demod.full("clear_integW2", Q2, 'out2'))

        assign(I, I1 - Q2)
        assign(Q, Q1 + I2)
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')
        adc_st.input1().with_timestamps().save_all("adc1")
        adc_st.input2().save_all("adc2")


# training + testing to get fidelity:
for i in [12]:
    qmm = QuantumMachinesManager()
    opt_readout = "C:\\_Lib\\python\\slab\\experiments\\qm_opx\\pulses\\" + str(i).zfill(5) + "_readout_optimal_pulse.h5"

    with File(opt_readout,'r') as a:
        opt_amp = np.array(a['I_wf'])
    opt_len = len(opt_amp)
    config['pulses']['clear_pulse']['length']=opt_len
    config['waveforms']['opt_wf']['samples']=0.45 * opt_amp
    config['integration_weights']['clear_integW1']['cosine']=[2.0] * int(opt_len / 4 )
    config['integration_weights']['clear_integW1']['sine']=[0.0] * int(opt_len / 4 )
    config['integration_weights']['clear_integW2']['cosine']=[0.0] * int(opt_len / 4 )
    config['integration_weights']['clear_integW2']['sine']=[2.0] * int(opt_len / 4 )
    discriminator = TwoStateDiscriminator(qmm, config, False, 'rr', 'ge_disc_params_opt.npz')
    discriminator.train(program=training_program, plot=True, dry_run=False, use_hann_filter=False)

    with program() as test_program:

        n = declare(int)
        res = declare(bool)
        statistic = declare(fixed)

        res_st = declare_stream()
        statistic_st = declare_stream()

        with for_(n, 0, n < N, n + 1):

            wait(wait_time, "rr")
            discriminator.measure_state("clear", "out1", "out2", res, statistic=statistic)
            save(res, res_st)
            save(statistic, statistic_st)


            align("qubit", "rr")

            wait(wait_time, "qubit")
            play("pi", "qubit")
            align("qubit", "rr")
            discriminator.measure_state("clear", "out1", "out2", res, statistic=statistic)
            save(res, res_st)
            save(statistic, statistic_st)

            seq0 = [0, 1] * N

        with stream_processing():
            res_st.save_all('res')
            statistic_st.save_all('statistic')

    qm = qmm.open_qm(config)
    job = qm.execute(test_program, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()['value']
    statistic = result_handles.get('statistic').fetch_all()['value']

    plt.figure()
    plt.hist(statistic[np.array(seq0) == 0], 50)
    plt.hist(statistic[np.array(seq0) == 1], 50)
    plt.plot([discriminator.get_threshold()] * 2, [0, 60], 'g')
    plt.show()

    p_s = np.zeros(shape=(2, 2))
    for i in range(2):
        res_i = res[np.array(seq0) == i]
        p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

    labels = ['g', 'e']
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.show()
