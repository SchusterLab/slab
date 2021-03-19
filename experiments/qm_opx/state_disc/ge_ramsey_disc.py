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

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)


qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params.npz')

ramsey_freq = 100e3
omega = 2*np.pi*ramsey_freq

dt = 250

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 0
T_max = 30000
times = np.arange(T_min, T_max + dt/2, dt)
avgs = 1000
reset_time = 500000

with program() as power_rabi:

    n = declare(int)
    t = declare(int)
    phi = declare(fixed)
    res = declare(bool)
    statistic = declare(fixed)

    res_st = declare_stream()
    statistic_st = declare_stream()

    with for_(n, 0, n < avgs, n + 1):
        assign(phi, 0)
        with for_(t, T_min, t < T_max + dt/2, t + dt):
            reset_frame("qubit", "rr")
            wait(reset_time//4, "qubit")
            play("pi2", "qubit")
            wait(t, "qubit")
            frame_rotation_2pi(phi, "qubit")
            play("pi2", "qubit")
            align("qubit", "rr")
            discriminator.measure_state("long_readout", "out1", "out2", res, statistic=statistic)

            save(res, res_st)
            save(statistic, statistic_st)


    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        statistic_st.buffer(len(times)).average().save('statistic')

qm = qmm.open_qm(config)
job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
statistic = result_handles.get('statistic').fetch_all()

plt.plot(times, res, 'b*')
plt.figure()
plt.plot(times, statistic, 'r*')