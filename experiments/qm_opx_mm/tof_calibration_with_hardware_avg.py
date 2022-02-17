from configuration_IQ import config, rr_IF, rr_LO
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager

simulation = 0
with program() as digital_train:

    with infinite_loop_():
        play("CW", "rr")  # 600ns
        # wait(100, "rr")   # 400ns

with program() as tof_calibration:
    adc_st = declare_stream(adc_trace=True)
    i = declare(int)
    with for_(i, 0, i < 10000, i+1):
        wait(20000//4, 'rr')
        reset_phase('rr')
        # reset_phase('jpa_pump')
        # align('rr', 'jpa_pump')
        # play('pump_square', 'jpa_pump')
        measure('readout', "rr", adc_st) # 400ns
        # measure('clear', "rr", adc_st) # 400ns


    with stream_processing():
        adc_st.input1().average().save("adcI")
        adc_st.input2().average().save("adcQ")
        adc_st.input1().fft().average().save('fft')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
if simulation:
    job = qm.simulate(tof_calibration, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(tof_calibration, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    adcI_handle = res_handles.get("adcI")
    adcQ_handle = res_handles.get("adcQ")
    I_avg = adcI_handle.fetch_all()
    Q_avg = adcQ_handle.fetch_all()
    print(np.mean(np.abs(I_avg)))
    print(np.mean(np.abs(Q_avg)))
    plt.figure()
    plt.plot(I_avg)
    plt.plot(Q_avg)
    plt.axhline(y=0)
    plt.show()

    # plt.figure()
    # plt.plot(res_handles.get('fft').fetch_all())
    # plt.show()