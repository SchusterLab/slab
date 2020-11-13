from configuration_IQ import config
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np


with program() as digital_train:

    with infinite_loop_():
        play("CW", "rr")  # 600ns
        wait(100, "rr")   # 400ns


with program() as tof_calibration:
    adc_st = declare_stream(adc_trace=True)
    i = declare(int)
    with for_(i, 0, i < 10, i+1):
        wait(300, "rr") # off 1.6 micro
        measure('readout', "rr", adc_st) # 400ns

    with stream_processing():
        adc_st.input1().save_all("adcI")
        adc_st.input2().save_all("adcQ")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(tof_calibration, duration_limit=0, data_limit=0)

res_handles = job.result_handles
res_handles.wait_for_all_values()
adcI_handle = res_handles.get("adcI")
adcQ_handle = res_handles.get("adcQ")

adcI_ = []
valI = adcI_handle.fetch_all()
adcQ_ = []
valQ = adcQ_handle.fetch_all()
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    adcI_.append(valI[i][0])
    adcQ_.append(valQ[i][0])

print (np.array(adcI_).shape)

adc_I_avg =  np.mean(np.array(adcI_),axis=0)
adc_Q_avg =  np.mean(np.array(adcQ_),axis=0)

ax = fig.add_subplot(111)
ax.plot(adc_I_avg, label='I')
ax.plot(adc_Q_avg, label='Q')



plt.legend()
plt.show()
