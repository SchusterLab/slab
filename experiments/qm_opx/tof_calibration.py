from configuration_IQ import config
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO = im['RF8']

# nu_q = 4.748488058822229e9
nu_q = 8.0518e9

nu_IF = 100e6
nu_LO = nu_q - nu_IF
LO.set_frequency(nu_LO)
LO.set_power(18)
LO.set_output(True)
LO.set_ext_pulse(mod=False)

with program() as digital_train:

    with infinite_loop_():
        play("CW", "rr")  # 600ns
        wait(100, "rr")   # 400ns

with program() as tof_calibration:
    adc_st = declare_stream(adc_trace=True)
    i = declare(int)
    update_frequency("rr", 1e6)
    with for_(i, 0, i < 1, i+1):

        wait(10000//4, "rr") # off 1.6 micro
        measure('long_readout'*amp(0.3), "rr", adc_st) # 400ns

    with stream_processing():
        adc_st.input1().average().save("adcI")
        adc_st.input2().average().save("adcQ")

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
# fig = plt.figure(figsize=(10, 5))
plt.plot(valI)
# plt.figure()
plt.plot(valQ)
# for i in range(10):
#     adcI_.append(valI[i][0])
#     adcQ_.append(valQ[i][0])
#
#     ax = fig.add_subplot(121)
#     ax.plot(adcI_[i], label='{}'.format(i))
#
#     ax = fig.add_subplot(122)
#     ax.plot(adcQ_[i], label='{}'.format(i))

# plt.legend()
plt.show()
