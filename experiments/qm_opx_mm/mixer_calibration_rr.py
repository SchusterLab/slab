from configuration_IQ import config, rr_IF, rr_LO, pump_IF
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
from h5py import File
import os
from slab.dataanalysis import get_next_filename
im = InstrumentManager()
# LO = im['RF8']
spec = im['SA']
# LO.set_frequency(rr_LO)
# LO.set_ext_pulse(mod=False)
# LO.set_power(16)

with program() as mixer_calibration:

    with infinite_loop_():
        play("readout"*amp(1.0), "rr")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(mixer_calibration, duration_limit=0, data_limit=0)

# qm.set_dc_offset_by_qe("rr", "Q", 0.0)
# qm.set_dc_offset_by_qe("rr", "I", 0.0)

delta_F = 50e3
rbw = 1e3
vbw = 10

spec.set_center_frequency(rr_LO-rr_IF)
spec.set_span(delta_F)
spec.set_resbw(rbw)
spec.set_vidbw(vbw)
# spec.set_query_sleep(50)
# spec.set_average_state(True)
# spec.set_averages(5)
# # # time.sleep(50)
# tr = spec.take_one()
# plt.plot(tr[0], tr[1], '.--')
# freq, amp = tr[0], tr[1]

# plt.axvline(x=rr_LO, linestyle='--', color='k')
# plt.axvline(x=rr_LO - rr_IF, linestyle='--', color='k')
# plt.axvline(x=rr_LO + rr_IF, linestyle='--', color='k')

def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

# qm.set_mixer_correction("mixer_RR", int(rr_IF), int(rr_LO), IQ_imbalance_corr(-0.01, 0.013*np.pi))

# delta_F = 300e6
# spec.set_center_frequency(rr_LO)
# spec.set_span(delta_F)
# # spec.set_resbw(100e3)
# spec.set_averages(10)
# tr = spec.take_one()
# freq, amp = tr[0], tr[1]
# plt.plot(freq, amp)
# a1 = amp[np.argmin(abs(freq-rr_LO+rr_IF))]
# a2 = amp[np.argmin(abs(freq-rr_LO))]
# a3 = amp[np.argmin(abs(freq-rr_LO-rr_IF))]
#
# print([a1, a2, a3])
# path = os.getcwd()
# data_path = os.path.join(path, "data/thesis")
# seq_data_file = os.path.join(data_path,
#                              get_next_filename(data_path, 'noise_temp_sa_amp_1.0_pump_on', suffix='.h5'))
# print(seq_data_file)
# with File(seq_data_file, 'w') as f:
#     f.create_dataset("power", data=amp)
#     f.create_dataset("freq", data=freq)
#     f.create_dataset("res_bw", data=rbw)
#     f.create_dataset("vid_bw", data=vbw)
#     f.create_dataset("delta", data=0e3) #pump detuning from the rr
#     f.create_dataset("pump", data=-22.3) #pump power in dBm