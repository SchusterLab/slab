from configuration_IQ import config, rr_IF, rr_LO
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO = im['RF8']
spec = im['SA']
LO.set_frequency(rr_LO)
LO.set_ext_pulse(mod=False)
LO.set_power(13)

with program() as mixer_calibration:

    with infinite_loop_():
        play("CW"*amp(1.05), "rr")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(mixer_calibration, duration_limit=0, data_limit=0)

# qm.set_dc_offset_by_qe("rr", "Q", 0.0)
# qm.set_dc_offset_by_qe("rr", "I", 0.0)

delta_F = 10e6
spec.set_center_frequency(rr_LO + rr_IF)
spec.set_span(delta_F)
spec.set_resbw(100e3)
tr = spec.take_one()
freq, amp = tr[0], tr[1]
plt.plot(freq, amp)
# plt.axvline(x=rr_LO, linestyle='--', color='k')
# plt.axvline(x=rr_LO - rr_IF, linestyle='--', color='k')
# plt.axvline(x=rr_LO + rr_IF, linestyle='--', color='k')

def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

# qm.set_mixer_correction("mixer_RR", int(rr_IF), int(rr_LO), IQ_imbalance_corr(0.0, 0.0125*np.pi))