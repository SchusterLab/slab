from configuration_IQ import config, ge_IF, ef_IF, qubit_LO, qubit_freq
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
# im = InstrumentManager()
# LO = im['RF5']
# spec = im['SA']

# nu_q = qubit_freq
# nu_IF = ge_IF
# nu_LO = qubit_LO
# LO.set_frequency(nu_LO)
# LO.set_power(18)
# LO.set_output(True)
# LO.set_ext_pulse(mod=False)

with program() as mixer_calibration:
    with infinite_loop_():
        play("CW"*amp(0.5), "qubit")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

job = qm.execute(mixer_calibration, duration_limit=0, data_limit=0)
# delta_F = 1e9
# spec.set_center_frequency(nu_LO)
# spec.set_span(delta_F)
# spec.set_resbw(100e3)
# time.sleep(5)
# tr = spec.take_one()
# plt.plot(tr[0], tr[1])
# plt.axvline(x=nu_LO, linestyle='--', color='k')
# plt.axvline(x=nu_LO - ge_IF, linestyle='--', color='k')
# plt.axvline(x=nu_LO + ge_IF, linestyle='--', color='k')

# qm.set_dc_offset_by_qe("qubit", "I", 0.0066)
# qm.set_dc_offset_by_qe("qubit", "Q", -0.0175)

def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

# qm.set_mixer_correction("mixer_qubit", int(ge_IF), int(qubit_LO), IQ_imbalance_corr(-0.0, 0.0 * np.pi))