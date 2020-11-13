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
atten = im["atten"]
# nu_q = 4.748488058822229e9
nu_q = 8.0518e9

nu_IF = 100e6
nu_LO = nu_q - nu_IF
LO.set_frequency(nu_LO)
LO.set_power(18)
LO.set_output(True)
LO.set_ext_pulse(mod=False)
# atten.set_attenuator(31.5)

with program() as mixer_calibration:

    with infinite_loop_():
        # play("CW", "qubit")
        play("CW"*amp(0.5), "rr")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(mixer_calibration, duration_limit=0, data_limit=0)

# qm.set_dc_offset_by_qe("rr", "Q", -0.035)
delta_F = 0.5e9
spec.set_center_frequency(nu_LO)
spec.set_span(delta_F)
spec.set_resbw(100e3)
def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

# qm.set_mixer_correction("mixer_RR", int(rr_IF), int(rr_LO), IQ_imbalance_corr(-0.05,-0.005*np.pi))