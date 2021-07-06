from configuration_IQ import config,  rr_LO, pump_IF
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO = im['RF8']

LO.set_frequency(rr_LO)
LO.set_power(18)
LO.set_ext_pulse(mod=False)

with program() as mixer_calibration:
    with infinite_loop_():
        play("CW"*amp(1.0/2), "jpa_pump")
        # play("my_control_op"*amp(0.02), "qubit2")
        # align('qubit', 'qubit3')
        # play("my_control_op"*amp(0.02), "qubit3")
        # play("my_control_op"*amp(0.02), "qubit")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
# job = qm.simulate(mixer_calibration, SimulationConfig(5000))
# samples = job.get_simulated_samples()
# samples.con1.plot()

job = qm.execute(mixer_calibration, duration_limit=0, data_limit=0)
# qm.set_dc_offset_by_qe("qubit", "I", 0.0259)
# qm.set_dc_offset_by_qe("qubit", "Q", -0.0366)


# plt.axvline(x=nu_LO, linestyle='--', color='k')
# plt.axvline(x=nu_LO - ge_IF, linestyle='--', color='k')
# plt.axvline(x=nu_LO + ge_IF, linestyle='--', color='k')


qm.set_dc_offset_by_qe("jpa_pump", "I", 0.0095)
qm.set_dc_offset_by_qe("jpa_pump", "Q", -0.075)

def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

qm.set_mixer_correction("mixer_jpa", int(pump_IF), int(rr_LO), IQ_imbalance_corr(0.0,0.0*np.pi))
#
# delta_F = 500e6
# spec.set_center_frequency(nu_LO)
# spec.set_span(delta_F)
# spec.set_resbw(100e3)
# time.sleep(5)
# tr = spec.take_one()
# plt.plot(tr[0], tr[1])