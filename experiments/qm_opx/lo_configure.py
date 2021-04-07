from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq, storage_LO
import numpy as np
from slab import*
from slab.instruments import instrumentmanager

im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
LO_s = im['sccav']
##################
# ramsey_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)

LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

LO_s.set_frequency(storage_LO)
LO_s.set_power(13)
