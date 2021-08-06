from configuration_IQ import config, qubit_LO, rr_LO, storage_LO
import numpy as np
from slab import*
from slab.instruments import instrumentmanager

im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
LO_s = im['SC1E48']
flux = im['YOKO1']

##################
# ramsey_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(15)
LO_q.set_output(True)

LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)
LO_r.set_output(True)

LO_s.set_frequency(storage_LO)
LO_s.set_power(13)
LO_s.set_output_state(True)

flux.set_current(0.3988e-3)