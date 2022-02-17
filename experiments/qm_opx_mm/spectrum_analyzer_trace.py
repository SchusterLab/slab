"""
Created on Jan 2022

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, rr_IF, rr_LO, pump_IF
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
from h5py import File
import os
from slab.dataanalysis import get_next_filename
im = InstrumentManager()
spec = im['SA']

delta_F = 50e3
rbw = 1e3
vbw = 10

spec.set_center_frequency(rr_LO-rr_IF)
spec.set_span(delta_F)
spec.set_resbw(rbw)
spec.set_vidbw(vbw)
"""It would be good to figure out how 
long each trace takes and set_query_sleep accordingly """
spec.set_query_sleep(50)
spec.set_average_state(True)
spec.set_averages(5)
# # time.sleep(50)
tr = spec.take_one()

plt.figure()
plt.plot(tr[0], tr[1], '.--')
plt.show()
freq, amp = tr[0], tr[1]