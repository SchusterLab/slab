# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:58:02 2011

@author: Dai
"""
from instruments import *
from rfgenerators import *
from spectrum_analyzer import *
import matplotlib.pyplot as plt
import time
from numpy import *

sa = SpectrumAnalyzer(address='128.135.35.166')
#sa = SpectrumAnalyzer(protocol='serial', port=2)

rf = E8257D(address='rfgen1.circuitqed.com')
lo = E8257D(address='rfgen2.circuitqed.com')

rf.set_output(False)
lo.set_output(False)

freq = 5e9
rf.set_frequency(freq)
lo.set_power(7)
lo.set_frequency(freq+sa.lo_offset)
lo.set_output(True)

p = []

pwrs = arange(-10.0, 10.0, 1)
for pwr in pwrs:
    rf.set_power(pwr)
    rf.set_output(True)
    time.sleep(0.2)
    p.append(sa.get_power())
    rf.set_output(False)
"""
rf.set_power(-10)
rf.set_output(True)
pwrs = []
t0 = time.time()
for i in range(50):
    pwrs.append(time.time()-t0)
    p.append(sa.get_avg_power())
    time.sleep(0.1)
"""
lo.set_output(False)
plt.scatter(pwrs, array(p))
plt.show()