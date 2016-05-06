# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:50:07 2011

@author: Dai
"""
from spectrum_analyzer import *
from rfgenerators import *
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

rf = E8257D(address='rfgen1.circuitqed.com')
lo = E8257D(address='rfgen2.circuitqed.com')
sa = SpectrumAnalyzer(address='128.135.35.167')

min_power = pickle.load(open('10dBm_min_pwr.data', 'r'))
cali = []
step = 10

lo.set_output(False)
rf.set_output(False)

lo.set_power(10)

for freq, mp in min_power:
    rf.set_frequency(freq)
    lo.set_frequency(freq + sa.lo_offset)
    lo.set_output(True)
    for pwr in range(mp, 10, step):
        rf.set_power(pwr)
        rf.set_output(True)
        time.sleep(0.2)
        #the calibration data is stored in a 2D array
        nd = [freq, sa.get_power(), pwr]
        cali.append(nd)
        print nd
        rf.set_output(False)
    lo.set_output(False)

print cali
pickle.dump(cali, open('10dBm_'+str(step)+'_cali.data', 'w'))

c = np.array(cali)
c = c.transpose()
plt.scatter(c[0], c[1])
plt.show()