# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:38:16 2011

@author: Dai
"""
from numpy import *
from guiqwt.pyplot import *
from time import *
import matplotlib.pyplot as plt

from spectrum_analyzer import *

pwr = []
ts = []
sa = SpectrumAnalyzer(address='128.135.35.166', query_sleep=0.2)
t0 = time.time();
for i in range(1000):
    pwr.append(sa.get_power())
    ts.append(time.time()-t0)
    print i
    
figure(0)
xlabel('time (s)')
ylabel('output')
title('Output Stability Test')
plot(ts, array(pwr), 'r+')
show()

plt.hist(pwr, 7)
plt.show()