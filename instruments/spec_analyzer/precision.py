# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:56:42 2011

@author: Dai
"""
from rfgenerators import *
from .spectrum_analyzer import *

sa = SpectrumAnalyzer(address='128.135.35.167')
rf = E8257D(address='rfgen1.circuitqed.com')
lo = E8257D(address='rfgen2.circuitqed.com')
rf.set_output(False)
lo.set_output(False)

lo.set_power(10)

freq = 6e9
rf.set_frequency(freq)
lo.set_frequency(freq+sa.lo_offset)
lo.set_output()

for pwr in range(-200, 0, 1):
    rf.set_power(pwr/10.0)
    rf.set_output()
    time.sleep(0.2)
    op = sa.get_power()
    print('SA output: '+str(op))
    print()
    rf.set_output(False)

"""  
for f0 in range(500, 510, 1):
    f = f0*1e7        
    rf.set_frequency(f)
    lo.set_frequency(f+sa.lo_offset)
    lo.set_output()
    for pwr in range(-10, 0, 2):
        rf.set_power(pwr)
        rf.set_output()
        time.sleep(0.2)
        op = sa.get_power()
        ppwr = get_rf_power(f, op)            
        print 'RF frequency: '+str(f)+', SA output: '+str(op)            
        print 'Measured: '+str(pwr)+', Predicted: '+str(ppwr)
        print 'Difference: '+str(ppwr-pwr)
        print
        rf.set_output(False)
    lo.set_output(False)
"""