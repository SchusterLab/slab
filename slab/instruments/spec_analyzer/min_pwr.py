# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:53:45 2011

@author: Dai
"""
from numpy import *
from guiqwt.pyplot import *

from .spectrum_analyzer import *
from rfgenerators import *
import pickle
import time

rf=E8257D(address='rfgen1.circuitqed.com')
lo=E8257D(address='rfgen2.circuitqed.com')

sa = SpectrumAnalyzer(address='128.135.35.167')

frequencies = [x*1e8 for x in range(50, 100, 1)]
#additional frequencies to sample
#frequencies.append(6.65e9)
#frequencies.append(6.75e9)
#frequencies.append(7.65e9)

minPower = []

lo.set_power(sa.lo_power+3)

for frequency in frequencies:
    measured_powers=[]
    lo.set_output(False)
    rf.set_output(False)

    rf.set_frequency(frequency)

    lo.set_frequency(frequency+sa.lo_offset)
    lo.set_power(10)
    lo.set_output(True)

    print("RF Frequency: "+str(frequency))

    lp = -80
    rp = 10
    while (rp - lp > 1):
        rf.set_power((lp+rp)/2)
        rf.set_output(True)
        time.sleep(0.2)
        m = sa.get_power()
        rf.set_output(False)        
        if (m > 50.0):
            rp = (lp + rp) / 2
        else:
            lp = (lp + rp) / 2
    minPower.append([frequency, (lp+rp)/2])
    print((frequency, (lp+rp)/2))

lo.set_output(False)
rf.set_output(False)

pickle.dump(minPower, open(str(int(lo.get_power()))+'dBm_min_pwr.data', 'w'))

figure(0)
mpp = []
for mp in minPower:
    mpp.append(mp[1])
mpp = array(mpp)
plot(frequencies, array(mpp), 'b+')
title('Minimal Detectable Power v. RF Frequency')
xlabel('RF Frequency (Hz)')
ylabel('Minimal Detectable Power (dBm)')
show()

"""
    for power in powers:
        rf.set_power(power)
        rf.set_output(True)
        mpow=sa.get_avg_power()
        print "Power: %f, measured: %f " % (power,mpow)
        measured_powers.append(mpow)
        if (mpow > 50.0) and not found: 
            minPower.append((frequency, power))
            found = True

    measured_powers=array(measured_powers)
    figure(i)
    plot(powers,measured_powers)
    title('RF Frequency %f' % frequency)
    xlabel("RF Power")
    ylabel("Spec Analyzer Output")
"""    