# -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nate E
"""

from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
from slab.instruments.PNAX import N5242A

from slab.instruments import InstrumentManager
from slab.instruments import Alazar, AlazarConfig
from slab.instruments import InstrumentManager
#from liveplot import LivePlotClient
#from slab.instruments import N5242A
from slab.dsfit import fithanger_new_withQc

im = InstrumentManager()
nwa = im['PNAX']

print( nwa.get_id())

print ('Deviced Connected')
expt_path = os.getcwd() + '\data'

sweep_pts =10001
ifbw = 500
#low power and high power in here
powerlist = [-55]
#high power centers
#centerlist = [6.0661e9,6.1054e9,6.1398e9,6.223e9,6.2662e9,6.277e9,6.332e9,6.3997e9]
centerlist = [6.4e9]
for i in range(len(powerlist)):
    power = powerlist[i]
    for j in range(len(centerlist)):
        span = 8e9
        start = centerlist[j]-span/2.0
        stop = centerlist[j]+span/2.0
        #start = 4.4762e9
        #stop = 4.525e9+2.5e6
        step = span/1000.0 # less than 1601, default to 1601 pts
        print( "Configuring the NWA")

        #nwa.set_remote_state()
        power = powerlist[i]
        avgs = 1
        #nwa.configure(power=power, ifbw=ifbw, sweep_points=sweep_pts, averages=avgs)
        #nwa.set_average_state(True)
        #print ("NWA Configured. IFBW = %f Hz, NumPts = %d, Avgs = %d " % (ifbw, sweep_pts, avgs))
        #fname = get_next_filename(expt_path,"F%s"%j+"POWER%s"%(powerlist[i]), suffix='.h5')
        fname = get_next_filename(expt_path,"BroadBand", suffix='.h5')
        print(fname)
        fname = os.path.join(expt_path, fname)
        with SlabFile(fname) as f:
            print("Configuring the PNAX")

            # turning off the pulses
            nwa.write("SENS:PULS0 0")
            nwa.write("SENS:PULS1 0")
            nwa.write("SENS:PULS2 0")
            nwa.write("SENS:PULS3 0")
            nwa.write("SENS:PULS4 0")
            # turning on the inverting
            nwa.write("SENS:PULS1:INV 1")
            nwa.write("SENS:PULS2:INV 1")
            nwa.write("SENS:PULS3:INV 1")
            nwa.write("SENS:PULS4:INV 1")

            nwa.set_timeout(10E3)
            nwa.clear_traces()
            nwa.setup_measurement("S21")

            dummy_freq = 5.0e9
            dummy_power = -50


            nwa.set_start_frequency(start)
            nwa.set_stop_frequency(stop)
            f.save_settings(nwa.get_settings())
            print(nwa.get_settings())
            nwa.set_ifbw(ifbw)
            nwa.set_sweep_points(sweep_pts)
            nwa.setup_take(averages_state=True)
            nwa.set_averages_and_group_count(avgs, True)
            nwa.setup_two_tone_measurement(read_frequency=dummy_freq, read_power=power,
                                           probe_start=start,
                                           probe_stop=stop, probe_power=dummy_power,
                                           two_tone=0)

            data = nwa.take_one_in_mag_phase()

            fpoints = data[0]
            mags = data[1]
            phases = data[2]
            print("finished downloading")

            #time.sleep(na.get_query_sleep())
            with SlabFile(fname) as f:
                f.append_line('fpts', fpoints)
                f.append_line('mags', mags)
                f.append_line('phases', phases)
                # f.append_line('read_freq',read_freqs)
                f.append_pt('read_power', power)
        print(fname)