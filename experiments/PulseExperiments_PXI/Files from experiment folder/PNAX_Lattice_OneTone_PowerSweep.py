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
#nwa = im['NWA']
#nwa = im['PNAX']
isPNAX = 1  # has to be PNAX
if isPNAX:
    nwa = im['PNAX2']


print( nwa.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'

sweep_pts =4001
ifbwlist = [200,300,500,500,1000,2000,2000,4000,5000,5000]
ifbwlist = [5000]*5
avglist = [1]*11
ifbw = 1000
#low power and high power in here
powerlist = [-75]
#high power centers
centerlist = [6.2e9 - 30e6,6.3024e9- 30e6,6.4224e9- 30e6,6.534e9- 30e6,6.6456e9- 30e6,6.7584e9- 30e6,6.8688e9- 30e6,6.984e9- 30e6]
centerlist = [6.162e9,6.2757e9,]
for i in range(len(powerlist)):
    power = powerlist[i]
    for j in range(len(centerlist)):
        span = 160e6
        start = centerlist[j]-span/2.0
        stop = centerlist[j]+span/2.0
        #start = 4.4762e9
        #stop = 4.525e9+2.5e6
        step = span/1000.0 # less than 1601, default to 1601 pts
        print( "Configuring the NWA")

        #nwa.set_remote_state()
        power = powerlist[i]
        avgs = avglist[i]
        ifbw = ifbwlist[i]
        #nwa.configure(power=power, ifbw=ifbw, sweep_points=sweep_pts, averages=avgs)
        #nwa.set_average_state(True)
        #print ("NWA Configured. IFBW = %f Hz, NumPts = %d, Avgs = %d " % (ifbw, sweep_pts, avgs))
        fname = get_next_filename(expt_path,"RNo%s"%j+"_SCANPOWER%s"%(int(powerlist[i])), suffix='.h5')
        #fname = get_next_filename(expt_path,"BroadBand", suffix='.h5')
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