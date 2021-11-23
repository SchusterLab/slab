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
#
im = InstrumentManager()

isPNAX = 1  # PNAX or NWA
if isPNAX:
     nwa = im['PNAX']
# else:
#     nwa = im['NWA']
#


print(nwa.get_id())
# drive = im['RF3']
print('Deviced Connected')
expt_path = os.getcwd() + '\data'

# #broad sweep
# sweep_pts = 20001
# ifbw = 200
# avgs = 1
# centerlist = [6.5e9]
# spanlist  = [1.2e9]*len(centerlist)
# powerlist = [-55]


# #Indiv. Res. Sweep
rrlist = np.array([6.19663, 6.3234, 6.42667, 6.55568, 6.6554, 6.778007, 6.8710, 6.9949])*1e9
plist = np.array([6.255,6.368,6.485,6.605,6.703,6.842,6.9319,7.056])*1e9
centerlist = rrlist

spanlist = [10e6]*len(centerlist)
sweep_pts = 5001
ifbw = 200
#powerlist = [-70.0, -65., -60.,-55., -50., -45., -40., -35., -30., -25., -20., -15., -10., -5.]
powerlist = [-70]
avgs = [2]
#avgs = [10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5,5, 5, 5]

for ii,center in enumerate(centerlist):
    for jj,power in enumerate(powerlist):
        prefix = "PHMIV3pt56LPSCANResNo%s" % (ii)
        start = center - spanlist[ii] / 2.0
        stop = center + spanlist[ii] / 2.0
        # start = 4.4762e9
        # stop = 4.525e9+2.5e6
        step = spanlist[ii] / 1000.0  # less than 1601, default to 1601 pts
        if not isPNAX:
            print( "Configuring the NWA")
            nwa.set_remote_state()
            nwa.configure(power=power, ifbw=ifbw, sweep_points=sweep_pts, averages=avgs, remote=True,start = start,stop = stop,center=center)
            nwa.set_average_state(True)
            print ("NWA Configured. IFBW = %f Hz, NumPts = %d, Avgs = %d " % (ifbw, sweep_pts, avgs))
            print( "DETAILS: ")
            nwa.get_settings()
        else:
            print ("Configuring the PNAX")
            #nwa.timeout = 2000
            nwa.set_center_frequency(center)
            nwa.set_span(spanlist[ii])
            nwa.write(":SOURCE:POWER1 %f" % (power))
            nwa.set_sweep_points(sweep_pts)
            nwa.clear_traces()
            nwa.setup_measurement("S21")
            nwa.set_ifbw(ifbw)
            nwa.setup_take(averages=avgs[jj],averages_state=True)

        fname = get_next_filename(expt_path, prefix, suffix='.h5')
        print(fname)
        fname = os.path.join(expt_path, fname)

        with SlabFile(fname) as f:

            print(nwa.get_settings())
            f.save_settings(nwa.get_settings())

            if not isPNAX:
                #data = nwa.segmented_sweep2(start=start,stop=stop,step=step)
                #Throwing errors - take_one workaround
                nwa.set_start_frequency(start)
                nwa.set_stop_frequency(stop)
                data = nwa.take_one_averaged_trace()
            else:
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

print("Reset the NWA")
if isPNAX == 0:
    nwa.set_remote_state()
else:
    pass
nwa.configure(power=-60, ifbw=ifbw, sweep_points=100, averages=1, remote=True, start=start, stop=stop,
              center=center)
nwa.set_average_state(True)
print("NWA Configured. IFBW = %f Hz, NumPts = %d, Avgs = %d " % (ifbw, sweep_pts, 1))
print("DETAILS: ")
nwa.get_settings()