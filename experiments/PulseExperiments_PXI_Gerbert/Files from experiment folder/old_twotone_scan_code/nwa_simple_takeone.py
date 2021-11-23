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

isPNAX = 1  # PNAX or NWA
if isPNAX:
    nwa = im['PNAX2']
else:
    nwa = im['NWA']



print( nwa.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'




sweep_pts = 10001
ifbw = 1000
avgs = 1
power = -5


center = 10e9
span = 20e9#1.0e9
start = center-span/2.0
stop = center+span/2.0
#start = 4.4762e9
#stop = 4.525e9+2.5e6
step = span/1000.0 # less than 1601, default to 1601 pts

prefix = "30K_Scan"

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
    nwa.set_span(span)
    nwa.write(":SOURCE:POWER1 %f" % (power))
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement("S21")
    nwa.set_ifbw(ifbw)
    nwa.setup_take(averages=avgs,averages_state=True)

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