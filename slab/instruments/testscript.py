__author__ = 'Nate'


from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
from instruments.PNAX import N5242A

from slab.instruments import InstrumentManager
from slab.instruments import Alazar, AlazarConfig
from slab.instruments import InstrumentManager
from liveplot import LivePlotClient
#from slab.instruments import N5242A
from slab.dsfit import fithanger_new_withQc



im = InstrumentManager()
#plotter = LivePlotClient()
dcflux = im['YOKO2']
nwa = im['PNAX']
# nwa = N5242A("N5242A", address="192.168.14.249", timeout=10.)
#drive = im['RF3']
print('Deviced Connected')
expt_path = os.getcwd() + '\data'



# initial NWA configuration values
ifbw = 20
read_power = -65.0
probe_power = -20.0
center = 4.35e9
span = 400e6
sweep_pts = 701
avgs = 5
delay = 0
print("Configuring the NWA")
nwa.set_timeout(10)
nwa.set_ifbw(ifbw)
nwa.set_center_frequency(center)
nwa.set_span(span)
nwa.write(":SOURCE:POWER1 %f" %(read_power))
nwa.write(":SOURCE:POWER3 %f" %(probe_power))
nwa.set_sweep_points(sweep_pts)
nwa.clear_traces()
nwa.setup_measurement("S21")
nwa.setup_take(averages_state=True)
nwa.set_averages_and_group_count(avgs, True)


nwa.set_power(-60, channel=1, port=3,state=0)

print("Done")