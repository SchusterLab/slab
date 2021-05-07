import slab
from slab import *
import time
import numpy as np

# make sure fridge server is running prior to running script
im=InstrumentManager()
fridge=im['whistler']
expt= Experiment(expt_path='S:\_SchusterFridge\Whistler_Data', prefix='cryolog', liveplot_enabled=False)

interval = 2.
start_time=time.time()

with expt.datafile() as f:
    f.attrs['start_time']=start_time
    f.attrs['interval']=interval
# generic cryocon channels
cryocon_channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
thermometer_labels=fridge.get_thermometers
channels=[]

#sets channels based on the channels present in the config
for i in range(0, len(thermometer_labels)):
    channels[i]=cryocon_channels[i]

print("Monitering temperature in"+expt.fname)
while True:

    try:
        temps=[fridge.get_temp(ch) for ch in channels]
        t=time.time()-start_time
        print(temps, t)
        time.sleep(interval)
        with expt.datafile() as f:
            f.append_pt('tpts', t)
            for ch, temp in zip(channels, temps):
                f.append_pt(ch, temp)

    except:
        print('Exception occured...trying to continue....')
