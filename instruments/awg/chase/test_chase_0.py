import numpy as np
from slab.instruments import DAx22000, DAx22000Segment,Alazar
from slab.instruments import InstrumentManager
from slab import Experiment
import matplotlib.pyplot as plt

expt=Experiment(path=r'C:\_Lib\python\slab\instruments\awg\chase',config_file='config.json')

expt.plotter.clear('dac1')
expt.plotter.clear('dac2')

dac = DAx22000('dac', '1')

im=InstrumentManager()
trig = im['trig']

print dac.initialize(ext_clk_ref=True)
print dac.set_clk_freq(freq=0.5e9)

xpts = np.arange(6400)
ypts = np.ceil(2047.5 + 2047.5 * np.sin(2.0 * np.pi * xpts / (32)))
print ypts
print dac.create_single_segment(1, 0, 2047, 2047, ypts, 1)
print dac.place_mrkr2(1)
print dac.set_ext_trig(ext_trig=True)

numsegs = 65
segs = []
waveforms = []

def waveform_compressor(waveform):
    print "compressing"


for ii in range(numsegs):
    # ii = numsegs - ii
    # gauss1=np.exp(-1.0 * (xpts - 1000) ** 2 / (2 * (2*ii+1) ** 2))
    # cutoff1_1=0.5 * (np.sign(xpts-(1000 - 3*(2*ii+1))) + 1)
    # cutoff1_2=0.5 * (np.sign((1000 + 3*(2*ii+1)) - xpts) + 1)
    # cutoff_gauss1=np.array([a*b*c for a,b,c in zip(gauss1,cutoff1_1,cutoff1_2)])
    # gauss2=np.exp(-1.0 * (xpts - 2500) ** 2 / (2 * (ii+1) ** 2))
    # cutoff2_1=0.5 * (np.sign(xpts-(2500 - 3*(ii+1))) + 1)
    # cutoff2_2=0.5 * (np.sign((2500 + 3*(ii+1)) - xpts) + 1)
    # cutoff_gauss2=np.array([a*b*c for a,b,c in zip(gauss2,cutoff2_1,cutoff2_2)])
    # waveform = 2047 + 2047*cutoff_gauss1#*np.exp(-1.0 * (xpts - 2000) ** 2 / (2 * (5*ii+1) ** 2)) #* 0.5 * (np.sign(xpts-2000 - 3*(5*ii+1)) + 1)#*0.5 * (np.sign(xpts-4000 - 3*(10*ii+1)) + 1) * 0.5 * (np.sign(4000 + 3*(10*ii+1) - xpts) + 1)
    # waveform += 2047*cutoff_gauss2
    # waveforms.append(waveform)
    # segs.append(DAx22000Segment(waveform, loops=0, triggered=True))
    segs.append(DAx22000Segment(xpts * 4095.0 / max(xpts)  * ii / numsegs, loops=1, triggered=True))


dac.create_segments(chan=1, segments=segs, loops=0)
dac.create_segments(chan=2, segments=segs, loops=0)

print dac.set_ext_trig(ext_trig=True)
print dac.run(trigger_now=False)

expt.cfg['alazar']["samplesPerRecord"]=8192
expt.cfg['alazar']["recordsPerBuffer"]=numsegs
expt.cfg['alazar']["recordsPerAcquisition"]=numsegs
adc = Alazar(expt.cfg['alazar'])

#expt.awg.stop()
trig.set_output(True)

def trig_stop():
    trig.set_output(False)
    dac.stop()
    print "trig stop"

def trig_start():
    trig.set_output(True)
    dac.run(trigger_now=False)
    print "trig start"

tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=None, start_function=None)

#trig.set_output(False)


dac.stop()
dac.close()

expt.plotter.plot_z('dac1',ch1_pts)
expt.plotter.plot_z('dac2',ch2_pts)
expt.plotter.plot_z('waveforms',waveforms)






