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
print dac.set_clk_freq(freq=2.5e9)

xpts = np.arange(8640)
ypts = np.ceil(2047.5 + 2047.5 * np.sin(2.0 * np.pi * xpts / (32)))
print ypts
print dac.create_single_segment(1, 0, 2047, 2047, ypts, 1)
print dac.place_mrkr2(1)
print dac.set_ext_trig(ext_trig=True)

numsegs = 100
segment_size = 160
segs = []
waveforms = []
generated_waveforms=[]

def segmentation(waveform,segment_size):
    zeros_segment_id_list=[]
    for ii in range(len(waveform)/segment_size):
        waveform_segment = waveform[ii*segment_size:(ii+1)*segment_size]
        #print waveform_segment
        if (np.array_equal(waveform_segment - 2047, np.zeros(segment_size))):
            zeros_segment_id_list.append(ii)
            #print "wow"
    return zeros_segment_id_list

def zero_segment_start_end(zeros_segment_id_list):
    segment_start_end_list=[]
    start_defined = False
    segment_next_expected = -1
    last_segment = -1
    for xx in zeros_segment_id_list:

        if start_defined and not (xx == segment_next_expected):
            segment_end = last_segment
            segment_start_end_list.append((segment_start,segment_end))
            start_defined=False
        if not start_defined:
            segment_start = xx
            start_defined=True
        segment_next_expected = xx+1
        last_segment = xx
    segment_start_end_list.append((segment_start,zeros_segment_id_list[-1]))

    return segment_start_end_list

def zero_segment_info(segment_start, segment_end, segment_size):
    loops = segment_end - segment_start + 1
    if segment_end == 179:
        triggered=0
        loops -=1
        #print "wow"
    else:
        triggered=0

    #loops = segment_end - segment_start + 1
    #print loops

    return DAx22000Segment(2047+np.zeros(segment_size),loops=loops,triggered=triggered)


for ii in range(numsegs):
    ii = numsegs-ii
    gauss1=np.exp(-1.0 * (xpts - 2000) ** 2 / (2 * (2*ii+1) ** 2))
    cutoff1_1=0.5 * (np.sign(xpts-(2000 - 3*(2*ii+1))) + 1)
    cutoff1_2=0.5 * (np.sign((2000 + 3*(2*ii+1)) - xpts) + 1)
    cutoff_gauss1=np.array([a*b*c for a,b,c in zip(gauss1,cutoff1_1,cutoff1_2)])
    gauss2=np.exp(-1.0 * (xpts - 4500) ** 2 / (2 * (ii+1) ** 2))
    cutoff2_1=0.5 * (np.sign(xpts-(4500 - 3*(ii+1))) + 1)
    cutoff2_2=0.5 * (np.sign((4500 + 3*(ii+1)) - xpts) + 1)
    cutoff_gauss2=np.array([a*b*c for a,b,c in zip(gauss2,cutoff2_1,cutoff2_2)])
    waveform = 2047 + 2047*cutoff_gauss1#*np.exp(-1.0 * (xpts - 2000) ** 2 / (2 * (5*ii+1) ** 2)) #* 0.5 * (np.sign(xpts-2000 - 3*(5*ii+1)) + 1)#*0.5 * (np.sign(xpts-4000 - 3*(10*ii+1)) + 1) * 0.5 * (np.sign(4000 + 3*(10*ii+1) - xpts) + 1)
    waveform += 2047*cutoff_gauss2
    zeros_segment_id_list = segmentation(waveform,segment_size)
    zero_segment_start_end_list = zero_segment_start_end(zeros_segment_id_list)

    generated_waveform=np.zeros(len(xpts))+2047

    last_segment_end = -1
    for segment_start, segment_end in zero_segment_start_end_list:
        if not (last_segment_end == -1):
            waveform_segment = waveform[(last_segment_end+1)*segment_size:(segment_start-1)*segment_size]
            #print last_segment_end,segment_start
            segs.append(DAx22000Segment(waveform_segment, loops=0, triggered=0))
            generated_waveform[(last_segment_end+1)*segment_size:(segment_start-1)*segment_size] = waveform_segment
        segs.append(zero_segment_info(segment_start, segment_end, segment_size))
        last_segment_end = segment_end
        #print last_segment_end

        if segment_end == 179:
            DAx22000Segment(2047+np.zeros(segment_size),loops=0,triggered=True)


    waveforms.append(waveform)
    generated_waveforms.append(generated_waveform)
    #segs.append(DAx22000Segment(waveform, loops=0, triggered=True))
    #segs.append(DAx22000Segment(xpts * 4095.0 / max(xpts)  * ii / numsegs, loops=0, triggered=True))

#print len(segs)
dac.create_segments(chan=1, segments=segs, loops=0)
dac.create_segments(chan=2, segments=segs, loops=0)

print dac.set_ext_trig(ext_trig=True)
#print dac.run(trigger_now=False)

expt.cfg['alazar']["samplesPerRecord"]=4096
expt.cfg['alazar']["recordsPerBuffer"]=numsegs
expt.cfg['alazar']["recordsPerAcquisition"]=numsegs
adc = Alazar(expt.cfg['alazar'])

#expt.awg.stop()
trig.set_output(False)

def trig_stop():
    trig.set_output(False)
    dac.stop()
    print "trig stop"

def trig_start():
    trig.set_output(True)
    dac.run(trigger_now=False)
    print "trig start"

tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=trig_stop, start_function=trig_start)

trig.set_output(False)


dac.stop()
dac.close()

expt.plotter.plot_z('dac1',ch1_pts)
expt.plotter.plot_z('dac2',ch2_pts)
expt.plotter.plot_z('waveforms',waveforms)
expt.plotter.plot_z('generated_waveforms',generated_waveforms)





