import numpy as np
from slab.instruments import DAx22000, DAx22000Segment

dac = DAx22000('dac', '1')

print dac.initialize(ext_clk_ref=False)
print dac.set_clk_freq(freq=2.5e9)

xpts = np.arange(8000)
ypts = np.ceil(2047.5 + 2047.5 * np.sin(2.0 * np.pi * xpts / (32)))
print ypts
print dac.create_single_segment(1, 0, 2047, 2047, ypts, 1)
print dac.place_mrkr2(1)
print dac.set_ext_trig(ext_trig=True)

numsegs = 100
segs = []
for ii in range(numsegs):
    segs.append(DAx22000Segment(xpts * 4095.0 / max(xpts) * ii / numsegs, loops=1, triggered=False))

dac.create_segments(chan=1, segments=segs, loops=0)

print dac.run(trigger_now=True)


