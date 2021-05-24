import os
from pynq import Overlay
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import re
from pynq import allocate

from qsystem0 import *
from qsystem0_asm import *

# Connect DAC229_T1_CH3 to ADC224_T0_CH0

# Load bitstream with custom overlay
soc = PfbSoc('qsystem_0.bit',force_init_clks=False)

print("soc.fs_dac",soc.fs_dac)
print("soc.fs_adc",soc.fs_adc)

with ASM_Program() as p:
    p.memri(0,1,0,"freq")
    p.regwi(0,2,30000,"gain")
    p.memri(0,3,2,"nsamp")
    p.regwi(0,4,0x9,"0b1001, stdysel = 1 (zero value), mode = 0 (nsamp), outsel = 01 (dds).")
    p.bitwi(0,4,4, "<<", 12)
    p.bitw(0,3,3,"|",4)
    p.regwi(0,5,0,"t")
    p.synci(1000)
    # is channel 5 an output that is triggering something? it seems like different
    # channels get different output settings from the seti command
    # for example why is a frequency value getting written here?
    p.memri(1,7,1) # Set the readout frequency
    p.seti(5,1,7,0)
    # what do the bits you are writing on page 1 registers 1 and 6 correspond to?
    p.regwi(1,1,0x8001,"bit 15 = 1, bit 0 = 1")
    p.regwi(1,6,0xc001,"bit 15 = 1, bit 14 = 1, bit 0 = 1")
    # does this command do something to whatever is on channel 5? if so what?
    p.regwi(1,7,100)
    # is channel 0 the ADC? page 1 register 0 is zero-initialized,
    # but what does it mean to set this channel to 0, i’m guessing this
    # is a specific case of the bit-pattern that we write to it later in the program?
    p.seti(0,1,0,0)
    p.memri(1,2,3,"Nsync")
    p.memri(1,3,4,"Loop")
    p.label("LOOP")
    p.set(3,0,1,0,0,2,3,5)
    p.set(4,0,1,0,0,2,3,5)
    # I see page 1 registers 1 and 6 make a return, but i’m not sure what this
    # instruction is actually doing. when we used seti on channel 5 earlier
    # the value we wrote looks a lot different than the bits we are writing to channel 0
    p.seti(0,1,6,0)
    p.seti(0,1,1,10)
    # i’m assuming Nsync is a wait time between executions?
    p.sync(1,2)
    p.loopnz(1,3,"LOOP")
    # see above, what does it mean to set the output of this channel to 0?
    p.seti(0,1,0,0)
    p.end("all done")
soc.tproc.load_asm_program(p)
