"""
rfsoc_avg_tutorial.py
connect DAC229_T1_CH3 to ADC224_T0_CH0
"""
import os
import sys
import time

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import scipy.io as sio

sys.path.append("/home/xilinx/repos/qsystem0/pynq")
from qsystem0 import *
from qsystem0_asm import *

def main():
    p = ASM_Program()
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

    soc = PfbSoc("/home/xilinx/repos/qsystem0/pynq/qsystem_0.bit")
    soc.tproc.load_asm_program(p)
    f_out = 5*30.72
    tempDac = freq2reg(soc.fs_dac,f_out)
    soc.tproc.single_write(addr=0,data=tempDac)
    # Set Readout frequency (same as output).
    tempAdc = freq2reg(soc.fs_adc,f_out)
    soc.tproc.single_write(addr=1,data=tempAdc) # Sets the DDS frequency

    # Duration of the output pulse on DAC domain.
    T_out = 60/f_out
    NS_out = T_out*soc.fs_dac
    NS_out_gen = int(NS_out/soc.gen0.NDDS)
    soc.tproc.single_write(addr=2,data=NS_out_gen)

    # Duration of the output pulse on ADC domain.
    NS_out_adc = NS_out*soc.fs_adc/soc.fs_dac
    NS_out_adc_dec = int(NS_out_adc/soc.readout.NDDS)
    Nsync = int(NS_out_adc_dec*1.3)
    soc.tproc.single_write(addr=3,data=Nsync)

    # Readout configuration to route input without frequency translation.
    soc.readout.set_out(sel="product")

    # Configure average+buffer window length.
    AVG_N = int(NS_out_adc_dec)
    soc.avg_buf.config(address=0,length=AVG_N)

    # Enable averager and buffer (will wait for trigger)
    soc.avg_buf.enable()

    # Number of repetitions.
    N = 10
    soc.tproc.single_write(addr=4,data=N-1)

    soc.setSelection("product") # "product", "dds", or "input"
    # Start tProc.
    soc.tproc.stop()
    soc.tproc.start()
    
    iacc,qacc =  soc.getAccumulated(length=N)
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(iacc, "*-")
    ax[0].set_ylabel("I")
    ax[1].plot(qacc, "*-")
    ax[1].set_ylabel("Q")
    ax[1].set_xlabel("sample number")
    ax[0].set_title("from getAccumulated")
    return
#ENDDEF
