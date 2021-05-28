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
from slab import generate_file_path

sys.path.append("/home/xilinx/repos/qsystem0/pynq")
from qsystem0 import *
from qsystem0_asm import *

DPI = 300

def main(plot=False):
    # parameters
    soc = PfbSoc("/home/xilinx/repos/qsystem0/pynq/qsystem_0.bit")
    f_out = 5*30.72
    T_out = 60/f_out
    NS_out = T_out*soc.fs_dac
    NS_out_adc = NS_out*soc.fs_adc/soc.fs_dac
    NS_out_adc_dec = int(NS_out_adc/soc.readout.NDDS)

    # create asm
    p = ASM_Program()
    p.synci(1000)
    p.regwi(1, 1, 0x8001, "bit 15 = 1, bit 0 = 1")
    p.regwi(1, 6, 0xc001, "bit 15 = 1, bit 14 = 1, bit 0 = 1")
    p.seti(0, 1, 0, 0)
    p.seti(0, 1, 6, 0)
    p.seti(0, 1, 1, 10)
    p.seti(0, 1, 0, 0)
    p.end()

    # load asm
    soc.tproc.load_asm_program(p)
    
    # Configure average+buffer window length.
    AVG_N = int(NS_out_adc_dec)
    soc.avg_buf.config(address=0, length=AVG_N)
    soc.avg_buf.enable()

    # "product", "dds", or "input"
    soc.setSelection("product")
    soc.tproc.stop()
    soc.tproc.start()
    
    iacc, qacc =  soc.getAccumulated(length=2)

    if plot:
        fig,ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(iacc, "*-")
        ax[0].set_ylabel("I")
        ax[1].plot(qacc, "*-")
        ax[1].set_ylabel("Q")
        ax[1].set_xlabel("sample number")
        ax[0].set_title("from getAccumulated")
        plot_file_path = generate_file_path(".", "avg_tut", "png")
        plt.savefig(plot_file_path, dpi=DPI)
        print("plotted to {}".format(plot_file_path))
    #ENDIF
    
    return
#ENDDEF
