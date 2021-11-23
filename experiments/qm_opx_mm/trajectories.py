from configuration_IQ import config, rr_LO, rr_IF
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from slab import*
import os
from slab.dataanalysis import get_next_filename
from slab import*

##################
# histogram_prog:
##################

reset_time = 500000
avgs = 5000
simulation = 0
with program() as trajectories:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    Ig = declare(fixed)
    Qg = declare(fixed)
    Ie = declare(fixed)
    Qe = declare(fixed)

    adc_st_g = declare_stream(adc_trace=True)
    adc_st_e = declare_stream(adc_trace=True)


    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "rr")
        measure("readout", "rr", adc_st_g)



        align("qubit_mode0", "rr")

        """Play a ge pi pulse and then readout"""
        wait(reset_time // 4, "qubit_mode0")
        play("pi", "qubit_mode0")
        align("qubit_mode0", "rr")
        measure("readout", "rr", adc_st_e)




    with stream_processing():
        adc_st_g.input1().with_timestamps().map(FUNCTIONS.demod(rr_IF, 2.0, 0, integrate=False)).average().save('adc_g_cos')

        # If_st.save_all('If')
        # Qf_st.save_all('Qf')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(trajectories, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(trajectories, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    adc_g_cos_handle = res_handles.get('adc_g_cos')
    adc_g_cos = adc_g_cos_handle.fetch_all()
    plt.plot(adc_g_cos)



