from configuration_IQ import config, rr_LO, rr_freq, rr_IF
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dsfit import*
from slab.dataanalysis import get_next_filename

f_min = -1.5e6+1
f_max = 0.5e6+1
df = 100e3# 25e3
f_vec = rr_freq - np.arange(f_min, f_max + df/2, df)

phi_min = 0
phi_max = 0.5
dphi = 0.5/20

phi_vec = np.arange(phi_min, phi_max + dphi/2, dphi)

avgs = 1000
reset_time = 100000
simulation = 0

with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    phi = declare(fixed)

    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(phi, phi_min, phi< phi_max + dphi/2, phi + dphi):
        with for_(i, 0, i < avgs, i+1):

            with for_(f, f_min + rr_IF, f < f_max + rr_IF + df/2, f + df):
                update_frequency("rr", f)
                reset_frame('jpa_pump')
                reset_frame('rr')
                frame_rotation_2pi(phi, 'jpa_pump')
                wait(reset_time//4, "rr")
                align('rr', 'jpa_pump')
                play('pump_square'*amp(0.02), 'jpa_pump')
                measure("long_readout"*amp(0.4), "rr", None,
                        demod.full("long_integW1", I1, 'out1'),
                        demod.full("long_integW2", Q1, 'out1'),
                        demod.full("long_integW1", I2, 'out2'),
                        demod.full("long_integW2", Q2, 'out2'))

                assign(I, I1-Q2)
                assign(Q, I2+Q1)

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(avgs, len(f_vec)).map(FUNCTIONS.average()).save_all('I')
        Q_st.buffer(avgs, len(f_vec)).map(FUNCTIONS.average()).save_all('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    # print ("Data collection done")
    #
    # job.halt()
    # amp = (I**2 + Q**2)
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'resonator_spec', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=Q)
    #     f.create_dataset("freqs", data=f_vec)