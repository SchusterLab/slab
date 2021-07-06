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

"""Resonator phase as a function of the pump phase"""

phi_min = 0.0
phi_max = 0.1
dphi = 0.1/10

phi = 0.0

phi_vec = np.arange(phi_min, phi_max + dphi/2, dphi)

avgs = 1000
reset_time = 100000
simulation = 0

with program() as resonator_spectroscopy:

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

    with for_(i, 0, i < avgs, i+1):

        with for_(phi, phi_min, phi < phi_max +  dphi/2, phi + dphi):
            # update_frequency("rr", f)
            # reset_frame('jpa_pump')
            # reset_frame('rr')
            # frame_rotation_2pi(phi, 'jpa_pump')
            wait(reset_time//4, "rr")
            align('rr', 'jpa_pump')
            play('pump_square'*amp(0.0625), 'jpa_pump')
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(phi_vec)).average().save('I')
        Q_st.buffer(len(phi_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()

    print ("Data collection done")
    job.halt()

    amps = Q**2 + I**2
    ph = np.arctan2(np.array(Q), np.array(I))
    ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
    # m = (ph[-1]-ph[0])/(x[-1] - x[0])
    # ph = ph - m*x*0.95
    # ph = ph -np.mean(ph)

    plt.figure()
    plt.plot(phi_vec, ph, '.')

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