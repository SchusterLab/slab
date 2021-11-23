from configuration_IQ import config
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from slab import*
##################
# histogram_prog:
##################
reset_time = 500000
avgs = 2000
simulation = 0

a_min = 0.005
a_max = 0.025
da = a_max/100

amp_vec = np.arange(a_min, a_max + da/2, da)

phi = 0.0

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    a = declare(fixed)
    Ig = declare(fixed)
    Qg = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    Ie = declare(fixed)
    Qe = declare(fixed)

    Ig_st = declare_stream()
    Qg_st = declare_stream()

    Ie_st = declare_stream()
    Qe_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        # reset_frame('jpa_pump')
        # reset_frame('rr')
        # frame_rotation_2pi(phi, 'jpa_pump')

        with for_(a, a_min, a < a_max + da/2, a + da):
            """Just readout without playing anything"""
            # reset_frame('jpa_pump')
            # reset_frame('rr')
            # frame_rotation_2pi(phi, 'jpa_pump')
            wait(reset_time // 4, "jpa_pump")
            align("rr", "jpa_pump")
            wait(16//4, 'jpa_pump')
            play('pump_square'*amp(a), 'jpa_pump')
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))
    
            assign(Ig, I1 - Q2)
            assign(Qg, I2 + Q1)
            save(Ig, Ig_st)
            save(Qg, Qg_st)
    
            align("qubit", "rr", "jpa_pump")
    
            """Play a ge pi pulse and then readout"""
            # reset_frame('jpa_pump')
            # reset_frame('rr')
            # frame_rotation_2pi(phi, 'jpa_pump')
            wait(reset_time // 4, "qubit")
            play("pi", "qubit")
            align("qubit", "jpa_pump")
            align("rr", "jpa_pump")
            wait(16//4, 'jpa_pump')
            play('pump_square'*amp(a), 'jpa_pump')
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))
            assign(Ie, I1 - Q2)
            assign(Qe, I2 + Q1)
            save(Ie, Ie_st)
            save(Qe, Qe_st)

    with stream_processing():

        Ig_st.buffer(len(amp_vec)).average().save('Ig')
        Qg_st.buffer(len(amp_vec)).average().save('Qg')

        Ie_st.buffer(len(amp_vec)).average().save('Ie')
        Qe_st.buffer(len(amp_vec)).average().save('Qe')


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(histogram, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(histogram, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig = np.array(res_handles.get("Ig").fetch_all())
    Qg = np.array(res_handles.get("Qg").fetch_all())

    Ie = np.array(res_handles.get("Ie").fetch_all())
    Qe = np.array(res_handles.get("Qe").fetch_all())

    job.halt()

    ph_g= np.arctan2(np.array(Qg), np.array(Ig))
    ph_g = np.unwrap(ph_g, discont=3.141592653589793, axis=-1)
    ph_g = ph_g - np.mean(ph_g)
    ph_e= np.arctan2(np.array(Qe), np.array(Ie))
    ph_e = np.unwrap(ph_e, discont=3.141592653589793, axis=-1)
    ph_e = ph_e - np.mean(ph_e)

    plt.figure()
    plt.plot(amp_vec, ph_g, '.--')
    plt.plot(amp_vec, ph_e, '.--')
    plt.show()

    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'histogram_jpa', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     dset = f.create_dataset("ig", data=Ig)
    #     dset = f.create_dataset("qg", data=Qg)
    #     dset = f.create_dataset("ie", data=Ie)
    #     dset = f.create_dataset("qe", data=Qe)