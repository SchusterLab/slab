from configuration_IQ import config, long_redout_len
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
avgs = 5000
simulation = 0

phi = 0
pump_amp = 1.0

with program() as histogram:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging

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
    # reset_frame('jpa_pump')
    # reset_frame('rr')
    # frame_rotation_2pi(phi, 'jpa_pump')

    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "jpa_pump")
        align("rr", "jpa_pump")
        play('pump_square'*amp(pump_amp), 'jpa_pump', duration=long_redout_len//4)
        measure("long_readout", "rr", None,
                demod.full("long_integW1", I1, 'out1'),
                demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),
                demod.full("long_integW2", Q2, 'out2'))

        assign(Ig, I1 - Q2)
        assign(Qg, I2 + Q1)
        save(Ig, Ig_st)
        save(Qg, Qg_st)

        align("qubit", "rr", "jpa_pump")

        """Play a ge pi pulse and then readout"""
        wait(reset_time // 4, "qubit")
        play("pi", "qubit")
        align("qubit", "jpa_pump", 'rr')
        play('pump_square'*amp(pump_amp), 'jpa_pump', duration=long_redout_len//4)
        measure("long_readout", "rr", None,
                demod.full("long_integW1", I1, 'out1'),
                demod.full("long_integW2", Q1, 'out1'),
                demod.full("long_integW1", I2, 'out2'),
                demod.full("long_integW2", Q2, 'out2'))

        assign(Ie, I1 - Q2)
        assign(Qe, I2 + Q1)
        save(Ie, Ie_st)
        save(Qe, Qe_st)

    with stream_processing():
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

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

    Ig = np.array(res_handles.get("Ig").fetch_all()['value'])
    Qg = np.array(res_handles.get("Qg").fetch_all()['value'])

    Ie = np.array(res_handles.get("Ie").fetch_all()['value'])
    Qe = np.array(res_handles.get("Qe").fetch_all()['value'])

    job.halt()

    plt.figure()
    plt.plot(Ig, Qg, '.')
    plt.plot(Ie, Qe, '.')
    plt.axis('equal')
    plt.show()

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

# """Extracting the qubit thermal population from Gaussian fitting of the histograms"""
# def gaus(x, a0, x0, sigma, a1, x1):
#     return a0*np.exp(-(x-x0)**2/(2*sigma**2)) + a1*np.exp(-(x-x1)**2/(2*sigma**2))
# from scipy.optimize import curve_fit
# y, x = np.histogram(Qg, 50)
# popt, pcov = curve_fit(gaus, x[:-1], y, p0=[1, 0.02, 0.001, 0, -0.002])