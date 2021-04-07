from configuration_IQ import config, qubit_LO, rr_LO, rr_IF, long_redout_len
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*

im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

##################
# histogram_prog:
##################
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

reset_time = 500000
avgs = 3000
simulation = 0
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

    with for_(n, 0, n < avgs, n + 1):

        """Just readout without playing anything"""
        wait(reset_time // 4, "rr")
        # reset_phase("jpa_pump")
        align("rr", "jpa_pump")
        play('pump_square', 'jpa_pump')
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
        wait(reset_time // 4, "qubit")
        # reset_phase("jpa_pump")
        play("pi", "qubit")
        align("qubit", "rr")
        align("rr", "jpa_pump")
        # frame_rotation_2pi(-np.pi/32, "jpa_pump")
        play('pump_square', 'jpa_pump')
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
        Ig_st.save_all('Ig')
        Qg_st.save_all('Qg')

        Ie_st.save_all('Ie')
        Qe_st.save_all('Qe')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(histogram, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(histogram, duration_limit=0, data_limit=0)
    print("Done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig_handle = res_handles.get("Ig")
    Qg_handle = res_handles.get("Qg")

    Ie_handle = res_handles.get("Ie")
    Qe_handle = res_handles.get("Qe")

    Ig = np.array(Ig_handle.fetch_all()['value'])
    Qg = np.array(Qg_handle.fetch_all()['value'])

    Ie = np.array(Ie_handle.fetch_all()['value'])
    Qe = np.array(Qe_handle.fetch_all()['value'])

    job.halt()

    plt.plot(Ig, Qg, '*')
    plt.plot(Ie, Qe, '*')

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'histogram_jpa', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        dset = f.create_dataset("ig", data=Ig)
        dset = f.create_dataset("qg", data=Qg)
        dset = f.create_dataset("ie", data=Ie)
        dset = f.create_dataset("qe", data=Qe)