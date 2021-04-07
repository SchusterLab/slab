from configuration_IQ import config, qubit_LO, rr_LO, ef_IF
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']

from slab.dsfit import*

###############
# qubit_spec_prog:
###############
LO_q.set_frequency(qubit_LO)
LO_q.set_ext_pulse(mod=False)
LO_q.set_power(18)
LO_r.set_frequency(rr_LO)
LO_r.set_ext_pulse(mod=False)
LO_r.set_power(18)

f_min = -20e6
f_max = 20e6
df = 400e3
freqs = np.arange(f_min, f_max + df/2, df)
# freqs = freqs + qubit_freq - 140e6

avgs = 500
reset_time = 500000
simulation = 0


simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

biased_th_g = 0.0014
qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    wait(5000//4, 'rr')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as qubit_ef_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ef_IF + f_min, f < ef_IF + f_max + df/2, f + df):

            update_frequency("qubit_ef", f)
            active_reset(biased_th_g)
            align("qubit", 'rr')
            play("pi", "qubit")
            align("qubit", "qubit_ef")
            play("saturation"*amp(0.1), "qubit_ef")
            align("qubit_ef", "rr")
            measure("long_readout", "rr", None,
                    demod.full("long_integW1", I1, 'out1'),
                    demod.full("long_integW2", Q1, 'out1'),
                    demod.full("long_integW1", I2, 'out2'),
                    demod.full("long_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():

        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(qubit_ef_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(qubit_ef_spec, duration_limit=0, data_limit=0)
    print("Experiment done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done!")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # fig.tight_layout()
    axs[0].plot(freqs/1e9, I)
    axs[0].plot(freqs/1e9, Q)
    axs[0].set_xlabel('Freq (GHz)')
    axs[0].set_ylabel('I/Q')
    amps = np.sqrt(np.array(I)**2 + np.array(Q)**2)
    ph = np.arctan2(np.array(Q), np.array(I))
    axs[1].plot(freqs/1e9, amps, 'b-')
    p = fitlor(freqs/1e9, -amps, showfit = False)
    axs[1].plot(freqs/1e9, -lorfunc(p, freqs/1e9), label=r'$\nu$ = %.3f GHz, $\Delta \nu$ = %.3f MHz'%(p[2], p[3]*1e3))
    print("fits = ", p)
    print("center freq", p[2], "GHz")
    print("linewidth", p[3]*1e3, "MHz")
    ax2 = axs[1].twinx()
    ax2.plot(freqs/1e9, ph, 'r-')
    axs[1].set_xlabel('Freq (GHz)')
    axs[1].set_ylabel('amp')
    ax2.set_ylabel('$\\varphi$')
    axs[1].legend()
    fig.show()