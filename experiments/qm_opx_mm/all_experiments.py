"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_dict import opx_config, qubit_params, readout_params, storage_params
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os, json
from slab.dataanalysis import get_next_filename

device_cfg = {**qubit_params, **readout_params, **storage_params}

with open("experiment_config.json", 'r') as f:
    expt_cfg = json.load(f)

"""Write experiment classses"""

def measurement(pulse="clear", amp=0.0, use_disc=True):
    if pulse=="readout":
        measure("readout"*amp(cfg["amp"]), "rr", None,
                demod.full("integW1", I1, 'out1'),
                demod.full("integW2", Q1, 'out1'),
                demod.full("integW1", I2, 'out2'),
                demod.full("integW2", Q2, 'out2'))

        assign(I, I1 - Q2)
        assign(Q, I2 + Q1)

        save(I, I_st)
        save(Q, Q_st)
    elif pulse=="clear":
        measure("clear", "rr", None,
                demod.full("clear_integW1", I1, 'out1'),
                demod.full("clear_integW2", Q1, 'out1'),
                demod.full("clear_integW1", I2, 'out2'),
                demod.full("clear_integW2", Q2, 'out2'))

        assign(I, I1 - Q2)
        assign(Q, I2 + Q1)

        save(I, I_st)
        save(Q, Q_st)
    elif use_disc:
        discriminator.measure_state("clear", "out1", "out2", res, I=I)

        save(res, Q_st)
        save(I, I_st)

class ResonatorSpectroscopy:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """readout resonator spectroscopy, varying the IF frequency"""
        cfg = self.expt_cfg["resonator_spectroscopy"]
        print(cfg)

        f_min = cfg["f_min"]
        f_max = cfg["f_max"]
        df = cfg["df"]

        f_vec = np.arange(f_min, f_max + df/2, df)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as resonator_spectroscopy:

            f = declare(int)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(f, f_min + device_cfg["rr_IF"], f <= f_max + device_cfg["rr_IF"], f + df):
                    update_frequency("rr", f)
                    wait(reset_time//4, "rr")
                    measure("readout"*amp(cfg["amp"]), "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(f, sweep_st)

            with stream_processing():
                I_st.buffer(len(f_vec)).average().save('I')
                Q_st.buffer(len(f_vec)).average().save('Q')
                sweep_st.buffer(len(f_vec)).average().save('f')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

        return job

class ResonatorSpectroscopyOptimal:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """readout resonator spectroscopy, varying the IF frequency"""
        cfg = self.expt_cfg["resonator_spectroscopy_optimal"]
        print(cfg)

        f_min = cfg["f_min"]
        f_max = cfg["f_max"]
        df = cfg["df"]

        f_vec = np.arange(f_min, f_max + df/2, df)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as resonator_spectroscopy:

            f = declare(int)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(f, f_min + device_cfg["rr_IF"], f <= f_max + device_cfg["rr_IF"], f + df):
                    update_frequency("rr", f)
                    wait(reset_time//4, "rr")
                    measure("clear", "rr", None,
                            demod.full("clear_integW1", I1, 'out1'),
                            demod.full("clear_integW2", Q1, 'out1'),
                            demod.full("clear_integW1", I2, 'out2'),
                            demod.full("clear_integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(f, sweep_st)

            with stream_processing():
                I_st.buffer(len(f_vec)).average().save('I')
                Q_st.buffer(len(f_vec)).average().save('Q')
                sweep_st.buffer(len(f_vec)).average().save('f')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

        return job

class QubitSpectroscopy:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """readout resonator spectroscopy, varying the IF frequency"""
        cfg = self.expt_cfg["qubit_spectroscopy"]
        print(cfg)

        f_min = cfg["f_min"]
        f_max = cfg["f_max"]
        df = cfg["df"]

        f_vec = np.arange(f_min, f_max + df/2, df)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            f = declare(int)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(f, f_min + self.device_cfg['ge_IF'][0], f <= f_max + self.device_cfg['ge_IF'][0], f + df):
                    update_frequency("qubit_mode0", f)
                    wait(reset_time//4, "qubit_mode0")
                    play(cfg["pulse_type"]*amp(cfg["amp"]), "qubit_mode0", duration=cfg["pulse_len"]//4)
                    align("qubit_mode0", "rr")
                    measure("readout", "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(f, sweep_st)

            with stream_processing():
                I_st.buffer(len(f_vec)).average().save('I')
                Q_st.buffer(len(f_vec)).average().save('Q')
                sweep_st.buffer(len(f_vec)).average().save('f')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class gePowerRabi:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """POwer Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_power_rabi"]
        print(cfg)

        a_min = cfg["a_min"]
        a_max = cfg["a_max"]
        da = cfg["da"]

        a_vec = np.arange(a_min, a_max + da/2, da)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            a = declare(fixed)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(a, a_min, a < a_max + da/2, a + da):
                    wait(reset_time//4, "qubit_mode0")
                    play(cfg["pulse_type"]*amp(a), "qubit_mode0")
                    align("qubit_mode0", "rr")
                    measure("readout", "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(a, sweep_st)

            with stream_processing():
                I_st.buffer(len(a_vec)).average().save('I')
                Q_st.buffer(len(a_vec)).average().save('Q')
                sweep_st.buffer(len(a_vec)).average().save('a')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geLengthRabi:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """Length Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_length_rabi"]
        print(cfg)

        l_min = int(cfg["l_min"])
        l_max = int(cfg["l_max"])
        dl = int(cfg["dl"])

        l_vec = np.arange(l_min, l_max + dl/2, dl)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            l = declare(int)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(l, l_min, l < l_max + dl/2, l + dl):
                    wait(reset_time//4, "qubit_mode0")
                    play(cfg["pulse_type"]*amp(cfg["pulse_amp"]), "qubit_mode0", duration=l//4)
                    align("qubit_mode0", "rr")
                    measure("readout", "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(l, sweep_st)

            with stream_processing():
                I_st.buffer(len(l_vec)).average().save('I')
                Q_st.buffer(len(l_vec)).average().save('Q')
                sweep_st.buffer(len(l_vec)).average().save('l')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class gePhaseRamsey:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """Length Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_phase_ramsey"]
        print(cfg)

        t_min = int(cfg["t_min"])
        t_max = int(cfg["t_max"])
        dt = int(cfg["dt"])

        t_vec = np.arange(t_min, t_max + dt/2, dt)

        omega = 2*np.pi*cfg["ramsey_freq"]

        dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            t = declare(int)
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
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):
                assign(phi, 0)
                with for_(t, t_min, t < t_max + dt/2, t + dt):
                    wait(reset_time//4, "qubit_mode0")
                    play("pi2", "qubit_mode0")
                    frame_rotation_2pi(phi, "qubit_mode0")
                    wait(t, "qubit_mode0")
                    play("pi2", "qubit_mode0")
                    align("qubit_mode0", "rr")
                    measure("readout", "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    assign(phi, phi + dphi)

                    save(I, I_st)
                    save(Q, Q_st)
                    save(t, sweep_st)

            with stream_processing():
                I_st.buffer(len(t_vec)).average().save('I')
                Q_st.buffer(len(t_vec)).average().save('Q')
                sweep_st.buffer(len(t_vec)).average().save('t')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geT1:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """Length Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_t1"]
        print(cfg)

        t_min = int(cfg["t_min"])
        t_max = int(cfg["t_max"])
        dt = int(cfg["dt"])

        t_vec = np.arange(t_min, t_max + dt/2, dt)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            t = declare(int)
            t_actual = declare(int)
            i = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(t, t_min, t < t_max + dt/2, t + dt):
                    wait(reset_time//4, "qubit_mode0")
                    play("pi", "qubit_mode0")
                    wait(t, "qubit_mode0")
                    align("qubit_mode0", "rr")
                    measure("readout", "rr", None,
                            demod.full("integW1", I1, 'out1'),
                            demod.full("integW2", Q1, 'out1'),
                            demod.full("integW1", I2, 'out2'),
                            demod.full("integW2", Q2, 'out2'))

                    assign(I, I1 - Q2)
                    assign(Q, I2 + Q1)

                    save(I, I_st)
                    save(Q, Q_st)
                    assign(t_actual, 4*t)
                    save(t_actual, sweep_st)

            with stream_processing():
                I_st.buffer(len(t_vec)).average().save('I')
                Q_st.buffer(len(t_vec)).average().save('Q')
                sweep_st.buffer(len(t_vec)).average().save('t')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class gePowerRabiActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config file for the QM

    def expt(self):
        """POwer Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_power_rabi"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        a_min = cfg["a_min"]
        a_max = cfg["a_max"]
        da = cfg["da"]

        a_vec = np.arange(a_min, a_max + da/2, da)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            a = declare(fixed)
            i = declare(int)

            I = declare(fixed)
            res = declare(bool)

            I_st = declare_stream()
            res_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(a, a_min, a < a_max + da/2, a + da):
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    wait(reset_time//50, 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    play(cfg["pulse_type"]*amp(a), "qubit_mode0")
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)
                    save(a, sweep_st)

            with stream_processing():
                res_st.boolean_to_int().buffer(len(a_vec)).average().save('res')
                I_st.buffer(len(a_vec)).average().save('I')
                sweep_st.buffer(len(a_vec)).average().save('a')

        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geWeakRabiActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config file for the QM

    def expt(self):
        """POwer Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_weak_rabi"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        a_min = cfg["a_min"]
        a_max = cfg["a_max"]
        da = cfg["da"]

        a_vec = np.arange(a_min, a_max + da/2, da)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            a = declare(fixed)
            i = declare(int)

            I = declare(fixed)
            res = declare(bool)

            I_st = declare_stream()
            res_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(a, a_min, a < a_max + da/2, a + da):
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    wait(reset_time//50, 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    play(cfg["pulse_type"]*amp(a), "qubit_mode0", duration=cfg["pule_le"]//4)
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)
                    save(a, sweep_st)

            with stream_processing():
                res_st.boolean_to_int().buffer(len(a_vec)).average().save('res')
                I_st.buffer(len(a_vec)).average().save('I')
                sweep_st.buffer(len(a_vec)).average().save('a')

        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geLengthRabiActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """POwer Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_length_rabi"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        l_min = int(cfg["l_min"])
        l_max = int(cfg["l_max"])
        dl = int(cfg["dl"])

        l_vec = np.arange(l_min, l_max + dl/2, dl)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            l = declare(int)
            i = declare(int)

            I = declare(fixed)
            res = declare(bool)

            I_st = declare_stream()
            res_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(l, l_min, l < l_max + dl/2, l + dl):
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    wait(reset_time//50, 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    play(cfg["pulse_type"]*amp(cfg["pulse_amp"]), "qubit_mode0", duration=l//4)
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)
                    save(l, sweep_st)

            with stream_processing():
                res_st.boolean_to_int().buffer(len(l_vec)).average().save('res')
                I_st.buffer(len(l_vec)).average().save('I')
                sweep_st.buffer(len(l_vec)).average().save('l')

        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class gePhaseRamseyActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """Length Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_phase_ramsey"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        t_min = int(cfg["t_min"])
        t_max = int(cfg["t_max"])
        dt = int(cfg["dt"])

        t_vec = np.arange(t_min, t_max + dt/2, dt)

        omega = 2*np.pi*cfg["ramsey_freq"]

        dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            t = declare(int)
            t_actual = declare(int)
            i = declare(int)

            phi = declare(fixed)

            I = declare(fixed)
            res = declare(bool)

            I_st = declare_stream()
            res_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):
                assign(phi, 0)
                with for_(t, t_min, t < t_max + dt/2, t + dt):
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    wait(reset_time//50, 'qubit_mode0')
                    play("pi2", "qubit_mode0")
                    frame_rotation_2pi(phi, "qubit_mode0")
                    wait(t, "qubit_mode0")
                    play("pi2", "qubit_mode0")
                    align("qubit_mode0", "rr")
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)

                    assign(phi, phi + dphi)
                    assign(t_actual, 4*t)

                    save(t_actual, sweep_st)

            with stream_processing():
                I_st.buffer(len(t_vec)).average().save('I')
                res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
                sweep_st.buffer(len(t_vec)).average().save('t')

        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geT1ActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """Length Rabi, varying the amplitude of a Gaussian pulse"""
        cfg = self.expt_cfg["ge_t1"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        t_min = int(cfg["t_min"])
        t_max = int(cfg["t_max"])
        dt = int(cfg["dt"])

        t_vec = np.arange(t_min, t_max + dt/2, dt)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        with program() as experiment:

            t = declare(int)
            t_actual = declare(int)
            i = declare(int)

            I = declare(fixed)
            res = declare(bool)

            I_st = declare_stream()
            res_st = declare_stream()
            sweep_st = declare_stream()

            with for_(i, 0, i < avgs, i+1):

                with for_(t, t_min, t < t_max + dt/2, t + dt):
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    wait(reset_time//50, 'qubit_mode0')
                    play("pi", "qubit_mode0")
                    wait(t, "qubit_mode0")
                    align("qubit_mode0", "rr")
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)

                    assign(t_actual, 4*t)
                    save(t_actual, sweep_st)

            with stream_processing():
                I_st.buffer(len(t_vec)).average().save('I')
                res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
                sweep_st.buffer(len(t_vec)).average().save('t')

        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

class geDisctiminator:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def training_measurement(self, readout_pulse, use_opt_weights, lsb=True):
        I = declare(fixed)
        Q = declare(fixed, value=0)
        I1 = declare(fixed)
        Q1 = declare(fixed)
        I2 = declare(fixed)
        Q2 = declare(fixed)
        res = declare(bool)

        I_st = declare_stream()
        Q_st = declare_stream()
        adc_st = declare_stream(adc_trace=True)
        if use_opt_weights:
            discriminator.measure_state(readout_pulse, "out1", "out2", res, I=I)
        else:
            if readout_pulse=="clear":
                measure("clear", "rr", adc_st,
                        demod.full("clear_integW1", I1, 'out1'),
                        demod.full("clear_integW2", Q1, 'out1'),
                        demod.full("clear_integW1", I2, 'out2'),
                        demod.full("clear_integW2", Q2, 'out2'))
            elif readout_pulse=="readout":
                measure("readout", "rr", adc_st,
                        demod.full("integW1", I1, 'out1'),
                        demod.full("integW2", Q1, 'out1'),
                        demod.full("integW1", I2, 'out2'),
                        demod.full("integW2", Q2, 'out2'))

            if lsb == False:
                assign(I, I1 + Q2)
                assign(Q, -Q1 + I2)
            else:
                assign(I, I1 - Q2)
                assign(Q, Q1 + I2)
        return I, Q

    def expt(self):
        cfg = self.expt_cfg["ge_discriminator"]
        print(cfg)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])
        use_opt_weights = False
        lsb = True
        qmm = QuantumMachinesManager()

        discriminator = TwoStateDiscriminator(qmm=qmm,
                                              config=self.opx_cfg,
                                              update_tof=False,
                                              rr_qe='rr',
                                              path=cfg["disc_file"],
                                              lsb=lsb)

        with program() as training_program:
            n = declare(int)
            # I = declare(fixed)
            # Q = declare(fixed, value=0)
            # I1 = declare(fixed)
            # Q1 = declare(fixed)
            # I2 = declare(fixed)
            # Q2 = declare(fixed)
            # res = declare(bool)
            #
            # I_st = declare_stream()
            # Q_st = declare_stream()
            # adc_st = declare_stream(adc_trace=True)

            with for_(n, 0, n < avgs, n + 1):

                wait(reset_time//4, "rr")
                I_st, Q_st = self.training_measurement(cfg["pulse"], use_opt_weights=use_opt_weights, lsb=lsb)
                # save(I, I_st)
                # save(Q, Q_st)

                align("qubit_mode0", "rr")

                wait(reset_time//4, "qubit_mode0")
                play("pi", "qubit_mode0")
                align("qubit_mode0", "rr")
                I_st, Q_st = self.training_measurement(cfg["pulse"], use_opt_weights=use_opt_weights, lsb=lsb)
                # save(I, I_st)
                # save(Q, Q_st)

            with stream_processing():
                I_st.save_all('I')
                Q_st.save_all('Q')
                adc_st.input1().with_timestamps().save_all("adc1")
                adc_st.input2().save_all("adc2")

        # training + testing to get fidelity:
        discriminator.train(program=training_program, plot=True, dry_run=False, use_hann_filter=False, correction_method='robust')

        with program() as benchmark_readout:
        
            n = declare(int)
            res = declare(bool)
            I = declare(fixed)
            Q = declare(fixed)
        
            res_st = declare_stream()
            I_st = declare_stream()
            Q_st = declare_stream()
        
            with for_(n, 0, n < avgs, n + 1):
        
                wait(reset_time//4, "rr")
                discriminator.measure_state(cfg["pulse"], "out1", "out2", res, I=I, Q=Q)
                save(res, res_st)
                save(I, I_st)
                save(Q, Q_st)
        
                align("qubit_mode0", "rr")
        
                wait(reset_time//4, "qubit_mode0")
                play("pi", "qubit_mode0")
                align("qubit_mode0", "rr")
                discriminator.measure_state(cfg["pulse"], "out1", "out2", res, I=I, Q=Q)
                save(res, res_st)
                save(I, I_st)
                save(Q, Q_st)
        
                seq0 = [0, 1] * N
        
            with stream_processing():
                res_st.save_all('res')
                I_st.save_all('I')
                Q_st.save_all('Q')
        
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(benchmark_readout, duration_limit=0, data_limit=0)

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        res = result_handles.get('res').fetch_all()['value']
        I = result_handles.get('I').fetch_all()['value']
        Q = result_handles.get('Q').fetch_all()['value']

        p_s = np.zeros(shape=(2, 2))
        for i in range(2):
            res_i = res[np.array(seq0) == i]
            p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

        if plot:
            plt.figure()
            plt.hist(I[np.array(seq0) == 0], 50)
            plt.hist(I[np.array(seq0) == 1], 50)
            plt.plot([discriminator.get_threshold()] * 2, [0, 60], 'g')
            plt.show()

            plt.figure()
            plt.plot(I, Q, '.')
            labels = ['g', 'e']

            plt.figure()
            ax = plt.subplot()
            sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')

            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('Prepared labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)

            plt.show()

        return p_s

class gePiCalActiveReset:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config file for the QM

    def expt(self):
        """Power Rabi, varying the amplitude of a Gaussian pulse. The sequence contains
        1. more than 1 pi or pi/1 pulse to amplify any voltage, decay, frequency errors
        2. Once calibrated, we can do all XY pulse sequence to tease out other error syndromes
        """
        cfg = self.expt_cfg["ge_pi_cal"]
        print(cfg)

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])
        n_pi_pulses = int(cfg["n_pi_pulses"])
        n_pi_2_pulses = int(cfg["n_pi_2_pulses"])

        a_pi_min= 0.95*self.device_cfg["qubit_params"]["pi_amps"][self.device_cfg["storage_params"]["storage_mode"]]
        a_pi_max = 1.05*self.device_cfg["qubit_params"]["pi_amps"][self.device_cfg["storage_params"]["storage_mode"]]

        a_pi_2_min= 0.95*self.device_cfg["qubit_params"]["pi_2_amps"][self.device_cfg["storage_params"]["storage_mode"]]
        a_pi_2_max = 1.05*self.device_cfg["qubit_params"]["pi_2_amps"][self.device_cfg["storage_params"]["storage_mode"]]

        da_pi = (a_pi_max - a_pi_min)/100
        a_pi_vec = np.arange(a_pi_min, a_pi_max + da_pi/2, da_pi)

        da_pi_2 = (a_pi_2_max - a_pi_2_min)/100
        a_pi_2_vec = np.arange(a_pi_2_min, a_pi_2_max + da_pi_2/2, da_pi_2)

        with program() as experiment:

            n = declare(int)
            i = declare(int)
            a = declare(fixed)
            res = declare(bool)
            I = declare(fixed)

            I_pi_st = declare_stream()
            I_pi_2_st = declare_stream()
            a_pi_st = declare_stream()
            a_pi_2_st = declare_stream()

            with for_(n, 0, n < N, n + 1):

                with for_(a, a_pi_min, a < a_pi_max + da_pi/2, a + da_pi):

                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    align('qubit_mode0', 'rr')
                    wait(reset_time//50, 'qubit_mode0')
                    play('pi2', 'qubit_mode0') #amplifies any errors on the equator
                    with for_(i, 0, i < n_pi_pulses, i+1):
                        play('gaussian'*amp(a), 'qubit_mode0')
                    play('pi2', 'qubit_mode0') #amplifies any errors on the equator
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, I_pi_st)
                    save(a, a_pi_st)

                with for_(a, a_pi_2_min, a < a_pi_2_max + da_pi_2/2, a + da_pi_2):

                    discriminator.measure_state("clear", "out1", "out2", res, I=I)
                    align('qubit_mode0', 'rr')
                    play('pi', 'qubit_mode0', condition=res)
                    align('qubit_mode0', 'rr')
                    wait(reset_time//50, 'qubit_mode0')
                    play('pi2', 'qubit_mode0')
                    with for_(i, 0, i < n_pi_2_pulses, i+1):
                        play('gaussian'*amp(a), 'qubit_mode0')
                    play('pi2', 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, I_pi_2_st)
                    save(a, a_pi_2_st)

            with stream_processing():
                I_pi_st.boolean_to_int().buffer(len(a_pi_vec)).average().save('I_pi')
                I_pi_2_st.boolean_to_int().buffer(len(a_pi_2_vec)).average().save('I_pi_2')
                a_pi_st.buffer(len(a_pi_vec)).average().save('a_pi')
                a_pi_2_st.buffer(len(a_pi_2_vec)).average().save('a_pi_2')

                qm = qmm.open_qm(self.opx_cfg)
                job = qm.execute(experiment, duration_limit=0, data_limit=0)

                return job

class StorageSpectroscopy:
    def __init__(self, device_cfg, expt_cfg, opx_config):
        self.device_cfg = device_cfg #all the device related configuration
        self.expt_cfg = expt_cfg #experiment related parameters
        self.opx_cfg = opx_config # config files for the QM

    def expt(self):
        """readout resonator spectroscopy, varying the IF frequency. The sequence is as follows:
        1. Drive/put a coherent state in the storage
        2. Use qubit as an ancilla by probing the qubit with a resolved pi pulse centered at n=0 peak
        3. When there is a >=1 photon, the qubit will not flip and the readout will give us a contrast
        """
        cfg = self.expt_cfg["storage_spectroscopy"]
        print(cfg)

        f_min = cfg["f_min"]
        f_max = cfg["f_max"]
        df = cfg["df"]

        f_vec = np.arange(f_min, f_max + df/2, df)

        avgs = cfg["avgs"]
        reset_time = int(cfg["reset_time"])

        desired_mode = cfg["mode"]
        self.device_cfg['storage_mode'] = desired_mode
        opx_mode = "storage_mode"+str(desired_mode)
        st_IF = self.device_cfg['storage_IF'][desired_mode]

        qmm = QuantumMachinesManager()
        discriminator = TwoStateDiscriminator(qmm=qmm, config=self.opx_cfg, update_tof=True, rr_qe='rr', path=self.device_cfg['disc_file_opt'], lsb=True)

        with program() as experiment:

            n = declare(int)        # Averaging
            f = declare(int)        # Frequencies
            res = declare(bool)
            I = declare(fixed)

            res_st = declare_stream()
            I_st = declare_stream()
            sweep_st = declare_stream()

            ###############
            # the sequence:
            ###############
            with for_(n, 0, n < avgs, n + 1):

                with for_(f,  st_IF + f_min, f <  st_IF + f_max + df/2, f + df):

                    update_frequency(opx_mode, f)
                    wait(reset_time// 4, opx_mode)# wait for the storage to relax, several T1s
                    play(cfg["pulse_type"]*amp(cfg["amp"]), opx_mode, duration=cfg["pulse_len"]//4)
                    align(opx_mode, "qubit_mode0")
                    play("res_pi", "qubit_mode0")
                    align("qubit_mode0", "rr")
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)
                    save(f, sweep_st)

            with stream_processing():

                res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
                I_st.buffer(len(f_vec)).average().save('I')
                sweep_st.buffer(len(f_vec)).average().save('f')

        qmm = QuantumMachinesManager()
        qm = qmm.open_qm(self.opx_cfg)
        job = qm.execute(experiment, duration_limit=0, data_limit=0)

        return job

test = StorageSpectroscopy(device_cfg=device_cfg, expt_cfg=expt_cfg, opx_config=opx_config)

job = test.expt()

res_handles = job.result_handles

res_handles.wait_for_all_values()

I = res_handles.get("I").fetch_all()
Q = res_handles.get("res").fetch_all()
a_vec = res_handles.get("f").fetch_all()
print ("Data collection done")

job.halt()

plt.plot(a_vec, I, '.-')
