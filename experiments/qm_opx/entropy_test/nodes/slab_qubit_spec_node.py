# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:56:21 2021

@author: paint
"""

import os
import qm
from entropylab import *
from entropyext_cal import *
import numpy as np
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from entropylab.api.execution import EntropyContext
from entropylab_qpudb import QuaCalNode
import numpy as np
from slab.dsfit import*

class QubitSpec(QuaCalNode):

    def __init__(self, f_min, f_max, df, pulse_len, pulse_volt,  avgs,  **kwargs):

        super(QubitSpec, self).__init__(**kwargs)
        self.rr=  'rr'
        self.qubit = 'qubit'
        self.f_min = f_min
        self.f_max = f_max
        self.df = df
        self.pulse_len = int(pulse_len)
        self.pulse_volt = pulse_volt
        self.avgs = avgs

    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        config['pulses']['CW']['length'] = self.pulse_len
        config['waveforms']['const_wf']['sample'] = self.pulse_volt
        return config

    def run_program(self, config, context: EntropyContext):

        qmm = context.get_resource("qmm")
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')
        reset_time = 500000
        f_vec = np.arange(self.f_min, self.f_max, self.df)

        with program() as prog:

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

            with for_(i, 0, i < self.avgs, i+1):

                with for_(f, self.f_min, f < self.f_max, f + self.df):

                    update_frequency(self.qubit, f)
                    wait(reset_time//4, self.qubit)
                    play("CW", self.qubit, duration=self.pulse_len//4)
                    align(self.qubit, self.rr)
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
                I_st.buffer(len(f_vec)).average().save('I')
                Q_st.buffer(len(f_vec)).average().save('Q')

        qm = qmm.open_qm(config)
        job = qm.execute(prog, duration_limit=0, data_limit=0)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        I_handle = res_handles.get("I")
        Q_handle = res_handles.get("Q")
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()

        power = I**2 + Q**2
        res_f = int(fitlor(f_vec, power, showfit=False)[2])

        plt.figure()
        plt.plot(f_vec, power, 'b.-')
        plt.axvline(x=res_f, linestyle='--', color='k')
        plt.pause(0.1)
        plt.draw()

        print('resonance frequency from the fit = %.3f' %res_f)
        ###################
        # qpu db updating #
        ###################

        qpu_db.set('qubit', 'ge_if', res_f, CalState.FINE)
        qpu_db.commit('qubit_spec')

    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')

        config['elements']['qubit']['intermediate_frequency'] = qpu_db.get('qubit', 'ge_if').value
        config['mixers']['mixer_qubit'][0]['intermediate_frequency'] = qpu_db.get('qubit', 'ge_if').value

        return config



