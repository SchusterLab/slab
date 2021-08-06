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

class PowerRabi(QuaCalNode):

    def __init__(self, a_min, a_max, da, pulse_len, avgs,  **kwargs):

        super(PowerRabi, self).__init__(**kwargs)
        self.rr=  'rr'
        self.qubit = 'qubit'
        self.a_min = a_min
        self.a_max = a_max
        self.da = da
        self.pulse_len = int(pulse_len)
        self.avgs = avgs

    def gauss(self, amplitude, mu, sigma, length):
        t = np.linspace(-length / 2, length / 2, length)
        gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        return [float(x) for x in gauss_wave]

    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        config['pulses']['gaussian_pulse']['length'] = self.pulse_len
        config['waveforms']['gauss_wf']['samples'] = self.gauss(0.45, 0, self.pulse_len//4, self.pulse_len)
        return config

    def run_program(self, config, context: EntropyContext):

        qmm = context.get_resource("qmm")
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')
        reset_time = 500000
        a_vec = np.arange(self.a_min, self.a_max, self.da)

        with program() as prog:

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

            with for_(i, 0, i < self.avgs, i+1):

                with for_(a, self.a_min, a < self.a_max, a + self.da):

                    wait(reset_time//4, self.qubit)
                    play('gaussian'*amp(a), self.qubit)
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
                I_st.buffer(len(a_vec)).average().save('I')
                Q_st.buffer(len(a_vec)).average().save('Q')

        qm = qmm.open_qm(config)
        job = qm.execute(prog, duration_limit=0, data_limit=0)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        I_handle = res_handles.get("I")
        Q_handle = res_handles.get("Q")
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()


        y = Q[2:]
        x = a_vec[2:]
        p = fitdecaysin(x, y,fitparams = None, showfit=False)

        plt.figure()
        plt.plot(x, y, 'bo',label = 'Q')
        plt.plot(x, decaysin(np.append(p, 0), x), 'r-', label=r'fit')
        pi_amp = 1/2/p[1]
        plt.axvline(pi_amp, color='k', linestyle='--')
        plt.xlabel('AWG Amp (a.u.)')
        plt.ylabel('Q')
        plt.legend()
        plt.pause(0.1)
        plt.draw()

        print('AWG amplitude for a pi pulse = %.4f' %pi_amp)
        ###################
        # qpu db updating #
        ###################

        qpu_db.set('qubit', 'ge_fast_pi_amp', pi_amp, CalState.COARSE)
        qpu_db.set('qubit', 'ge_fast_pi_len', self.pulse_len, CalState.COARSE)
        qpu_db.set('qubit', 'ge_fast_pi2_amp', pi_amp/2, CalState.COARSE)
        qpu_db.set('qubit', 'ge_fast_pi2_len', self.pulse_len, CalState.COARSE)

        qpu_db.commit('ge power Rabi')

    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')
        config['pulses']['pi_pulse']['length'] = qpu_db.get('qubit', 'ge_fast_pi_len').value
        config['waveforms']['pi_wf']['samples'] = self.gauss(0.45 * qpu_db.get('qubit', 'ge_fast_pi_amp').value, 0, qpu_db.get('qubit', 'ge_fast_pi_len').value//4, qpu_db.get('qubit', 'ge_fast_pi_len').value)
        config['pulses']['pi2_pulse']['length'] = qpu_db.get('qubit', 'ge_fast_pi2_len').value
        config['waveforms']['pi2_wf']['samples'] = self.gauss(0.45 * qpu_db.get('qubit', 'ge_fast_pi2_amp').value , 0, qpu_db.get('qubit', 'ge_fast_pi2_len').value//4, qpu_db.get('qubit', 'ge_fast_pi2_len').value)
        return config



