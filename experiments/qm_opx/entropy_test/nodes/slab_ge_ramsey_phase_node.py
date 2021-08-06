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

class RamseyPhase(QuaCalNode):

    def __init__(self, t_min, t_max, dt, ramsey_freq, avgs,  **kwargs):

        super(RamseyPhase, self).__init__(**kwargs)
        self.rr =  'rr'
        self.qubit = 'qubit'
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt
        self.ramsey_freq = int(ramsey_freq)
        self.avgs = avgs

    def gauss(self, amplitude, mu, sigma, length):
        t = np.linspace(-length / 2, length / 2, length)
        gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        return [float(x) for x in gauss_wave]

    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        return config

    def run_program(self, config, context: EntropyContext):

        omega = 2*np.pi*self.ramsey_freq
        dphi = omega*self.dt*1e-9/(2*np.pi)
        t_vec = np.arange(self.t_min, self.t_max, self.dt)

        qmm = context.get_resource("qmm")
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')
        reset_time = 500000

        with program() as prog:

            t = declare(int) #array of time delays
            phi = declare(fixed)
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

                assign(phi, 0)

                with for_(t, self.t_min, t < self.t_max, t + self.dt):

                    wait(reset_time//4, self.qubit)
                    play("pi2", self.qubit)
                    wait(t, self.qubit)
                    frame_rotation_2pi(phi, self.qubit) #2pi is already multiplied to the phase
                    play("pi2", self.qubit)
                    align(self.qubit, self.rr)
                    measure("clear", "rr", None,
                            demod.full("clear_integW1", I1, 'out1'),
                            demod.full("clear_integW2", Q1, 'out1'),
                            demod.full("clear_integW1", I2, 'out2'),
                            demod.full("clear_integW2", Q2, 'out2'))

                    assign(I, I1-Q2)
                    assign(Q, I2+Q1)

                    assign(phi, phi + dphi)

                    save(I, I_st)
                    save(Q, Q_st)

            with stream_processing():
                I_st.buffer(len(t_vec)).average().save('I')
                Q_st.buffer(len(t_vec)).average().save('Q')

        qm = qmm.open_qm(config)
        job = qm.execute(prog, duration_limit=0, data_limit=0)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        I_handle = res_handles.get("I")
        Q_handle = res_handles.get("Q")
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()


        y = Q[:]
        x = t_vec[:]/1e3
        p = fitdecaysin(x, y, fitparams=None, showfit=False)
        offset = self.ramsey_freq/1e6 - p[1]
        qubit_freq = qpu_db.get('qubit', 'ge_freq').value/1e9
        qubit_if = qpu_db.get('qubit', 'ge_if').value/1e6

        nu_q_new = (qubit_freq + offset/1e3)
        if_q_new = (qubit_if + offset/1e3)

        plt.figure()
        plt.plot(x, y, 'bo',label = 'Q')
        plt.plot(x, decaysin(np.append(p, 0), x), 'r-', label=r'fit, $T_{2}^{*}$ = %.2f $\mu$s' % p[3])
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel('Q')
        plt.legend()
        plt.pause(0.1)
        plt.draw()

        print("Original qubit frequency choice =", qubit_freq,"GHz")
        print ("Oscillation freq = ", p[1]," MHz")
        print("Offset freq =", offset,"MHz")
        print("Suggested qubit frequency choice =", nu_q_new,"GHz")
        print("T2* =", p[3],"us")
        ###################
        # qpu db updating #
        ###################

        qpu_db.set('qubit', 'ge_freq', 1e9*nu_q_new, CalState.FINE)
        qpu_db.set('qubit', 'ge_if', 1e6*if_q_new, CalState.FINE)

        qpu_db.commit('ge Ramsey')

    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')
        config['elements']['qubit']['intermediate_frequency'] = qpu_db.get('qubit', 'ge_if').value
        config['mixers']['mixer_qubit'][0]['intermediate_frequency'] = qpu_db.get('qubit', 'ge_if').value

        return config



