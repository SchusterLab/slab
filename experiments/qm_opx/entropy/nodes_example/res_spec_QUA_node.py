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
from scope import my_scope

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qpu_resolver import resolve
from entropylab.api.execution import EntropyContext

import numpy as np

class ResSpec(QuaCalNode):
    
      
    def __init__(self, element, fmin, fmax, df, **kwargs):
        
        super(ResSpec, self).__init__(**kwargs)
        self.element =  resolve.res(element)
        self.fmin = fmin
        self.fmax = fmax
        self.df = df
        self.execute_kwargs = {'data_limit': 0, 'duration_limit': 0}


    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        return config 
    
    def run_program(self, config, context: EntropyContext):

        qmm = context.get_resource("qmm")

        f_vec = np.arange(self.fmin, self.fmax, self.df)


        with program() as prog:
            i = declare(int)
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(i, 0, i<1000, i+1):
                with for_(f, self.fmin, f<self.fmax, f+self.df):
                    update_frequency(self.element, f)
                    wait(1000, self.element)
                    measure("readout", self.element, None, dual_demod.full('integW_cos', 'out1', 'integW_sine', 'out2', I),
                                                           dual_demod.full('integW_min_sine', 'out1', 'integW_cos', 'out2', Q))
                    save(I, I_st)
                    save(Q, Q_st)
            with stream_processing():
                I_st.buffer(len(f_vec)).average().save('I')
                Q_st.buffer(len(f_vec)).average().save('Q')
                
        qm = qmm.open_qm(config)
        job = qm.execute(prog, duration_limit=0, data_limit=0)
        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        I = result_handles.I.fetch_all()
        Q = result_handles.Q.fetch_all()
        sig = I + 1j*Q
        plt.figure()
        plt.title(self.element)
        mag = np.abs(sig)
        plt.plot(f_vec, mag)
        p1, p2 = plt.ginput(2, timeout=60)
        x1, y1 = p1
        x2, y2 = p2
        x1, x2 = np.min([x1, x2]), np.max([x1, x2])
        
        mask = (x1 < f_vec) * (f_vec < x2)
        minidx = np.argmin(mag[mask])
        f_res = f_vec[mask][minidx]
        
        plt.title(self.element + ": %.4f MHz" % (f_res / 1e6))
        plt.axvline(f_res, color='black', ls='--')
        plt.pause(0.1)
        plt.draw()
        
        ###################
        # qpu db updating #
        ###################
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db1')
        
        qpu_db.set(self.element, 'f_intermediate', f_res)
        f_lo = qpu_db.get(self.element, 'f_lo').value
        f_rf = f_lo + f_res
        qpu_db.set(self.element, 'f_rf', f_rf)
                
        qpu_db.commit()
        print("Resonator Frequency %.4f GHz" % (f_rf / 1e9))

        
    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db1')
        f_res = qpu_db.get(self.element, 'f_intermediate').value
        f_lo = qpu_db.get(self.element, 'f_lo').value
        config["elements"][self.element]['intermediate_frequency'] = f_res
        config['mixers'][resolve.res_mixer(1)].append({'lo_frequency': f_lo,
                                                       'intermediate_frequency': f_res,
                                                       'correction': [1, 0, 0, 1]})
        # config.update_intermediate_frequency(self.element, self.f_res,
        #                                       strict=False)
        print("Updated IF frequency ", config["elements"][self.element]['intermediate_frequency'])
        return config
        
    
            
