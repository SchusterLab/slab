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

class AdcOffsetsNode(QuaCalNode):

    # def __init__(self, jpa=False, **kwargs):
    #
    #     super(AdcOffsetsNode, self).__init__(**kwargs)
    #     # self.element =  'rr'
    #     self.jpa = jpa

    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        return config

    def run_program(self, config, context: EntropyContext):

        qmm = context.get_resource("qmm")
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')

        with program() as prog:
            adc_st = declare_stream(adc_trace=True)
            i = declare(int)
            with for_(i, 0, i < 1000, i+1):
                wait(200000//4, 'rr')
                reset_phase('rr')
                if True:
                    reset_phase('jpa_pump')
                    align('rr', 'jpa_pump')
                    play('pump_square', 'jpa_pump')
                measure('clear', "rr", adc_st) # 400ns

            with stream_processing():
                adc_st.input1().average().save("adcI")
                adc_st.input2().average().save("adcQ")

        qm = qmm.open_qm(config)
        job = qm.execute(prog, duration_limit=0, data_limit=0)
        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        adcI = result_handles.adcI.fetch_all()
        adcQ = result_handles.adcQ.fetch_all()
        adcI_offset = np.mean(adcI) * 2**-12
        adcQ_offset = np.mean(adcQ) * 2**-12
        plt.figure()
        plt.title('TOF and ADC offset calibration - raw data')
        plt.plot(adcI, label = 'ADCI offset: {}'.format(adcI_offset))
        plt.plot(adcQ, label = 'ADCQ offset: {}'.format(adcQ_offset))
        plt.legend()
        plt.pause(0.1)
        plt.draw()

        ###################
        # qpu db updating #
        ###################

        qpu_db.set('readout', 'adc1_offset', config['controllers']['con1']['analog_inputs'][1]['offset']-adcI_offset, CalState.FINE)
        qpu_db.set('readout', 'adc2_offset', config['controllers']['con1']['analog_inputs'][2]['offset']-adcQ_offset, CalState.FINE)
        qpu_db.commit('commit_test')

    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db')

        config['controllers']['con1']['analog_inputs'][1]['offset'] = qpu_db.get('readout', 'adc1_offset').value
        config['controllers']['con1']['analog_inputs'][2]['offset'] = qpu_db.get('readout', 'adc2_offset').value
        return config



