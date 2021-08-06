import sys
sys.path.append(r"C:\Users\paint\QM\StateDiscriminator")
import os
import qm
from entropylab import *
from entropyext_cal import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scope import my_scope
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qpu_resolver import resolve
from entropylab.api.execution import EntropyContext
import tensorflow as tf
import seaborn as sns
import itertools
from TwoStateDiscriminator import TwoStateDiscriminator
import pickle

class MultiReadoutTrainBenchmark(QuaCalNode):
    def __init__(self, elements, n_pts, wait_time, train=True, **kwargs):
        super(MultiReadoutTrainBenchmark, self).__init__(**kwargs)
        self.elements = elements
        self.q_xy = [resolve.q(element, channel='xy') for element in elements]
        self.res = [resolve.res(element) for element in elements]
        self.n_pts = n_pts
        self.wait_time = wait_time
        self.execute_kwargs = {'data_limit': 0, 'duration_limit': 0}
        self.train = train

    def prepare_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:

        return config 
    
    def run_program(self, config, context: EntropyContext):
        
        qmm = context.get_resource("qmm")
        
        if self.train:
            with program() as training_program:
                
                adc_stream = declare_stream(adc_trace=True)
                I_st = declare_stream()
                Q_st = declare_stream()
                
                with for_(n, 0, n < self.n_pts, n + 1):
                    wait(self.wait_time, *self.q_xy, *self.res)
                    align()
                    for i in range(len(self.elements)):
                        reset_phase(self.res[i])
                    
                    measure("readout_pulse_g", f"rr{res_i}a", "adc",
                            dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                            dual_demod.full("integW_min_sin", "out1", "integW_cos", "out2", Q))
            save(I, 'I')
            save(Q, 'Q')
            wait(wait_time, "rr1a")

        with for_(n, 0, n < g2e, n + 1):

            align("rr1a", "rr2a")
            reset_phase(f"rr{res_i}a")
            measure("readout_pulse_e", f"rr{res_i}a", "adc",
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_min_sin", "out1", "integW_cos", "out2", Q))
            save(I, 'I')
            save(Q, 'Q')
            wait(wait_time, "rr1a")
        ##################################
        # initialize discriminator class #
        ##################################
        # path = ["training_data_" + self.q_xy[i] + ".npz" for i in range(len(self.elements))]
        # discriminator = [TwoStateDiscriminator(qmm=qmm, config=config, update_tof=False,
        #                                        rr_qe=self.res[i], path=path[i], lsb=False) for i in range(len(self.elements))]
        qubits = self.q_xy
        wait_time = self.wait_time

        all_prep_combinations = list(itertools.product(*[range(2) for k in range(len(qubits))]))
        
        with program() as benchmark_readout:
        
            n = declare(int)
            m = declare(int)
            k = declare(int)
            k_vec = declare(int, size=len(self.elements))
            a = declare(fixed)
            res = [declare(bool) for i in range(len(self.elements))]
            I = [declare(fixed) for i in range(len(self.elements))]
            Q = [declare(fixed) for i in range(len(self.elements))]
            
            prep_st = declare_stream()
            res_st = [declare_stream() for i in range(len(self.elements))]
            I_st = [declare_stream() for i in range(len(self.elements))]
            Q_st = [declare_stream() for i in range(len(self.elements))]
        
            with for_(n, 0, n < self.n_pts, n + 1):
                
                with for_(k, 0, k < 2 ** len(self.elements), k + 1):
                    decimal_to_binary(k, k_vec)
                    align()
                    wait(wait_time, *self.res, *self.q_xy)
                    for i in range(len(self.elements)):
                        assign(a, Cast.to_fixed(k_vec[i]))
                        play('pi' * amp(a), self.q_xy[i])
                    align()
                    for i in range(len(self.elements)):
                        measure("readout", self.res[i], None,
                                dual_demod.full('demod1_iw_' + self.res[i], 'out1',
                                                'demod2_iw_' + self.res[i], 'out2', I[i]),
                                dual_demod.full('minus_demod2_iw_' + self.res[i], 'out1',
                                                'demod1_iw_' + self.res[i], 'out2', Q[i]))
                    for i in range(len(self.elements)):
                        assign(res[i], I[i] < threshold[i])
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        save(res[i], res_st[i])
                        save(k_vec[i], prep_st)
                    
                # for state_prep in all_prep_combinations:
                #     align()
                #     wait(wait_time, *self.res, *self.q_xy)
                #     multi_prepare_state(state_prep, qubits)
                #     align()
                #     for i in range(len(self.elements)):
                #         discriminator[i].measure_state("readout", "out1", "out2",
                #                                        res[i], I=I[i], Q=Q[i])
                #         save(I[i], I_st[i])
                #         save(Q[i], Q_st[i])
                #         save(res[i], res_st)
                #         save(state_prep[i], prep_st)
              
            with stream_processing():
                prep_st.buffer(2 ** len(self.elements), len(self.elements)).save_all('prep')
                for i in range(len(self.elements)):
                    res_st[i].buffer(2 ** len(self.elements)).save_all('res_' + self.res[i])
                    I_st[i].buffer(2 ** len(self.elements)).save_all('I' + self.q_xy[i])
                    Q_st[i].buffer(2 ** len(self.elements)).save_all('Q' + self.q_xy[i])
        
        qm = qmm.open_qm(config)
        job = qm.execute(benchmark_readout, duration_limit=0, data_limit=0, flags=['auto-element-thread'])
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        measured_states = np.array([res_handles.get("res_" + self.res[i]).fetch_all()['value'] for i in range(len(self.res))]).swapaxes(0, -1).swapaxes(0, 1)
        prepared_states = res_handles.get('prep').fetch_all()['value']
        I = [res_handles.get("I" + self.q_xy[i]).fetch_all()['value'] for i in range(len(self.elements))]
        Q = [res_handles.get("Q" + self.q_xy[i]).fetch_all()['value'] for i in range(len(self.elements))]

        prep_state_index = np.array([all_prep_combinations.index(tuple(prepared_states[i, j])) for i in range(self.n_pts) for j in range(len(all_prep_combinations))]).flatten()
        meas_state_index = np.array([all_prep_combinations.index(tuple(measured_states[i, j])) for i in range(self.n_pts) for j in range(len(all_prep_combinations))]).flatten()

        conf_matrix = np.zeros((len(all_prep_combinations), len(all_prep_combinations)))
        for i in range(len(all_prep_combinations)):
            mask = (prep_state_index == i)
            for j in range(len(all_prep_combinations)):
                conf_matrix[i, j] = np.mean(meas_state_index[mask] == j)

        L = int(np.ceil(np.sqrt(len(all_prep_combinations))))
        for i in range(len(self.elements)):
            fig = plt.figure()
            for j in range(len(all_prep_combinations)):
                plt.subplot(L, L, j + 1)
                plt.plot(I[i][:, j], Q[i][:, j], '.', color='C%d' % j)
                plt.axis('equal')
                plt.axvline(threshold[i], color='black')
                plt.title("State Prep. " + str(all_prep_combinations[j]))
                fig.suptitle(self.res[i] + " readout signal")

        
        labels = [("{}" * len(self.q_xy)).format(*all_prep_combinations[i]) for i in range(len(all_prep_combinations))]
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        ax.set_xlabel('Measured States')
        ax.set_ylabel('Prepared States')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        fidelity = np.mean(np.diagonal(conf_matrix))
        print("Readout Fidelity: ", fidelity)

        ###################
        # qpu db updating #
        ###################
        
        # f_lo = qpu_db.get(self.res, 'f_lo').value
        # qpu_db.set(self.res, 'f_opt_rf', f_opt_if + f_lo)
                
        # qpu_db.commit()
        # print("Optimal Readout Frequency %.4f GHz" % ((f_opt_if + f_lo)/ 1e9))
        
    def update_config(self, config: QuaConfig, context: EntropyContext) -> QuaConfig:
        # qpu_db: QpuDatabaseConnection = context.get_resource('qpu_db1')
        # f_opt_rf = qpu_db.get(self.res, 'f_opt_rf').value
        # f_lo = qpu_db.get(self.res, 'f_lo').value
        # f_opt_if = f_opt_rf - f_lo
        # config["elements"][self.res]['intermediate_frequency'] = f_opt_if
        # config['mixers'][resolve.res_mixer(1)].append({'lo_frequency': f_lo,
        #                                                'intermediate_frequency': f_opt_if,
        #                                                'correction': [1, 0, 0, 1]})
        # config.update_intermediate_frequency(self.element, self.f_res,
        #                                       strict=False)
        # print("Updated IF frequency ", config["elements"][self.res]['intermediate_frequency'])
        return config
        
    