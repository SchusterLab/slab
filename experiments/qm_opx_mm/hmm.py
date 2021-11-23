import numpy as np

##----------------------------------------------------------------##
def forward(meas_seq, T, E):
    num_meas = len(meas_seq)
    N = T.shape[0]
    alpha = np.zeros((num_meas, N))
    pi = [0.25, 0.25, 0.25, 0.25]
    alpha[0] = pi*E[:,meas_seq[0]] #No information in the first measurement with the prior above
    for t in range(1, num_meas):
        alpha[t] = alpha[t-1].dot(T) * E[:, meas_seq[t]]
    return alpha

def backward(meas_seq, T, E):
    N = T.shape[0]
    num_meas = len(meas_seq)
    beta = np.zeros((N,num_meas))
    beta[:,-1] = 0.25 #No information in the last measuremnt
    for t in reversed(range(num_meas-1)):
        for n in range(N):
            beta[n,t] = sum(beta[:,t+1] * T[n,:] * E[:, meas_seq[t+1]])
    return beta

def likelihood(meas_seq, T, E):
    # returns log P(Y  \mid  model)
    # using the forward part of the forward-backward algorithm
    return  forward(meas_seq, T, E)[-1].sum()

def gamma(meas_seq, T, E):
    alpha = forward(meas_seq, T, E)
    beta  = backward(meas_seq, T, E)
    obs_prob = likelihood(meas_seq, T, E)
    return (np.multiply(alpha, beta.T) / obs_prob)

qubit_params = {'t1':100, 't2':130, 'nth':5e-2}
cavity_params = {'t1':5e2, 'nth':0.001}
readout_params = {'length':3.2, 'trigger':7.2, 'pi_pulse':3, 'g_inf':0.02, 'e_inf':0.05}

qubit_t1 = qubit_params['t1']
qubit_t2 = qubit_params['t2']
qubit_nth = qubit_params['nth']
cavity_t1 = cavity_params['t1']
cavity_nth = cavity_params['nth']

readout_len = readout_params['length']
trigger_period = readout_params['trigger']
pi_length = readout_params['pi_pulse']

"""Readout fidelities from an independent measurement"""
g_infidelity, e_infidelity = 0.0194, 0.06647

fstate_in = 0

if fstate_in == 0:
    cavity_t1 = cavity_t1/1
    Pnm =  cavity_nth * (1-np.exp(-trigger_period/cavity_t1))
else:
    cavity_t1 = cavity_t1/fstate_in
    Pnm =  (1-np.exp(-trigger_period/cavity_t1)) + cavity_nth * (1-np.exp(-strigger_period/cavity_t1))

Pmn = 0 + 0 #assuming that the population at (n+1) is negligible and (n-1) we will estimate\
Pge = qubit_nth * (1-np.exp(-trigger_period/qubit_t1)) +\
    (1-np.exp(-pi_length/qubit_t2))
Peg = (1-np.exp(-trigger_period/qubit_t1)) + \
    (1-np.exp(-pi_length/qubit_t2))

T = np.asarray([[(1-Pmn)*(1-Pge), (1-Pmn)*Pge, Pmn*Pge, Pmn*(1-Pge)],
     [(1-Pmn)*Peg, (1-Pmn)*(1-Peg), Pmn*(1-Peg), Pmn*Peg],
     [Pnm*(1-Pge), Pnm*Pge, (1-Pnm)*Pge, (1-Pnm)*(1-Pge)],
     [Pnm*Peg, Pnm*(1-Peg), (1-Pnm)*(1-Peg), (1-Pnm)*Peg]])

E = 0.5*np.asarray([[1-g_infidelity, g_infidelity],
    [e_infidelity, 1- e_infidelity],
    [1-g_infidelity, g_infidelity],
    [e_infidelity, 1- e_infidelity]])

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_IQ import config
import matplotlib.pyplot as plt

###################
# The QUA program #
###################

beta_vec = [0.25]*4
# meas_seq_vec = [0, 0, 0, 1, 0]

# expected #
# plt.figure()
# backward_probs = backward(meas_seq_vec, T, E)
# Norm = np.sum(backward_probs, axis=0)
# NormMat = np.reshape(np.tile(Norm, 4), (5, 4)).transpose()
# print(Norm)
# for i in range(4):
#     plt.plot(backward_probs[i], '.')

def backward_qua(meas_vec, T,E,save_all=True):
    t = declare(int)
    n = declare(int)

    E_0 = declare(fixed, value=E[:, 0].tolist())
    E_1 = declare(fixed, value=E[:, 1].tolist())
    T_vec0 = declare(fixed, value=T[0, :].tolist())
    T_vec1 = declare(fixed, value=T[1, :].tolist())
    T_vec2 = declare(fixed, value=T[2, :].tolist())
    T_vec3 = declare(fixed, value=T[3, :].tolist())
    beta = declare(fixed, value=beta_vec)
    b0_stream = declare_stream()
    b1_stream = declare_stream()
    b2_stream = declare_stream()
    b3_stream = declare_stream()
    E_temp = declare(fixed)

    beta0_temp = declare(fixed)
    beta1_temp = declare(fixed)
    beta2_temp = declare(fixed)
    beta3_temp = declare(fixed)

    with for_(t, meas_vec.length()  - 1, t >= 0, t - 1):
        assign(beta0_temp, 0)
        assign(beta1_temp, 0)
        assign(beta2_temp, 0)
        assign(beta3_temp, 0)
        with for_(n, 0, n < T.shape[1], n + 1):
            assign(E_temp, Util.cond(meas_vec [t] == 1, E_1[n], E_0[n]))
            assign(beta0_temp, beta0_temp + (beta[n] * T_vec0[n] * E_temp))
            assign(beta1_temp, beta1_temp + (beta[n] * T_vec1[n] * E_temp))
            assign(beta2_temp, beta2_temp + (beta[n] * T_vec2[n] * E_temp))
            assign(beta3_temp, beta3_temp + (beta[n] * T_vec3[n] * E_temp))

        assign(beta[0], beta0_temp)
        assign(beta[1], beta1_temp)
        assign(beta[2], beta2_temp)
        assign(beta[3], beta3_temp)
        if save_all:
            save(beta[0], b0_stream)
            save(beta[1], b1_stream)
            save(beta[2], b2_stream)
            save(beta[3], b3_stream)
    if not save_all:
        save(beta[0], b0_stream)
        save(beta[1], b1_stream)
        save(beta[2], b2_stream)
        save(beta[3], b3_stream)

    with stream_processing():
        b0_stream.save_all('b0')
        b1_stream.save_all('b1')
        b2_stream.save_all('b2')
        b3_stream.save_all('b3')


# with program() as hmm:
#     ####
#     backward_qua(meas_seq_vec, T, E, save_all=True)
#     ####
#
#
# ######################################
# # Open Communication with the Server #
# ######################################
# qmm = QuantumMachinesManager()
#
# #####################
# # Simulate Program #
# #####################
#
# simulation_config = SimulationConfig(
#     duration=10000)
#
# job = qmm.simulate(config, hmm, simulation_config)
#
# ### Results
# samples = job.get_simulated_samples()
# res_handles = job.result_handles
# res_handles.wait_for_all_values()
# back_prob_qua = np.zeros((T.shape[0], len(meas_seq_vec)))
# back_prob_qua[0, :] = res_handles.b0.fetch_all()['value']
# back_prob_qua[1, :] = res_handles.b1.fetch_all()['value']
# back_prob_qua[2, :] = res_handles.b2.fetch_all()['value']
# back_prob_qua[3, :] = res_handles.b3.fetch_all()['value']
#
# ### Comparison
# plt.figure()
# for i in range(T.shape[0]):
#     plt.plot(back_prob_qua[i, 3::-1], '.')
#     print(f'Qua beta[{i}] = ', back_prob_qua[i, 3::-1])
#     print(f'Python beta[{i}] = ', backward_probs[i, :-1])
#     print('Python - Qua / Python = ', (backward_probs[i, :-1] - back_prob_qua[i, 3::-1]) / backward_probs[i, :-1])
