from pylab import*
import matplotlib.pyplot as plt
import seaborn as sns
from h5py import File
import pandas as pd
import json
import numpy as np
from datetime import datetime
import scipy as sc
from scipy.optimize import curve_fit
from h5py import File
import os
from slab.dataanalysis import get_next_filename

########################
class hmm_analysis_2:

    def __init__(self, qubit_params = None, cavity_params = None, readout_params = None):
        
        self.qubit_params = qubit_params
        self.cavity_params = cavity_params
        self.readout_params = readout_params

        """All the timescales are in μs"""
        self.qubit_t1 = qubit_params['t1']
        self.qubit_t2 = qubit_params['t2']
        self.qubit_nth = qubit_params['nth']
        
        self.cavity_t1 = cavity_params['t1']
        self.cavity_nth = cavity_params['nth']
        
        self.readout_len = readout_params['length']        
        self.trigger_period = readout_params['trigger']
        self.pi_length = readout_params['pi_pulse']
        self.g_inf = readout_params['g_inf']        
        self.e_inf = readout_params['e_inf']        

    ##----------------------------------------------------------------##
    def forward(self, meas_seq, T, E):
        num_meas = len(meas_seq)
        N = T.shape[0]
        alpha = zeros((num_meas, N))
        pi = [0.25, 0.25, 0.25, 0.25]
        alpha[0] = pi*E[:,meas_seq[0]]
        for t in range(1, num_meas):
            alpha[t] = alpha[t-1].dot(T) * E[:, meas_seq[t]]
        return alpha

    def backward(self, meas_seq, T, E):
        N = T.shape[0]
        num_meas = len(meas_seq)
        beta = zeros((N,num_meas))
        beta[:,-1:] = 1
        for t in reversed(range(num_meas-1)):
            for n in range(N):
                beta[n,t] = sum(beta[:,t+1] * T[n,:] * E[:, meas_seq[t+1]])
        return beta

    def likelihood(self, meas_seq, T, E):
        # returns log P(Y  \mid  model)
        # using the forward part of the forward-backward algorithm
        return  self.forward(meas_seq, T, E)[-1].sum()

    def gamma(self, meas_seq, T, E):
        alpha = self.forward(meas_seq, T, E)
        beta  = self.backward(meas_seq, T, E)
        obs_prob = self.likelihood(meas_seq, T, E)
        return (multiply(alpha, beta.T) / obs_prob)

    def viterbi(self, meas_seq, T, E):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        num_meas = len(meas_seq)
        N = T.shape[0]
        delta = zeros((num_meas, N))
        psi = zeros((num_meas, N))
        pi = [0.25,0.25,0.25,0.25]
        delta[0] = pi*E[:,meas_seq[0]]
        for t in range(1, num_meas):
            for j in range(N):
                delta[t,j] = max(delta[t-1]*T[:,j]) * E[j, meas_seq[t]]
                psi[t,j] = argmax(delta[t-1]*T[:,j])

        # backtrack
        states = zeros(num_meas, dtype=int32)
        states[num_meas-1] = argmax(delta[num_meas-1])
        for t in range(num_meas-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states
    ##----------------------------------------------------------------##
    def alpha_awg_cal(self, cav_amp=0.4, cav_len=250):
        # takes input array of amps and length and converts them to output array of alphas,
        # using a calibration h5 file defined in the experiment config
        # pull calibration data from file, handling properly in case of multimode cavity
        cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

        fn_file = cal_path + '\\00000_2021_7_30_cavity_square.h5'

        with File(fn_file, 'r') as f:
            omegas = np.array(f['omegas'])
            amps = np.array(f['amps'])

        # assume zero frequency at zero amplitude, used for interpolation function
        omegas = np.append(omegas, 0.0)
        amps = np.append(amps, 0.0)

        o_s = omegas
        a_s = amps

        # interpolate data, transfer_fn is a function that for each amp returns the corresponding omega
        transfer_fn = sc.interpolate.interp1d(a_s, o_s)

        omega_desired = transfer_fn(cav_amp)
        alpha = omega_desired * cav_len

        """Returns alpha in the cavity"""
        return alpha
        
    ##----------------------------------------------------------------##
    def openfile(self, filename):

        return File(filename,'r')

    def stateprep(self, fstate_in=0, data_filename=None, at_end=True, t2_err =0.0):
        
        """Readout fidelities from an independent measurement"""
        g_infidelity, e_infidelity = self.g_inf, self.e_inf
        
        self.a = self.openfile(data_filename)

        bit3 = pd.DataFrame(self.a['bit3'])[:]
        
        cav_amp = np.array(self.a['amp'])
        cav_len = np.array(self.a['time'])
        npi_m = int(np.array(self.a['pi_m']))
        npi_n = int(np.array(self.a['pi_n']))

        self.a.close()
        
        df = bit3
        alpha = self.alpha_awg_cal(cav_amp, cav_len)
        print('# of π at m= {}, at n = {}'.format(npi_m, npi_n))
        print('Coherent drive: amp = {}, length = {} ns'.format(cav_amp, cav_len))

        nx, ny = np.shape(df)
                
        """Renaming the columns of repeated pi pulses"""
        l = []
        for i in range(ny):
            l.append('π%i'%i)
        df.columns = l

        """Find out the unique Fock levels and their occurences"""
        
        p_m_counts = []
        
        """T and E matrices for the state prep part"""
        fstate_in = 0

        if fstate_in ==0:
            cavity_t1 = self.cavity_t1/(fstate_in+1)
            Pnm =  self.cavity_nth * (1-np.exp(-self.trigger_period/cavity_t1))
        else: 
            cavity_t1 = self.cavity_t1/(fstate_in)
            Pnm =  (1-np.exp(-self.trigger_period/cavity_t1)) + self.cavity_nth * (1-np.exp(-self.trigger_period/cavity_t1))

        Pmn = 0 + 0 #assuming that the population at (n+1) is negligible and (n-1) we will estimate\

        Pge = self.qubit_nth * (1-np.exp(-self.trigger_period/self.qubit_t1)) +\
            (1-np.exp(-self.pi_length/self.qubit_t2) - t2_err)
        Peg = (1-np.exp(-self.trigger_period/self.qubit_t1)) + \
            (1-np.exp(-self.pi_length/self.qubit_t2) - t2_err)

        T = np.asarray([[(1-Pmn)*(1-Pge), (1-Pmn)*Pge, Pmn*Pge, Pmn*(1-Pge)],
             [(1-Pmn)*Peg, (1-Pmn)*(1-Peg), Pmn*(1-Peg), Pmn*Peg],
             [Pnm*(1-Pge), Pnm*Pge, (1-Pnm)*Pge, (1-Pnm)*(1-Pge)],
             [Pnm*Peg, Pnm*(1-Peg), (1-Pnm)*(1-Peg), (1-Pnm)*Peg]])

        E = 0.5*np.asarray([[1-g_infidelity, g_infidelity],
            [e_infidelity, 1- e_infidelity],
            [1-g_infidelity, g_infidelity],
            [e_infidelity, 1- e_infidelity]])
            
        for jj in range(len(df)):
            """State preparation probabilities at the end or at the beginning of m π pulses"""
            meas_seq = df.iloc[jj]
            gamma_matrix = self.gamma(meas_seq, T, E)
            if at_end==True: #Probablitity of state surviving till the end
                P0_last = gamma_matrix[-1,0] + gamma_matrix[-1,1]
                P1_last = gamma_matrix[-1,2] + gamma_matrix[-1,3]
            else:#Probability at the beginnning
                P0_last = gamma_matrix[0,0] + gamma_matrix[0,1]
                P1_last = gamma_matrix[0,2] + gamma_matrix[0,3]   
            p_m_counts.append(P1_last/P0_last)

        return alpha, cav_amp, p_m_counts
########################

qubit_params = {'t1':120, 't2':130, 'nth':2.2e-2}
cavity_params = {'t1':6.0e2, 'nth':0.001}
readout_params = {'length':3.2, 'trigger':4.768, 'pi_pulse':565e-3, 'g_inf':0.0246, 'e_inf':0.0408}
expt_name = 'photon_counting_parity_alpha'

filelist = np.arange(151, 183, 1)

filelist = [157]

t2_err_arr = np.arange(0.0, 0.2, 0.05)

for ii, i in enumerate(filelist):
    p_m_counts = []
    filename = "..\\data\\" + str(i).zfill(5) + "_"+expt_name.lower()+".h5"
    print(filename)
    for err in t2_err_arr:
        p_m_temp = []
        obj = hmm_analysis_2(qubit_params=qubit_params, cavity_params=cavity_params, readout_params=readout_params)
        alpha, c_amp, p_m_temp = obj.stateprep(data_filename=filename, at_end=False, t2_err=err)
        p_m_counts.append(p_m_temp)

    path = "../data/photon_counting/g0_20210731/test/"
    filename = path + "n" + str(0) +"_camp_" + str(c_amp)+"_len_"+str(400)+".h5"
    with File(filename, 'w') as f:
        f.create_dataset("p_m_counts", data=p_m_counts)
        f.create_dataset('alpha', data=alpha)
