import copy
import json
from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from h5py import File
import json
from slab.dsfit import *
from slab.aofit import *
from slab.gerb_fit.gerb_fit_210519 import *
import os
import os.path

import math
from numpy.linalg import inv
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Tuning:
    def __init__(self, file_names, N=8, log_tuning_files_name=None):
        lattice_cfg = file_names
        os.chdir("C:\\210801 - PHMIV3_56 - BF4 cooldown 4\\ipython notebook")
        self.N = N
        if log_tuning_files_name==None:
            log_tuning_files_name = "S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\log_tuning_files.json"
        self.json_log_tuning_files(log_tuning_files_name, file_names)

        try:
            self.energylistarray = np.load(lattice_cfg["energylistarray_name"],allow_pickle = True)
            self.flxquantaarray = np.load(lattice_cfg["flxquantaarray_name"],allow_pickle = True)
        except:
            print("Couldn't load energy list and flux quanta arrays")

        try:
            self.reslistarray = np.load(lattice_cfg["reslistarray_name"],allow_pickle = True)
        except:
            print("Couldn't load res list arrays")

        try:
            self.FF_SWCTM = np.load(lattice_cfg["FF_SWCTM_name"])
            self.FF_SWCTM  = self.FF_SWCTM[0:self.N,0:self.N]
            print(self.FF_SWCTM.shape)
        except:
            print("Couldn't load FF_SWCTM")
        try:
            self.DC_CTM = np.load(lattice_cfg["DC_CTM_name"])[0:self.N,0:self.N]
        except:
            print("Couldn't load DC_CTM")

        try:
            self.FF_LocalSlopes = np.load(lattice_cfg["FF_LocalSlopes_name"])[0:self.N] # should be GHz / V
        except:
            print("Couldn't laod local slopes")
        try:
            self.FF_dVdphi = np.load(lattice_cfg["FF_dVdphi_name"])[0:self.N] # dV/dphi
        except:
            print("Couldn't load FF_dVdphi")
        try:
            self.DC_dVdphi = np.load(lattice_cfg["DC_dVdphi_name"])[0:self.N] # dV/dphi
        except:
            print("Couldn't load DC_dVdphi")

        try:
            self.DC_CTMinv = inv(self.DC_CTM)
        except:
            pass
        try:
            self.FF_SWCTMinv = inv(self.FF_SWCTM)
        except:
            pass

        try:
            self.FF_SWCTMN = np.zeros((self.N, self.N))
            for ii in range(len(self.FF_SWCTM)):
                diag = self.FF_SWCTM[ii, ii]
                for jj in range(len(self.FF_SWCTM)):
                    if diag==0:
                        print("zero diag element! setting to zero")
                        self.FF_SWCTMN[ii, jj] = 0.00005
                    else:
                        self.FF_SWCTMN[ii, jj] = self.FF_SWCTM[ii, jj] / (diag * self.FF_dVdphi[ii])
            self.FF_SWCTMNinv = inv(self.FF_SWCTMN)
        except:
            pass

        self.DC_CTMN = np.zeros((self.N,self.N))
        # temp = np.zeros((self.N,self.N))

        # for ii in np.arange(self.N):
            # for jj in np.arange(self.N):
                # temp[ii,jj] = self.DC_CTM[ii,jj]

        for ii in range(len(self.DC_CTM)):
            diag = self.DC_CTM[ii, ii]
            for jj in range(len(self.DC_CTM)):
                self.DC_CTMN[ii, jj] = self.DC_CTM[ii, jj] / (diag * self.DC_dVdphi[ii])
        self.DC_CTMNinv = inv(self.DC_CTMN)

        # temp = np.zeros((self.N,self.N))
        # temp2 = np.zeros((self.N, self.N))
        # for ii in np.arange(self.N):
        #     for jj in np.arange(self.N):
        #         temp[ii,jj] = self.DC_CTM[ii,jj]
        #         temp2[ii,jj] = self.DC_CTMNinv[ii,jj]
        #
        # self.DC_CTM = temp
        # self.DC_CTMNinv = temp2



        self.generate_omegaphi_functions(self.energylistarray, self.flxquantaarray)

    def omega_to_V_thru_CTM(self, Vtype, freq_list):
        vec0 = []
        for ii in np.arange(self.N):
            #print('omegatophi%s' % ii + '(freq_list[%s' % ii + '])')
            vec0.append(self.omegatophi_list[ii](freq_list[ii]))
        if Vtype=="DC":
            return np.array(self.DC_CTMNinv.dot(vec0))
        if Vtype=="FF":
            return np.array(self.FF_SWCTMNinv.dot(vec0))


    def omega_to_V_thru_CTM_direct(self, Vtype, freq_list):
        if Vtype=="DC":
            return np.array(self.DC_CTMinv.dot(np.asarray(freq_list)))
        if Vtype=="FF":
            return np.array(self.FF_SWCTMinv.dot(np.asarray(freq_list)))


    def generate_omegaphi_functions(self, energylistarray, flxquantaarray):
        self.phitoomega_list = []
        self.omegatophi_list = []
        self.dphidomega_list = []
        self.domegadphi_list = []

        for i in range(self.N):
            self.phitoomega_list.append(
                scipy.interpolate.interp1d(flxquantaarray[i], energylistarray[i], kind='cubic', ))
            self.omegatophi_list.append(
                scipy.interpolate.interp1d(energylistarray[i], flxquantaarray[i], kind='cubic', ))
            self.dphidomega_list.append(
                scipy.interpolate.interp1d(energylistarray[i][0:-1],np.diff(flxquantaarray[i])/np.diff(energylistarray[i]), kind='cubic'))
            self.domegadphi_list.append(
                scipy.interpolate.interp1d(flxquantaarray[i][0:-1],
                                     np.diff(energylistarray[i]) / np.diff(flxquantaarray[i]) , kind='cubic'))

        return [self.phitoomega_list,self.omegatophi_list,self.dphidomega_list,self.domegadphi_list]


    def omega_to_V_thru_LocalSlopes(self, Vtype, jump_freq_list):
        # This should be dphi given local slope
        vec0 = (1 / (np.array(self.FF_dVdphi)[0:self.N])) * (1 / (np.array(self.FF_LocalSlopes))) * np.array(jump_freq_list)

        # voltage to get those values
        return np.array(self.FF_SWCTMNinv.dot(vec0))

    def json_log_phi_diff(self, filename, phi_diff):
        t_obj = time.localtime()
        t = time.asctime(t_obj)
        with open(filename, 'r+') as f:
            data = json.load(f)
            data[t] = phi_diff
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def json_log_tuning_files(self, filename, tuning_file_names):
        t_obj = time.localtime()
        t = time.asctime(t_obj)
        with open(filename, 'r+') as f:
            data = json.load(f)
            data[t] = tuning_file_names
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def correct_flux_offsets(self, intended_freq, measured_freq, filename=None):
        if filename==None:
            filename = "S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\phi_diff_log.json"
        print("calculating new flxquantaarray")
        freq_diff = np.array(intended_freq) - np.array(measured_freq)

        phi_diff = np.zeros(len(freq_diff))
        for ii in np.arange(self.N):
            phi_diff[ii] =  self.omegatophi_list[ii](intended_freq[ii]) - self.omegatophi_list[ii](measured_freq[ii])

        new_flxquantaarray = copy.deepcopy(self.flxquantaarray)
        for i in range(self.N):
            new_flxquantaarray[i] = np.array(self.flxquantaarray[i]) + phi_diff[i]
        self.json_log_phi_diff(filename, list(phi_diff))

        t = time.localtime()
        month = t[1]
        day = t[2]
        hour = t[3]
        min = t[4]
        flx_quanta_filename = time.localtime()
        flx_name =  str(month).zfill(2) + str(day).zfill(2) + str(hour) + str(min).zfill(2) + "_flxquantaarray.npy"
        print("New fluxqanta array filename: " + flx_name)
        np.save(flx_name, new_flxquantaarray)

        return new_flxquantaarray

    def plot_colored_CTM(self, Vtype, save_file = None):
        if Vtype=="DC":
            CTM = copy.copy(self.DC_CTM)
        if Vtype=="FF":
            CTM = copy.copy(self.FF_SWCTM)

        for ii in range(len(CTM)):
            diag = CTM[ii, ii]
            for jj in range(len(CTM)):
                CTM[ii, jj] = (CTM[ii, jj] / (diag)) * 100
                if ii==jj:
                    CTM[ii,jj] = 0

        qlist = ["Q0", "Q1", "Q2", 'Q3', "Q4", "Q5", "Q6", "Q7"]
        flist = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]


        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        im = ax.imshow(CTM, cmap="RdBu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(CTM)))
        ax.set_yticks(np.arange(len(CTM)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(flist[0:len(CTM)])
        ax.set_yticklabels(qlist[0:len(CTM)])

        # Loop over data dimensions and create text annotations.
        for ii in range(len(CTM)):
            for jj in range(len(CTM)):
                text = ax.text(ii, jj, round(CTM[jj, ii], 1),
                               ha="center", va="center", color="w")

        plt.colorbar(im, cax=cax)

        if save_file !=None:
            plt.savefig(save_file,dpi=600,bbox_inches = "tight",transparent = "false")

        plt.show()

    def plot_flux_dispersion(self, save_file=None):
        plt.figure(figsize=(12, 8))
        for ii in range(len(self.flxquantaarray)):
            plt.plot(self.flxquantaarray[ii], self.energylistarray[ii], label='Q%s' % (ii))
        plt.xlabel('flux quanta')
        plt.ylabel('Frequency (GHz)')
        plt.legend()
        plt.grid(True)
        if save_file !=None:
            plt.savefig(save_file,dpi=600,bbox_inches = "tight",transparent = "false")
        plt.show()

