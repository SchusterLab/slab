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
    def __init__(self, file_names):
        lattice_cfg = file_names
        os.chdir("C:\\210701 - PHMIV3_56 - BF4 cooldown 3\\ipython notebook")

        try:
            self.energylistarray = np.load(lattice_cfg["energylistarray_name"],allow_pickle = True)
            self.flxquantaarray = np.load(lattice_cfg["flxquantaarray_name"],allow_pickle = True)
        except:
            print("Couldn't load energy list and flux quanta arrays")

        try:
            self.FF_SWCTM = np.load(lattice_cfg["FF_SWCTM_name"])
        except:
            print("Couldn't load FF_SWCTM")
        try:
            self.DC_CTM = np.load(lattice_cfg["DC_CTM_name"])
        except:
            print("Couldn't load DC_CTM")

        try:
            self.FF_LocalSlopes = np.load(lattice_cfg["FF_LocalSlopes_name"]) # should be GHz / V
        except:
            print("Couldn't laod local slopes")
        try:
            self.FF_dVdphi = np.load(lattice_cfg["FF_dVdphi_name"]) # dV/dphi
        except:
            print("Couldn't load FF_dVdphi")
        try:
            self.DC_dVdphi = np.load(lattice_cfg["DC_dVdphi_name"]) # dV/dphi
        except:
            print("Couldn't load DC_dVdphi")

        try:
            self.FF_SWCTMN = np.zeros((8, 8))
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


        self.DC_CTMN = np.zeros((8, 8))
        for ii in range(len(self.DC_CTM)):
            diag = self.DC_CTM[ii, ii]
            for jj in range(len(self.DC_CTM)):
                self.DC_CTMN[ii, jj] = self.DC_CTM[ii, jj] / (diag * self.DC_dVdphi[ii])
        self.DC_CTMNinv = inv(self.DC_CTMN)


        self.generate_omegaphi_functions(self.energylistarray, self.flxquantaarray)

    def omega_to_V_thru_CTM(self, Vtype, freq_list):
        vec0 = []
        for ii in np.arange(8):
            #print('omegatophi%s' % ii + '(freq_list[%s' % ii + '])')
            vec0.append(self.omegatophi_list[ii](freq_list[ii]))
        if Vtype=="DC":
            return np.array(self.DC_CTMNinv.dot(vec0))
        if Vtype=="FF":
            return np.array(self.FF_SWCTMNinv.dot(vec0))

    def generate_omegaphi_functions(self, energylistarray, flxquantaarray):
        self.phitoomega_list = []
        self.omegatophi_list = []
        self.dphidomega_list = []
        self.domegadphi_list = []

        for i in range(8):
            self.phitoomega_list.append(
                interpolate.interp1d(flxquantaarray[i], energylistarray[i], kind='cubic', ))
            self.omegatophi_list.append(
                interpolate.interp1d(energylistarray[i], flxquantaarray[i], kind='cubic', ))
            self.dphidomega_list.append(
                interpolate.interp1d(energylistarray[i][0:-1],np.diff(flxquantaarray[i])/np.diff(energylistarray[i]), kind='cubic'))
            self.domegadphi_list.append(
                interpolate.interp1d(flxquantaarray[i][0:-1],
                                     np.diff(energylistarray[i]) / np.diff(flxquantaarray[i]) , kind='cubic'))


    def omega_to_V_thru_LocalSlopes(self, Vtype, jump_freq_list):
        # This should be dphi given local slope
        vec0 = (1 / (np.array(self.FF_dVdphi))) * (1 / (np.array(self.FF_LocalSlopes))) * np.array(jump_freq_list)

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

    def correct_flux_offsets(self, intended_freq, measured_freq, filename=None):
        if filename==None:
            filename = "S:\\_Data\\210412 - PHMIV3_56 - BF4 cooldown 2\\phi_diff_log.json"
        print("calculating new flxquantaarray")
        freq_diff = np.array(intended_freq) - np.array(measured_freq)

        phi_diff = np.zeros(len(freq_diff))
        for ii in np.arange(8):
            phi_diff[ii] =  self.omegatophi_list[ii](intended_freq[ii]) - self.omegatophi_list[ii](measured_freq[ii])

        new_flxquantaarray = copy.deepcopy(self.flxquantaarray)
        for i in range(8):
            new_flxquantaarray[i] = np.array(self.flxquantaarray[i]) + phi_diff[i]
        self.json_log_phi_diff(filename, list(phi_diff))

        t = time.localtime()
        month = t[1]
        day = t[2]
        hour = t[3]
        flx_quanta_filename = time.localtime()
        flx_name =  str(month).zfill(2) + str(day).zfill(2) + str(hour).zfill(2) + "_flxquantaarray.npy"
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
        ax.set_xticks(np.arange(len(flist)))
        ax.set_yticks(np.arange(len(qlist)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(flist)
        ax.set_yticklabels(qlist)

        # Loop over data dimensions and create text annotations.
        for ii in range(len(qlist)):
            for jj in range(len(flist)):
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

