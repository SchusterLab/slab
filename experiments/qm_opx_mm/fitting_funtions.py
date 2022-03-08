"""
Created on feb 2022

@author: Ankur Agrawal, Schuster Lab

Collection of fitting function used in the stimulated emission experiment and in qubit-cavity experiments

Slight improvement in certain functions for bettet fitting

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, mu, sigma, a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def decaysin(x, a, b, c, d, e, f):
    return a*np.sin(2*np.pi*b*x+c*np.pi/180.)*np.exp(-1.*(x-f)/d)+e

def coherent_state(n, alpha):
    return np.exp(-abs(alpha)**2)*abs(alpha)**(2*n)/scipy.special.factorial(n)

def storage_t1_coherent(t, a, k):
    """define alpha for this data right before you call this fit function
    T1 = 1/k. Decay rate of a coherent state is the same as T1 wheras, the Fock states decay faster by n*k
    """
    return a*np.exp(-alpha**2*np.exp(-k*t))

def exponential(x, a, b, c):
    return a*np.exp(-x/b) + c

def number_splitting(x, y, n=0):
    """Fits an individual Gaussian around each number peak and returns an array
    of the peak values and frequency with error bars.
    'x': in units of χ so please pass it correctly, >0
    'y': qubit excitation probability
    """
    # number of peaks
    num_peaks = int(max(x))

    fitdata = {}
    peaks = []
    freqs = []
    peaks_err = []
    freqs_err = []

    fig, axs = plt.subplots(1, 1, dpi=300)
    plt.tight_layout()
    axs.set_xlabel(r'$n$', fontsize=20)
    axs.set_ylabel(r'$P_{e}$', fontsize=20)
    axs.grid(True, which='major', axis='both', color='grey', linestyle='--', linewidth=0.4)

    for ind in range(num_peaks-n):
        #         fitdata['n%.i'%ind] = {}
        """Fits the peak with a Gaussian spanning 1 unit of χ"""
        min_ind, max_ind =  np.argmin(abs(x-ind-0.5)), np.argmin(abs(x-ind+0.5))

        x_domain = x[min_ind:max_ind]
        y_domain = y[min_ind:max_ind]

        axs.plot(x_domain, y_domain, 'ks')

        a = (y_domain[0] + y_domain[-1])/2
        b = (np.max(y_domain) - np.min(y_domain))
        c = x_domain[np.argmax(y_domain)]
        d = (np.max(x_domain) - np.min(x_domain))/10
        p = [a, b, c, d]
        popt, pcov = curve_fit(gauss_baseline, x_domain, y_domain, p0=(a, b, c, d))
        axs.plot(x_domain, gauss_baseline(x_domain, *popt), 'r')
        freqs.append(popt[2])
        freqs_err.append(np.sqrt(np.diag(pcov)[2]))
        peaks.append(popt[1])
        peaks_err.append(np.sqrt(np.diag(pcov)[1]))

    fitdata['freq'] = freqs
    fitdata['peak'] = peaks
    fitdata['freq_err'] = freqs_err
    fitdata['peak_err'] = peaks_err

    plt.tick_params(direction='in', length=6, width=2, colors='k', \
                    grid_color='grey', grid_alpha=0.5, labelsize=20, labelbottom=True, right=True, top=True, left=True)
    plt.show()
    """Note: Don't forget to scale the returned frequency values with χ """
    return fitdata

def fitcoherentstate(peak_val, peak_val_err=0):
    xdata = np.arange(len(peak_val))
    ydata = np.array(peak_val)
    if peak_val_err:
        ydata_err = np.array(peak_val_err)
        popt, pcov = curve_fit(coherent_state, xdata, ydata, sigma=ydata_err)
    else:
        popt, pcov = curve_fit(coherent_state, xdata, ydata)

    return popt[0], np.sqrt(pcov[0][0])

def histogram():
    return