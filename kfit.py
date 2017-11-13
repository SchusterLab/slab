"""
Note on adding fit functions - Gerwin Koolstra May 2016

The form of fit functions that are added, from now on, should be

    fitfunc(x, *p)

where x is the x-data and p is a list containing all the parameters. This function should be documented, especially
the order in which the parameters appear in p. The function should return the value of the
function at point x, and nothing else. fitfunc may then be used as argument in fitbetter, to create the actual
fitting procedure.
"""
import numpy as np
import math as math
import matplotlib.pyplot as plt
import scipy, cmath
import scipy.fftpack
from scipy import optimize
from tabulate import tabulate

def argselectdomain(xdata, domain):
    ind = np.searchsorted(xdata, domain)
    return (ind[0], ind[1])

def selectdomain(xdata, ydata, domain):
    ind = np.searchsorted(xdata, domain)
    return xdata[ind[0]:ind[1]], ydata[ind[0]:ind[1]]

def zipsort(xdata, ydata):
    inds = np.argsort(xdata)
    return np.take(xdata, inds), np.take(ydata, inds, axis=0)

def get_rsquare(ydata, ydatafit):
    """
    Get the rsquare goodness of fit measure. This is a value between 0 and 1, indicating how well the fit
    approximates the data. A value of 0 indicating extremely bad, 1 the best fit. For further reading:
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    :param ydata: Data
    :param ydatafit: Fit evaluated at the data points
    :return:R squared value
    """
    ybar = np.mean(ydata)
    total_sum_of_squares = np.sum((ydata - ybar) ** 2)
    residual_sum_of_squares = np.sum((ydata - ydatafit) ** 2)
    return 1 - residual_sum_of_squares / total_sum_of_squares

def plot_fitresult(xdata, ydata, bestfitparams, fitparam_errors, fitparam_names=None):
    # To print the fit result we use the legend feature from pyplot. This is nice because it has an algorithm that
    # can determine the best place to print the text within the figure. We plot the first few datapoints (not visible)
    # and assign labels to those. Then we can use the loc = 0 to let pyplot figure out the best location.
    if fitparam_names is None:
        fitparam_names = ["par%d" % k for k in range(len(bestfitparams))]

    # Remember the limits of the y-axis so that we don't change it
    ylims = plt.ylim()
    for k in range(len(bestfitparams)):
        plt.plot(xdata[k], ydata[k],
                 label=r"%s = %.2e $\pm$ %.2e" % (fitparam_names[k], bestfitparams[k], fitparam_errors[k]), alpha=0)

    plt.legend(loc=0, frameon=False, prop={'size': 8}, title="Fit result")
    plt.ylim(ylims)

def fitbetter(xdata, ydata, fitfunc, fitparams, parambounds=None, domain=None, showfit=False, showstartfit=False,
              showdata=True, mark_data='ko', mark_fit='r-', **kwargs):
    """
    Uses curve_fit from scipy.optimize to fit a non-linear least squares function to ydata, xdata
    Note: when applying bounds the fit method used is a different one than with an unconstrained fit. It's good
    practice to not apply bounds to parameters if it's not needed.
    :param xdata: x-axis
    :param ydata: y-axis
    :param fitfunc: One of the fitfunctions below
    :param fitparams: Parameters for the fitfunction
    :param parambounds: Tuple of bounds for each of the parameters: ([par1_min, par2_min, ...], [par1_max, par2_max, ...])
    :param domain: Domain for the xdata
    :param showfit: Show the fit
    :param showstartfit: Show the curve with initial guesses
    :param showdata: Plot the data.
    :param label: Label for the data
    :param mark_data: Marker format for the data
    :param mark_fit: Marker format for the fit
    :return:
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if parambounds is None:
        parambounds = (-np.inf, +np.inf)

    # New in scipy 0.17:
    # * Parameter bounds: constrain fit parameters.
    #   Example: if there are 3 fit parameters which have to be constrained to (0, inf) we can use
    #   parambounds = ([0, 0, 0], [np.inf, np.inf, np.inf]), Alternatively, one may set: parambounds = (0, np.inf).
    #   Default is of course (-np.inf, np.inf)

    startparams = fitparams
    bestfitparams, covmatrix = optimize.curve_fit(fitfunc, fitdatax, fitdatay, startparams, bounds=parambounds, **kwargs)

    try:
        fitparam_errors = np.sqrt(np.diag(covmatrix))
    except:
        print(covmatrix)
        print(
            "Error encountered in calculating errors on fit parameters. This may result from a very flat parameter space")

    if showfit:
        if showdata:
            plt.plot(fitdatax, fitdatay, mark_data, label="data")
        if showstartfit:
            plt.plot(fitdatax, fitfunc(fitdatax, *startparams), label="startfit")
        plt.plot(fitdatax, fitfunc(fitdatax, *bestfitparams), mark_fit, label="fit")

    return bestfitparams, fitparam_errors

#######################################################################
#######################################################################
#################### WRAPPERS FOR FITFUNCTIONS ########################
#######################################################################
#######################################################################

def fit_lor(xdata, ydata, fitparams=None, no_offset=False, domain=None, showfit=False, showstartfit=False,
            verbose=True, **kwarg):
    """
    Fit a Lorentzian; returns
    The quality factor can be found by Q = center/fwhm = center/(2*hwhm)
    :param xdata: Frequency
    :param ydata: Power in W
    :param fitparams: [offset,amplitude,center,hwhm]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: Prints the fit results
    :return: [fitresult, fiterrors] if successful
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0, 0, 0, 0]
        fitparams[0] = (fitdatay[0] + fitdatay[-1]) / 2.
        fitparams[1] = max(fitdatay) - min(fitdatay)
        fitparams[2] = fitdatax[np.argmax(fitdatay)]
        fitparams[3] = (max(fitdatax) - min(fitdatax)) / 10.

        if no_offset:
            fitparams.pop(0)
    elif len(fitparams) == 4 and no_offset:
        raise ValueError("no_offset is True, fitparams should be a list of length 3.")

    def lorfunc_jac(x, *p):
        if no_offset:
            amp, f0, hwhm = p
        else:
            offset, amp, f0, hwhm = p

        df_damp = 1 / (1 + (x-f0)**2 / hwhm**2)
        df_df0 = 2 * amp * (x-f0) / (hwhm**2 * (1 + (x-f0)**2/hwhm**2)**2)
        df_dhwhm = 2 * amp * (x-f0)**2 / (hwhm**3 * (1 + (x-f0)**2/hwhm**2)**2)

        if no_offset:
            return np.transpose(np.vstack([df_damp, df_df0, df_dhwhm]))
        else:
            df_doffset = np.ones(len(x))
            return np.transpose(np.vstack([df_doffset, df_damp, df_df0, df_dhwhm]))

    params, param_errs = fitbetter(fitdatax, fitdatay, lorfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, jac=lorfunc_jac, **kwarg)

    if verbose:
        parnames = ['Offset', 'Amplitude', 'f0', 'HWHM']
        if no_offset:
            parnames.pop(0)

        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))

        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_kinetic_fraction(xdata, ydata, fitparams=None, Tc_fixed=False, domain=None, showfit=False, showstartfit=False,
                         verbose=True, **kwarg):
    """
    Fits resonance frequencies (absolute, not shifts) vs. temperature due to kinetic inductance. Uses kinfunc
    Returns [f0, alpha, Tc]
    :param xdata: Temperature
    :param ydata: Resonance frequency
    :param fitparams: [f0_guess, alpha_guess, Tc_guess]
    :param Tc_fixed: True/False
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: List of optimal fitparameters (if successful) / None (if not successful)
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if fitparams is None:
        print("Please provide some initial guesses.")

    if Tc_fixed:
        fitparams = fitparams[:2]

    def kinfunc_jac(x, *p):
        f0, alpha, Tc = p
        a = 1 - (x/Tc)**4
        b = (1 + alpha/a)**(-3/2.)

        df_dalpha = -f0*b/(2*a)
        df_dTc = 2 * b * f0 * x**4 * alpha / (a**2 * Tc**5)
        df_df0 = b**(1/3.)
        return np.transpose(np.vstack([df_df0, df_dalpha, df_dTc]))

    params, param_errs = fitbetter(fitdatax, fitdatay, kinfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, jac=kinfunc_jac, **kwarg)

    if verbose:
        parnames = ['f0', 'Kinetic Inductance fraction', 'Tc']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))

        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_double_lor(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
                   verbose=True, **kwarg):
    """
    Fits two lorentzians. Uses twolorfunc. Convert to Q: center1/2*hwhm1, center2/2*hwhm2
    :param xdata: Frequency
    :param ydata: Power in W
    :param fitparams: [offset, amplitude 1, center 1, hwhm 1, amplitude 2, center 2, hwhm 2]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: List of optimal fitparameters (if successful) / None (if not successful)
    """

    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if fitparams is None:
        print("Please provide some initial guesses.")

    params, param_errs = fitbetter(fitdatax, fitdatay, twolorfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit)

    if verbose:
        parnames = ['Offset', 'A1', 'f1', 'HWHM1', 'A2', 'f2', 'HWHM2']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))

        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_N_gauss(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
                verbose=True, no_offset=False, **kwarg):
    """
    Fits a series of N Gaussian peaks or dips.
    If no_offset = True : Uses Ngaussfunc_no_offset
    If no_offset = False : uses Ngaussfunc
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [offset, amplitude 1, res freq 1, sigma 1, ...] or [amplitude 1, res freq 1, sigma 1, ...] if no_offset=True
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param no_offset: True/False
    :return: Optimal fit result (if successful).
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if fitparams is None:
        print("Please provide some initial guesses.")

    if no_offset:
        params, param_errs = fitbetter(fitdatax, fitdatay, Ngaussfunc_no_offset, fitparams, domain=None,
                                       showfit=showfit,
                                       showstartfit=showstartfit, **kwarg)
    else:
        params, param_errs = fitbetter(fitdatax, fitdatay, Ngaussfunc, fitparams, domain=None, showfit=showfit,
                                       showstartfit=showstartfit, **kwarg)

    if verbose:
        idx = 0
        for par, err in zip(params, param_errs):
            print("Parameter {} : {} +/- {}".format(idx, par, err))
            idx += 1

        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=None)

    return params, param_errs

def fit_exp(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
            verbose=True, **kwarg):
    """
    Fit exponential decay of the form (p[0]+p[1]*exp(-x/p[2])). Uses expfunc.
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [offset, amplitude, t0, tau]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: Optimal fit parameters.
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0., 0., 0.]
        fitparams[0] = fitdatay[-1]
        fitparams[1] = fitdatay[0] - fitdatay[-1]
        fitparams[2] = (fitdatax[-1] - fitdatax[0]) / 5.

    # Note that one may use the Jacobian, provided no bounds are supplied. There's still an error in the output covariance.
    def jacobian(fitparams, xdata, ydata, expfunc):
        return [np.ones(len(xdata)), np.exp(-xdata/fitparams[2]), fitparams[1]/fitparams[2]**2 * np.exp(-xdata/fitparams[2])]

    params, param_errs = fitbetter(fitdatax, fitdatay, expfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)#, Dfun=jacobian, col_deriv=1)

    if verbose:
        parnames = ['Offset', 'Amplitude', chr(964)]
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))

        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_pulse_err(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False):
    """
    Fit pulse error decay (p[0]+p[1]*(1-p[2])^x). Uses pulse_errfunc
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [offset, amplitude, ?]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: Optimal fitresult.
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0., 0.]
        fitparams[0] = fitdatay[-1]
        fitparams[1] = fitdatay[0] - fitdatay[-1]
        fitparams[1] = fitdatay[0] - fitdatay[-1]

    params, param_errs = fitbetter(fitdatax, fitdatay, pulse_errfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit)

    return params, param_errs

def fit_decaysin(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False, verbose=True,
                 **kwarg):
    """
    Fits decaying sine wave of form: p[0]*np.sin(2.*pi*p[1]*x+p[2]*pi/180.)*np.e**(-1.*(x-p[5])/p[3])+p[4]
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [A, f, phi (deg), tau, offset, t0]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: Optimal fit parameters.
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        FFT = scipy.fft(fitdatay)
        fft_freqs = scipy.fftpack.fftfreq(len(fitdatay), fitdatax[1] - fitdatax[0])
        max_ind = np.argmax(abs(FFT[4:len(fitdatay) / 2.])) + 4
        fft_val = FFT[max_ind]

        fitparams = [0, 0, 0, 0, 0]
        fitparams[4] = np.mean(fitdatay)
        fitparams[0] = (max(fitdatay) - min(fitdatay)) / 2.
        fitparams[1] = fft_freqs[max_ind]
        fitparams[2] = (cmath.phase(fft_val) - np.pi / 2.) * 180. / np.pi
        fitparams[3] = (max(fitdatax) - min(fitdatax))

    params, param_errs = fitbetter(fitdatax, fitdatay, decaysin, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ['Amplitude', 'Frequency', chr(966), chr(964), 'Offset', 'Start time']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_sin(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False, verbose=True,
            **kwarg):
    """
    Fits sin wave of form: p[0]*np.sin(2.*pi*p[1]*x+p[2]*pi/180.)+p[3].
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [Amplitude, Frequency (Hz), Phi (deg), Offset]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: Optimal fit parameters.
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        FFT = scipy.fft(fitdatay)
        fft_freqs = scipy.fftpack.fftfreq(len(fitdatay), fitdatax[1] - fitdatax[0])
        max_ind = np.argmax(abs(FFT[4:len(fitdatay) / 2.])) + 4
        fft_val = FFT[max_ind]

        fitparams = [0, 0, 0, 0]
        fitparams[3] = np.mean(fitdatay)
        fitparams[0] = (max(fitdatay) - min(fitdatay)) / 2.
        fitparams[1] = fft_freqs[max_ind]
        fitparams[2] = (cmath.phase(fft_val) - np.pi / 2.) * 180. / np.pi

    params, param_errs = fitbetter(fitdatax, fitdatay, sinfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ['Amplitude', 'Frequency (Hz)', chr(966), 'Offset']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_gauss(xdata, ydata, fitparams=None, no_offset=False, domain=None, showfit=False, showstartfit=False,
              verbose=True, **kwarg):
    """
    Fit a gaussian. You can choose to include an offset, using no_offset=True/False. Adjust fitparams accordingly:
    no_offset = True:   p[1] exp(- (x-p[2])**2/p[3]**2/2) (uses gaussfunc_nooffset)
    no_offset = False:  p[0]+p[1] exp(- (x-p[2])**2/p[3]**2/2) (uses gaussfunc)
    :param xdata: x points
    :param ydata: y points
    :param fitparams: [offset, amplitude, center, std] or [amplitude, center, std] if no_offset=True
    :param no_offset: True/False
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :return: Optimal fit parameters, if successful
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0, 0, 0, 0]
        fitparams[0] = (fitdatay[0] + fitdatay[-1]) / 2.
        fitparams[1] = max(fitdatay) - min(fitdatay)
        fitparams[2] = fitdatax[np.argmax(fitdatay)]
        fitparams[3] = (max(fitdatax) - min(fitdatax)) / 3.

    if no_offset:
        fitfunc = gaussfunc_nooffset
        if len(fitparams) > 3:
            fitparams = fitparams[1:]
    else:
        fitfunc = gaussfunc

    params, param_errs = fitbetter(fitdatax, fitdatay, fitfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        if no_offset:
            parnames = ['Amplitude', chr(956), chr(963)]
        else:
            parnames = ['Offset', 'Amplitude', chr(956), chr(963)]

        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_hanger(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
               verbose=True, **kwarg):
    """
    Fit Hanger Transmission (S21) data taking into account asymmetry. Uses hangerfunc.
    :param xdata: Frequency points
    :param ydata: Power in W
    :param fitparams: [f0, Qi, Qc, df, scale]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param printresult: True/False
    :param label: String
    :return: Optimal fit parameters [f0, Qi, Qc, df, scale] if successful.
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        peakloc = np.argmin(fitdatay)
        ymax = (fitdatay[0] + fitdatay[-1]) / 2.
        ymin = fitdatay[peakloc]
        f0 = fitdatax[peakloc]
        Q0 = abs(fitdatax[peakloc] / ((max(fitdatax) - min(fitdatax)) / 3.))
        scale = ymax
        Qi = Q0 * (1. + ymax)
        Qc = Qi / (ymax)
        fitparams = [f0, abs(Qi), abs(Qc), 0., scale]

    params, param_errs = fitbetter(fitdatax, fitdatay, hangerfunc, fitparams, domain=domain, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ['f0', 'Qi', 'Qc', 'df', 'scale']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_parabola(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
                 verbose=True, **kwarg):
    """
    Fit a parabola. Uses parabolafunc. Specify fitparams as [p0, p1, p2] where y = p0 + p1*(x-p2)**2
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [p0, p1, p2] where y = p0 + p1*(x-p2)**2
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False, prints the fitresult
    :return: Fitresult, Fiterror
    """
    if fitparams is None:
        print("Please specify fit parameters in function input")
        return

    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    params, param_errs = fitbetter(fitdatax, fitdatay, parabolafunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ["a%d" % idx for idx in range(len(params))]
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_s11(xdata, ydata, mode='oneport', fitparams=None, domain=None, showfit=False, showstartfit=False,
            verbose=True, **kwarg):
    """
    Fit an S11 curve. For mode='oneport' this code uses s11_mag_func_asymmetric. If mode='twoport' the code uses
    s11_mag_twoport. In both cases the fit function can fit asymmetric line shapes, represented by the parameter df.
    NB: fits the voltage signal, not a power (i.e. use this functionto fit |S11| instead of |S11|**2.
    For mode='oneport', Note Qi = f0/(2*eps), Qc = f0/kr.
    :param xdata: Frequency points
    :param ydata: S11 voltage data
    :param fitparams: [f0, kr, eps, df, scale]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False, prints the fitresults
    :return: Fitresult, Fiterror
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if fitparams is None and mode == 'oneport':
        f0_guess = fitdatax[np.argmin(fitdatay)]
        kr_guess = (fitdatax[-1] - fitdatax[0]) / 5.
        eps_guess = (fitdatax[-1] - fitdatax[0]) / 5.
        df_guess = 0
        scale_guess = np.max(fitdatay)
        fitparams = [f0_guess, kr_guess, eps_guess, df_guess, scale_guess]
    if fitparams is None and mode == 'twoport':
        f0_guess = fitdatax[np.argmin(fitdatay)]
        Qc_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.)
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.)
        df_guess = 0
        scale_guess = np.max(fitdatay)
        fitparams = [f0_guess, Qc_guess, Qi_guess, df_guess, scale_guess]

    if mode == 'oneport':
        params, param_errs = fitbetter(fitdatax, fitdatay, s11_mag_func_asymmetric, fitparams,
                                       parambounds=([0, 0, 0, -np.inf, -np.inf], np.inf),
                                       domain=None, showfit=showfit,
                                       showstartfit=showstartfit, **kwarg)
        names = ['f0', chr(954), chr(949), 'df', 'scale']
    else:
        params, param_errs = fitbetter(fitdatax, fitdatay, s11_mag_twoport, fitparams,
                                       parambounds=([0, 0, 0, -np.inf, -np.inf], np.inf),
                                       domain=None, showfit=showfit,
                                       showstartfit=showstartfit, **kwarg)
        names = ['f0', 'Qc', 'Qi', 'df', 'scale']

    if verbose:
        print(tabulate(zip(names, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=names)

    return params, param_errs

def fit_fano(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
             verbose=True, **kwarg):
    """
    Fit a fano lineshape. Uses fano_func.
    :param xdata: Frequency points
    :param ydata: Power in W
    :param fitparams: [w0, fwhm, q (fano factor), scale]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False. Prints the fitresult
    :return: Fitresult, Fiterror
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0, 0, 0, 0]
        fitparams[3] = max(fitdatay) - min(fitdatay)
        fitparams[0] = fitdatax[np.argmax(fitdatay)]
        fitparams[1] = (max(fitdatax) - min(fitdatax)) / 10.
        fitparams[2] = 10.

    params, param_errs = fitbetter(fitdatax, fitdatay, fano_func, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ['f0', 'FWHM', 'Fano factor', 'Amplitude']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_lor_asym(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
                 verbose=True, **kwarg):
    """
    Fit asymmetric lorentzian lineshape, derived from a capacitor in series with the LC circuit. Uses asym_lorfunc.
    See also fit_fano
    :param xdata: Frequency points
    :param ydata: S_21 Power in W
    :param fitparams: [peak amplitude, f0, fwhm, parallel capacitance]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False. Prints the fitresult.
    :return: Fitresult, Fiterror
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = [0, 0, 0, 0]
        fitparams[3] = max(fitdatay) - min(fitdatay)
        fitparams[0] = fitdatax[np.argmax(fitdatay)]
        fitparams[1] = (max(fitdatax) - min(fitdatax)) / 10.
        fitparams[2] = fitparams[0] / 10.

    params, param_errs = fitbetter(fitdatax, fitdatay, asym_lorfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        parnames = ['Amplitude', 'f0', 'FWHM', 'Parallel capacitance']
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_poly(xdata, ydata, mode=None, fitparams=None, domain=None, showfit=False, showstartfit=False,
             verbose=True, **kwarg):
    """
    Fit a polynomial. Uses polyfunc. Specify fitparams as [p0, p1, p2, ...] where
    y = p0 + p1*x + p2*x**2 + ...
    :param xdata: x-data
    :param ydata: y-data
    :param mode: 'even' or 'odd' restricts the fitfunction to only even or odd powers.
    :param fitparams: [a0, a1, a2, a3, ...] where y = a0 + a1*x + a2*x**2 + ...
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False. Prints the fitresults.
    :return: Fitresult, Fiterror
    """
    if fitparams is None:
        print("Please specify fit parameters in function input")
        return

    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if mode == 'even':
        fitfunction = polyfunc_even
        fitfunc_string = "Fit function: y = a0 + a1*x**2 + a2*x**4 + ..."
    elif mode == 'odd':
        fitfunction = polyfunc_odd
        fitfunc_string = "Fit function: y = a0 + a1*x + ..."
    else:
        fitfunction = polyfunc
        fitfunc_string = "Fit function: y = a0 + a1*x + a2*x**3 + ..."

    params, param_errs = fitbetter(fitdatax, fitdatay, fitfunction, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        print(fitfunc_string)
        parnames = ["a%d" % idx for idx in range(len(params))]
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs

def fit_powerlaw(xdata, ydata, fitparams=None, domain=None, showfit=False, showstartfit=False,
                 verbose=True, **kwarg):
    """
    Fit a power law function of the form y = p[0] + p[1] * x ** p[2]
    :param xdata: x-data
    :param ydata: y-data
    :param fitparams: [Offset, Multiplicative factor, Exponent]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False. Prints the fitresults.
    :return: Fitresult, Fiterror
    """
    if fitparams is None:
        print("Please specify fit parameters in function input")
        return

    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    fitfunc_string = "y = p[0] + p[1] * x ^ p[2]"
    params, param_errs = fitbetter(fitdatax, fitdatay, powerlawfunc, fitparams, domain=None, showfit=showfit,
                                   showstartfit=showstartfit, **kwarg)

    if verbose:
        print(fitfunc_string)
        parnames = ["Offset", "Multiplicator", "Exponent"]
        print(tabulate(zip(parnames, params, param_errs), headers=["Parameter", "Value", "Std"],
                       tablefmt="rst", floatfmt="", numalign="center", stralign='left'))
        plot_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=parnames)

    return params, param_errs
###########################################################
###########################################################
#################### FIT FUNCTIONS ########################
###########################################################
###########################################################

def lorfunc(x, *p):
    """
    Lorentzian with or without offset
    :param p: [offset, peak amplitude, center, hwhm] or [peak amplitude, center, hwhm]
    :param x: Frequency points
    :return: p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)
    """
    if len(p) == 3:
        return p[0] / (1 + (x - p[1]) ** 2 / p[2] ** 2)
    else:
        return p[0] + p[1] / (1 + (x - p[2]) ** 2 / p[3] ** 2)

def kinfunc(x, *p):
    """
    Function describing a resonance frequency due to kinetic inductance as function of temperature.
    :param p: [f0, alpha, Tc] or [f0, alpha]
    :param x: Temperature points
    :return: f0*(1-alpha/2.*1/(1-(x/Tc)**4))
    """
    f0 = p[0]
    alpha = p[1]

    if len(p) == 3:
        Tc = p[2]
    else:
        Tc = 1.2
        print("Assuming Tc = %.2f K" % Tc)

    f0s = f0 * (1 + alpha / (1 - (x / Tc) ** 4)) ** (-1 / 2.)
    return f0s

def twolorfunc(x, *p):
    """
    Two lorentzian functions (magnitude)
    :param p: [offset, amplitude 1, center 1, hwhm 1, amplitude 2, center 2, hwhm 2]
    :param x: Frequency points
    :return: p[0] + p[1]/(1+(x-p[2])**2/p[3]**2) + p[4]/(1+(x-p[5])**2/p[6]**2)
    """
    return p[0] + p[1] / (1 + (x - p[2]) ** 2 / p[3] ** 2) + p[4] / (1 + (x - p[5]) ** 2 / p[6] ** 2)

def asym_lorfunc(x, *p):
    """
    Asymmetric Lorentzian profile derived with a parallel capacitor that directly couples the input and output lines.
    :param x: Frequency points
    :param p: [peak amplitude, f0, fwhm, parallel capacitance]
    :return: np.abs(np.sqrt(p[0])/(1 + 1j * (x - p[1]) / p[2]) + np.sqrt(p[0]) * 2 * x * p[3] / p[1] / (+1j + 2 * x * p[3] / p[1])) ** 2
    """
    return np.abs(np.sqrt(p[0]) / (1 + 1j * 2 * (x - p[1]) / p[2]) + np.sqrt(p[0]) * 2 * x * p[3] / p[1] / (+1j + 2 * x * p[3] / p[1])) ** 2

def fano_func(x, *p):
    """
    Fano function. q describes the asymmetry.
    :param x: Frequency points
    :param p: [w0, fwhm, q, scale]
    :return: p[3] * (p[2]*p[1]/2. + (x-p[0]))**2/((p[1]/2.)**2 + (x-p[0])**2)
    """
    return p[3] * (p[2] * p[1] / 2. + (x - p[0])) ** 2 / ((p[1] / 2.) ** 2 + (x - p[0]) ** 2)

def print_cavity_Q(fit):
    """
    Prints the Q values given center and HWHM
    :param fit: Optimal fitparameters found by fitlor
    :return: fit[2]/(2*fit[3])
    """
    print(fit[2] / 2 / fit[3])
    return fit[2] / 2 / fit[3]

def gaussfunc(x, *p):
    """
    Gaussian function, including an offset
    :param p: [offset, amplitude, center, standard deviation]
    :return: p[0]+p[1]*math.e**(-1./2.*(x-p[2])**2/p[3]**2)
    """
    return p[0] + p[1] * math.e ** (-1. / 2. * (x - p[2]) ** 2 / p[3] ** 2)

def gaussfunc_nooffset(x, *p):
    """
    Gaussian function, no offset
    :param p: [amplitude, center, standard deviation]
    :return: p[0]*math.e**(-1./2.*(x-p[1])**2/p[2]**2)
    """
    return p[0] * math.e ** (-1. / 2. * (x - p[1]) ** 2 / p[2] ** 2)

def Ngaussfunc(x, *p):
    """
    Gaussian function with N peaks, including an offset
    :param p: [offset, A1, f1, sigma1, A2, f2, sigma2, ...]
    :return: p[3*n+1]*math.e**(-1./2.*(x-p[3*n+2])**2/p[3*n+3]**2)
    """
    N = int((len(p) - 1) / 3.)
    Ngauss = p[0]
    for n in range(N):
        Ngauss += p[3 * n + 1] * math.e ** (-1. / 2. * (x - p[3 * n + 2]) ** 2 / p[3 * n + 3] ** 2)
    return Ngauss


def Ngaussfunc_no_offset(x, *p):
    """
    Gaussian function with N peaks, no offset
    :param p: [A1, f1, sigma1, A2, f2, sigma2, ...]
    :return: p[3*n+1]*math.e**(-1./2.*(x-p[3*n+2])**2/p[3*n+3]**2)
    """
    N = int((len(p) - 1) / 3.)
    Ngauss = 0
    for n in range(N):
        Ngauss += p[3 * n + 1] * math.e ** (-1. / 2. * (x - p[3 * n + 2]) ** 2 / p[3 * n + 3] ** 2)
    return Ngauss

def expfunc(x, *p):
    """
    Exponential function, including an offset
    :param p: [offset, amplitude, t0, tau]
    :param x: time
    :return: p[0]+p[1]*math.e**(-x/p[2])
    """
    return p[0] + p[1] * math.e ** (-x / p[2])

def pulse_errfunc(x, *p):
    """
    Pulse error function
    :param p: [offset, ?]
    :param x: x-axis
    :return: p[0]+0.5*(1-((1-p[1])**x))
    """
    return p[0] + 0.5 * (1 - ((1 - p[1]) ** x))

def decaysin(x, *p):
    """
    Exponential decaying sine function.
    :param p: [A, f, phi (deg), tau, offset, t0]
    :param x: Time
    :return: p[0]*np.sin(2.*np.pi*p[1]*x+p[2]*np.pi/180.)*np.e**(-1.*(x-p[5])/p[3])+p[4]
    """
    return p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - p[5]) / p[3]) + p[4]

def sinfunc(x, *p):
    """
    Sine function
    :param p: [Amplitude, Frequency (Hz), Phi (deg), Offset]
    :param x: Time points
    :return: p[0]*np.sin(2.*np.pi*p[1]*x+p[2]*np.pi/180.)+p[3]
    """
    return p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) + p[3]

def hangerfunc(x, *p):
    """
    Hanger function
    :param p: [f0, Qi, Qc, df, scale]
    :param x: Frequency points
    :return: scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))
    """
    f0, Qi, Qc, df, scale = p
    a = (x - (f0 + df)) / (f0 + df)
    b = 2 * df / f0
    Q0 = 1. / (1. / Qi + 1. / Qc)
    return scale * (-2. * Q0 * Qc + Qc ** 2. + Q0 ** 2. * (1. + Qc ** 2. * (2. * a + b) ** 2.)) / (
        Qc ** 2 * (1. + 4. * Q0 ** 2. * a ** 2.))

def s11_mag_func(x, *p):
    """
    Symmetric S11 magnitude function (reflection from resonator) in voltage.
    :param x: Frequency points
    :param p: [w0, Qi, Qc]
    :return: np.abs(((p[2]-p[1])/p[2] + 2*1j*(x-p[0])*p[1]/p[0])/((p[1]+p[2])/p[2] + 2*1j*(x-p[0])*p[1]/p[0]))
    """
    return np.abs(((p[2] - p[1]) / p[2] + 2 * 1j * (x - p[0]) * p[1] / p[0]) / (
        (p[1] + p[2]) / p[2] + 2 * 1j * (x - p[0]) * p[1] / p[0]))

def s11_phase_func(x, *p):
    """
    Symmetric S11 phase function (reflection from resonator) in radians.
    :param x: Frequency points
    :param p: [w0, Qi, Qc]
    :return: np.angle(((p[2]-p[1])/p[2] + 2*1j*(x-p[0])*p[1]/p[0])/((p[1]+p[2])/p[2] + 2*1j*(x-p[0])*p[1]/p[0]))
    """
    return np.angle(((p[2] - p[1]) / p[2] + 2 * 1j * (x - p[0]) * p[1] / p[0]) / (
        (p[1] + p[2]) / p[2] + 2 * 1j * (x - p[0]) * p[1] / p[0]))

def s11_mag_func_asymmetric(x, *p):
    """
    Asymmetric S11 magnitude function (reflection from 1 port resonator), in voltage!
    :param x: Frequency points
    :param p: [f0, kr, eps, df, scale]
    :return: p[4]*np.abs((1j*(x-p[0]) + (p[2]-p[1]/2.))/(1j*(x-p[0]) + 1j*p[3] + (p[2]+p[1]/2.)))
    """
    return p[4] * np.abs((1j * (x - p[0]) + (p[2] - p[1] / 2.)) / (1j * (x - p[0]) + 1j * p[3] + (p[2] + p[1] / 2.)))

def s11_phase_func_asymmetric(x, *p):
    """
    Asymmetric S11 phase function (reflection from 1 port resonator)
    :param x: Frequency points
    :param p: [f0, kr, eps, df, scale]
    :return: np.angle((1j*(x-p[0]) + (p[2]-p[1]/2.))/(1j*(x-p[0]) + 1j*p[3] + (p[2]+p[1]/2.)))
    """
    return np.angle((1j * (x - p[0]) + (p[2] - p[1] / 2.)) / (1j * (x - p[0]) + 1j * p[3] + (p[2] + p[1] / 2.)))

def s11_mag_twoport(x, *p):
    """
    Reflection off a 2 port resonator
    :param x: fpoints
    :param p: [f0, Qc, Qi, df, scale]
    :return: scale*(-1j*dw + 1j*ki - eps)/(1j*dw + kr + eps)
    """
    f0, Qc, Qi, df, scale = p
    dw = x - f0
    kr = f0 / Qc
    eps = f0 / Qi
    ki = df
    return scale * np.abs((-1j * dw + 1j * ki - eps) / (1j * dw + kr + eps))

def s11_phase_twoport(x, *p):
    """
    Reflection off a 2 port resonator
    :param x: fpoints
    :param p:  [f0, Qc, Qi, df, scale]
    :return: np.angle((-1j*dw + 1j*ki - eps)/(1j*dw + kr + eps))
    """
    f0, Qc, Qi, df, scale = p
    dw = x - f0
    kr = f0 / Qc
    eps = f0 / Qi
    ki = df
    return np.angle((-1j * dw + 1j * ki - eps) / (1j * dw + kr + eps))

def parabolafunc(x, *p):
    """
    Parabola function
    :param x: x-data
    :param p: [a0, a1, a2] where y = a0 + a1 * (x-a2)**2
    :return: p[0] + p[1]*(x-p[2])**2
    """
    return p[0] + p[1] * (x - p[2]) ** 2

def polyfunc(x, *p):
    """
    Polynomial of arbitrary order. Order is specified by the length of p
    :param x: x-data
    :param p: [a0, a1, a2, a3, ...] where y = a0 + a1*x + a2*x**2 + ...
    :return: p[0] + p[1]*x + p[2]*x**2 + ...
    """
    y = 0
    for n, P in enumerate(p):
        y += P * x ** n
    return y

def polyfunc_even(x, *p):
    """
    Even polynomial of arbitrary order. Order is specified by the length of p
    :param x: x-data
    :param p: [a0, a1, a2, a3, ...] where y = a0 + a1*x**2 + a2*x**4 + ...
    :return: p[0] + p[1]*x**2 + p[2]*x**4 + ...
    """
    y = 0
    for n, P in enumerate(p):
        y += P * x ** (2*n)
    return y

def polyfunc_odd(x, *p):
    """
    Odd polynomial of arbitrary order. Order is specified by the length of p
    :param x: x-data
    :param p: [a0, a1, a2, a3, ...] where y = a0 + a1*x + a2*x**3 + ...
    :return: p[0] + p[1]*x + p[2]*x**3 + ...
    """
    y = p[0]
    for n, P in enumerate(p[1:]):
        y += P * x ** (2*n+1)
    return y

def powerlawfunc(x, *p):
    """
    Power law of order p[2]
    :param x: x-data
    :param p: [Offset, Multiplicative factor, Exponent]
    :return: p[0] + p[1] * x ** (p[2])
    """
    return p[0] + p[1] * x ** (p[2])

if __name__ == '__main__':
    pass
