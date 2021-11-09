from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from h5py import File
import json
from slab.dsfit import *
from slab.aofit import *
import math
from slab import *
import numpy as np
from scipy.special import ellipk
from IPython.display import Image
import matplotlib
import matplotlib.pyplot as plt
from slab.kfit import fit_lor
from slab.kfit import fit_hanger
import os
from scipy.interpolate import interp1d
notebookdirectory = os.getcwd()
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
print(os.getcwd())
from scipy.signal import argrelextrema
import copy
from qutip import *

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

## Json formatter for lattice_device_cfg
# Changed basestring to str, and dict uses items() instead of iteritems().

def to_json(o, level=0,):
    INDENT = 3
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
          ret += comma
          comma = ",\n"
          ret += SPACE * INDENT * (level + 1)
          ret += '"' + str(k) + '":' + SPACE
          ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def decaysin_lm(x, amp, f, phi_0, T1, offset, exp0):
    return amp*np.sin(2.*np.pi*f*x+phi_0*np.pi/180.)*np.e**(-1.*(x-exp0)/T1)+offset

def decaysin3(p, x):
    return p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) * np.e ** (-1. * (x - x[0]) / p[3]) + p[4]

def iq_rot(I, Q, phi):
    """Digitially rotates IQdata by phi, calcualting phase as np.unrwap(np.arctan2(Q, I))
    :param I: I data from h5 file
    :param Q: Q data from h5 file
    :param phi: iq rotation desired (in degrees)
    :returns: rotated I, Q
    """
    phi = phi * np.pi / 180
    Irot = I * np.cos(phi) + Q * np.sin(phi)
    Qrot = -I * np.sin(phi) + Q * np.cos(phi)
    return Irot, Qrot


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2. ** 0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2. * numpy.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}

def iq_process(f, raw_I, raw_Q, ran=1, phi=0, sub_mean=True):
    """Converts digitial data to voltage data, rotates iq, subtracts off mean, calculates mag and phase
    :param f: frequency
    :param raw_I: I data from h5 file
    :param raw_Q: Q data from h5 file
    :param ran: range of DAC. If set to -1, doesn't convert data to voltage
    :param phi: iq rotation desired (in degrees)
    :param sub_mean: boolean, if True subtracts out average background in IQ data
    :returns: I, Q, mag, phase
    """
    if sub_mean:
        # ie, if want to subtract mean
        I = array(raw_I).flatten() - mean(array(raw_I).flatten())
        Q = array(raw_Q).flatten() - mean(array(raw_Q).flatten())
    else:
        I = array(raw_I).flatten()
        Q = array(raw_Q).flatten()

    # divide by 2**15 to convert from bits to voltage, *ran to get right voltage range
    if ran > 0:
        I = I / 2 ** 15 * ran
        Q = Q / 2 ** 15 * ran

    # calculate mag and phase
    phase = np.unwrap(np.arctan2(Q, I))
    mag = np.sqrt(np.square(I) + np.square(Q))

    # IQ rotate
    I, Q = iq_rot(I, Q, phi)

    return I, Q, mag, phase

def return_I_Q_res(filenb, pi=False, ff=False):
    expt_name = "resonator_spectroscopy"

    if pi:
        expt_name = expt_name + "_pi"
    if ff:
        expt_name = "ff_" + expt_name

    fname = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"

    with File(fname, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]

        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']

        try:
            I_raw = a['I']
            Q_raw = a['Q']
        except:
            I_raw = a['qb{}_I'.format(on_qb)]
            Q_raw = a['qb{}_Q'.format(on_qb)]

        f = arange(expt_params['start'] + readout_f, expt_params['stop'] + readout_f, expt_params['step'])[:len(I_raw)]

        # process I, Q data
        (I, Q, mag, phase) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=0, sub_mean=False)

        a.close()
    return f, I, Q


def plot_freq_data(f, I, Q, mag, phase, expected_f, show=['I', 'Q'], domain=None, mag_phase_plot=False, polar=False, title='', marker=False,plot=True):
    """Fits frequency data to a lorentzian, then plots data and prints result

    :param f -- data frequency
    :param I -- I data
    :param Q -- Q data
    :param mag -- mag data
    :param phase -- phase data
    :param expected_f -- expected frequency
    :param show -- if you want to plot I, Q, or both
    :param mag_phase_plot -- boolean, determines whether or not you plot mag and phase as well
    :param polar -- adds a plot of I and Q in polar coordinates
    :param title -- title of plot
    :param marker -- plots a vertical line at marker
    :returns: dsfit result
    """

    p = fitlor(f, np.square(mag), showfit=False)

    if plot==True:
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, title=title)
        if 'I' in show:
            ax.plot(f, I, 'b.-', label='I')
        if 'Q' in show:
            ax.plot(f, Q, 'r.-', label='Q')
        if marker:
            ax.axvline(x=marker)
        ax.set_xlabel('Freq(GHz)')
        ax.set_ylabel('I and Q (V)')
        ax.set_xlim(f[0], f[-1])
        ax.legend(loc='upper right')
        if mag_phase_plot:
            fig = plt.figure(figsize=(14, 5))
            ax3 = fig.add_subplot(111, title=' Mag and phase')
            ax3.plot(f, mag, 'g.-', label='Magnitude')
            ax3.set_xlabel('Freq(GHz)')
            ax3.set_ylabel('Magnitude (V)')
            ax3.set_xlim(f[0], f[-1])
            p = fitlor(f, np.square(mag), showfit=False, domain=domain)
            ax3.plot(f, sqrt(lorfunc(p, f)), 'k.-', label='fit')
            ax3.axvline(p[2], color='k', linestyle='dashed', label="fit freq")
            ax3.axvline(expected_f, color='r', linestyle='dashed', label="expected freq")
            ax3.legend(loc='upper right')
            ax4 = ax3.twinx()
            ax4.plot(f, phase, 'm.-', label='Phase')
            ax4.set_ylabel('Phase')
            ax4.legend(loc='lower right')

        else:
            ax2 = ax.twinx()
            ax2.plot(f, mag, 'k.-', alpha=0.3, label='mag')
            ax2.set_ylabel('Mag (V)')
            p = fitlor(f, np.square(mag), showfit=False, domain=domain)
            ax2.plot(f, sqrt(lorfunc(p, f)), 'k--', label='fit')
            ax2.axvline(p[2], color='k', linestyle='dashed', label="fit  freq")
            ax2.axvline(expected_f, color='r', linestyle='dashed', label="expected  freq")
            ax2.legend(loc='lower right')

    maxpt = np.max(mag)
    maxpt_x = np.where(mag == maxpt)
    print("ANALYSIS")
    print("expected frequency: ", expected_f)
    print("Resonant frequency from fitting mag squared: ", p[2], "GHz")
    print("Max Point: ", f[maxpt_x], "GHz")
    print("HWHM: ", p[3] * 1e3, "MHz")
    print()

    if plot==True:
        fig.tight_layout()
        plt.show()

    if polar and plot==True:
        plt.title("I vs Q polar plot")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.grid()
        plt.plot(I, Q)
        print("Qmin = " + str(np.min(Q)))
        print("Qmax = " + str(np.max(Q)))
        print("Imin = " + str(np.min(I)))
        print("Imax = " + str(np.max(I)))
        plt.show()

    return p

def get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=False):

    params = {}

    params['ran'] = hardware_cfg['awg_info']['keysight_pxi']['digtzr_vpp_range']
    params['dt_dig'] = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
    params['readout_params'] = lattice_cfg['readout']

    if one_qb:
        on_qb = on_qbs[0]
        rd_qb = on_rds[0]
        params["rd_setup"] = lattice_cfg["qubit"]["setup"][rd_qb]
        params["qb_setup"] = lattice_cfg["qubit"]["setup"][on_qb]
        rd_setup = params["rd_setup"]
        qb_setup = params["qb_setup"]
        params['readout_freq'] = params['readout_params'][rd_setup]["freq"][rd_qb]
        params['readout_window'] = params['readout_params'][rd_setup]["window"][rd_qb] * params['dt_dig']

        params['dig_atten_qb'] = lattice_cfg['powers'][qb_setup]['drive_digital_attenuation'][on_qb]
        params['dig_atten_rd'] = lattice_cfg['powers'][rd_setup]['readout_drive_digital_attenuation'][rd_qb]
        params['read_lo_pwr'] = lattice_cfg['powers'][rd_setup]['readout_drive_lo_powers'][rd_qb]
        params['qb_lo_pwr'] = lattice_cfg['powers'][qb_setup]['drive_lo_powers'][on_qb]

        params['qb_freq'] = lattice_cfg['qubit']['freq'][on_qb]

    else:
        on_qb = on_qbs[0]
        rd_qb = on_rds[0]
        params["rd_setup"] = lattice_cfg["qubit"]["setup"][rd_qb]
        params["qb_setup"] = lattice_cfg["qubit"]["setup"][on_qb]
        rd_setup = params["rd_setup"]
        qb_setup = params["qb_setup"]
        params['readout_freq'] = params['readout_params'][rd_setup]["freq"][rd_qb]
        params['readout_window'] = params['readout_params'][rd_setup]["window"][rd_qb] * params['dt_dig']

        params['dig_atten_qb'] = lattice_cfg['powers'][qb_setup]['drive_digital_attenuation'][on_qb]
        params['dig_atten_rd'] = lattice_cfg['powers'][rd_setup]['readout_drive_digital_attenuation'][rd_qb]
        params['read_lo_pwr'] = lattice_cfg['powers'][rd_setup]['readout_drive_lo_powers'][rd_qb]
        params['qb_lo_pwr'] = lattice_cfg['powers'][qb_setup]['drive_lo_powers'][on_qb]

        params['qb_freq'] = lattice_cfg['qubit']['freq'][on_qb]


    return params

def get_config(filenb, expt_name):
    """Takes in any experiemnt, and returns config

    Keyword arguments:
    filenb -- the data runs you want to analyze and plot
    expt_name -- the name of the experiment you are doing this with
    """
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
    return hardware_cfg, experiment_cfg, lattice_cfg

def check_sync(filenb, expt_name, expt_num=0):
    """Takes in any experiemnt, and plots averaged dig sample per record + window for each expt

    Keyword arguments:
    filenb -- the data runs you want to analyze and plot
    expt_name -- the name of the experiment you are doing this with
    expt--num -- the number of the expeirment that you want plotted out fully
    """
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)

        readout_window = params["readout_window"]
        dt_dig = params["dt_dig"]
        data_1 = np.array(a['I'])
        data_2 = np.array(a['Q'])
        data_1 = data_1  / 2 ** 15 * 1
        data_2 = data_2  / 2 ** 15 * 1

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131, title='I')
        plt.pcolormesh(data_1)
        ax.set_xlabel('Digitizer bins')
        ax.set_ylabel('Experiment number')
        ax2 = fig.add_subplot(132, title='Q')
        plt.pcolormesh(data_2)
        ax2.set_xlabel('Digitizer bins')
        ax2.set_ylabel('Experiment number')
        ax3 = fig.add_subplot(133, title='Expt num = ' + str(expt_num))
        ax3.plot(np.arange(data_1[0].size*dt_dig, step=dt_dig), data_1[expt_num])
        ax3.plot(np.arange(data_1[0].size*dt_dig, step=dt_dig), data_2[expt_num])
        ax3.axvspan(readout_window[0]*dt_dig, readout_window[1]*dt_dig, alpha=0.2, color='b')
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Signal')
        fig.tight_layout()
        plt.show()

def resonator_spectroscopy(filenb, phi=0, sub_mean=True, show=['I', 'Q'], mag_phase_plot=False, polar=False, debug=False, marker=False,plot=True, ff=False):
    """Fits resonator_spectroscopoy data, then plots data and prints result

    Keyword arguments:
    filelist -- the data runs you want to analyze and plot
    phi -- in degrees, angle by which you want to rotate IQ
    sub_mean -- if True, subtracts out the average background of IQ measurements
    show -- if plot I, Q, or both
    mag_phase_plot -- boolean, determines whether or not you plot mag and phase as well
    plar -- adds a plot of I and Q in polar coordinates
    debug -- print out experiment attributes
    marker -- plot vertical line where you expect
    """
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    if ff:
        expt_name = "ff_resonator_spectroscopy"
    else:
        expt_name = "resonator_spectroscopy"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        I_raw = a['I']
        Q_raw = a['Q']
        f = arange(expt_params['start'] + readout_f, expt_params['stop'] + readout_f, expt_params['step'])[:len(I_raw)]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)
        # process I, Q data
        (I, Q, mag, phase) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=phi, sub_mean=sub_mean)

        # plot and fit data
        title = expt_name
        plot_freq_data(f=f, I=I, Q=Q, mag=mag, phase=phase,
                       expected_f=readout_f, show=show, mag_phase_plot=mag_phase_plot, polar=polar, title=title, marker=marker,plot=plot)


def resonator_spectroscopy_pi(filenb, phi=0, sub_mean=True, show=['I', 'Q'], mag_phase_plot=False, polar=False, debug=False, marker=False,plot=True, ff=False):
    """Fits resonator_spectroscopoy data, then plots data and prints result

    Keyword arguments:
    filelist -- the data runs you want to analyze and plot
    phi -- in degrees, angle by which you want to rotate IQ
    sub_mean -- if True, subtracts out the average background of IQ measurements
    show -- if plot I, Q, or both
    mag_phase_plot -- boolean, determines whether or not you plot mag and phase as well
    plar -- adds a plot of I and Q in polar coordinates
    debug -- print out experiment attributes
    marker -- plot vertical line where you expect
    """
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    if ff:
        expt_name = "ff_resonator_spectroscopy_pi"
    else:
        expt_name = "resonator_spectroscopy_pi"

    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        I_raw = a['I']
        Q_raw = a['Q']
        f = arange(expt_params['start'] + readout_f, expt_params['stop'] + readout_f, expt_params['step'])[:len(I_raw)]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)
        # process I, Q data
        (I, Q, mag, phase) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=phi, sub_mean=sub_mean)

        # plot and fit data
        title = expt_name
        plot_freq_data(f=f, I=I, Q=Q, mag=mag, phase=phase,
                       expected_f=readout_f, show=show, mag_phase_plot=mag_phase_plot, polar=polar, title=title, marker=marker,plot=plot)


def resonator_spectroscopy_chi(filenb_g, filenb_e, marker=None, phi=0, sub_mean=True, show=['I', 'Q'], mag_phase_plot=False, polar=False, debug=False):
    expt_name_g = "resonator_spectroscopy"
    expt_name_e = "resonator_spectroscopy_pi"
    expt_name = "resonator_spectroscopy_pi"
    fname_g = "..\\data\\" + str(filenb_g).zfill(5) + "_" + expt_name_g.lower() + ".h5"
    fname_e = "..\\data\\" + str(filenb_e).zfill(5) + "_" + expt_name_e.lower() + ".h5"

    with File(fname_g, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        I_raw = a['I']
        Q_raw = a['Q']
        f = arange(expt_params['start'] + readout_f, expt_params['stop'] + readout_f, expt_params['step'])[:len(I_raw)]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)
        # process I, Q data
        (I, Q, mag, phase) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=phi, sub_mean=sub_mean)

        a.close()

    with File(fname_e, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        I_raw = a['I']
        Q_raw = a['Q']
        f = arange(expt_params['start'] + readout_f, expt_params['stop'] + readout_f, expt_params['step'])[
            :len(I_raw)]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)
        # process I, Q data
        (Ie, Qe, mage, phasee) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=phi, sub_mean=sub_mean)

        a.close()

    I_diff = Ie - I
    Q_diff = Qe - Q
    Icentered_diff = I_diff - mean(I_diff)
    Qcentered_diff = Q_diff - mean(Q_diff)
    magcentered_diff = sqrt(Icentered_diff ** 2 + Qcentered_diff ** 2)
    mag_diff = sqrt(I_diff ** 2 + Q_diff ** 2)
    phase_diff = np.arctan2(I_diff, Q_diff)
    phase_diff = np.unwrap(phase_diff)
    phasecentered_diff = np.arctan2(Icentered_diff, Qcentered_diff)
    phasecentered_diff = np.unwrap(phasecentered_diff)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, title=expt_name_e)
    if marker != None:
        ax.axvline(marker, color='r')
        print("marker is " + str(marker))
    ax.plot(f, I_diff, 'b.-', label='I_diff')
    ax.set_xlabel('Freq(GHz)')
    ax.set_ylabel('I')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(f, Q_diff, 'r.-', label='Q_diff')
    ax2.set_ylabel('Q')
    ax2.legend(loc='upper right')
    fig.tight_layout()

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, title=expt_name_e)
    ax.plot(f, magcentered_diff, 'k.-', label='Centered Magnitude e-g')
    if marker != None:
        ax.axvline(marker, color='r')
        print("marker is " + str(marker))
    ax.set_xlabel('Freq(GHz)')
    ax.set_ylabel('Centered Magnitude')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(f, phasecentered_diff, 'g.-', label='Centered Phase e-g')
    ax2.set_ylabel('Centered Phase')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

def ff_track_traj(filenb, phi=0, sub_mean=True, iq_plot=False, debug=False, marker=None, slices=False,plot=True):
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    expt_name = "ff_track_traj"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        flux_vec = expt_params['ff_vec']
        print("flux vec {}".format(flux_vec))

        nu_q = params['qb_freq']  # expected qubit freq
        ppiqstart = experiment_cfg[expt_name]['qbf_start']
        ppiqstop = experiment_cfg[expt_name]['qbf_stop']
        ppiqstep = experiment_cfg[expt_name]['qbf_step']
        freqvals = np.arange(ppiqstart, ppiqstop, ppiqstep) + nu_q

        per_start = experiment_cfg[expt_name]['perc_flux_vec_start']
        per_stop = experiment_cfg[expt_name]['perc_flux_vec_stop']
        per_step = experiment_cfg[expt_name]['perc_flux_vec_step']
        per_vals = np.arange(per_start, per_stop, per_step)

        I_raw = a['I']
        Q_raw = a['Q']

        magarray = []
        parray = []
        for i in range(I_raw.shape[0]):
            # process I, Q data
            (I, Q, mag, phase) = iq_process(f=freqvals, raw_I=I_raw[i], raw_Q=Q_raw[i], ran=ran, phi=phi,
                                            sub_mean=sub_mean)
            magarray.append(mag)
            p = fitlor(freqvals, np.square(mag), showfit=False)
            parray.append(p)


        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

        x, y = np.meshgrid(freqvals, per_vals[:len(I_raw)])
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(4)
        fig.set_figwidth(8)
        #freq_new = np.append(freqvals, 0)
        #tvals_new = np.append(t_vals[:len(I_raw)], 0)
        # print(freq_new.shape)
        # print(tvals_new.shape)
        #ax.pcolormesh(x, y, magarray, shading='flat', cmap='RdBu')
        ax.pcolormesh(x , y, magarray, shading='nearest', cmap='RdBu')
        ax.set_ylabel('% of vector')
        ax.set_xlabel('freq (GHz)')
        ax.set_title('PPIQ Sweep Across Flux Pulse')
        fig.tight_layout()
        plt.show()
        # figure(figsize=(12, 4))
        # plt.pcolormesh(x, y, magarray, shading='nearest', cmap='RdBu')
        # plt.show()

        if iq_plot:
            x, y = np.meshgrid(freqvals, t_vals[:len(I_raw)])
            figure(figsize=(12, 4))
            plt.pcolormesh(x, y, I_raw, shading='nearest', cmap='RdBu')
            plt.show()
            figure(figsize=(12, 4))
            plt.pcolormesh(x, y, Q_raw, shading='nearest', cmap='RdBu')
            plt.show()

        if slices:
            for i in range(I_raw.shape[0]):
                # process I, Q data
                (I, Q, mag, phase) = iq_process(f=freqvals, raw_I=I_raw[i], raw_Q=Q_raw[i], ran=ran, phi=phi,
                                                sub_mean=sub_mean)

                # plot and fit data
                title = expt_name
                plot_freq_data(f=freqvals, I=I, Q=Q, mag=mag, phase=phase,
                               expected_f=nu_q, mag_phase_plot=False, polar=False, title=title, marker=marker,plot=plot)
        return per_vals[:len(I_raw)], parray[2]

def ff_ramp_cal_ppiq(filenb, phi=0, sub_mean=True, iq_plot=False, debug=False, marker=None, mag_phase_plot=False, polar=False,
                     slices=False,plot=True):
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    expt_name = "ff_ramp_cal_ppiq"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        flux_vec = expt_params['ff_vec']
        print("flux vec {}".format(flux_vec))

        nu_q = params['qb_freq']  # expected qubit freq
        ppiqstart = experiment_cfg[expt_name]['start']
        ppiqstop = experiment_cfg[expt_name]['stop']
        ppiqstep = experiment_cfg[expt_name]['step']
        freqvals = np.arange(ppiqstart, ppiqstop, ppiqstep) + nu_q

        dtstart = experiment_cfg[expt_name]['dt_start']
        dtstop = experiment_cfg[expt_name]['dt_stop']
        dtstep = experiment_cfg[expt_name]['dt_step']
        t_vals = np.arange(dtstart, dtstop, dtstep)

        I_raw = a['I']
        Q_raw = a['Q']

        magarray = []
        parray = []
        for i in range(I_raw.shape[0]):
            # process I, Q data
            (I, Q, mag, phase) = iq_process(f=freqvals, raw_I=I_raw[i], raw_Q=Q_raw[i], ran=ran, phi=phi,
                                            sub_mean=sub_mean)
            magarray.append(mag)
            p = fitlor(freqvals, np.square(mag), showfit=False)
            parray.append(p)
            #magarray.append(np.append(mag, 0))
        #magarray.append(np.zeros(len(magarray[0])))

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

        if plot==True:
            x, y = np.meshgrid(freqvals, t_vals[:len(I_raw)])
            fig, ax = plt.subplots(1,1)
            fig.set_figheight(4)
            fig.set_figwidth(8)
            #freq_new = np.append(freqvals, 0)
            #tvals_new = np.append(t_vals[:len(I_raw)], 0)
            # print(freq_new.shape)
            # print(tvals_new.shape)
            #ax.pcolormesh(x, y, magarray, shading='flat', cmap='RdBu')
            ax.pcolormesh(x , y, magarray, shading='nearest', cmap='RdBu')
            ax.set_ylabel('dt (ns)')
            ax.set_xlabel('freq (GHz)')
            ax.set_title('PPIQ Sweep Across Flux Pulse')
            fig.tight_layout()
            plt.show()
        # figure(figsize=(12, 4))
        # plt.pcolormesh(x, y, magarray, shading='nearest', cmap='RdBu')
        # plt.show()

        if iq_plot:
            x, y = np.meshgrid(freqvals, t_vals[:len(I_raw)])
            figure(figsize=(12, 4))
            plt.pcolormesh(x, y, I_raw, shading='nearest', cmap='RdBu')
            plt.show()
            figure(figsize=(12, 4))
            plt.pcolormesh(x, y, Q_raw, shading='nearest', cmap='RdBu')
            plt.show()

        if slices:
            for i in range(I_raw.shape[0]):
                # process I, Q data
                (I, Q, mag, phase) = iq_process(f=freqvals, raw_I=I_raw[i], raw_Q=Q_raw[i], ran=ran, phi=phi,
                                                sub_mean=sub_mean)

                # plot and fit data
                title = expt_name
                plot_freq_data(f=freqvals, I=I, Q=Q, mag=mag, phase=phase,
                               expected_f=nu_q, mag_phase_plot=mag_phase_plot, polar=polar, title=title, marker=marker,plot=plot)
        return t_vals[:len(I_raw)], parray


def melting_single_readout_full_ramp(filenb, phi=0, sub_mean=False, show=["I"], fitparams=None, domain=None,
                                     debug=False, pi_cal=False, norm_plot=False, rabi_vals=None, fit_J=False):
    expt_name = "melting_single_readout_full_ramp"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = np.array(a["I"])
        Q_raw = np.array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['evolution_t_start'], expt_params['evolution_t_stop'], expt_params['evolution_t_step'])[
            :(len(I_raw))]

        J = []

        for s in show:
            if pi_cal:
                plt.plot(t, eval(s)[0:-2])
                if rabi_vals != None:
                    plt.axhline(y=rabi_vals[0], c='grey', label="g rabi")
                    plt.axhline(y=rabi_vals[1], c='m', label="e rabi")
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.axhline(y=eval(s)[-2], c='g', label="g")
                plt.axhline(y=eval(s)[-1], c='r', label="e")
                plt.legend()
                plt.xlabel('time (ns)')
                plt.ylabel(s + " volts")
                plt.show()

                if norm_plot:
                    plt.plot(t, (eval(s)[0:-2] - eval(s)[-1]) / (eval(s)[-2] - eval(s)[-1]))
                    plt.ylim((-0.1, 1))
                    plt.xlabel('time (ns)')
                    plt.ylabel(s + ' normalized')
                    plt.legend()
            #             plt.show()
            else:
                plt.title(s)
                plt.plot(t, eval(s))
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.show()
        if fit_J:
            return J

def melting_single_readout_full_ramp_v2(filenb, phi=0, sub_mean=False, show=["I"], fitparams=None, domain=None,
                                     debug=False, pi_cal=True, norm_plot=False, rabi_vals=None, fit_J=False):

    expt_name = "melting_single_readout_full_ramp"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = np.array(a["I"])
        Q_raw = np.array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['evolution_t_start'], expt_params['evolution_t_stop'], expt_params['evolution_t_step'])[
            :(len(I_raw))]

        J = []

        for s in show:
            if pi_cal:
                pi_file = a.attrs['pi_cal_fname']
                with File(pi_file, 'r') as p:
                    I_raw_P = np.array(p["I"])
                    Q_raw_P = np.array(p["Q"])
                    (I_P, Q_P, mag_P, phase_P) = iq_process(f, I_raw_P, Q_raw_P, ran, phi, sub_mean)
                print(I_P)
                I = np.concatenate((I, I_P))
                Q = np.concatenate((Q, Q_P))
                plt.plot(t, eval(s)[0:-2])
                if rabi_vals != None:
                    plt.axhline(y=rabi_vals[0], c='grey', label="g rabi")
                    plt.axhline(y=rabi_vals[1], c='m', label="e rabi")
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.axhline(y=eval(s)[-2], c='g', label="g")
                plt.axhline(y=eval(s)[-1], c='r', label="e")
                plt.legend()
                plt.xlabel('time (ns)')
                plt.ylabel(s + " volts")
                plt.show()

                if norm_plot:
                    plt.plot(t, (eval(s)[0:-2] - eval(s)[-2]) / (eval(s)[-1] - eval(s)[-2]))
                    plt.ylim((-0.1, 1))
                    plt.xlabel('time (ns)')
                    plt.ylabel(s + ' normalized')
                    plt.legend()
            #             plt.show()
            else:
                plt.title(s)
                plt.plot(t, eval(s))
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.show()
        if fit_J:
            return J
        else:
            return I,Q

def melting_single_readout_full_ramp_Q3(filenb, phi=0, sub_mean=False, show=["I"], fitparams=None, domain=None,
                                     debug=False, pi_cal=True, norm_plot=False, rabi_vals=None, fit_J=False):

    expt_name = "melting_single_readout_full_ramp_Q3"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = np.array(a["I"])
        Q_raw = np.array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['evolution_t_start'], expt_params['evolution_t_stop'], expt_params['evolution_t_step'])[
            :(len(I_raw))]

        J = []

        for s in show:
            if pi_cal:
                pi_file = a.attrs['pi_cal_fname']
                with File(pi_file, 'r') as p:
                    I_raw_P = np.array(p["I"])
                    Q_raw_P = np.array(p["Q"])
                    (I_P, Q_P, mag_P, phase_P) = iq_process(f, I_raw_P, Q_raw_P, ran, phi, sub_mean)
                print(I_P)
                I = np.concatenate((I, I_P))
                Q = np.concatenate((Q, Q_P))
                plt.plot(t, eval(s)[0:-2])
                if rabi_vals != None:
                    plt.axhline(y=rabi_vals[0], c='grey', label="g rabi")
                    plt.axhline(y=rabi_vals[1], c='m', label="e rabi")
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.axhline(y=eval(s)[-2], c='g', label="g")
                plt.axhline(y=eval(s)[-1], c='r', label="e")
                plt.legend()
                plt.xlabel('time (ns)')
                plt.ylabel(s + " volts")
                plt.show()

                if norm_plot:
                    plt.plot(t, (eval(s)[0:-2] - eval(s)[-2]) / (eval(s)[-1] - eval(s)[-2]))
                    plt.ylim((-0.1, 1))
                    plt.xlabel('time (ns)')
                    plt.ylabel(s + ' normalized')
                    plt.legend()
            #             plt.show()
            else:
                plt.title(s)
                plt.plot(t, eval(s))
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.show()
        if fit_J:
            return J
        else:
            return I,Q

def melting_single_readout_full_ramp_2setups(filenb, phi=0, sub_mean=False, show=["I"], fitparams=None, domain=None,
                                     debug=False, pi_cal=True, norm_plot=False, rabi_vals=None, fit_J=False):

    expt_name = "melting_single_readout_full_ramp_Q3"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = np.array(a["I"])
        Q_raw = np.array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['evolution_t_start'], expt_params['evolution_t_stop'], expt_params['evolution_t_step'])[
            :(len(I_raw))]

        J = []

        for s in show:
            if pi_cal:
                pi_file = a.attrs['pi_cal_fname']
                with File(pi_file, 'r') as p:
                    I_raw_P = np.array(p["I"])
                    Q_raw_P = np.array(p["Q"])
                    (I_P, Q_P, mag_P, phase_P) = iq_process(f, I_raw_P, Q_raw_P, ran, phi, sub_mean)
                print(I_P)
                I = np.concatenate((I, I_P))
                Q = np.concatenate((Q, Q_P))
                plt.plot(t, eval(s)[0:-2])
                if rabi_vals != None:
                    plt.axhline(y=rabi_vals[0], c='grey', label="g rabi")
                    plt.axhline(y=rabi_vals[1], c='m', label="e rabi")
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.axhline(y=eval(s)[-2], c='g', label="g")
                plt.axhline(y=eval(s)[-1], c='r', label="e")
                plt.legend()
                plt.xlabel('time (ns)')
                plt.ylabel(s + " volts")
                plt.show()

                if norm_plot:
                    plt.plot(t, (eval(s)[0:-2] - eval(s)[-2]) / (eval(s)[-1] - eval(s)[-2]))
                    plt.ylim((-0.1, 1))
                    plt.xlabel('time (ns)')
                    plt.ylabel(s + ' normalized')
                    plt.legend()
            #             plt.show()
            else:
                plt.title(s)
                plt.plot(t, eval(s))
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.show()
        if fit_J:
            return J
        else:
            return I,Q

def melting_single_readout_full_ramp(filenb, phi=0, sub_mean=False, show=["I"], fitparams=None, domain=None,
                                     debug=False, pi_cal=True, norm_plot=False, rabi_vals=None, fit_J=False):

    expt_name = "melting_single_readout_full_ramp"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = np.array(a["I"])
        Q_raw = np.array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['evolution_t_start'], expt_params['evolution_t_stop'], expt_params['evolution_t_step'])[
            :(len(I_raw))]

        J = []

        for s in show:
            if pi_cal:
                pi_file = a.attrs['pi_cal_fname']
                with File(pi_file, 'r') as p:
                    I_raw_P = np.array(p["I"])
                    Q_raw_P = np.array(p["Q"])
                    (I_P, Q_P, mag_P, phase_P) = iq_process(f, I_raw_P, Q_raw_P, ran, phi, sub_mean)
                print(I_P)
                I = np.concatenate((I, I_P))
                Q = np.concatenate((Q, Q_P))
                plt.plot(t, eval(s)[0:-2])
                if rabi_vals != None:
                    plt.axhline(y=rabi_vals[0], c='grey', label="g rabi")
                    plt.axhline(y=rabi_vals[1], c='m', label="e rabi")
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.axhline(y=eval(s)[-2], c='g', label="g")
                plt.axhline(y=eval(s)[-1], c='r', label="e")
                plt.legend()
                plt.xlabel('time (ns)')
                plt.ylabel(s + " volts")
                plt.show()

                if norm_plot:
                    plt.plot(t, (eval(s)[0:-2] - eval(s)[-2]) / (eval(s)[-1] - eval(s)[-2]))
                    plt.ylim((-0.1, 1))
                    plt.xlabel('time (ns)')
                    plt.ylabel(s + ' normalized')
                    plt.legend()
            #             plt.show()
            else:
                plt.title(s)
                plt.plot(t, eval(s))
                if fit_J:
                    res = fit_sin(t, eval(s)[0:-2])
                    J.append(res['omega'] * 10 ** 9 / ((2 * 2 * np.pi * 10 ** 6)))
                    plt.plot(t, res['fitfunc'](t))
                plt.show()
        if fit_J:
            return J
        else:
            return I,Q


def pulse_probe_iq(filenb, phi=0, sub_mean=True, show=['I', 'Q'], domain=None,mag_phase_plot=False, polar=False, debug=False, marker=None,plot = True, ff=False):
    """Fits pulse_probe_iq data, then plots data and prints result
    :param filelist -- the data runs you want to analyze and plot
    :paramphi -- in degrees, angle by which you want to rotate IQ
    :paramsub_mean -- if True, subtracts out the average background of IQ measurements
    :param show -- if plot I or Q
    :parammag_phase_plot -- boolean, determines whether or not you plot mag and phase as well
    :paramplar -- adds a plot of I and Q in polar coordinates
    :paramdebug -- print out experiment attributes
    """
    # the 'with' statement automatically opens and closes File a, even if code in the with block fails
    expt_name = "pulse_probe_iq"
    if ff:
        expt_name = "ff_pulse_probe_iq"

    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        # get data in from json file
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        # print(quantum_device_cfg.keys())
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']

        nu_q = params['qb_freq']  # expected qubit freq
        if ff:
            ff_vec = expt_params['ff_vec']
            print("fast flux vec is {}".format(ff_vec))

        I_raw = a['I']
        Q_raw = a['Q']
        f = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I_raw))] + nu_q

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

        # process I, Q data
        (I, Q, mag, phase) = iq_process(f=f, raw_I=I_raw, raw_Q=Q_raw, ran=ran, phi=phi, sub_mean=sub_mean)

        # plot and fit data
        title = expt_name
        p = plot_freq_data(f=f, I=I, Q=Q, mag=mag, phase=phase,
                       expected_f=nu_q, show=show, domain=domain, mag_phase_plot=mag_phase_plot, polar=polar, title=title, marker=marker,plot=plot)
        return p

def ff_pulse_probe_iq(filenb, phi=0, sub_mean=True, show=['I', 'Q'], mag_phase_plot=False, polar=False, debug=False, marker=None):
    print("This version has been deprecated, just pass ff=True to pulse probe iq")

def rabi(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False, ff=False,decay = True):
    """
    takes in rabi data, processes it, plots it, and fits it
    :param filenb: filenumber
    :param phi: desired iq rotation
    :param sub_mean: boolean, if you want to subract mean off data
    :param show: array of strings, whether you want to fit to I or Q or both
    :param fitparams: array of starting values for parameters
    :param domain: domain over which you want to fit, ie [1000, 2000]
    :param debug: boolean, prints out all experient parameters
    """

    # if you use gaussian pulses for Rabi, the time that you plot is the sigma -the actual pulse time is 4*sigma (4 being set in the code
    SIGMA_CUTOFF = 4

    expt_name = "rabi"
    if ff:
        expt_name = "ff_rabi"

    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        #####IMPORT STUFF#####
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]

        pulse_type = expt_params['pulse_type']
        amp = expt_params['amp']

        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran'] # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']


        ####GET IQ #####
        if "threshold" in expt_params.keys() and expt_params["threshold"]:
            state = array(a["state"])
        I_raw = array(a["I"])
        Q_raw = array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I))]
        if pulse_type == "gauss":
            t = t * 4

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

            #####PLOT####
        title = expt_name + '$, \\nu_q$ = ' + str(around(nu_q, 3)) + ' GHz ' + 'amp = ' + str(amp)
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, title=title)
        for s in show:
            if s == "Q":
                ax.plot(t, eval(s), 'ro-', label=s)
            else:
                ax.plot(t, eval(s), 'bo-', label=s)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(s + " (V)")

            ####FIT AND ANALYZE###
            if decay == True:
                p = fitdecaysin(t, eval(s), showfit=False, fitparams=fitparams, domain=domain)
                ax.plot(t, decaysin3(p, t), 'k-', label="fit")
            else:
                sin3 = lambda p, x: p[0] * np.sin(2. * np.pi * p[1] * x + p[2] * np.pi / 180.) + p[3]
                p = fitsin(t, eval(s), showfit=False, fitparams=fitparams, domain=domain)
                ax.plot(t, sin3(p, t), 'k-', label="fit")


            t_pi = 1 / (2 * p[1])
            t_half_pi = 1 / (4 * p[1])
            ax.axvline(t_pi, color='k', linestyle='dashed')
            ax.axvline(t_half_pi, color='k', linestyle='dashed')

            print("ANALYSIS " + s)
            if pulse_type == 'gauss':
                print("pulse type is GAUSS! pi_len in sigma to input into code to get right pulse out is:")
                print("Half pi length in sigma =", t_half_pi / 4, "sigma")
                print("pi length in sigma =", t_pi / 4, "sigma")
                print("\n")

            print("Half pi length in ns =", t_half_pi, "ns")
            print("pi length =", t_pi, "ns")
            print("suggested_pi_length = ", int(t_pi) + 1, "    suggested_pi_amp = ",
                  amp * (t_pi) / float(int(t_pi) + 1))
            if decay == True:
                print("Rabi decay time = ", p[3], "ns")
            if debug:
                print(p)

        ax.legend()
        plt.show()
    return p


def t1(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False):
    """
    takes in t1 data, processes it, plots it, and fits it
    :param filenb: filenumber
    :param phi: desired iq rotation
    :param sub_mean: boolean, if you want to subract mean off data
    :param show: array of strings, whether you want to fit to I or Q or both
    :param fitparams: array of starting values for parameters
    :param domain: domain over which you want to fit, ie [1000, 2000]
    :param debug: boolean, prints out all experient parameters
    """

    expt_name = "t1"
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = array(a["I"])
        Q_raw = array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)

        # check for old version/new version for back-compatibility
        try:
            t1_len_array = expt_params["t1_len_array"]
        except:
            t1_len_array = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I))]

        # new version
        if t1_len_array =="auto":
            t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I))]
        else:
            t = t1_len_array

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

            ####PLOT####
        title = expt_name + '$, \\nu_q$ = ' + str(around(nu_q, 3)) + ' GHz '
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, title=title)
        t = np.array(t) / 1000  # convert to us
        for s in show:
            if s == "Q":
                ax.plot(t, eval(s), 'ro-', label=s)
            else:
                ax.plot(t, eval(s), 'bo-', label=s)
            ax.set_xlabel('Time (us)')
            ax.set_ylabel(s + " (V)")

            ####ANALYZE
            p = fitexp(t, eval(s), fitparams=fitparams, domain=domain, showfit=False)
            T1 = p[3]
            ax.plot(t, expfunc(p, t), 'k-', label='fit')
            print("ANALYSIS " + s)
            print("T1  =", p[3], "us")
            if debug:
                print(p)

        ax.legend()
        plt.show()
    return p

def ramsey(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False):
    """
    takes in ramsey data, processes it, plots it, and fits it
    :param filenb: filenumber
    :param phi: desired iq rotation
    :param sub_mean: boolean, if you want to subract mean off data
    :param show: array of strings, whether you want to fit to I or Q or both
    :param fitparams: array of starting values for parameters
    :param domain: domain over which you want to fit, ie [1000, 2000]
    :param debug: boolean, prints out all experient parameters
    """
    expt_name = 'ramsey'
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]

        ramsey_freq = expt_params['ramsey_freq'] * 1e3

        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = array(a["I"])
        Q_raw = array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I))]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

        title = expt_name + '$, \\nu_q$ = ' + str(around(nu_q, 3)) + ' GHz '
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, title=title)
        t = t / 1000

        for s in show:
            if s == "Q":
                ax.plot(t, eval(s), 'ro-', label=s)
            else:
                ax.plot(t, eval(s), 'bo-', label=s)
            ax.set_xlabel('Time ($\mu$s)')
            ax.set_ylabel(s + " (V)")

            ######ANALYZE
            # p = fitdecaysin(t[3:],s[3:],fitparams = fitparams, domain = domain, showfit=False)
            p = fitdecaysin(t, eval(s), fitparams=fitparams, domain=domain, showfit=False)
            ax.plot(t, decaysin3(p, t), 'k-', label='fit')

            offset = ramsey_freq - p[1]
            nu_q_new = nu_q + offset * 1e-3

            print("ANALYSIS")
            print("Original qubit frequency choice =", nu_q, "GHz")
            print("ramsey freq = ", ramsey_freq, "MHz")
            print("Oscillation freq = ", p[1], " MHz")
            print("Offset freq between data and ramsey =", offset, "MHz")
            print("Suggested qubit frequency choice =", nu_q_new, "GHz")
            print("T2* =", p[3], "us")
        ax.legend()
        plt.show()
    return p, nu_q_new

def echo(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False):
    """
    takes in echo data, processes it, plots it, and fits it
    :param filenb: filenumber
    :param phi: desired iq rotation
    :param sub_mean: boolean, if you want to subract mean off data
    :param show: array of strings, whether you want to fit to I or Q or both
    :param fitparams: array of starting values for parameters
    :param domain: domain over which you want to fit, ie [1000, 2000]
    :param debug: boolean, prints out all experient parameters
    """
    expt_name = 'echo'
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"
    with File(filename, 'r') as a:
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]

        ramsey_freq = expt_params['ramsey_freq'] * 1e3
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        I_raw = array(a["I"])
        Q_raw = array(a["Q"])
        (I, Q, mag, phase) = iq_process(f, I_raw, Q_raw, ran, phi, sub_mean)
        t = arange(expt_params['start'], expt_params['stop'], expt_params['step'])[:(len(I))]

        if debug:
            print("DEBUG")
            print("averages =", expt_params['acquisition_num'])
            print("Rd LO pwr= ", read_lo_pwr, "dBm")
            print("Qb LO pwr= ", qb_lo_pwr, "dBm")
            print("Rd atten= ", dig_atten_rd, "dB")
            print("Qb atten= ", dig_atten_qb, "dB")
            print("Readout params", readout_params)
            print("experiment params", expt_params)

        title = expt_name + '$, \\nu_q$ = ' + str(around(nu_q, 3)) + ' GHz '
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, title=title)
        t = t / 1000

        for s in show:
            if s == "Q":
                ax.plot(t, eval(s), 'ro-', label=s)
            else:
                ax.plot(t, eval(s), 'bo-', label=s)
            ax.set_xlabel('Time ($\mu$s)')
            ax.set_ylabel(s + " (V)")

            ######ANALYZE
            # p = fitdecaysin(t[3:],s[3:],fitparams = fitparams, domain = domain, showfit=False)
            p = fitdecaysin(t, eval(s), fitparams=fitparams, domain=domain, showfit=False)
            print(p)
            ax.plot(t, decaysin3(p, t), 'k-', label='fit')

            offset = ramsey_freq - p[1]
            nu_q_new = nu_q + offset * 1e-3

            print("ANALYSIS")
            print("Original qubit frequency choice =", nu_q, "GHz")
            print("ramsey freq = ", ramsey_freq, "MHz")
            print("Oscillation freq = ", p[1], " MHz")
            print("Offset freq between data and ramsey =", offset, "MHz")
            print("Suggested qubit frequency choice =", nu_q_new, "GHz")
            print("Echo experiment: CP = ", expt_params['cp'], "CPMG = ", expt_params['cpmg'])
            print("Number of echoes = ", expt_params['echo_times'])
            print("T2 =", p[3], "us")
        ax.legend()
        plt.show()
        return p


def fid_func(Vval, ssbinsg, ssbinse, sshg, sshe):
    ind_g = np.argmax(ssbinsg > Vval)
    ind_e = np.argmax(ssbinse > Vval)
    return np.abs(((np.sum(sshg[:ind_g]) - np.sum(sshe[:ind_e])) / sshg.sum()))


def histogram_fit(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False, rancut=120,
                  details=True, numbins=200, ff=False, IQrot = True):
    expt_name = 'histogram'
    if ff:
        expt_name = 'ff_histogram'

    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"

    with File(filename, 'r') as a:

        tags = ''

        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        expt_cfg = (json.loads(a.attrs['experiment_cfg']))[expt_name.lower()]
        # numbins = expt_cfg['numbins']
        a_num = expt_cfg['acquisition_num']
        ns = expt_cfg['num_seq_sets']
        readout_length = readout_params['length']
        window = params['readout_window']
        atten = '0'

        if details:
            print('Readout length = ', readout_length)
            print('Readout window = ', window)
            print("Digital atten = ", atten)
            print("Readout freq = ", readout_f)
        I = array(a['I'])
        Q = array(a['Q'])
        sample = a_num

        I, Q = I / 2 ** 15 * ran, Q / 2 ** 15 * ran

        colors = ['b', 'r', 'g']
        labels = ['g', 'e', 'f']
        titles = ['I', 'Q']

        IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3], 1)
        IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[
                                                                                                    2::3], Q.T.flatten()[
                                                                                                           2::3]

        x0g, y0g = mean(IQsss[0][::int(a_num / sample)]), mean(IQsss[1][::int(a_num / sample)])
        x0e, y0e = mean(IQsss[2][::int(a_num / sample)]), mean(IQsss[3][::int(a_num / sample)])
        vec = [(x0g+x0e)/2, (y0g+y0e)/2]
        if IQrot:
            phi = arctan((y0e - y0g) / (x0e - x0g))
            if x0g>x0e:
                phi = phi+np.pi
        else:
            phi = 0
        if details:
            fig = plt.figure(figsize=(10, 5))

            ax = fig.add_subplot(221, title='length,window = ' + str(readout_length) + ',' + str(window))
            for ii in range(2):
                ax.plot(IQsss[2 * ii][:], IQsss[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

            ax.set_xlabel('I (V)')
            ax.set_ylabel('Q (V)')
            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

            ax = fig.add_subplot(222, title=tags + ', atten = ' + str(atten))

            for ii in range(2):
                ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
                            yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')

            ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
            ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

        I = I
        Q = Q
        print(phi)
        IQsssrot = (I.T.flatten()[0::3] * cos(phi) + Q.T.flatten()[0::3] * sin(phi),
                    -I.T.flatten()[0::3] * sin(phi) + Q.T.flatten()[0::3] * cos(phi),
                    I.T.flatten()[1::3] * cos(phi) + Q.T.flatten()[1::3] * sin(phi),
                    -I.T.flatten()[1::3] * sin(phi) + Q.T.flatten()[1::3] * cos(phi),
                    I.T.flatten()[2::3] * cos(phi) + Q.T.flatten()[2::3] * sin(phi),
                    -I.T.flatten()[2::3] * sin(phi) + Q.T.flatten()[2::3] * cos(phi))

        x0g, y0g = mean(IQsssrot[0][:]), mean(IQsssrot[1][:])
        x0e, y0e = mean(IQsssrot[2][:]), mean(IQsssrot[3][:])

        if details:
            for kk in range(2):

                ax = fig.add_subplot(2, 2, kk + 3, title=expt_name + titles[kk ])
                ax.hist(IQsssrot[kk], bins=numbins, alpha=0.75, color=colors[0], label=labels[0],histtype=u'step', linewidth = 1.5)
                ax.hist(IQsssrot[kk+2], bins=numbins, alpha=0.75, color=colors[1], label=labels[1],
                        histtype=u'step', linewidth=1.5)
                ax.axvline(x=0)
                ax.set_xlabel(titles[kk] + '(V)')
                ax.set_ylabel('Number')
                ax.legend()
                if kk == 0:
                    ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
                else:
                    ax.set_xlim(y0g - ran / rancut, y0g + ran / rancut)

    for ii, i in enumerate(['I', 'Q']):
        if ii is 0:
            lims = [x0g - ran / rancut, x0g + ran / rancut]
        else:
            lims = [y0g - ran / rancut, y0g + ran / rancut]
        sshg, ssbinsg = np.histogram(IQsssrot[ii], bins=numbins, range=lims)
        sshe, ssbinse = np.histogram(IQsssrot[ii + 2], bins=numbins, range=lims)
        fid_list = [fid_func(V, ssbinsg, ssbinse, sshg, sshe) for V in ssbinsg[0:-1]]
        fid = max(fid_list)
        if ii == 0:
            temp = fid

        if details:
            print("Single shot readout fidility from channel ", i, " after rotation = ", fid)
            print("Optimal angle =", phi)

    if details:
        print('---------------------------')

        fig.tight_layout()
        plt.show()

    return temp, phi, vec, IQsssrot[0], IQsssrot[2]


def histogram_og(filenb, phi=0, sub_mean=True, show=['I'], fitparams=None, domain=None, debug=False,rancut=120):
    expt_name = 'histogram'
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"

    with File(filename, 'r') as a:

        tags = ''


        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing
        readout_f = params['readout_freq']
        dig_atten_qb = params['dig_atten_qb']
        dig_atten_rd = params['dig_atten_rd']
        read_lo_pwr = params['read_lo_pwr']
        qb_lo_pwr = params['qb_lo_pwr']
        nu_q = params['qb_freq']

        expt_cfg = (json.loads(a.attrs['experiment_cfg']))[expt_name.lower()]
        numbins = expt_cfg['numbins']
        print(numbins)
        numbins = 200
        a_num = expt_cfg['acquisition_num']
        ns = expt_cfg['num_seq_sets']
        readout_length = readout_params['length']
        window = params['readout_window']
        atten = '0'
        print('Readout length = ', readout_length)
        print('Readout window = ', window)
        print("Digital atten = ", atten)
        print("Readout freq = ", readout_f)
        I = array(a['I'])
        Q = array(a['Q'])
        sample = a_num

        I, Q = I / 2 ** 15 * ran, Q / 2 ** 15 * ran

        colors = ['r', 'b', 'g']
        labels = ['g', 'e', 'f']
        titles = ['I', 'Q']

        IQs = mean(I[::3], 1), mean(Q[::3], 1), mean(I[1::3], 1), mean(Q[1::3], 1), mean(I[2::3], 1), mean(Q[2::3], 1)
        IQsss = I.T.flatten()[0::3], Q.T.flatten()[0::3], I.T.flatten()[1::3], Q.T.flatten()[1::3], I.T.flatten()[
                                                                                                    2::3], Q.T.flatten()[
                                                                                                           2::3]

        fig = plt.figure(figsize=(15, 7 * 5))

        ax = fig.add_subplot(621, title='length,window = ' + str(readout_length) + ',' + str(window))
        x0g, y0g = mean(IQsss[0][::int(a_num / sample)]), mean(IQsss[1][::int(a_num / sample)])
        x0e, y0e = mean(IQsss[2][::int(a_num / sample)]), mean(IQsss[3][::int(a_num / sample)])
        phi = arctan((y0e - y0g) / (x0e - x0g))
        for ii in range(2):
            ax.plot(IQsss[2 * ii][:], IQsss[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

        ax.set_xlabel('I (V)')
        ax.set_ylabel('Q (V)')
        ax.set_xlim(x0g - ran / rancut, x0g + ran / rancut)
        ax.set_ylim(y0g - ran / rancut, y0g + ran / rancut)

        ax = fig.add_subplot(622, title=tags  + ', atten = ' + str(atten))

        for ii in range(3):
            ax.errorbar(mean(IQsss[2 * ii]), mean(IQsss[2 * ii + 1]), xerr=std(IQsss[2 * ii]),
                        yerr=std(IQsss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

        ax.set_xlim(x0g - ran / rancut , x0g + ran / rancut )
        ax.set_ylim(y0g - ran / rancut , y0g + ran / rancut )

        IQsssrot = (I.T.flatten()[0::3] * cos(phi) + Q.T.flatten()[0::3] * sin(phi),
                    -I.T.flatten()[0::3] * sin(phi) + Q.T.flatten()[0::3] * cos(phi),
                    I.T.flatten()[1::3] * cos(phi) + Q.T.flatten()[1::3] * sin(phi),
                    -I.T.flatten()[1::3] * sin(phi) + Q.T.flatten()[1::3] * cos(phi),
                    I.T.flatten()[2::3] * cos(phi) + Q.T.flatten()[2::3] * sin(phi),
                    -I.T.flatten()[2::3] * sin(phi) + Q.T.flatten()[2::3] * cos(phi))

        ax = fig.add_subplot(623, title='rotated')
        x0g, y0g = mean(IQsssrot[0][:]), mean(IQsssrot[1][:])
        x0e, y0e = mean(IQsssrot[2][:]), mean(IQsssrot[3][:])

        for ii in range(3):
            ax.plot(IQsssrot[2 * ii][:], IQsssrot[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

        ax.set_xlabel('I (V)')
        ax.set_ylabel('Q (V)')
        ax.set_xlim(x0g - ran / rancut , x0g + ran / rancut )
        ax.set_ylim(y0g - ran / rancut , y0g + ran / rancut )

        ax = fig.add_subplot(624)

        for ii in range(3):
            ax.errorbar(mean(IQsssrot[2 * ii]), mean(IQsssrot[2 * ii + 1]), xerr=std(IQsssrot[2 * ii]),
                        yerr=std(IQsssrot[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

        ax.set_xlim(x0g - ran / rancut , x0g + ran / rancut )
        ax.set_ylim(y0g - ran / rancut , y0g + ran / rancut )

        for kk in range(4):

            ax = fig.add_subplot(6, 2, kk % 2 + 5, title=expt_name + titles[kk % 2])
            ax.hist(IQsssrot[kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
            ax.set_xlabel(titles[kk % 2] + '(V)')
            ax.set_ylabel('Number')
            ax.legend()
            if kk % 2 == 0:
                ax.set_xlim(x0g - ran / rancut , x0g + ran / rancut )
            else:
                ax.set_xlim(y0g - ran / rancut , y0g + ran / rancut )

    for ii, i in enumerate(['I', 'Q']):
        sshg, ssbinsg = np.histogram(IQsss[ii], bins=numbins)
        sshe, ssbinse = np.histogram(IQsss[ii + 2], bins=numbins)
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        print("Single shot readout fidility from channel ", i, " = ", fid)

    print('---------------------------')

    for ii, i in enumerate(['I', 'Q']):
        if ii is 0:
            lims = [x0g - ran / rancut , x0g + ran / rancut ]
        else:
            lims = [y0g - ran / rancut , y0g + ran / rancut ]
        sshg, ssbinsg = np.histogram(IQsssrot[ii], bins=numbins, range=lims)
        sshe, ssbinse = np.histogram(IQsssrot[ii + 2], bins=numbins, range=lims)
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        print("Single shot readout fidility from channel ", i, " after rotation = ", fid)
        if ii == 0:
            temp = fid
        print("Optimal angle =", phi)

        ax = fig.add_subplot(6, 2, 7 + ii)
        ax.plot(ssbinse[:-1], cumsum(sshg) / sshg.sum(), color='r')
        ax.plot(ssbinse[:-1], cumsum(sshe) / sshg.sum(), color='b')
        ax.plot(ssbinse[:-1], np.abs(cumsum(sshe) - cumsum(sshg)) / sshg.sum(), color='k')
        if ii == 0:
            ax.set_xlim(x0g - ran / rancut , x0g + ran / rancut )
        else:
            ax.set_xlim(y0g - ran / rancut , y0g + ran / rancut )
        ax.set_xlabel(titles[ii] + '(V)')
        ax.set_ylabel('$F$')

    print('---------------------------')

    fig.tight_layout()
    plt.show()

    return temp

def histogram_IQ_2setups(filenb):
    expt_name = 'histogram_2setups'
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"

    with File(filename, 'r') as a:
        tags = ''
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_params = experiment_cfg[expt_name.lower()]
        on_qbs = expt_params['on_qbs']
        on_rds = expt_params['on_rds']
        rd_qb = on_rds[0]

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing

        expt_cfg = (json.loads(a.attrs['experiment_cfg']))[expt_name.lower()]
        a_num = expt_cfg['acquisition_num']
        ns = expt_cfg['num_seq_sets']
        window = params['readout_window']

        num_seq = 1 + len(on_qbs) + expt_cfg["f"]

        labels = ['gQ{}'.format(rd_qb)]
        if len(on_qbs)>0:
            temp = ''
            for i in pi_qbs:
                temp = temp + "piQ{}".format(i)
            labels.append(temp)

        try:
            rd_setup = params["rd_setup"]
            I = array(a["qb{}_I".format(rd_setup)])
            Q = array(a["qb{}_Q".format(rd_setup)])
        except:
            I = array(a['I'])
            Q = array(a['Q'])

        I, Q = I / 2 ** 15 * ran, Q / 2 ** 15 * ran

        IQ_means = []
        IQ_ss = []
        for seq in range(num_seq):
            IQ_means.append(mean(I[seq::num_seq], 1))
            IQ_means.append(mean(A[seq::num_seq], 1))

            IQ_ss.append(I.T.flatten()[seq::num_seq])
            IQ_ss.append(Q.T.flatten()[seq::num_seq])

    return labels, IQ_ss


def histogram_IQ_crosstalk(filenb, phi=0):
    """returns:
    labels: [g, e, piQcross, pipiQcross]
    IQss: [ [I_g, Q_g], [I_e, Q_e], [I_piQcross, Q_piQcross], [I_pipiQcross, Q_pipiQcross]]
    IQssrot: same, but rotated """
    expt_name = 'histogram_crosstalk'
    filename = "..\\data\\" + str(filenb).zfill(5) + "_" + expt_name.lower() + ".h5"

    with File(filename, 'r') as a:
        tags = ''
        hardware_cfg = (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg = (json.loads(a.attrs['experiment_cfg']))
        lattice_cfg = (json.loads(a.attrs['lattice_cfg']))
        expt_cfg = experiment_cfg[expt_name.lower()]

        cross_qb = expt_cfg['cross_qb']
        on_qbs = expt_cfg['on_qbs']
        on_rds = expt_cfg['on_rds']
        on_qb = on_qbs[0]
        rd_qb = on_rds[0]

        params = get_params(hardware_cfg, experiment_cfg, lattice_cfg, on_qbs, on_rds, one_qb=True)
        readout_params = params['readout_params']
        ran = params['ran']  # range of DAC card for processing

        a_num = expt_cfg['acquisition_num']
        ns = expt_cfg['num_seq_sets']
        window = params['readout_window']

        num_seq = 4
        labels = ['gQ{}'.format(rd_qb), "eQ{}".format(rd_qb), "gQ{}piQ{}".format(rd_qb, cross_qb),
                  "eQ{}piQ{}".format(rd_qb, cross_qb)]

        try:
            I = array(a['I'])
            Q = array(a['Q'])
        except:
            rd_setup = lattice_cfg["qubit"]["setup"][rd_qb]
            I = array(a["qb{}_I".format(params["rd_setup"])])
            Q = array(a["qb{}_Q".format(params["rd_setup"])])

        I, Q = I / 2 ** 15 * ran, Q / 2 ** 15 * ran

        IQ_means = []
        IQss = []
        for seq in range(num_seq):
            IQss.append(
                [I.T.flatten()[seq::num_seq],
                 Q.T.flatten()[seq::num_seq]]
            )
        IQssrot = []

        for IQs in IQss:
            Irot = IQs[0] * cos(phi) + IQs[1] * sin(phi)
            Qrot = -IQs[0] * sin(phi) + IQs[1] * cos(phi)
            IQssrot.append([Irot, Qrot])

    return labels, IQss, IQssrot

def histogram_IQ_analyzer(IQss, labels=['g', 'e'], ran=1, tags=None, numbins=200):
    fig = plt.figure(figsize=(15, 7 * 5))
    colors = ['r', 'b']

    ax = fig.add_subplot(621, title=tags)
    x0g, y0g = mean(IQss[0]), mean(IQss[1])
    x0e, y0e = mean(IQss[2]), mean(IQss[3])
    phi = arctan((y0e - y0g) / (x0e - x0g))

    for ii in range(2):
        ax.plot(IQss[2 * ii][:], IQss[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

    ax.set_xlabel('I (V)')
    ax.set_ylabel('Q (V)')
    ax.set_xlim(x0g - ran , x0g + ran )
    ax.set_ylim(y0g - ran , y0g + ran )

    ax = fig.add_subplot(622, title=tags)

    for ii in range(2):
        ax.errorbar(mean(IQss[2 * ii]), mean(IQss[2 * ii + 1]), xerr=std(IQss[2 * ii]),
                    yerr=std(IQss[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_xlim(x0g - ran, x0g + ran)
    ax.set_ylim(y0g - ran, y0g + ran)

    IQssrot = (IQss[0] * cos(phi) + IQss[1] * sin(phi),
                -IQss[0] * sin(phi) + IQss[1] * cos(phi),
               IQss[2] * cos(phi) + IQss[3] * sin(phi),
               -IQss[2] * sin(phi) + IQss[3] * cos(phi))

    ax = fig.add_subplot(623, title='rotated')
    x0g, y0g = mean(IQssrot[0][:]), mean(IQssrot[1][:])
    x0e, y0e = mean(IQssrot[2][:]), mean(IQssrot[3][:])

    for ii in range(2):
        ax.plot(IQssrot[2 * ii][:], IQssrot[2 * ii + 1][:], '.', color=colors[ii], alpha=0.85)

    ax.set_xlabel('I (V)')
    ax.set_ylabel('Q (V)')
    ax.set_xlim(x0g - ran, x0g + ran)
    ax.set_ylim(y0g - ran, y0g + ran)

    ax = fig.add_subplot(624)

    for ii in range(2):
        ax.errorbar(mean(IQssrot[2 * ii]), mean(IQssrot[2 * ii + 1]), xerr=std(IQssrot[2 * ii]),
                    yerr=std(IQssrot[2 * ii + 1]), fmt='o', color=colors[ii], markersize=10)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_xlim(x0g - ran, x0g + ran)
    ax.set_ylim(y0g - ran, y0g + ran)

    for kk in range(4):

        ax = fig.add_subplot(6, 2, kk % 2 + 5, title=expt_name + labels[kk % 2])
        ax.hist(IQssrot[kk], bins=numbins, alpha=0.75, color=colors[int(kk / 2)], label=labels[int(kk / 2)])
        ax.set_xlabel(labels[kk % 2] + '(V)')
        ax.set_ylabel('Number')
        ax.legend()
        if kk % 2 == 0:
            ax.set_xlim(x0g - ran , x0g + ran )
        else:
            ax.set_xlim(y0g - ran , y0g + ran )


    for ii, i in enumerate(['I', 'Q']):
        sshg, ssbinsg = np.histogram(IQss[ii], bins=numbins)
        sshe, ssbinse = np.histogram(IQss[ii + 2], bins=numbins)
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        print("Single shot readout fidility from channel ", i, " = ", fid)

    print('---------------------------')

    for ii, i in enumerate(['I', 'Q']):
        if ii is 0:
            lims = [x0g - ran , x0g + ran ]
        else:
            lims = [y0g - ran, y0g + ran ]
        sshg, ssbinsg = np.histogram(IQssrot[ii], bins=numbins, range=lims)
        sshe, ssbinse = np.histogram(IQssrot[ii + 2], bins=numbins, range=lims)
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        print("Single shot readout fidility from channel ", i, " after rotation = ", fid)
        if ii == 0:
            temp = fid
        print("Optimal angle =", phi)

        ax = fig.add_subplot(6, 2, 7 + ii)
        ax.plot(ssbinse[:-1], cumsum(sshg) / sshg.sum(), color='r')
        ax.plot(ssbinse[:-1], cumsum(sshe) / sshg.sum(), color='b')
        ax.plot(ssbinse[:-1], np.abs(cumsum(sshe) - cumsum(sshg)) / sshg.sum(), color='k')
        if ii == 0:
            ax.set_xlim(x0g - ran , x0g + ran )
        else:
            ax.set_xlim(y0g - ran , y0g + ran )
        ax.set_xlabel(labels[ii] + '(V)')
        ax.set_ylabel('$F$')

    print('---------------------------')

    fig.tight_layout()
    plt.show()

def twotoneanalysis(fname):
    with SlabFile(fname) as f:
        magnitudedata = True
        phasedata = True
        rfreqpts = f['fpts-r'][:]
        rmagpts = f['mags-r'][:]
        rphasepts = f['phases-r'][:]
        qfreqpts = f['fpts-q'][:]
        qmagpts = f['mags-q'][:]
        qphasepts = f['phases-q'][:]
        tempstr = 'flux%s_actual_pts' % fname[-4]
        fluxpts = f[tempstr][:]
        # fluxpts = f['flux0_actual_pts'][:]
        fluxpts = fluxpts * 10 ** 3

        def padder(axis):
            paddedaxis = np.zeros(np.size(axis) + 1)
            diff = axis[1] - axis[0]
            for ii, elem in enumerate(axis):
                paddedaxis[ii] = elem
            paddedaxis[-1] = axis[-1] + diff
            return paddedaxis

        padrfreqpts = padder(rfreqpts[0])
        padqfreqpts = padder(qfreqpts[0])
        padfluxpts = padder(fluxpts)

        qmagptsnew = np.zeros(np.shape(qmagpts))

        refpt = qmagpts[0, 0]
        for i in range(np.size(qmagpts[:, 0])):
            qmagptsnew[i, :] = qmagpts[i, :]
            qmagptsnew[i, :] = savgol_filter(qmagptsnew[i, :], 101, 9)
            delta = np.average(qmagptsnew[i, 0:2000])
            qmagptsnew[i, :] = qmagptsnew[i, :] - delta
        for ii, current in enumerate(fluxpts):
            rphasepts[ii] = rphasepts[ii] * (np.pi / 180)
            rphasepts[ii] = np.unwrap(rphasepts[ii], np.pi - np.pi * .0001)
            mr, br = np.polyfit(rfreqpts[ii, 100:200], rphasepts[ii, 100:200], 1)
            rphasepts[ii] = (rphasepts[ii] - (rfreqpts[ii] * mr + br))
            delta = np.average(rphasepts[ii, 0:10]) - np.pi
            rphasepts[ii] = rphasepts[ii] - delta
        for ii, current in enumerate(fluxpts):
            qphasepts[ii] = qphasepts[ii] * (np.pi / 180)
            delta = np.average(qphasepts[ii, 0:20]) - np.pi
            qphasepts[ii] = qphasepts[ii] - delta
            mq, bq = np.polyfit(qfreqpts[ii, 10:-10], qphasepts[ii, 10:-10], 1)
            qphasepts[ii] = (qphasepts[ii] - (qfreqpts[ii] * mq + bq))
            #             delta = np.average(qphasepts[ii,0:20])-np.pi
            #             qphasepts[ii] = (qphasepts[ii] - delta)%(2*np.pi)
            qphasepts[ii] = savgol_filter(qphasepts[ii], 51, 3)

        start = 1
        stop = -1
        # plot mag vs freq single tone
        figure(figsize=(15, 4))
        subplot(111, title='S21 R%s' % (fname[-4]) + ' Single-Tone mag data', xlabel='Frequency (GHz)',
                ylabel='magnitude (dBm)')
        for ii, current in enumerate(fluxpts):
            print("Min peak at {}".format(rfreqpts[ii][np.argmin(rmagpts[ii])]))
            plt.plot(rfreqpts[ii], rmagpts[ii], label='Voltage (mV): %s' % (current))
        plt.show()
        # plot mag vs freq for all currents all on same graph
        figure(figsize=(15, 4))
        subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone mag data', xlabel='Frequency (GHz)',
                ylabel='magnitude (dBm)')
        for ii, current in enumerate(fluxpts):
            print("Max peak at {}".format(qfreqpts[ii][start:stop][np.argmax(qmagptsnew[ii][start:stop])]))
            plt.plot(qfreqpts[ii][start:stop], qmagptsnew[ii][start:stop], label='Voltage (mV): %s' % (current))
        plt.show()
        # plot phase vs freq for all currents all on same graph
        figure(figsize=(15, 4))
        subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone phase data', xlabel='Frequency (GHz)',
                ylabel='phase shift (rad)')
        for ii, current in enumerate(fluxpts):
            plt.plot(qfreqpts[ii][start:stop], qphasepts[ii][start:stop])
        plt.show()

        # Plot mag,phase for each current value
        #         for ii,current in enumerate(fluxpts):
        #             figure(figsize=(15,2*4))
        #             ax1=plt.subplot(211,title='S11 R1 Two-Tone phase data MAG')
        #             ax1.plot(qfreqpts[ii][start:stop],qmagptsnew[ii][start:stop],label = 'CURRENT: %s'%(str(current)))
        #             print('FLUX: ' + str(current) + ' QUBIT FREQ: ' + str(qfreqpts[ii][start:stop][np.argmax(qmagptsnew[ii][start:stop])]))
        #             ax2 = plt.subplot(212,title='S11 R1 Two-Tone phase data PHASE',xlabel='Frequency (GHz)',sharex=ax1)
        #             ax2.plot(qfreqpts[ii][start:stop],qphasepts[ii][start:stop])
        #             plt.show()

        figure(figsize=(15, 4))
        subplot(111, title='S21 Two-Tone magnitude data for Q %s' % fname[-4], xlabel='Frequency (GHz)',
                ylabel='Voltage (mV)')
        plt.pcolor(padqfreqpts, padfluxpts, qmagpts, cmap='RdBu')
        plt.show()

        figure(figsize=(15, 4))
        subplot(111, title='S21 Two-Tone phase data for Q %s' % fname[-4], xlabel='Frequency (GHz)',
                ylabel='Voltage (mV)')
        plt.pcolor(padqfreqpts, padfluxpts, qphasepts, cmap='RdBu')
        plt.show()

        ## plot qubit peaks vs current
        qmagpeaks = []
        qphasepeaks = []
        for ii, current in enumerate(fluxpts):
            qmagpeaks.append(qfreqpts[0][np.argmin(qmagpts[ii])])
            qphasepeaks.append(qfreqpts[0][np.argmin(qphasepts[ii])])

        figure(figsize=(15, 4))
        subplot(111, title='S21 Two-tone qubit peak vs current for Q%s' % fname[-4], xlabel='Voltage (mV)',
                ylabel='Freq (GHz)')
        plt.plot(fluxpts, qmagpeaks, label='mag')
        plt.plot(fluxpts, qphasepeaks, label='phase')
        plt.legend()
        plt.show()

def polynomial(p,x):
    return p[0]+p[1]*(x-p[-1])+p[2]*(x-p[-1])**2+p[3]*(x-p[-1])**3+p[4]*(x-p[-1])**4+p[5]*(x-p[-1])**5+p[6]*(x-p[-1])**6+p[7]*(x-p[-1])**7+p[8]*(x-p[-1])**8+p[9]*(x-p[-1])**9

def fitpolynomial(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata

        fitparams=[0,0,0,0,0,0,0,0,0,0,0,0]

    p1 = fitgeneral(fitdatax,fitdatay,polynomial,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    return p1

def invertedcos(p,x):
    return p[0]+p[1]/cos(p[2]*x-p[-1])

def fitinvertedcos(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata

        fitparams=[0,0,0,0,0,0,0,0,0,0,0,0]

    p1 = fitgeneral(fitdatax,fitdatay,invertedcos,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    return p1


# qubit H
def hamiltonian(Ec, Ej, d, flux, N=100):
    """
    Return the charge qubit hamiltonian as a Qobj instance.
    """
    m = np.diag(4 * Ec * (arange(-N, N + 1)) ** 2) - Ej * (cos(pi * flux) * 0.5 * (np.diag(np.ones(2 * N), -1) +
                                                                                   np.diag(np.ones(2 * N), 1)) +
                                                           1j * d * sin(pi * flux) * 0.5 * (
                                                                       np.diag(np.ones(2 * N), -1) -
                                                                       np.diag(np.ones(2 * N), 1)))
    return Qobj(m)


# JC H
def jc_hamiltonian(f_c, f_qs, g, flux, N_r=5, N_q=5):
    f_q = Qobj(diag(f_qs[0:N_q] - f_qs[0]))
    a = tensor(destroy(N_r), qeye(N_q))
    b = tensor(qeye(N_r), destroy(N_q))
    H = f_c * a.dag() * a + tensor(qeye(N_r), f_q) + g * (a.dag() + a) * (b.dag() + b)
    return Qobj(H)

def plot_energies(ng_vec, energies, ymax=(20, 3), ymin=(0,0)):
    """
    Plot energy levels as a function of bias parameter ng_vec.
    """
    fig, axes = plt.subplots(1,2, figsize=(16,6))

    for n in range(len(energies[0,:])):
        axes[0].plot(ng_vec, (energies[:,n]-energies[:,0])/(2*pi))
    axes[0].set_ylim(ymin[0], ymax[0])
    axes[0].set_xlabel(r'$flux$', fontsize=18)
    axes[0].set_ylabel(r'$E_n$', fontsize=18)

    for n in range(len(energies[0,:])):
        axes[1].plot(ng_vec, (energies[:,n]-energies[:,0])/(energies[:,1]-energies[:,0]))
    axes[1].set_ylim(ymin[1], ymax[1])
    axes[1].set_xlabel(r'$flux$', fontsize=18)
    axes[1].set_ylabel(r'$(E_n-E_0)/(E_1-E_0)$', fontsize=18)
    return fig, axes

def plot_energies_v2(ng_vec, energies, ymax=(20, 3), ymin=(0,0)):
    """
    Plot energy levels as a function of bias parameter ng_vec.
    """

    for n in range(len(energies[0,:])):
        plt.plot(ng_vec, (energies[:,n]-energies[:,0])/(2*pi),'--',linewidth=1)
    plt.ylim(ymin[0], ymax[0])
    plt.xlabel(r'$flux$', fontsize=20)
    plt.ylabel(r'$E_n$', fontsize=20)


def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]


def twotoneHJC(fname, details=True):
    with SlabFile(fname) as f:
        rfreqpts = f['fpts-r'][:]
        rmagpts = f['mags-r'][:]
        rphasepts = f['phases-r'][:]
        qfreqpts = f['fpts-q'][:]
        qmagpts = f['mags-q'][:]
        qphasepts = f['phases-q'][:]
        tempstr = 'flux%s_actual_pts' % fname[-4]
        fluxpts = f[tempstr][:]
        fluxpts = fluxpts * 10 ** 3

        def padder(axis):
            paddedaxis = np.zeros(np.size(axis) + 1)
            diff = axis[1] - axis[0]
            for ii, elem in enumerate(axis):
                paddedaxis[ii] = elem
            paddedaxis[-1] = axis[-1] + diff
            return paddedaxis

        padrfreqpts = padder(rfreqpts[0])
        padqfreqpts = padder(qfreqpts[0])
        padfluxpts = padder(fluxpts)
        qmagptsnew = np.zeros(np.shape(qmagpts))
        refpt = qmagpts[0, 0]
        for i in range(np.size(qmagpts[:, 0])):
            qmagptsnew[i, :] = qmagpts[i, :]
            qmagptsnew[i, :] = savgol_filter(qmagptsnew[i, :], 11, 3)
            delta = np.average(qmagptsnew[i, 0:2000])
            qmagptsnew[i, :] = qmagptsnew[i, :] - delta
        for ii, current in enumerate(fluxpts):
            rphasepts[ii] = rphasepts[ii] * (np.pi / 180)
            rphasepts[ii] = np.unwrap(rphasepts[ii], np.pi - np.pi * .0001)
            mr, br = np.polyfit(rfreqpts[ii, 100:200], rphasepts[ii, 100:200], 1)
            rphasepts[ii] = (rphasepts[ii] - (rfreqpts[ii] * mr + br))
            delta = np.average(rphasepts[ii, 0:10]) - np.pi
            rphasepts[ii] = rphasepts[ii] - delta
        for ii, current in enumerate(fluxpts):
            qphasepts[ii] = qphasepts[ii] * (np.pi / 180)
            delta = np.average(qphasepts[ii, 0:20]) - np.pi
            qphasepts[ii] = qphasepts[ii] - delta
            mq, bq = np.polyfit(qfreqpts[ii, 10:-10], qphasepts[ii, 10:-10], 1)
            qphasepts[ii] = (qphasepts[ii] - (qfreqpts[ii] * mq + bq))
            qphasepts[ii] = savgol_filter(qphasepts[ii], 11, 3)

        start = 1
        stop = -1
        # plot mag vs freq single tone

        if details == True:
            figure(figsize=(15, 4))
            subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone mag data', xlabel='Frequency (GHz)',
                    ylabel='magnitude (dBm)')
            for ii, current in enumerate(fluxpts):
                plt.plot(rfreqpts[ii], rmagpts[ii], label='Voltage (mV): %s' % (current))
            plt.show()

        if details == True:
            readout_freqs = []
            for ii, current in enumerate(fluxpts):
                min_freq = rfreqpts[0][np.argmin(rmagpts[ii])]
                readout_freqs.append(min_freq + 750e3)
                print("READOUT : %s" % (readout_freqs[ii]))
                print("FLUX PTS: %s" % (fluxpts[ii]))

        if details == True:
            # plot mag vs freq for all currents all on same graph
            figure(figsize=(15, 4))
            subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone mag data', xlabel='Frequency (GHz)',
                    ylabel='magnitude (dBm)')
            for ii, current in enumerate(fluxpts):
                plt.plot(qfreqpts[ii][start:stop], qmagptsnew[ii][start:stop], label='Voltage (mV): %s' % (current))
            plt.show()
            # plot phase vs freq for all currents all on same graph
            figure(figsize=(15, 4))
            subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone phase data', xlabel='Frequency (GHz)',
                    ylabel='phase shift (rad)')
            for ii, current in enumerate(fluxpts):
                plt.plot(qfreqpts[ii][start:stop], qphasepts[ii][start:stop])
            plt.show()

            figure(figsize=(15, 4))
            subplot(111, title='S21 Two-Tone magnitude data for Q %s' % fname[-4], xlabel='Frequency (GHz)',
                    ylabel='Voltage (mV)')
            plt.pcolor(padqfreqpts, padfluxpts, qmagpts, cmap='RdBu')
            plt.show()

            figure(figsize=(15, 4))
            subplot(111, title='S21 Two-Tone phase data for Q %s' % fname[-4], xlabel='Frequency (GHz)',
                    ylabel='Voltage (mV)')
            plt.pcolor(padqfreqpts, padfluxpts, qphasepts, cmap='RdBu')
            plt.show()

        resminima = []
        qubitpeaks = []
        for ii in np.arange(len(rmagpts)):
            resminima.append(rfreqpts[ii][np.argmin(rmagpts[ii])])
            qubitpeaks.append(qfreqpts[ii][np.argmax(qmagptsnew[ii])])
        if details == True:
            for ii, current in enumerate(fluxpts):
                figure(figsize=(15, 4))
                subplot(111, title='S21 R%s' % (fname[-4]) + ' Two-Tone mag data', xlabel='Frequency (GHz)',
                        ylabel='magnitude (dBm)')
                plt.plot(qfreqpts[ii][start:stop], qmagptsnew[ii][start:stop], label='Voltage (mV): %s' % (current))
                sigma = np.std(qmagptsnew[ii][start:stop])
                peaks, _ = find_peaks(qmagptsnew[ii][start:stop], height=None, prominence=sigma * 6, threshold=None, )
                print(qfreqpts[ii][peaks])
                plt.legend()
                plt.grid(True)
                plt.show()
    return [np.array(fluxpts) / 1e3, resminima, qubitpeaks]


def twotoneHJCfitter(fname, Ec, wqmax, wqmin, fc, g, flxoffset, flxscale, nb_pts=2001, flux_vec_bounds=1, details=True, RS=False):
    fluxpts, rmin, qpk = twotoneHJC(fname, details=False)
    flux = fluxpts
    Ec = Ec  # 2*pi* 0.250 #GHz
    wq_max_guess = wqmax  # GHz
    wq_min_guess = wqmin  # GHz  # 3.85
    f_c = fc  # 2*pi*6.15735 #GHz
    g = g  # 2*pi*0.074 #GHz

    Q_flux_offset = flxoffset  # -1.640328
    Q_flux_scale = flxscale  # 5.225

    Ej = (2 * pi * wq_max_guess + Ec) ** 2 / 8 / Ec
    d = (wq_min_guess / wq_max_guess) ** 2

    if details == True:
        print('Ec = 2*pi*', Ec / 2 / pi, ' GHz')
        print('Ej = 2*pi*', Ej / 2 / pi, ' GHz')
        print('d = ', d)
        print('Ec/Ej = ', Ej / Ec)

    f_qs_upper = np.asarray(hamiltonian(Ec, Ej, d, flux=0.0).eigenenergies())
    vr_energies_upper = np.asarray(jc_hamiltonian(f_c, f_qs_upper, g, flux=0.0).eigenenergies())

    if details == True:
        print('\nUpper sweet spot:')
        print('cavity freq = ', vr_energies_upper[argmin(abs(vr_energies_upper - f_c))] / 2 / pi, 'GHz')
        print('qubit freq = ', vr_energies_upper[argmin(abs(vr_energies_upper - f_c)) + 1] / 2 / pi, 'GHz')

    f_qs_lower = np.asarray(hamiltonian(Ec, Ej, d, flux=0.5).eigenenergies())
    vr_energies_lower = np.asarray(jc_hamiltonian(f_c, f_qs_lower, g, flux=0.5).eigenenergies())
    if details == True:
        print('\nLower sweet spot:')
        print('cavity freq = ', vr_energies_lower[argmin(abs(vr_energies_lower - f_c))] / 2 / pi, 'GHz')
        print('qubit freq = ', vr_energies_lower[argmin(abs(vr_energies_lower - f_c)) - 1] / 2 / pi, 'GHz')
    flux_vec = np.linspace(-flux_vec_bounds, flux_vec_bounds, nb_pts)

    # qubit energy levels
    energies = np.asarray([hamiltonian(Ec, Ej, d, flux).eigenenergies() for flux in flux_vec])

    # vacuum rabi energy levels
    vr_energies = np.asarray(
        [jc_hamiltonian(f_c, energies[ii], g, flux).eigenenergies() for (ii, flux) in enumerate(flux_vec)])

    #     ng_vec_old = Q_flux_scale*(flux_vec+Q_flux_offset_old/Q_flux_scale)
    ng_vec = Q_flux_scale * (flux_vec + Q_flux_offset / Q_flux_scale)
    energies = vr_energies

    if details == True:
        figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        for n in range(1, 3):
            omegavsV = scipy.interpolate.interp1d(ng_vec, (energies[:, n] - energies[:, 0]) / (2 * pi), kind='cubic', )
            plt.plot(ng_vec, omegavsV(ng_vec), '--', linewidth=1, label='new_fit')
        plt.ylim(fc / (2 * np.pi) - .001, fc / (2 * np.pi) + .035)
        # plt.xlim(-0.1, 0.1)
        plt.xlabel(r'$flux$', fontsize=20)
        plt.ylabel(r'$E_n$', fontsize=20)
        plt.legend()
    else:
        for n in range(1, 3):
            omegavsV = scipy.interpolate.interp1d(ng_vec, (energies[:, n] - energies[:, 0]) / (2 * pi), kind='cubic', )
    if details == True:
        print('FLX: ')
        print(flux)
        print('rmin')
        print(rmin)
    if details == True:
        if RS:
            plt.scatter(np.array(flux), np.array(rmin), marker='o', s=5)
        else:
            plt.scatter(np.array(flux), np.array(rmin) / 1e9, marker='o', s=5)
        figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        for n in range(1, 3):
            omegavsV = scipy.interpolate.interp1d(ng_vec, (energies[:, n] - energies[:, 0]) / (2 * pi), kind='cubic', )
            plt.plot(ng_vec, omegavsV(ng_vec), '--', linewidth=1, label='new_fit')
        plt.ylim(4.1, 6.0)
        # plt.xlim(-0.1, 0.1)
        plt.xlabel(r'$flux$', fontsize=20)
        plt.ylabel(r'$E_n$', fontsize=20)
        plt.legend()
    else:
        for n in range(1, 3):
            omegavsV = scipy.interpolate.interp1d(ng_vec, (energies[:, n] - energies[:, 0]) / (2 * pi), kind='cubic', )

    if details == True:
        if RS:
            plt.scatter(np.array(flux), np.array(qpk) , marker='o', s=5)
        else:
            plt.scatter(np.array(flux), np.array(qpk)/1e9, marker='o', s=5)

    energylist = (energies[:, 1] - energies[:, 0]) / (2 * np.pi)
    reslist = (energies[:, 2] - energies[:, 0]) / (2 * np.pi)
    voltlist = Q_flux_scale * (flux_vec + Q_flux_offset / Q_flux_scale)
    flxquantalist = voltlist / Q_flux_scale
    print('startpt for Voltage branch cut: ')
    startpt = np.argmin(np.abs(voltlist))
    print(startpt)

    lss = closest(argrelextrema(energylist, np.less)[0], startpt)
    uss = closest(argrelextrema(energylist, np.greater)[0], startpt)

    if lss > uss:
        temp = lss
        lss = uss
        uss = temp
    if details == True:
        print("Lower cut: ")
        print(lss)
        print("lower voltage:")
        print(voltlist[lss])
        print("lower energy:")
        print(energylist[lss])
        print("Upper cut: ")
        print(uss)
        print("Upper voltage: ")
        print(voltlist[uss])
        print('Upper energy: ')
        print(energylist[uss])

    if details == True:
        plt.figure(figsize=(12, 8))
        plt.plot(energylist[lss:uss], flxquantalist[lss:uss])
        plt.xlabel('frequency(GHz)')
        plt.ylabel('Bias (Flx Qnta)')
        plt.title('Inverse bias function')
        plt.show()

    #     energylistQ = energylist[lss:uss]
    #     voltlistQ = voltlist[lss:uss]
    #     flxquantalistQ = flxquantalist[lss:uss]

    reslistQ = reslist[lss:uss]
    energylistQ = energylist[lss:uss]
    voltlistQ = voltlist[lss:uss]
    flxquantalistQ = flxquantalist[lss:uss]

    return [reslistQ, energylistQ, voltlistQ, flxquantalistQ]


def HJCinterpolation(fname, Ec, wqmax, wqmin, fc, g, flxoffset, flxscale, details=True):
    fluxpts, rmin, qpk = twotoneHJC(fname, details=details)
    flux = fluxpts
    Ec = Ec  # 2*pi* 0.250 #GHz
    wq_max_guess = wqmax  # GHz
    wq_min_guess = wqmin  # GHz  # 3.85
    f_c = fc  # 2*pi*6.15735 #GHz
    g = g  # 2*pi*0.074 #GHz

    Q_flux_offset = flxoffset  # -1.640328
    Q_flux_scale = flxscale  # 5.225

    Ej = (2 * pi * wq_max_guess + Ec) ** 2 / 8 / Ec
    d = (wq_min_guess / wq_max_guess) ** 2

    f_qs_upper = np.asarray(hamiltonian(Ec, Ej, d, flux=0.0).eigenenergies())
    vr_energies_upper = np.asarray(jc_hamiltonian(f_c, f_qs_upper, g, flux=0.0).eigenenergies())

    f_qs_lower = np.asarray(hamiltonian(Ec, Ej, d, flux=0.5).eigenenergies())
    vr_energies_lower = np.asarray(jc_hamiltonian(f_c, f_qs_lower, g, flux=0.5).eigenenergies())
    flux_vec = np.linspace(-5, 5, 1001)

    # qubit energy levels
    energies = np.asarray([hamiltonian(Ec, Ej, d, flux).eigenenergies() for flux in flux_vec])

    # vacuum rabi energy levels
    vr_energies = np.asarray(
        [jc_hamiltonian(f_c, energies[ii], g, flux).eigenenergies() for (ii, flux) in enumerate(flux_vec)])

    #     ng_vec_old = Q_flux_scale*(flux_vec+Q_flux_offset_old/Q_flux_scale)
    ng_vec = Q_flux_scale * (flux_vec + Q_flux_offset / Q_flux_scale)
    energies = vr_energies

    for n in range(1, 2):
        omegavsV = scipy.interpolate.interp1d(ng_vec, (energies[:, n] - energies[:, 0]) / (2 * pi), kind='cubic', )

    energylist = (energies[:, 1] - energies[:, 0]) / (2 * np.pi)
    voltlist = Q_flux_scale * (flux_vec + Q_flux_offset / Q_flux_scale)
    flxquantalist = voltlist / Q_flux_scale
    startpt = np.argmin(np.abs(voltlist))

    energylistQ = energylist
    voltlistQ = voltlist
    flxquantalistQ = flxquantalist

    return [energylistQ, voltlistQ, flxquantalistQ]