__author__ = 'Nitrogen'

from slab import *
from slab.dsfit import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from pylab import *



def lorfuncsum(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]
    for i in range(13):
        out = out + p[3*i+1]/sqrt((1+(x-p[3*i+2])**2/p[3*i+3]**2))
    return out

def fitlorsum(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False):
    fitlorlist = [array([  7.43851682e-01,   1.69836859e+04,   1.53449010e+00,
          1.43880036e-04]),
 array([  1.02521272e+00,   1.09122336e+01,   1.62684776e+00,
          2.95913027e-03]),
 array([  1.13422115e+00,   1.76680551e+01,   1.71461304e+00,
          2.78435841e-03]),
 array([  1.02474302e+00,   1.70988586e+01,   1.80053378e+00,
          3.29249551e-03]),
 array([  9.64606963e-01,   1.16531462e+01,   1.88905033e+00,
          2.89196346e-03]),
 array([ 0.60415585,  5.20785507,  1.97570119,  0.00805273]),
 array([  7.87376253e-01,   1.02769963e+01,   2.06219280e+00,
          4.27439548e-03]),
 array([ 1.10927308,  4.10836591,  2.14661783,  0.00680325]),
 array([  1.10640040e+00,   1.61846240e+01,   2.23588758e+00,
          4.16423577e-03]),
 array([ 1.00683879,  4.2016006 ,  2.32200894,  0.01478233]),
 array([  1.29656221e+00,   1.55385189e+01,   2.40977008e+00,
          3.03861565e-03]),
 array([  1.06292665e+00,   1.17724128e+01,   2.49728347e+00,
          4.26434953e-03]),
 array([  1.05282972e+00,   8.96137293e+00,   2.58299100e+00,
          3.67959338e-03])]
    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams = zeros(40)
        fitparams[0]=min(fitdatay)
        for i in range(13):
            fitparams[3*i+1] = fitlorlist[i][1]
            fitparams[3*i+2] = fitlorlist[i][2]
            fitparams[3*i+3] = fitlorlist[i][3]

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, lorfuncsum, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label)

    for i in range(13):
        p1[3*i+3]=abs(p1[3*i+3])
    return p1


def harmfuncsum(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]
    for i in range(13):
        out = out + p[3*i+1]/sqrt(((x**2-p[3*i+2]**2)**2 + x**2*p[3*i+3]**2))
    return out

def fitharmsum(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False):
    fitlorlist = [array([  7.43851682e-01,   1.69836859e+04,   1.53449010e+00,
          1.43880036e-04]),
 array([  1.02521272e+00,   1.09122336e+01,   1.62684776e+00,
          2.95913027e-03]),
 array([  1.13422115e+00,   1.76680551e+01,   1.71461304e+00,
          2.78435841e-03]),
 array([  1.02474302e+00,   1.70988586e+01,   1.80053378e+00,
          3.29249551e-03]),
 array([  9.64606963e-01,   1.16531462e+01,   1.88905033e+00,
          2.89196346e-03]),
 array([ 0.60415585,  5.20785507,  1.97570119,  0.00805273]),
 array([  7.87376253e-01,   1.02769963e+01,   2.06219280e+00,
          4.27439548e-03]),
 array([ 1.10927308,  4.10836591,  2.14661783,  0.00680325]),
 array([  1.10640040e+00,   1.61846240e+01,   2.23588758e+00,
          4.16423577e-03]),
 array([ 1.00683879,  4.2016006 ,  2.32200894,  0.01478233]),
 array([  1.29656221e+00,   1.55385189e+01,   2.40977008e+00,
          3.03861565e-03]),


 array([  1.06292665e+00,   1.17724128e+01,   2.49728347e+00,
          4.26434953e-03]),
 array([  1.05282972e+00,   8.96137293e+00,   2.58299100e+00,
          3.67959338e-03])]
    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams = zeros(40)
        fitparams[0]=min(fitdatay)
        for i in range(13):
            fitparams[3*i+1] = fitlorlist[i][1]
            fitparams[3*i+2] = fitlorlist[i][2]
            fitparams[3*i+3] = fitlorlist[i][3]

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, harmfuncsum, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label)

    for i in range(13):
        p1[3*i+3]=abs(p1[3*i+3])
    return p1




def absharmsum(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]
    for i in range(15):
        out = out + p[3*i+1]/((x**2-p[3*i+2]**2) -1j*x*p[3*i+3])
    return abs(out)

def fitabsharmsum(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False):
    fitlorlist = array([  6.13206023e-01,   1.31867209e+04,   1.53483358e+00,
         1.28079651e-04]), array([  8.77794575e-01,   7.74695362e+00,   1.62684274e+00,
         2.66425042e-03]), array([  9.11248475e-01,   1.23247131e+01,   1.71461478e+00,
         2.70856909e-03]), array([  7.47248313e-01,   1.19577697e+01,   1.80053359e+00,
         3.27045098e-03]), array([  7.50282987e-01,   8.13310146e+00,   1.88904316e+00,
         2.81316095e-03]), array([  7.94192331e-01,   1.12826186e+01,   1.97495967e+00,
         3.08990217e-03]), array([  6.51675225e-01,   7.15481326e+00,   2.06215083e+00,
         4.11260786e-03]), array([  1.46532047e+00,   6.15388813e+00,   2.14089962e+00,
         2.28514906e-03]), array([  8.28362458e-01,   1.13036846e+01,   2.23588630e+00,
         4.11590310e-03]), array([  6.79824799e-01,   1.28232196e+01,   2.32383465e+00,
         5.05750222e-03]), array([  1.03935821e+00,   1.08598336e+01,   2.40976077e+00,
         2.91488348e-03]), array([  7.48060565e-01,   8.24271924e+00,   2.49728018e+00,
         4.25647566e-03]), array([  9.05856100e-01,   6.26171410e+00,   2.58300380e+00,
         3.34953614e-03]), array([  1.12234702e+00,   1.78038378e+01,   2.67382055e+00,
         5.00488065e-03]), array([  1.51160354e+00,   1.46875420e+01,   2.76016318e+00,
         3.78841910e-03])
    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams = zeros(46)
        fitparams[0]=min(fitdatay)
        for i in range(15):
            fitparams[3*i+1] = fitlorlist[i][1]
            fitparams[3*i+2] = fitlorlist[i][2]
            fitparams[3*i+3] = fitlorlist[i][3]

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, absharmsum, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label)

    for i in range(15):
        p1[3*i+3]=abs(p1[3*i+3])
    return p1

def absharmsum2(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]+1j*p[53]

    for i in range(13):
        out = out + (p[4*i+1] + 1j*p[4*i+4])/((x**2-p[4*i+2]**2) -1j*x*p[4*i+3])
    return abs(out)



def fitabsharmsum2(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False):

    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams = array([ -1.53353272e+00,   1.60228714e-01,   1.53475119e+00,
         9.89949024e-05,   0.00000000e+00,   6.45282184e-02,
         1.62690342e+00,   3.05579578e-03,   0.00000000e+00,
         1.02328698e-01,   1.71461396e+00,   3.08096264e-03,
         0.00000000e+00,   1.18762220e-01,   1.80050933e+00,
         3.49782611e-03,   0.00000000e+00,   7.80841129e-02,
         1.88898572e+00,   3.18484650e-03,   0.00000000e+00,
         9.10111573e-02,   1.97569196e+00,   6.82463442e-03,
         0.00000000e+00,   1.03089631e-01,   2.06168317e+00,
         4.19832430e-03,   0.00000000e+00,   1.05203028e-01,
         2.14661222e+00,   9.74279955e-03,   0.00000000e+00,
         1.76147811e-01,   2.23583668e+00,   4.42316840e-03,
         0.00000000e+00,   2.22963602e-01,   2.32189341e+00,
         1.83919246e-02,   0.00000000e+00,   1.40213580e-01,
         2.40973351e+00,   3.33481492e-03,   0.00000000e+00,
         1.67414492e-01,   2.49728905e+00,   4.95079333e-03,
         0.00000000e+00,   1.63946713e-01,   2.58309409e+00,
         6.39406043e-03,   0.00000000e+00,   0.000])

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, absharmsum2, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label)

    for i in range(13):
        p1[4*i+3]=abs(p1[4*i+3])
    return p1


def harmfunccomplexsum(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]
    for i in range(15):
        out = out + p[3*i+1]/((x**2-p[3*i+2]**2) - 1j*x*p[3*i+3])
    return out

def absharmfunccomplexsum(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0]
    for i in range(15):
        out = out + p[3*i+1]/((x**2-p[3*i+2]**2) - 1j*x*p[3*i+3])
    return abs(out)

def fitabsharmcomplexsum(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False):

    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams = array([  1.95119318e-01,   1.48711751e-01,   1.53483539e+00,
         1.52158201e-04,   7.25849447e-02,   1.62684835e+00,
         3.41254602e-03,   1.21194486e-01,   1.71463776e+00,
         3.14296448e-03,   8.68741763e-02,   1.80050879e+00,
         3.57667674e-03,   2.59936204e-02,   1.88897670e+00,
         3.43332822e-03,   1.31902880e-01,   1.97485817e+00,
         3.47424343e-03,   1.15657190e-01,   2.06144752e+00,
         4.17123577e-03,   1.05153977e-01,   2.14098839e+00,
         4.51683433e-03,   1.72537625e-01,   2.23595112e+00,
         4.54133763e-03,   1.61313254e-01,   2.32372700e+00,
         5.04861443e-03,   5.77600276e-02,   2.40967747e+00,
         3.39506542e-03,   4.03222692e-02,   2.49721818e+00,
         4.51533849e-03,   1.86954073e-02,   2.58312924e+00,
         4.10638807e-03,  -7.97484012e-02,   2.67375167e+00,
         5.09114072e-03,  -2.11385267e-01,   2.75991459e+00,
         4.65346217e-03])
    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, absharmfunccomplexsum, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label)

    for i in range(15):
        p1[3*i+3]=abs(p1[3*i+3])
    return p1


def harmfunccomplexsum2(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    out = p[0] + 1j*p[61]
    for i in range(15):
        out = out + (p[4*i+1] + 1j*p[4*i+4])/((x**2-p[4*i+2]**2) -1j*x*p[4*i+3])
    return out


def square(t, a, t0, w, sigma=0):
    if sigma>0:
        return a * (
            (t >= t0) * (t < t0 + w) +  # Normal square pulse
            (t >= t0-2*sigma) * (t < t0) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2)) +  # leading gaussian edge
            (t >= t0 + w)* (t <= t0+w+2*sigma) * np.exp(-(t - (t0 + w)) ** 2 / (2 * sigma ** 2))  # trailing edge
        )
    else:
        return a * (t >= t0) * (t < t0 + w)

def sideband(t, plus, minus, freq=0, phase=0, offset=False, offset_fit_lin=0,offset_fit_quad=0):

    return ( np.cos(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus - np.cos(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus,
             +np.sin(2 * np.pi * (freq/1.0e9 * t)+ phase*np.pi/180.0) * plus +np.sin(2 * np.pi * (freq/1.0e9 * t) + phase*np.pi/180.0) * minus)

def dc_offset(p, x):
    return p[0]*x**2 + p[1]*x**4


def fitdc_offset(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False,defaultlabel=True):

    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams=[0,0]
        fitparams[0] = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]**2-fitdatax[0]**2)
        fitparams[1] = 0

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, dc_offset, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label,defaultlabel = defaultlabel)

    return p1


def power(p, x):
    return p[0]*x**p[1]


def fitpower(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label="",debug=False,defaultlabel=True):

    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:
        fitparams=[0,0]
        fitparams[0] = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]**2-fitdatax[0]**2)
        fitparams[1] = 2

    if debug==True: print fitparams
    p1 = fitgeneral(fitdatax, fitdatay, power, fitparams, domain=None, showfit=showfit, showstartfit=showstartfit,
                    label=label,defaultlabel = defaultlabel)

    return p1