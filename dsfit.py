# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 15:41:58 2011

@author: David Schuster
"""

import numpy as np
from guiqwt.pyplot import *
#from matplotlib.pyplot import *
from scipy import optimize

def selectdomain(xdata,ydata,domain):
    ind=np.searchsorted(xdata,domain)
    return xdata[ind[0]:ind[1]],ydata[ind[0]:ind[1]]

def zipsort(xdata,ydata):
    inds=np.argsort(xdata)
    return take(xdata,inds),take(ydata,inds,axis=0)
    
"""Wrapper around scipy.optimize.leastsq"""
def fitgeneral(xdata,ydata,fitfunc,fitparams,domain=None,showfit=False,showstartfit=False,label=""):
    """Uses optimize.leastsq to fit xdata ,ydata using fitfunc and adjusting fit params"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    errfunc = lambda p, x, y: (fitfunc(p,x) - y)**2 # Distance to the target function
    startparams=fitparams # Initial guess for the parameters
    bestfitparams, success = optimize.leastsq(errfunc, startparams[:], args=(fitdatax,fitdatay))
    if showfit:
        plot(fitdatax,fitdatay,'bo',label=label+" data")
        if showstartfit:
            plot(fitdatax,fitfunc(startparams,fitdatax),label=label+" startfit")
        plot(fitdatax,fitfunc(bestfitparams,fitdatax),'r-',label=label+" fit")
    return bestfitparams
    
def lorfunc(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    return p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)

def fitlor (xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    """fit lorentzian:
        returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        fitparams=[0,0,0,0]
        fitparams[0]=(fitdatay[0]+fitdatay[-1])/2.
        fitparams[1]=max(fitdatay)-min(fitdatay)
        fitparams[2]=fitdatax[np.argmax(fitdatay)]
        fitparams[3]=(max(fitdatax)-min(fitdatax))/3.

    p1 = fitgeneral(fitdatax,fitdatay,lorfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    p1[3]=abs(p1[3])
    return p1
    
def print_cavity_Q(fit):
    print fit[2]/2/fit[3]
    return fit[2]/2/fit[3]
    
def gaussfunc(p, x):
    """p[0]+p[1] exp(- (x-p[2])**2/p[3]**2/2)"""
    return p[0]+p[1] *exp(-1./2.*(x-p[2])**2/p[3]**2)

def fitgauss (xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    """fit lorentzian"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        fitparams=[0,0,0,0]
        fitparams[0]=(fitdatay[0]+fitdatay[-1])/2.
        fitparams[1]=max(fitdatay)-min(fitdatay)
        fitparams[2]=fitdatax[np.argmax(fitdatay)]
        fitparams[3]=(max(fitdatax)-min(fitdatax))/3.

    p1 = fitgeneral(fitdatax,fitdatay,gaussfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    return p1    

def hangerfunc_old(p,x):
    """p=[f0,Q,S21Min,Tmax]
       (4*(x-p[0])**2/p[0]**2 * p[1]**2+p[2]**2/(1+4*(x-p[0])**2/p[0]**2 * p[1]**2))*p[3]
    """
    return ((4.*((x-p[0])* p[1]/p[0])**2. +p[2]**2.)/(1.+4.*((x-p[0])* p[1]/p[0])**2.))*p[3]

def hangerqs_old(fitparams):
    """Converts fitparams into Qi and Qc"""
    return abs(fitparams[1]/fitparams[2]), abs(fitparams[1])/(1-abs(fitparams[2]))
    
def fithanger_old (xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    """Fit's Hanger Transmission (S21) data without taking into account asymmetry
       returns p=[f0,Q,S21Min,Tmax]
    """
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        fitparams=[0,0,0,0]
        peakloc=np.argmin(fitdatay)
        ymax=(fitdatay[0]+fitdatay[-1])/2.
        fitparams[0]=fitdatax[peakloc]
        fitparams[1]=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/3.))
        if fitdatay[peakloc]>0: fitparams[2]=(fitdatay[peakloc]/ymax)**0.5
        else: fitparams[2]=0.001
        fitparams[3]=ymax
    return fitgeneral(fitdatax,fitdatay,hangerfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)

def hangerfunc(p,x):
    """p=[f0,Qi,Qc,df,scale]"""
    f0,Qi,Qc,df,scale = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    Q0=1./(1./Qi+1./Qc)
    return scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))

def fithanger(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        peakloc=np.argmin(fitdatay)
        ymax=(fitdatay[0]+fitdatay[-1])/2.
        f0=fitdatax[peakloc]
        Q0=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/3.))
        Qi=Q0*(1.+ymax)
        Qc=Q0*Qi/ymax
        scale=ymax
        fitparams=[f0,Qi,Qc,f0/1e6,scale]
    return fitgeneral(fitdatax,fitdatay,hangerfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    


if __name__ =='__main__':
    figure(1)
    xdata=np.linspace(-15,25,1000)
    
    params=[1.,20.,5.,2.]
    ydata=gaussfunc(params,xdata)-1+2*np.random.rand(len(xdata))
    #plot(xdata,ydata-2.5+5*random.rand(xdata.__len__()),'bo')
    subplot(1,2,1)
    p1=fitgauss(xdata,ydata,showfit=True)
    subplot(1,2,2)
    p2=fitlor(xdata,ydata,showfit=True)
    #plot(xdata,lorfunc(p1,xdata),'r-')
    
    
    noise=0.00
    figure(2)
    params2=[7.8,200,200.,0.005,1.]
#    params2=[7.8,200,0.01,1.]
    xdata2=np.linspace(7,9,500)
    ydata2=hangerfunc(params2,xdata2)-noise/2.+noise*np.random.rand(len(xdata2))
    p3=fithanger(xdata2,ydata2,showfit=True,showstartfit=True)   
    print p3
    #print hangerqs(p3)
    
    show()