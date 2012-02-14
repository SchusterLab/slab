# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 15:41:58 2011

@author: David Schuster
"""

import numpy as np
import math as math
import guiqwt.pyplot as plt1 # Original version doesn't seem to work. Changed to a better coding style 
import matplotlib.pyplot as plt2
plt=plt1

from scipy import optimize

def set_fit_plotting(pkg='matplotlib'):
    global plt
    plt={'guiqwt':plt1,'matplotlib':plt2}[pkg]
    

def selectdomain(xdata,ydata,domain):
    ind=np.searchsorted(xdata,domain)
    return xdata[ind[0]:ind[1]],ydata[ind[0]:ind[1]]

def zipsort(xdata,ydata):
    inds=np.argsort(xdata)
    return plt.take(xdata,inds),plt.take(ydata,inds,axis=0)
    
"""Wraplter around scipy.optimize.leastsq"""
def fitgeneral(xdata,ydata,fitfunc,fitparams,domain=None,showfit=False,showstartfit=False,label=""):
    """Uses optimize.leastsq to fit xdata ,ydata using fitfunc and adjusting fit params"""
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
#    print 'minimum', np.min(fitdatay)
#    ymin=np.min(fitdatay)
    errfunc = lambda p, x, y: (fitfunc(p,x) - y) #there shouldn't be **2 # Distance to the target function
    startparams=fitparams # Initial guess for the parameters
    bestfitparams, success = optimize.leastsq(errfunc, startparams[:], args=(fitdatax,fitdatay))
    if showfit:
        plt.plot(fitdatax,fitdatay,'bo',label=label+" data")
        if showstartfit:
            plt.plot(fitdatax,fitfunc(startparams,fitdatax),label=label+" startfit")
        plt.plot(fitdatax,fitfunc(bestfitparams,fitdatax),'r-',label=label+" fit")
        plt.legend()
        err=math.fsum(errfunc(bestfitparams,fitdatax,fitdatay))
        print 'the best fit has an RMS of {0}'.format(err)
#    plt.t
#    plt.figtext()    
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
    return p[0]+p[1]*math.e**(-1./2.*(x-p[2])**2/p[3]**2)
    

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
    return fitgeneral(fitdatax,fitdatay,hangerfunc_old,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)

def hangerfunc(p,x):
    """p=[f0,Qi,Qc,df,scale]"""
    #print p    
    f0,Qi,Qc,df,scale = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    Q0=1./(1./Qi+1./Qc)
    return scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))

def hangerfunctilt(p,x):
    """Ge Editing  p=[f0,Qi,Qc,df,scale,slope, offset]"""
    f0, Qi, Qc, df, scale, slope, offset = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    Q0=1./(1./Qi+1./Qc)
#    return slope*x+offset+scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))
    return slope*x+offset+scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))


def fithanger(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    """Fit Hanger Transmission (S21) data taking into account asymmetry.
        fitparams = []
        returns p=[f0,Qi,Qc,df,scale]
        Uses hangerfunc. 
    """
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        peakloc=np.argmin(fitdatay)
        ymax=(fitdatay[0]+fitdatay[-1])/2.
        ymin=fitdatay[peakloc]        
        f0=fitdatax[peakloc]
        Q0=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/3.))
        Qi=Q0*(1.+ymax)
        Qc=Qi/ymax
        scale= ymax-ymin
        #slope = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]-fitdatax[0])
        #offset= ymin-slope*f0
        fitparams=[f0,Qi,Qc,0.,scale]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    return fitgeneral(fitdatax,fitdatay,hangerfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
    
def fithangertilt(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    """Fit Hanger Transmission (S21) data taking into account asymmetry.
        fitparams = []
        returns p=[f0, Q, S21Min, Tmax]
        Uses hangerfunctilt instead of hangerfunc. 
    """
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        peakloc=np.argmin(fitdatay)
        ymax=(fitdatay[0]+fitdatay[-1])/2.
        ymin=fitdatay[peakloc]        
        f0=fitdatax[peakloc]
        Q0=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/3.))
        Qi=Q0*(1.+ymax)
        Qc=Qi/ymax
        scale= ymax-ymin
        slope = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]-fitdatax[0])
        offset= ymin-slope*f0
        fitparams=[f0,Qi,Qc,0.,scale,slope, offset]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    return fitgeneral(fitdatax,fitdatay,hangerfunctilt,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
 

if __name__ =='__main__':
    plt.figure(1)
    xdata=np.linspace(-15,25,1000)
    
    params=[1.,20.,5.,2.]
    ydata=gaussfunc(params,xdata)-1+2*np.random.rand(len(xdata))
    #plot(xdata,ydata-2.5+5*random.rand(xdata.__len__()),'bo')
    plt.subplot(1,2,1)
    p1=fitgauss(xdata,ydata,showfit=True)
    plt.subplot(1,2,2)
    p2=fitlor(xdata,ydata,showfit=True)
    #plot(xdata,lorfunc(p1,xdata),'r-')
    
    
    noise=0.
    plt.figure(2)
    params2=[7.8,200,200.,0.005,1.,0.,0.]
    print '{0}\n--------------Test Parameter---------- \nf0: {1}\nQi: {2}\nQc: {3}\ndf: {4}\nScale: {5}\nSlope: {6}\nOffset:{7}\n'.format\
          ('',params2[0],params2[1],params2[2],params2[3],params2[4],params2[5],params2[6])
#    params2=[7.8,200,0.01,1.]
    xdata2=np.linspace(7,9,1000)
    ydata2=hangerfunc(params2,xdata2)-noise/2.+noise*np.random.rand(len(xdata2))
    fit=fithanger(xdata2,ydata2,showfit=True,showstartfit=True)   
    print '{0}\n--------------Best Fit---------- \nf0: {1}\nQi: {2}\nQc: {3}\ndf: {4}\nScale: {5}\nSlope: {6}\nOffset:{7}\n'.format\
          ('hanger',fit[0],fit[1],fit[2],fit[3],fit[4],fit[5],fit[6])
    #print hangerqs(p3)
    
    plt.show()