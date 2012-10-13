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
    
def argselectdomain(xdata,domain):
    ind=np.searchsorted(xdata,domain)
    return (ind[0],ind[1])

def selectdomain(xdata,ydata,domain):
    ind=np.searchsorted(xdata,domain)
    return xdata[ind[0]:ind[1]],ydata[ind[0]:ind[1]]

def zipsort(xdata,ydata):
    inds=np.argsort(xdata)
    return np.take(xdata,inds),np.take(ydata,inds,axis=0)
    
"""Wraplter around scipy.optimize.leastsq"""
def fitgeneral(xdata,ydata,fitfunc,fitparams,domain=None,showfit=False,showstartfit=False,label="",mark_data='bo',mark_fit='r-'):
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
        plt.plot(fitdatax,fitdatay,mark_data,label=label+" data")
        if showstartfit:
            plt.plot(fitdatax,fitfunc(startparams,fitdatax),label=label+" startfit")
        plt.plot(fitdatax,fitfunc(bestfitparams,fitdatax),mark_fit,label=label+" fit")
        if label!='': plt.legend()
        err=math.fsum(errfunc(bestfitparams,fitdatax,fitdatay))
        #print 'the best fit has an RMS of {0}'.format(err)
#    plt.t
#    plt.figtext()    
    return bestfitparams
    
def lorfunc(p, x):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    return p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)

def fitlor(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
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
    
def decaysin(p,x):
    
    return p[0]*np.sin(p[1]*x+p[2])*np.e**(-1*x/p[3])+p[4]

def fitdecaysin(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,label=""):
    if domain is not None:
        fitdatax,fitdatay = selectdomain(xdata,ydata,domain)
    else:
        fitdatax=xdata
        fitdatay=ydata
    if fitparams is None:    
        fitparams=[0,0,0,0,0]
        fitparams[4]=(max(fitdatay)+min(fitdatay))/2
        fitparams[0]=(max(fitdatay)-min(fitdatay))/2
        fitparams[1]=2*np.pi/((max(fitdatax)-min(fitdatax))/20)
        fitparams[2]=0.
        fitparams[3]=(max(fitdatax)-min(fitdatax))/2.

    p1 = fitgeneral(fitdatax,fitdatay,decaysin,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
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

def hangerfunc_new(p,x):
    """p=[f0,Qi,Qc,df,scale]"""
    #print p    
    f0,Qi,df,scale = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    Qc=4000.
    y=10*np.log10(scale*(Qc**2. + Qi**2.*Qc**2.*(2.*a + b)**2.)/((Qc+Qi)**2 + 4.*Qi**2.*Qc**2.*a**2.))
    return y
    
def hangerfunc_new_withQc(p,x):
    """p=[f0,Qi,Qc,df,scale]"""
    #print p    
    f0,Qi,Qc,df,scale = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    y=10*np.log10(scale*(Qc**2. + Qi**2.*Qc**2.*(2.*a + b)**2.)/((Qc+Qi)**2 + 4.*Qi**2.*Qc**2.*a**2.))
    return y

def hangerfunctilt(p,x):
    """Ge Editing  p=[f0,Qi,Qc,df,scale,slope, offset]"""
    f0, Qi, Qc, df, slope, offset = p
    a=(x-(f0+df))/(f0+df)
    b=2*df/f0
    Q0=1./(1./Qi+1./Qc)
    #y=math.exp(slope*x+offset)
    y=[math.exp(slope*i+offset) for i in x]
#    return slope*x+offset+scale*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))
    return y*(-2.*Q0*Qc + Qc**2. + Q0**2.*(1. + Qc**2.*(2.*a + b)**2.))/(Qc**2*(1. + 4.*Q0**2.*a**2.))

def fithanger_new(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,printresult=False,label="",mark_data='bo',mark_fit='r-'):
    """Fit Hanger Transmission (S21) data taking into account asymmetry.
        needs a given Qc, which is assumed to be constant.
        You need to define the Qc in hangerfunc_new()
        fitparams = []
        returns p=[f0,Qi,df,scale]
        Uses hangerfunc_new. 
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
        Q0=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/5.))
        scale= ymax-ymin
        Qi=2*Q0

        #slope = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]-fitdatax[0])
        #offset= ymin-slope*f0
        fitparams=[f0,abs(Qi),0.,scale]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    fitresult=fitgeneral(fitdatax,fitdatay,hangerfunc_new,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label, mark_data=mark_data, mark_fit=mark_fit)        
    fitresult[1]=abs(fitresult[1])
    #fitresult[2]=abs(fitresult[2])
    if printresult: print '-- Fit Result --\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}'.format(fitresult[0],fitresult[1],fitresult[2],fitresult[3])
    return fitresult
    
def fithanger_new_withQc(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,printresult=False,label="",mark_data='bo',mark_fit='r-'):
    """Fit Hanger Transmission (S21) data taking into account asymmetry.
        use the same parameters as old one 'fithanger', but a different interpretation of the fit formula
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
        Q0=abs(fitdatax[peakloc]/((max(fitdatax)-min(fitdatax))/5.))
        scale= ymax-ymin
        Qi=2*Q0
        Qc=Q0
        #slope = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]-fitdatax[0])
        #offset= ymin-slope*f0
        fitparams=[f0,abs(Qi),abs(Qc),0.,scale]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    fitresult=fitgeneral(fitdatax,fitdatay,hangerfunc_new_withQc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label, mark_data=mark_data, mark_fit=mark_fit)        
    fitresult[1]=abs(fitresult[1])
    fitresult[2]=abs(fitresult[2])
    if printresult: print '-- Fit Result --\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nscale: {4}'.format(fitresult[0],fitresult[1],fitresult[2],fitresult[3],fitresult[4])
    return fitresult

def fithanger(xdata,ydata,fitparams=None,domain=None,showfit=False,showstartfit=False,printresult=False,label="",mark_data='bo',mark_fit='r-'):
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
        scale= ymax
        Qi=Q0*(1.+ymax)
        Qc=Qi/(ymax)
        #slope = (fitdatay[-1]-fitdatay[0])/(fitdatax[-1]-fitdatax[0])
        #offset= ymin-slope*f0
        fitparams=[f0,Qi,Qc,0.,scale]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    fitresult=fitgeneral(fitdatax,fitdatay,hangerfunc,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label, mark_data=mark_data, mark_fit=mark_fit)        
    if printresult: print '-- Fit Result --\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}'.format(fitresult[0],fitresult[1],fitresult[2],fitresult[3],fitresult[4])
    return fitresult
    
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
        fitparams=[f0,Qi,Qc,0.,slope, offset]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    return fitgeneral(fitdatax,fitdatay,hangerfunctilt,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)

def polynomial(p,x):
    return p[0]+p[1]*(x-p[-1])+p[2]*(x-p[-1])**2+p[3]*(x-p[-1])**3+p[4]*(x-p[-1])**4+p[5]*(x-p[-1])**5+p[6]*(x-p[-1])**6+p[7]*(x-p[-1])**7+p[8]*(x-p[-1])**8+p[9]*(x-p[-1])**9

def fitbackground(xdata,ydata,fitparams=None, showfit=False,showstartfit=False,label=""):
    """Fit Hanger Transmission (S21) data taking into account asymmetry.
        fitparams = []
        returns p=[f0,Qi,Qc,df,scale]
        Uses hangerfunc. 
    """
    fitdatax=xdata
    fitdatay=ydata
    if fitparams is None:    
        fitparams=[-6,0,0,0,0,0,0,0,0,0,6.9e+9]
        #print '--------------Initial Parameter Set--------------\nf0: {0}\nQi: {1}\nQc: {2}\ndf: {3}\nScale: {4}\nSlope: {5}\nOffset:{6}\n'.format(f0,Qi,Qc,0.,scale,slope, offset)
    return fitgeneral(fitdatax,fitdatay,polynomial,fitparams,domain=None,showfit=showfit,showstartfit=showstartfit,label=label)
     

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise (ValueError, 
                'Input vectors y_axis and x_axis must have same length')
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    
def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]


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
    
    