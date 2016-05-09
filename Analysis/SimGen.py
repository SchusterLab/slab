# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:58:26 2012
Generating an exponentially decaying Sin signal, with typical frequency f and 
decay constant d
@author: Ge Yang
"""

from numpy import *
import pylab as pl
import scipy as sp
import matplotlib as mpl

def ExpDecaySine(f,tau,cycle,step):
    x=arange(0,cycle,step)
    y=sin(x*2*pi*f)*exp(-x/tau)
#    pl.plot(x,y,'bo')
#    pl.show()
    return [x,y]
    

    
if __name__ == "__main__":
    s1=ExpDecaySine(f=1.,tau=5,cycle=30,step=0.001)
    s2=ExpDecaySine(f=2.,tau=5,cycle=30,step=0.001)
    s3=ExpDecaySine(f=3.,tau=5,cycle=30,step=0.001)
        
    a= s1[1]
    b= s2[1]
    c= s3[1]
    s=array([s1[0],s1[0]])
    s[1] = a+b+c
    
    f= abs(sp.fft(s[1]))
    pl.plot(s1[0],s[1])
    pl.show()
    
    pl.plot(s[1], f)
    pl.show()
    