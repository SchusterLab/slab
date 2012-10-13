# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:21:37 2012

@author: Phil
"""

import qutip
from matplotlib.pyplot import *
import numpy as np
import multiprocessing
import time

class cavity:
    def __init__(self,f0,Zc,levels=None):
        self.f0 = f0
        self.Zc = Zc
        self.levels=levels
        self.V0=f0*sqrt(6.62248e-34 * pi*Zc)
        if levels is not None:
            self.levels=levels
            self.a=qutip.destroy(levels)
            self.H=f0 *( self.a.dag() * self.a + 0.5 * qutip.qeye(levels))        
    
class transmon(object):
    def __init__(self,Ej,Ec,ng=1e-6,charges=5,levels=None):
        self.getH(Ej,Ec,ng,charges,levels)

    def getH(self,Ej,Ec,ng=1e-6,charges=5,levels=None):
        self.charges=charges
        self.levels=levels
        self.ng=ng
        self.Ej=Ej
        self.Ec=Ec
        self.nhat=qutip.num(2*charges+1)-charges*qutip.qeye(2*charges+1)
        a=np.ones(2*charges)
        self.Hj=qutip.Qobj(Ej/2.0*(np.diag(a,1)+np.diag(a,-1)))
        self.Hc=4.*Ec*(self.nhat-qutip.qeye(2*charges+1)*ng/2.0)**2
#        cm = np.linspace (-1*charges,charges,2*charges+1)
#        Hc = qutip.Qobj(4.*Ec*np.diag((cm-ng/2.0)**2))
        self.H= self.Hc+self.Hj
        self.basis,self.energies=self.H.eigenstates()
        if levels is not None:
            self.basis=self.basis[:levels]
            self.energies=self.energies[:levels]
            
    def charge(self,i,j):
       return (self.basis[i].dag() * self.nhat * self.basis[j]).norm()
       
    def Emn(self,m=0,n=1):
        return np.real(self.energies[n]-self.energies[m])
        
    def alpha(self,m=1):
        return self.Emn(m,m+1)-self.Emn(m-1,m)
                
if __name__=="__main__":
    Ej=30.3
    Ec=5
    t=transmon(Ej,Ec,charges=20)
    print "E_01 = %f\t<0|n|1> = %f\talpha = %f" % (t.Emn(),t.charge(0,1),t.alpha())

    nglist=np.linspace(-4,4.,100)
    levels=20
    transmonEnergies=np.transpose([transmon(Ej,Ec,ng,charges=20,levels=5).energies for ng in nglist])

    for te in transmonEnergies:
        plot(nglist,te)
    show()

    