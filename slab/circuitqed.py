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

from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import pi,linspace,cos,sin,ones,transpose,reshape,array,argsort,sort, \
                  meshgrid,amax,amin
from numpy.linalg import eig

class Schrodinger:

    def __init__(self,sparse_args=None,solve=True):
        self.solved=False
        self.sparse_args=sparse_args
        self.solved=False
        if solve: self.solve()

    @staticmethod
    def uv(vec):
        return vec/sqrt(dot(vec,vec))

    @staticmethod
    def Dmat(numpts,delta=1):
        a=0.5/delta*ones(numpts); a[0]=0;a[-2]=0;
        #b=-2./delta**2*ones(numpts); b[0]=0;b[-1]=0;
        c=-0.5/delta*ones(numpts); c[1]=0;c[-1]=0;
        return sparse.spdiags([a,c],[-1,1],numpts,numpts)

    @staticmethod
    def D2mat(numpts,delta=1,periodic=True):
        a=1./delta**2*ones(numpts)
        b=-2./delta**2*ones(numpts)
        c=1./delta**2*ones(numpts)
        #print "delta = %f" % (delta)
        if periodic:
            return sparse.spdiags([c,a,b,c,c],[-numpts+1,-1,0,1,numpts-1],numpts,numpts)
        else:
            return sparse.spdiags([a,b,c],[-1,0,1],numpts,numpts)

    def Hamiltonian(self):
        return None

    def solve(self,sparse_args=None):
        Hmat=self.Hamiltonian()
        if sparse_args is not None: self.sparse_args=sparse_args
        if self.sparse_args is None:
            en,ev=eig(Hmat.todense())
        else:
            en,ev=eigsh(Hmat,**self.sparse_args)
        ev=transpose(array(ev))[argsort(en)]
        en=sort(en)
        self.en=en
        self.ev=ev
        self.solved=True
        return self.en,self.ev

    def energies(self,num_levels=-1):
        if not self.solved: self.solve()
        return self.en[:num_levels]

    def psis(self,num_levels=-1):
        if not self.solved: self.solve()
        return self.ev[:num_levels]

    def reduced_operator(self,operator,num_levels=-1):
        if not self.solved: self.solve()
        if sparse.issparse(operator):
            return array([array([dot(psi1,operator.dot(psi2)) for psi2 in self.psis(num_levels)]) for psi1 in self.psis(num_levels)])
        else:
            return array([array([dot(psi1,dot(operator,psi2)) for psi2 in self.psis(num_levels)]) for psi1 in self.psis(num_levels)])

class Schrodinger1D(Schrodinger):
    """x is array of locations
       U is array of potential at x
       KE is kinetic energy prefactor
       num_levels (None)...number of levels for sparse solver or None for dense solve...sparse not working right yet...+
    """
    def __init__(self,x,U,KE=1,sparse_args=None,solve=True):
        self.x=x
        self.U=U
        self.KE=KE
        Schrodinger.__init__(self,sparse_args,solve)


    def Hamiltonian(self):
        Vmat=sparse.spdiags([self.U],[0],len(self.U),len(self.U))
        Kmat=-self.KE*Schrodinger.D2mat(numpts=len(self.x),delta=self.x[1]-self.x[0])
        return Kmat+Vmat

    def plot(self,num_levels=None):
        if not self.solved: self.solve()
        if num_levels is None:
            num_levels=len(self.energies)-1
        plot(self.x,self.U)
        for ind in range(num_levels):
            plot(array([self.x[0],self.x[-1]]),array([self.energies()[ind],self.energies()[ind]]),label="$E_%d$" % ind)
            plot(self.x,self.psis()[ind]/max(abs(self.psis()[ind]))*max(abs(self.energies()[ind+1]-self.energies()[ind])/2.,1)+self.energies()[ind],label="$\psi_%d$" % ind)

    def plot_wavefunctions(self,num_levels=-1):
        for psi in self.psis(num_levels):
            plot(self.x,psi*sign(psi[1]-psi[0]),label="$\psi_%d$" % ind)

class Schrodinger2D(Schrodinger):
    def __init__(self,x,y,U,KEx=1,KEy=1,sparse_args=None,solve=True):
        """x is array of locations
           U is array of potential at x
           KE is kinetic energy prefactor
           num_levels (None)...number of levels for sparse solver or None for dense solve...sparse not working right yet...+
        """
        self.x=x
        self.y=y
        self.U=U
        self.KEx=KEx
        self.KEy=KEy
        Schrodinger.__init__(self,sparse_args=sparse_args,solve=solve)

    def Hamiltonian(self):
        U=self.U.flatten()
        Vmat=sparse.spdiags([U],[0],len(U),len(U))
        Kmat=sparse.kron(-self.KEy*Schrodinger.D2mat(len(self.y),self.y[1]-self.y[0]),sparse.identity(len(self.x))) + \
             sparse.kron(sparse.identity(len(self.y)),-self.KEx*Schrodinger.D2mat(len(self.x),self.x[1]-self.x[0]))
        return Kmat+Vmat

    def get_2Dpsis(self,num_levels=-1):
        psis=[]
        for psi in self.psis(num_levels):
            psis.append(reshape(psi,(len(self.y),len(self.x))))
        return psis

    def plot(self,num_levels=None):
        if num_levels is None:
            num_levels=len(self.energies())
        print self.energies(num_levels)
        figure(figsize=(20,5))
        subplot(1,num_levels+1,1)
        self.plot_potential()
        #xlabel('$\phi$')
        for ii,psi2D in enumerate(self.get_2Dpsis(num_levels)):
            subplot(1,num_levels+1,ii+2)
            #imshow(psi2D.real,extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]),interpolation="None",aspect='auto')
            imshow(psi2D.real,interpolation="None",aspect='auto')
            xlabel(ii)

    def plot_potential(self):
        imshow(self.U,extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]),aspect='auto',interpolation='None')

class FluxQubit(Schrodinger1D):
    def __init__(self,Ej,El,Ec,phi,phis=None,sparse_args=None,solve=True):
        if phis is None: self.phis=2*pi*linspace(-2,2,201)
        else: self.phis=phis

        self.Ej=Ej
        self.El=El
        self.Ec=Ec
        self.phi=phi

        Schrodinger1D.__init__(self,x=self.phis,U=self.flux_qubit_potential(),KE=4*Ec,sparse_args=sparse_args,solve=solve)

    def flux_qubit_potential(self):
        return -self.Ej*cos(self.phis)+self.El*(2.*pi*self.phi-self.phis)**2

    def plot(self,num_levels=None):
        Schrodinger1D.plot(self,num_levels)
        xlabel('$\delta/2\pi$')
        ylabel('E/h (GHz)')
        title('Ej=%.2f GHz, El=%.2f GHz, Ec=%.2f GHz, $\Phi=%.2f \, \Phi_0$' % (self.Ej,self.El,self.Ec,self.phi))

    def phi_operator(self,num_levels=-1):
        """phi matrix element <0|phi|1> in eigenbasis"""
        phi_mat=sparse.spdiags([self.phis],[0],len(self.phis),len(self.phis))
        return self.reduced_operator(phi_mat,num_levels)

    def n_operator(self,num_levels=-1):
        """number matrix element <0|n|1> in eigenbasis"""
        return self.reduced_operator(Schrodinger.Dmat(len(self.phis),self.phis[1]-self.phis[0]),num_levels)


class ZeroPi(Schrodinger2D):

    #good sparse_args={'k':6,'which':'LM','sigma':gnd_state_energy,'maxiter':None}
    def __init__(self,Ej,El,Ect,Ecp,numtpts,numphipts,numwells,sparse_args=None,solve=True):
        self.Ej=Ej
        self.El=El
        self.Ect=Ect
        self.Ecp=Ecp
        thetas=linspace(-pi/2,3*pi/2,numtpts)
        phis=linspace(-2*pi*numwells/2.,2*pi*numwells/2.,numphipts)
        T,P=meshgrid(thetas,phis)
        Vxy=-2*Ej*cos(T)*cos(P)+El*P**2
        #Vxy=Ej*T**2+El*P**2
        Vxy+=amax(Vxy)
        Schrodinger2D.__init__(self,x=thetas,y=phis,U=Vxy,KEx=8*Ect,KEy=8*Ecp,sparse_args=sparse_args,solve=solve)

    def plot(self,num_levels=None):
        #title('Ej=%.2f GHz, El=%.2f GHz, Ect=%.2f GHz, Ecp= %.2f' % (self.Ej,self.El,self.Ect,self.Ecp))
        #xlabel('$\phi$')
        #ylabel('$\theta$')
        Schrodinger2D.plot(self,num_levels)

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

def test_zeropi():
    numtpts=201
    numphipts=501
    numwells=10
    Ej=111.
    Ect=.007
    Ecp=500.
    El=1

    ZP=ZeroPi(Ej,El,Ect,Ecp,numtpts,numphipts,numwells,sparse_args={'k':6,'which':'LM','sigma':1265,'maxiter':None})
    #ZP=ZeroPi(Ej,El,Ect,Ecp,numtpts,numphipts,numwells,sparse_args=None)
    ZP.plot(5)
    show()

def test_fluxqubit():
    flux_qubit=FluxQubit(Ej=17.,El=1.5,Ec=1.,phi=.49,solve=True)
    flux_qubit.plot(5)
    ylim(-10,30)
    xlim(-5,10)
    show()

def test_transmon():
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


if __name__=="__main__":
    #test_transmon()
    print "Testing flux qubit solver"
    test_fluxqubit()
    print "Testing Zero-Pi qubit solver"
    test_zeropi()