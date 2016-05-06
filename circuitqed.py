# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:21:37 2012

@author: Phil
"""

try:
    import qutip
except:
    print "Warning no qutip!"
from matplotlib.pyplot import *
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import pi, linspace, cos, sin, ones, transpose, reshape, array, argsort, sort, \
    meshgrid, amax, amin, dot, sqrt, exp, tanh, sign, argmax
from numpy.linalg import eig


class Schrodinger:
    """Abstract class for solving the 1D and 2D Schrodinger equation 
    using finite differences and sparse matrices"""

    def __init__(self, sparse_args=None, solve=True):
        """ @param sparse_args arguments for the eigsh sparse solver
            @param solve if solve=True then it will immediately construct the Hamiltonian and solve for the eigenvalues
        """
        self.solved = False
        self.sparse_args = sparse_args
        self.solved = False
        if solve: self.solve()

    @staticmethod
    def uv(vec):
        """normalizes a vector
            @param vec vector to normalize
        """
        return vec / sqrt(dot(vec, vec))

    @staticmethod
    def Dmat(numpts, delta=1):
        """Derivative matrix
            @param numpts dimension of derivative matrix
            @param delta optional scaling of point spacing
        """
        a = 0.5 / delta * ones(numpts)
        a[0] = 0
        a[-2] = 0
        #b=-2./delta**2*ones(numpts); b[0]=0;b[-1]=0
        c = -0.5 / delta * ones(numpts)
        c[1] = 0
        c[-1] = 0
        return sparse.spdiags([a, c], [-1, 1], numpts, numpts)

    @staticmethod
    def D2mat(numpts, delta=1, periodic=True, q=0):
        """2nd Derivative matrix
            @param numpts dimension of derivative matrix
            @param delta spacing between points
            @param periodic whether derivative wraps around (default True) 
            @param q is a quasimomentum between -pi and pi, which is used if periodic=True
        """

        a = 1. / delta ** 2 * ones(numpts)
        b = -2. / delta ** 2 * ones(numpts)
        c = 1. / delta ** 2 * ones(numpts)
        #print "delta = %f" % (delta)
        if periodic:
            if q == 0:
                return sparse.spdiags([c, a, b, c, c], [-numpts + 1, -1, 0, 1, numpts - 1], numpts, numpts)
            else:
                return sparse.spdiags([exp(-(0. + 1.j) * q) * c, a, b, c, exp((0. + 1.j) * q) * c],
                                      [-numpts + 1, -1, 0, 1, numpts - 1], numpts, numpts)
        else:
            return sparse.spdiags([a, b, c], [-1, 0, 1], numpts, numpts)

    def Hamiltonian(self):
        """Abstract method used by solver"""
        return None

    def solve(self, sparse_args=None):
        """Constructs and solves for eigenvalues and eigenvectors of Hamiltonian
            @param sparse_args if present used in eigsh sparse solver"""
        Hmat = self.Hamiltonian()
        if sparse_args is not None: self.sparse_args = sparse_args
        if self.sparse_args is None:
            en, ev = eig(Hmat.todense())
        else:
            en, ev = eigsh(Hmat, **self.sparse_args)
        ev = transpose(array(ev))[argsort(en)]
        en = sort(en)
        self.en = en
        self.ev = ev
        self.solved = True
        return self.en, self.ev

    def energies(self, num_levels=-1):
        """returns eigenvalues of Hamiltonian (solves if not already solved)"""
        if not self.solved: self.solve()
        return self.en[:num_levels]

    def psis(self, num_levels=-1):
        """returns eigenvectors of Hamiltonian (solves if not already solved)"""
        if not self.solved: self.solve()
        return self.ev[:num_levels]

    def reduced_operator(self, operator, num_levels=-1):
        """Finds operator in eigenbasis of the hamiltonian truncated to num_levels
        @param operator a (sparse) matrix representing an operator in the x basis
        @num_levels number of levels to truncate Hilbert space
        """
        if not self.solved: self.solve()
        if sparse.issparse(operator):
            return array([array([dot(psi1, operator.dot(psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
                          self.psis(num_levels)])
        else:
            return array([array([dot(psi1, dot(operator, psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
                          self.psis(num_levels)])


class Schrodinger1D(Schrodinger):
    """1D Schrodinger solver class"""

    def __init__(self, x, U, KE=1, periodic=True, q=0, **kwargs):
        """@param x is array of locations
           @param U is array of potential at x
           @param KE is kinetic energy prefactor
           @param periodic True/False for boundary conditions
           @param q, if periodic=True then use exp(i q) for boundary condition phase
           @param num_levels (None)...number of levels for sparse solver or None for dense solve...sparse not working right yet...+
        """
        self.x = x
        self.U = U
        self.KE = KE
        self.periodic = periodic
        self.q = q
        Schrodinger.__init__(self, **kwargs)

    def Hamiltonian(self):
        """Constructs Hamiltonian using the potential and Kinetic energy terms"""
        Vmat = sparse.spdiags([self.U], [0], len(self.U), len(self.U))
        Kmat = -self.KE * Schrodinger.D2mat(numpts=len(self.x), delta=self.x[1] - self.x[0], periodic=self.periodic,
                                            q=self.q)
        return Kmat + Vmat

    def plot(self, num_levels=10,psi_size=None):
        """Plots potential, energies, and wavefunctions
        @param num_levels (-1 by default) number of levels to plot"""
        psize=psi_size
        if not self.solved: self.solve()
        if num_levels == -1:
            num_levels = len(self.energies()) - 1
        plot(self.x/(2*pi), self.U)
        for ind in range(num_levels):
            plot(array([self.x[0], self.x[-1]])/(2*pi), array([self.energies()[ind], self.energies()[ind]]),
                 label="$E_%d$" % ind)
            if psi_size is None:
                psize=max(abs(self.energies()[ind + 1] - self.energies()[ind]) / 2., 1)/ max(abs(self.psis()[ind]))
            plot(self.x/(2*pi), self.psis()[ind] * psize + self.energies()[ind],
                 label="$\psi_%d$" % ind)

    def plot_wavefunctions(self, num_levels=10):
        """plots wavefunctions, for internal use"""
        for ind, psi in enumerate(self.psis(num_levels)):
            plot(self.x, psi * sign(psi[1] - psi[0]), label="$\psi_%d$" % ind)


class Schrodinger2D(Schrodinger):
    def __init__(self, x, y, U, KEx=1, KEy=1, periodic_x=False, periodic_y=False, qx=0, qy=0, sparse_args=None,
                 solve=True):
        """@param x is array of locations in x direction
           @param y is array of locations in y direction
           @param U is array of potential at x
           @param KEx is kinetic energy prefactor in x direction
           @param KEy is kinetic energy prefactor in y direction
           @param periodic_x True/False for x boundary conditions
           @param periodic_y True/False for y boundary conditions
           @param qx, if periodic_x=True then use exp(i qx) for boundary condition phase
           @param qy, if periodic_y=True then use exp(i qy) for boundary condition phase
           @param num_levels (None)...number of levels for sparse solver or None for dense solve...sparse not working right yet...+
        """
        self.x = x
        self.y = y
        self.U = U
        self.KEx = KEx
        self.KEy = KEy
        self.qx = qx
        self.qy = qy
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        Schrodinger.__init__(self, sparse_args=sparse_args, solve=solve)

    def Hamiltonian(self):
        """Constructs Hamiltonian using the potential and Kinetic energy terms"""
        U = self.U.flatten()
        Vmat = sparse.spdiags([U], [0], len(U), len(U))
        Kmat = sparse.kron(-self.KEy * Schrodinger.D2mat(len(self.y), self.y[1] - self.y[0], self.periodic_y, self.qy),
                           sparse.identity(len(self.x))) + \
               sparse.kron(sparse.identity(len(self.y)),
                           -self.KEx * Schrodinger.D2mat(len(self.x), self.x[1] - self.x[0], self.periodic_x, self.qx))
        return Kmat + Vmat

    def get_2Dpsis(self, num_levels=-1):
        psis = []
        for psi in self.psis(num_levels):
            psis.append(reshape(psi, (len(self.y), len(self.x))))
        return psis

    def plot(self, num_levels=10):
        """Plots potential, energies, and wavefunctions
        @param num_levels (-1 by default) number of levels to plot"""
        if num_levels == -1:
            num_levels = len(self.energies())
        print self.energies(num_levels)-self.energies()[0]
        figure(figsize=(20, 5))
        subplot(1, num_levels + 1, 1)
        self.plot_potential()
        #xlabel('$\phi$')
        for ii, psi2D in enumerate(self.get_2Dpsis(num_levels)):
            subplot(1, num_levels + 1, ii + 2)
            #imshow(psi2D.real,extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]),interpolation="None",aspect='auto')
            imshow(psi2D.real, interpolation="None", aspect='auto')
            xlabel(ii)

    def plot_potential(self):
        """Plots potential energy landscape"""
        imshow(self.U, extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]), aspect='auto', interpolation='None')
        xlabel('x')
        ylabel('y')

class Fluxonium(Schrodinger1D):
    """Customized 1D Schrodinger solver class for flux qubits,
       allows you to specify properties using conventional circuit parameters"""

    def __init__(self, Ej, El, Ec, phi, phiL, d, periodic=False,q=0,phis=None, sparse_args=None, solve=True):
        """
        @param Ej Josephson energy
        @param El Inductance energy
        @param Ec Charging energy
        @param phi Flux bias phi/phi_0
        @param phis if specified used as basis for solving (default is -2,2 with 201 pts)
        """
        if phis is None:
            self.phis = 2 * pi * linspace(-2, 2, 201)
        else:
            self.phis = phis

        self.Ej = Ej
        self.El = El
        self.Ec = Ec
        self.phi = phi
        self.phiL = phiL
        self.d = d


        Schrodinger1D.__init__(self, x=self.phis, U=self.fluxonium_potential(), KE=4 * Ec, periodic=periodic,q=q,sparse_args=sparse_args,
                               solve=solve)

    def fluxonium_potential(self):
        """Flux qubit with a squid loop; the phi here is phi_j and not phi_l unlike in the flux_qubit"""
        return -0.5*(self.Ej * ((1+self.d)*cos(self.phis - 2. * pi * self.phi - 2. * pi * self.phiL) + (1-self.d)*cos(self.phis-2. * pi * self.phiL))) + self.El/2. * (self.phis) ** 2
        #return -0.5*(self.Ej * cos(self.phis - 2. * pi * self.phi) + self.Ej * cos(self.phis)) + self.El/2. * (self.phis-self.phiL)** 2

    def plot(self, num_levels=10,**kwargs):
        """Plot potential, energies, eigenvectors"""
        Schrodinger1D.plot(self, num_levels,**kwargs)
        xlabel('$\delta/2\pi$')
        ylabel('E/h (GHz)')
        ylim(min(self.fluxonium_potential()),min(2*self.energies(num_levels)[-1],max(self.fluxonium_potential())))
        title('(Ej=%.2f , El=%.2f , Ec=%.2f)GHz , ($\Phi_J=%.2f $,$\Phi_L=%.2f \,) \Phi_0$ ' % (self.Ej, self.El, self.Ec, self.phi,self.phiL))

    def phi_operator(self, num_levels=-1):
        """phi matrix element <0|phi|1> in eigenbasis"""
        phi_mat = sparse.spdiags([self.phis], [0], len(self.phis), len(self.phis))
        return self.reduced_operator(phi_mat, num_levels)

    def n_operator(self, num_levels=-1):
        """number matrix element <0|n|1> in eigenbasis"""
        return self.reduced_operator(Schrodinger.Dmat(len(self.phis), self.phis[1] - self.phis[0]), num_levels)


class FluxQubit(Schrodinger1D):
    """Customized 1D Schrodinger solver class for flux qubits, 
       allows you to specify properties using conventional circuit parameters"""

    def __init__(self, Ej, El, Ec, phi, periodic=False,q=0,phis=None, sparse_args=None, solve=True):
        """
        @param Ej Josephson energy
        @param El Inductance energy
        @param Ec Charging energy
        @param phi Flux bias phi/phi_0
        @param phis if specified used as basis for solving (default is -2,2 with 201 pts)
        """
        if phis is None:
            self.phis = 2 * pi * linspace(-2, 2, 201)
        else:
            self.phis = phis

        self.Ej = Ej
        self.El = El
        self.Ec = Ec
        self.phi = phi

        Schrodinger1D.__init__(self, x=self.phis, U=self.flux_qubit_potential(), KE=4 * Ec, periodic=periodic,q=q,sparse_args=sparse_args,
                               solve=solve)

    def flux_qubit_potential(self):
        """Make Flux qubit potential from circuit parameters"""
        return -self.Ej * cos(self.phis - 2. * pi * self.phi) + self.El/2. * (self.phis) ** 2


    def plot(self, num_levels=10,**kwargs):
        """Plot potential, energies, eigenvectors"""
        Schrodinger1D.plot(self, num_levels,**kwargs)
        xlabel('$\delta/2\pi$')
        ylabel('E/h (GHz)')
        ylim(min(self.flux_qubit_potential()),min(2*self.energies(num_levels)[-1],max(self.flux_qubit_potential())))
        title('Ej=%.2f GHz, El=%.2f GHz, Ec=%.2f GHz, $\Phi=%.2f \, \Phi_0$' % (self.Ej, self.El, self.Ec, self.phi))

    def phi_operator(self, num_levels=-1):
        """phi matrix element <0|phi|1> in eigenbasis"""
        phi_mat = sparse.spdiags([self.phis], [0], len(self.phis), len(self.phis))
        return self.reduced_operator(phi_mat, num_levels)

    def n_operator(self, num_levels=-1):
        """number matrix element <0|n|1> in eigenbasis"""
        return self.reduced_operator(Schrodinger.Dmat(len(self.phis), self.phis[1] - self.phis[0]), num_levels)


class ZeroPi(Schrodinger2D):
    """Customized version of Schrodinger2D for Zero-Pi qubit"""
    #good sparse_args={'k':6,'which':'LM','sigma':gnd_state_energy,'maxiter':None}
    def __init__(self, Ej, El, Ecs, Ecj, ng, phi, numxpts, numypts, numwells, sparse_args=None, solve=True):
        """
        @param Ej Josephson Energy
        @param El Inductance Energy
        @param Ecs Sum Charging Energy (Ecsum)
        @param Ecj Junction Charging Energy (Ecj)
        @param xpts number of points in heavy direction
        @param ypts number of points in light direction
        @param ng is the gate charge (-1 to 1)
        @param phi is the dimensionless flux (2*pi*Phi/Phi_0) from -pi to pi
        @param numxpts number of points in the x direction
        @param numypts number of points in the y direction
        @param numwells number of wells to simulate
        """
        self.Ej = Ej
        self.El = El
        self.Ecs = Ecs
        self.Ecj = Ecj
        self.numxpts = numxpts
        self.numypts = numypts
        self.numwells = numwells
        self.ng = ng
        self.phi = phi
        Vxy = self.make_potential()
        Schrodinger2D.__init__(self, x=self.x, y=self.y, U=Vxy, KEx=2 * Ecs, KEy=2 * Ecj, periodic_x=True,
                               qx=self.ng * pi, sparse_args=sparse_args, solve=solve)

    def make_potential(self):
        self.x = linspace(-pi / 2, 3 * pi / 2, self.numxpts)
        self.y = linspace(-2 * pi * self.numwells / 2., 2 * pi * self.numwells / 2., self.numypts)
        X, Y = meshgrid(self.x, self.y)
        Vxy = -2 * self.Ej * cos(X) * cos(Y - self.phi) + self.El * Y ** 2
        #Vxy=Ej*T**2+El*P**2
        Vxy += amax(Vxy)
        return Vxy

    def sparsify(self, numxpts=None, numypts=None, num_levels=10):
        if numxpts is not None: self.numxpts = numxpts
        if numypts is not None: self.numypts = numypts
        self.U = self.make_potential()
        self.sparse_args = {'k': num_levels, 'which': 'LM', 'sigma': self.energies()[0], 'maxiter': None}


#    def plot(self,num_levels=None):
#        #title('Ej=%.2f GHz, El=%.2f GHz, Ect=%.2f GHz, Ecp= %.2f' % (self.Ej,self.El,self.Ect,self.Ecp))
#        #xlabel('$\phi$')
#        #ylabel('$\theta$')
#        Schrodinger2D.plot(self,num_levels)

class Rydberg(Schrodinger1D):
    """Schrodinger1D class to evaluate Rydberg states of electrons on helium, by default Energy units are GHz and length units are nm's"""

    def __init__(self, x=None, KE=9212.5, UE=2407.95, Efield=0., BarrierHeight=100, BarrierWidth=0.01,
                 level_potential=False, **kwargs):
        """
            @param x are points on which to evaluate potential
            @param KE is kinetic energy scale in units of Energy*length**2 (default 9212.5 GHz/nm**2 for electrons on helium)
            @param UE is the potential energy scale which gives potential -U/z in units of Energy*length (default 2407.95 GHz*nm)
            @param Efield is the applied electric field in units of Energy/length
            @param BarrierHeight how tall to make the barrier at the surface (default 1e4)
            @param BarrierWidth how wide to make the barrier at the surface (default 0.01 nm)
            @param level_potential determines whether to level the potential at the classical turning point in the case of negative E fields)
        """
        if x is None:
            numpts = 501
            x = linspace(1e-5, 600., numpts)  #all the z points
        self.x = x
        self.UE = UE
        self.Efield = Efield
        self.BarrierHeight = BarrierHeight
        self.BarrierWidth = BarrierWidth
        self.level_potential = level_potential
        Schrodinger1D.__init__(self, x, U=self.make_potential(), KE=KE, periodic=False, **kwargs)

    def make_potential(self):
        """Makes Rydberg potential"""
        B = self.BarrierHeight
        Bw = self.BarrierWidth
        Bpts = B * (1 + tanh(-self.x / Bw)) / self.x ** 2  #Helium Outer shell repulsion
        Upts = -self.UE / self.x  #Rydberg Atom electric potential
        Epts = self.Efield * self.x  #Ionizing electric field
        Vpts = Upts + Epts + Bpts  #all the potentials combined
        if self.level_potential:
            vmaxind = argmax(Vpts[50:]) + 50
            Vpts2 = array(Vpts)
            for ii, v in enumerate(Vpts):
                if ii < vmaxind:
                    Vpts2[ii] = v
                else:
                    Vpts2[ii] = Vpts[vmaxind]
        else:
            Vpts2 = Vpts
        return Vpts2

    def plot(self, num_levels=5):
        """Plots levels/wavefunctions with customized labels/ranges for Rydberg"""
        Schrodinger1D.plot(self, num_levels)
        xlabel('Electron height, z (nm)')
        ylabel('Energy (GHz)')
        mine = amin(self.energies(num_levels))
        maxe = amax(self.energies(num_levels))
        ylim(mine - (maxe - mine) * .5, maxe + (maxe - mine) * .5)

    def dipole_moment(self, psi1=0, psi2=1):
        """Returns x matrix element between two psi's in nanometers"""
        return dot(self.ev[psi1], self.x * self.ev[psi2])


class cavity:
    def __init__(self, f0, Zc, levels=None):
        self.f0 = f0
        self.Zc = Zc
        self.levels = levels
        self.V0 = f0 * sqrt(6.62248e-34 * pi * Zc)
        if levels is not None:
            self.levels = levels
            self.a = qutip.destroy(levels)
            self.H = f0 * ( self.a.dag() * self.a + 0.5 * qutip.qeye(levels))


class transmon(object):
    def __init__(self, Ej, Ec, ng=1e-6, charges=5, levels=None):
        self.getH(Ej, Ec, ng, charges, levels)

    def getH(self, Ej, Ec, ng=1e-6, charges=5, levels=None):
        self.charges = charges
        self.levels = levels
        self.ng = ng
        self.Ej = Ej
        self.Ec = Ec
        self.nhat = qutip.num(2 * charges + 1) - charges * qutip.qeye(2 * charges + 1)
        a = np.ones(2 * charges)
        self.Hj = qutip.Qobj(Ej / 2.0 * (np.diag(a, 1) + np.diag(a, -1)))
        self.Hc = 4. * Ec * (self.nhat - qutip.qeye(2 * charges + 1) * ng / 2.0) ** 2
        #        cm = np.linspace (-1*charges,charges,2*charges+1)
        #        Hc = qutip.Qobj(4.*Ec*np.diag((cm-ng/2.0)**2))
        self.H = self.Hc + self.Hj
        self.basis, self.energies = self.H.eigenstates()
        if levels is not None:
            self.basis = self.basis[:levels]
            self.energies = self.energies[:levels]

    def charge(self, i, j):
        return (self.basis[i].dag() * self.nhat * self.basis[j]).norm()

    def Emn(self, m=0, n=1):
        return np.real(self.energies[n] - self.energies[m])

    def alpha(self, m=1):
        return self.Emn(m, m + 1) - self.Emn(m - 1, m)


def test_zeropi():
    numtpts = 201
    numphipts = 501
    numwells = 10
    Ej = 111.
    Ect = .007
    Ecp = 500.
    El = 1

    ZP = ZeroPi(Ej, El, Ect, Ecp, numtpts, numphipts, numwells,
                sparse_args={'k': 6, 'which': 'LM', 'sigma': 1265, 'maxiter': None})
    #ZP=ZeroPi(Ej,El,Ect,Ecp,numtpts,numphipts,numwells,sparse_args=None)
    ZP.plot(5)
    show()


def test_fluxqubit():
    flux_qubit = FluxQubit(Ej=17., El=1.5, Ec=1., phi=.49, solve=True)
    print flux_qubit.n_operator(5)
    flux_qubit.plot(5)
    ylim(-10, 30)
    xlim(-5, 10)
    show()


def test_transmon():
    Ej = 30.3
    Ec = 5
    t = transmon(Ej, Ec, charges=20)
    print "E_01 = %f\t<0|n|1> = %f\talpha = %f" % (t.Emn(), t.charge(0, 1), t.alpha())

    nglist = np.linspace(-4, 4., 100)
    levels = 20
    transmonEnergies = np.transpose([transmon(Ej, Ec, ng, charges=20, levels=5).energies for ng in nglist])

    for te in transmonEnergies:
        plot(nglist, te)
    show()


if __name__ == "__main__":
    #test_transmon()
    print "Testing flux qubit solver"
    test_fluxqubit()
    print "Testing Zero-Pi qubit solver"
    test_zeropi()