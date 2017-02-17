from slab import *
from slab.dsfit import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import json
from h5py import File
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import *
from matplotlib.colors import colorConverter
import scipy.optimize as spo
import random

class idealprocesstomographycorrelationsEC():

    def __init__(self,gate_num=0,xdata_p = None,ydata_p = None):

        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
#         self.Z2 = np.array([[1,0],[0,0]])
        self.Z2 = np.array([[1,0],[0,0]])
        self.P = np.array([self.I,self.X,self.Y,self.Z])
        self.UCZ = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        self.UCX = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        self.UCY = array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]])
        self.I4 = kron(self.I,self.I)
        self.M = np.kron(self.Z2,self.I4)
        self.markerlist = ['s','o','d','^']
        self.colorlist = ['r','c','orange','b']
        self.gate_num = gate_num
        if xdata_p != None:
            self.xdata_p = xdata_p
            self.ydata_p = ydata_p
        else:
            self.xdata_p = None
            self.ydata_p = None

        if gate_num == 0:
            self.gate = self.I4
        elif gate_num == 1:
            self.gate = self.UCZ
        elif gate_num == 2:
            self.gate = self.UCX
        elif gate_num == 3:
            self.gate = self.UCY

        self.PP = []
        for i in arange(4):
            for j in arange(4):
                self.PP.append(kron(self.P[i],self.P[j]))

        self.PP = array(self.PP)




    def expand(self,psi_mm):

        psi_q = array([1,0])
        self.psi = np.kron(psi_q,psi_mm)
        return self.psi

    def U_sb_ge(self,mode,phi):

        self.U = empty([8,8],dtype= complex)

        if mode == 1:
            n1 = 0
            n2 = 1
        else:
            n1 = 1
            n2 = 0

        self.U.fill(0)
        self.U[0][0] = 1.0
        self.U[1][1] = n1
        self.U[2][2] = n2
        self.U[5][5] = n1
        self.U[6][6] = n2
        self.U[7][7] = 1
        self.U[mode][4] = -1j*exp(-1j*phi)
        self.U[4][mode] = -1j*exp(+1j*phi)
        self.U[3][7-mode] = -1j*exp(-1j*phi)
        self.U[7-mode][3] = -1j*exp(+1j*phi)

        return self.U

    def U_q_ge(self,theta,phi):

        self.U = empty([8,8],dtype= complex)
        self.U.fill(0)

        for i in arange(4):

            self.U[i][i] = cos(theta/2.0)
            self.U[i+4][i+4] = cos(theta/2.0)
            self.U[i][i+4] = -1j*sin(theta/2.0)*exp(1j*phi)
            self.U[i+4][i] = -1j*sin(theta/2.0)*exp(-1j*phi)

        return self.U

    def preparestate(self,state_num):

        U_list = [self.U_q_ge(0,0),self.U_q_ge(pi/2.0,pi/2.0),self.U_q_ge(pi/2.0,0),self.U_q_ge(pi,0)]
        psi = array([1,0,0,0,0,0,0,0])


        psi = dot(U_list[state_num%4],psi)
        psi = dot(self.U_sb_ge(1,-pi/2.0),psi)
        psi = dot(U_list[state_num/4],psi)
        psi = dot(self.U_sb_ge(2,-pi/2.0),psi)
        return psi

    def actgate(self,state_num):

        return dot(self.gate,self.preparestate(state_num)[:4])

    def finalstate(self,tom_num,state_num,phi):

        self.psi_mm = self.actgate(state_num)

        if tom_num == 0:

            # IX

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(1,phi),self.psi))

        elif tom_num ==1:

            # IY

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(1,phi),self.psi))

        elif tom_num ==2:

            # IZ

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(1,phi),self.psi)

        elif tom_num ==3:

            # XI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==4:

            # XX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==5:

            # XY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==6:

            # XZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==7:

            # YI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==8:

            # YX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==9:

            # YY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==10:

            # YZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0.0),dot(self.U_sb_ge(2,phi),self.psi))

        elif tom_num ==11:

            # ZI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(2,phi),self.psi)

        elif tom_num ==12:

            # ZX

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, pi/2.0),dot(self.U_sb_ge(1,phi),self.psi))

        elif tom_num ==13:

            # ZY

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(1,phi),self.psi))

        elif tom_num ==14:

            # ZZ

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_sb_ge(1,phi),self.psi)


    def transmon_meas(self,tom_num,state_num,phi):

        psif  = self.finalstate(tom_num,state_num,phi)
        rho = outer(psif,conjugate(psif))
        return trace(dot(self.M,rho))

    def plot_corr_vs_phase(self, tom_num, state_num):

        lab=''

        if((tom_num +1)/4 == 0):
            lab = lab + ' -- I'
        elif((tom_num + 1)/4 == 1):
            lab = lab + ' -- X'
        elif((tom_num + 1)/4 == 2):
            lab = lab + ' -- Y'
        elif((tom_num + 1)/4 == 3):
            lab = lab + ' -- Z'
        if((tom_num + 1)%4 == 0):
            lab = lab + 'I'
        elif(((tom_num + 1))%4 == 1):
            lab = lab + 'X'
        elif(((tom_num + 1))%4 == 2):
            lab = lab + 'Y'
        elif(((tom_num + 1))%4 == 3):
            lab = lab + 'Z'

        figure(figsize = (10,5))
        subplot(111, title=lab)
        phipts = linspace(-pi,pi,50)
        out = []
        for phi in phipts:
            out.append(self.transmon_meas(tom_num,state_num,phi))
        plot(phipts,out)
        xlabel('Phase (deg)')
        ylabel('Probability')
        xlim(-pi,pi)



    def plot_corr_all_vs_phase3D(self,start,stop, tom_num):



        lab=''

        if((tom_num +1)/4 == 0):
            lab = lab + ' -- I'
        elif((tom_num + 1)/4 == 1):
            lab = lab + ' -- X'
        elif((tom_num + 1)/4 == 2):
            lab = lab + ' -- Y'
        elif((tom_num + 1)/4 == 3):
            lab = lab + ' -- Z'
        if((tom_num + 1)%4 == 0):
            lab = lab + 'I'
        elif(((tom_num + 1))%4 == 1):
            lab = lab + 'X'
        elif(((tom_num + 1))%4 == 2):
            lab = lab + 'Y'
        elif(((tom_num + 1))%4 == 3):
            lab = lab + 'Z'

        fig = plt.figure(figsize = (10,8))
        ax = fig.gca(projection='3d')
        ax.set_title('Measured Correlator: ' + str(lab))

        out_list=[]
        state_num_list = arange(start,stop)
        verts = []

        for state_num in state_num_list:

            lab = str(state_num)
            if((state_num)%4 == 0):
                lab = lab + ' -- $I$'
            elif((state_num)%4 == 1):
                lab = lab + ' -- $Y_{\pi/2}$'
            elif((state_num)%4 == 2):
                lab = lab + ' -- $X_{\pi/2}$'
            elif((state_num)%4 == 3):
                lab = lab + ' -- $X_{\pi}$'
            lab = lab + '$\otimes$'
            if((state_num)/4 == 0):
                lab = lab + '$I$'
            elif(((state_num))/4 == 1):
                lab = lab + '$Y_{\pi/2}$'
            elif(((state_num))/4 == 2):
                lab = lab + '$X_{\pi/2}$'
            elif(((state_num))/4 == 3):
                lab = lab + '$X_{\pi}$'


            phipts = linspace(-pi,pi,50)
            out = []


            for phi in phipts:
                out.append(2*(self.transmon_meas(tom_num, state_num, phi + pi/2.0)-0.5))

            verts.append(list(zip(phipts,out)))

        edgecolor = ['r','b','g','y','orange','pink','gray','c','k','r','b','g','y','orange','pink','gray']
        edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']
        poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor )
        poly.set_alpha(0.5)
        ax.add_collection3d(poly, zs=state_num_list, zdir='y')
        ax.set_xlabel('Final Sideband Phase (deg)')
        ax.set_xlim3d(-pi, pi)
        ax.set_ylabel('State #')
        ax.set_ylim3d(-1, 16)
        ax.set_zlabel('Probability')
        ax.set_zlim3d(-1,1)

        plt.show()

#         legend(bbox_to_anchor = (1.45,1.35))


    def plot_corr_all_vs_phase(self,start,stop, tom_num):

        for pl in arange(2):

            lab=''

            if((tom_num +1)/4 == 0):
                lab = lab + ' -- I'
            elif((tom_num + 1)/4 == 1):
                lab = lab + ' -- X'
            elif((tom_num + 1)/4 == 2):
                lab = lab + ' -- Y'
            elif((tom_num + 1)/4 == 3):
                lab = lab + ' -- Z'
            if((tom_num + 1)%4 == 0):
                lab = lab + 'I'
            elif(((tom_num + 1))%4 == 1):
                lab = lab + 'X'
            elif(((tom_num + 1))%4 == 2):
                lab = lab + 'Y'
            elif(((tom_num + 1))%4 == 3):
                lab = lab + 'Z'

            sta  =0
            sto= 100

            if pl == 0:

                fig = plt.figure(figsize = (8,5))
                ax = fig.add_subplot(111,title= 'Expected Correlator: ' + str(lab))

            elif pl == 1:

                fig = plt.figure(figsize = (8,5))
                ax = fig.add_subplot(111,title= 'Measured Correlator: ' + str(lab))

            for state_num in arange(start,stop):

                lab = str(state_num)
                if((state_num)/4 == 0):
                    lab = lab + ' -- $I$'
                elif((state_num)/4 == 1):
                    lab = lab + ' -- $Y_{\pi/2}$'
                elif((state_num)/4 == 2):
                    lab = lab + ' -- $X_{\pi/2}$'
                elif((state_num)/4 == 3):
                    lab = lab + ' -- $X_{\pi}$'
                lab = lab + '$\otimes$'
                if((state_num)%4 == 0):
                    lab = lab + '$I$'
                elif(((state_num))%4 == 1):
                    lab = lab + '$Y_{\pi/2}$'
                elif(((state_num))%4 == 2):
                    lab = lab + '$X_{\pi/2}$'
                elif(((state_num))%4 == 3):
                    lab = lab + '$X_{\pi}$'



                phipts = linspace(-5/3.0*pi,pi*5/3.0,100)
#                 phipts = linspace(-pi,pi,100)
                out = []

                for phi in phipts:
                    out.append(2*(self.transmon_meas(tom_num, state_num, phi +pi/2.0)-0.5))


                legend(bbox_to_anchor = (1.5,1))
                xlim(-pi,pi)
                ylim(-1,1)


                if state_num < 8:
                    if pl == 0:
                        ax.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                markersize = 7.5,color=self.colorlist[state_num/4],label=lab)
                    elif pl == 1:
                        ax.plot(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5),'-',
                                marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=lab)
                    xlabel('Final Sideband Phase (deg)')
                    ylabel('Probability')
                elif state_num ==8:
                    ax = plt.gca()
                    ax2 = ax.twinx()
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = lab)
                    elif pl ==1 :
                        ax2.plot(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=lab)

                else:
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = lab)
                    elif pl == 1:
                        ax2.plot(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=lab)

                legend(bbox_to_anchor = (1.9,1))

                xlim(-300.0,300.0)
                xlim(-180.0,180.0)
                ylim(-1,1)



        for pl in arange(2):

            sta  = 0
            sto= 30

            lab=''



            if((tom_num +1)/4 == 0):
                lab = lab + ' -- I'
            elif((tom_num + 1)/4 == 1):
                lab = lab + ' -- X'
            elif((tom_num + 1)/4 == 2):
                lab = lab + ' -- Y'
            elif((tom_num + 1)/4 == 3):
                lab = lab + ' -- Z'
            if((tom_num + 1)%4 == 0):
                lab = lab + 'I'
            elif(((tom_num + 1))%4 == 1):
                lab = lab + 'X'
            elif(((tom_num + 1))%4 == 2):
                lab = lab + 'Y'
            elif(((tom_num + 1))%4 == 3):
                lab = lab + 'Z'

            fig = plt.figure(figsize = (10,8))
            fig.add_subplot(111)
            ax = fig.gca(projection='3d')
            if pl ==0:
                ax.set_title('Expected Correlator: ' + str(lab))
            else:
                ax.set_title('Measured Correlator: ' + str(lab))

            out_list=[]
            state_num_list = arange(start,stop)
            verts = []
            verts2 = []

            for state_num in state_num_list:

                lab = str(state_num)
                if((state_num)%4 == 0):
                    lab = lab + ' -- $I$'
                elif((state_num)%4 == 1):
                    lab = lab + ' -- $Y_{\pi/2}$'
                elif((state_num)%4 == 2):
                    lab = lab + ' -- $X_{\pi/2}$'
                elif((state_num)%4 == 3):
                    lab = lab + ' -- $X_{\pi}$'
                lab = lab + '$\otimes$'
                if((state_num)/4 == 0):
                    lab = lab + '$I$'
                elif(((state_num))/4 == 1):
                    lab = lab + '$Y_{\pi/2}$'
                elif(((state_num))/4 == 2):
                    lab = lab + '$X_{\pi/2}$'
                elif(((state_num))/4 == 3):
                    lab = lab + '$X_{\pi}$'


                phipts = linspace(-5/3.0*pi,5/3.0*pi,100)
                phipts = linspace(-pi,pi,100)
                out = []


                for phi in phipts:
                    out.append(2*(self.transmon_meas(tom_num, state_num, phi+pi/2.0)-0.5))

                verts.append(list(zip(phipts*180.0/pi,out)))
                verts2.append(list(zip(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5))))

            edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']

            if pl == 0:
                poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor = edgecolor,alpha=0.5)
                poly.set_alpha(0.25)
                ax.add_collection3d(poly, zs=state_num_list, zdir='y')
                z = linspace(-1,1,10)
                x = 180*ones(len(z))
                y = 1*ones(len(z))
                for i in arange(16):
                    ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
                for i in arange(16):
                    ax.plot(z*180,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
            else:
                poly = LineCollection(verts2, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor,alpha=0.5)
                poly.set_alpha(0.25)
                ax.add_collection3d(poly, zs=state_num_list, zdir='y')
                z = linspace(-1,1,10)
                x = 180*ones(len(z))
                y = 1*ones(len(z))
                for i in arange(16):
                    ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
                for i in arange(16):
                    ax.plot(z*180,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)

            ax.set_xlabel('Phase (deg)')
            ax.set_xlim3d(-180.0, 180.0)
            ax.set_ylabel('State #')
            ax.set_ylim3d(-1, 16)
            ax.set_zlabel('Probability')
            ax.set_zlim3d(-1,1)


            plt.show()


    def plot_corr_all(self,tom_num):

        figure(figsize = (10,5))
        subplot(111, title='Two-mode correlations at $\phi = 0$ for $\Psi$ = ' + str(around(self.psi_mm,2)))

        state_num_list = arange(16)
        phipts = linspace(-pi,pi,50)
        out = []
        for tom_num in tom_num_list:
            out.append(2*(self.transmon_meas(tom_num,state_num + pi/2)-0.5))
        plot(tom_num_list,out,'o-')

        xlabel('Tomography correlator #')
        ylabel('Probability')
        legend(bbox_to_anchor = (1.4,1))
        ylim(-1,1)

    def theory_matrix(self):
        m_ab=[]

        for state_num in arange(16):
            for tom_num in arange(15):
                m_ab.append(2*(self.transmon_meas(tom_num, state_num, pi/2.0)-0.5))

        m_ab = around(real(array(m_ab)),4).reshape((16,15))

        return m_ab

    def corr_meas_all(self,state_num):
        ans = []
        for tom_num in arange(15):
            ans.append(2*(self.transmon_meas(tom_num,state_num,pi/2.0) - 0.5))
        return ans

    def tomography(self,state_num):

        den_mat=0.25*self.PP[0]
        avg = self.corr_meas_all(state_num)
        for i in arange(15):
            den_mat+= 0.25*(avg[i]*self.PP[i+1])


        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121, title='Real', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(real(den_mat).flatten())))

        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Reds(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
#         plt.show()


#         fig = plt.figure(figsize=(5,5))

        ax = fig.add_subplot(122, title='Imaginary', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(imag(den_mat).flatten())))


        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Blues(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0.3,1.3,2.3,3.3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0.3,1.3,2.3,3.3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
        plt.show()

def two_qubit_gate_inversion(m_ab):


    def get_B_array():
        I = matrix([[1,0],[0,1]])
        X = matrix([[0,1],[1,0]])
        Y = matrix([[0,-1j],[1j,0]])
        Z = matrix([[1,0],[0,-1]])

        P=[]
        P.append(I)
        P.append(X)
        P.append(Y)
        P.append(Z)

        B=[]

        for P_i in P:
            for P_j in P:
                B.append(np.kron(P_i,P_j))

        return array(B)

    def get_phi_array():
        phi_1=[]
        phi_1.append(array([[1,0],[0,0]]))
        phi_1.append(array([[0.5,-0.5],[-0.5,0.5]]))
        phi_1.append(array([[0.5,0.5j],[-0.5j,0.5]]))
        phi_1.append(array([[0,0],[0,1]]))


        phi = []

        for phi_i in phi_1:
            for phi_j in phi_1:
                phi.append(np.kron(phi_i,phi_j))

        return array(phi)

    B = get_B_array()
    phi = get_phi_array()


    def den_mat(state_num):

        rho = 0.25*m_ab[state_num][0]*B[0]

        for i in arange(1,16):
            rho+= 0.25*(m_ab[state_num][i]*B[i])

        return rho


    def get_lambdas():

        lambdas = []
        for j in arange(16):
            rho = den_mat(j)
            for k in arange(16):
                lambdas.append(trace(dot(phi[k],rho)))

        return array(lambdas)

#     print around(real(get_lambdas()),1)

    def get_B_jk_mn():

        B_jk_mn = []

        for j in arange(16):
             for k in arange(16):
                row = []
                for m in arange(16):
                    for n in arange(16):
                        ans = trace(dot(dot(phi[k],B[m]),dot(phi[j],conjugate(transpose(B[n])))))
                        row.append(ans)

                B_jk_mn.append(row)

        return inv(array(B_jk_mn))


    def get_alpha():
        B_inv = get_B_jk_mn()
        lambdas = get_lambdas()

        return dot(B_inv,lambdas)


    return get_alpha().reshape(16,16)

def two_qubit_gate_maximum_likelihood(m_ab):

    # Making Chi equal to CZ
    Xguess_real = zeros((16,16))
    Xguess_imag = zeros((16,16))

    Xguess_real[0][0] = 1

    #Xguess_real[0][0] = 0.25
    #Xguess_real[3][0] = 0.25
    #Xguess_real[0][3] = 0.25
    #Xguess_real[3][3] = 0.25

    #Xguess_real[0][12] = 0.25
    #Xguess_real[3][12] = 0.25
    #Xguess_real[0][15] = -0.25
    #Xguess_real[3][15] = -0.25

    #Xguess_real[12][0] = 0.25
    #Xguess_real[15][0] = -0.25
    #Xguess_real[12][3] = 0.25
    #Xguess_real[15][3] = -0.25

    #Xguess_real[12][12] = 0.25
    #Xguess_real[12][15] = -0.25
    #Xguess_real[15][12] = -0.25
    #Xguess_real[15][15] = 0.25

    Xguess = vstack((Xguess_real,Xguess_imag))




    # Set boundary for allocation of each Chi to be between -1 and 1
    bnds = []
    for ii in range(0,512):
        bnds.append((long(-1),long(1)))

    def get_B_array():
        I = matrix([[1,0],[0,1]])
        X = matrix([[0,1],[1,0]])
        Y = matrix([[0,-1j],[1j,0]])
        Z = matrix([[1,0],[0,-1]])

        P=[]
        P.append(I)
        P.append(X)
        P.append(Y)
        P.append(Z)

        B=[]

        for P_i in P:
            for P_j in P:
                B.append(np.kron(P_i,P_j))

        return B

    def convert_array_to_matrix(x_array):
        x_real = x_array[0:256]
        x_imag = np.multiply(x_array[256:512],1j)
        x = x_real+x_imag
        x = np.reshape(x,(16,16))
        return x



    # states
    def get_phi_array():
        phi_1=[]
        phi_1.append(matrix([[1,0],[0,0]]))
        phi_1.append(matrix([[0.5,-0.5],[-0.5,0.5]]))
        phi_1.append(matrix([[0.5,0.5j],[-0.5j,0.5]]))
        phi_1.append(matrix([[0,0],[0,1]]))


        phi = []

        for phi_i in phi_1:
            for phi_j in phi_1:
                phi.append(matrix(np.kron(phi_i,phi_j)))

        return phi

    # measurements
    def get_M_array():
        X = matrix([[0,1],[1,0]])
        Y = matrix([[0,-1j],[1j,0]])

        M_1=[]
        M_1.append(matrix([[1,0],[0,1]]))
        M_1.append(X)
        M_1.append(Y)
        M_1.append(matrix([[1,0],[0,-1]]))

        M = []

        for M_i in M_1:
            for M_j in M_1:
                M.append(matrix(np.kron(M_i,M_j)))

        return M

    def get_MMT_array(B,phi,M):
        MMT = []

        for a in arange(0,16):
            for b in arange(0,16):
                for m in arange(0,16):
                    for n in arange(0,16):
                        matrix_multiply_trace = np.trace(np.dot(M[b],np.dot(B[m],np.dot(phi[a],B[n].getH()))))
                        MMT.append(matrix_multiply_trace)


        return np.array(MMT).reshape((16,16,16,16))

    B=get_B_array()
    phi=get_phi_array()
    M=get_M_array()
    MMT = get_MMT_array(B,phi,M)


    def get_BndBm_array(B):
        BndBm = []
        for m in arange(0,16):
            BndBm_n = []
            for n in arange(0,16):
                BndBm_n.append(np.dot(B[n].getH(),B[m]))
            BndBm.append(BndBm_n)

        return BndBm

    BndBm = get_BndBm_array(B)


    def get_cons_mat(x_array):
        x = convert_array_to_matrix(x_array)
        B = get_B_array()

        cons_mat = matrix(zeros((4,4)))
        for m in arange(0,16):
            for n in arange(0,16):
                XmnBndBm = np.multiply(x[m][n],BndBm[m][n])
                cons_mat = np.add(cons_mat,XmnBndBm)

        return cons_mat


    cons_list = []
    cons_list.append({'type': 'eq', 'fun': lambda x: np.sum(np.abs(get_cons_mat(x)-np.identity(4)))})
    cons = tuple(cons_list)

    # error function
    def error_function(x_array):
        x = convert_array_to_matrix(x_array)



        err = 0
        for a in arange(0,16):
            for b in arange(0,16):

                sum_ab=0
                for m in arange(0,16):
                    for n in arange(0,16):
                        sum_ab+=x[m][n]*MMT[a][b][m][n]

                err += np.absolute(m_ab[a][b]-sum_ab)**2

        return err



    def callbackF(Xi):
        global Nfeval
        Nfeval += 1

        cost = error_function(Xi)
        print str(Nfeval)+ ': cost function: ' + str(cost)

        global Xeval
        Xeval.append(Xi)

        global Cost
        Cost.append(cost)



    min_result = spo.minimize(error_function, Xguess, method='SLSQP', options={'disp': True,'maxiter':120}
                              , bounds=bnds,constraints=cons,callback=callbackF)

    # optimize allocation and Sharpe ratio
    optimized_fun = min_result.fun
    optimized_x = min_result.x

    #print optimized_x
    return convert_array_to_matrix(optimized_x)

def simulate_process_tomography_experiment(gate):
    # states

    def get_phi_array():
        phi_1=[]
        phi_1.append(matrix([[1,0],[0,0]]))
        phi_1.append(matrix([[0.5,-0.5],[-0.5,0.5]]))
        phi_1.append(matrix([[0.5,0.5j],[-0.5j,0.5]]))
        phi_1.append(matrix([[0,0],[0,-1]]))

        phi = []

        for phi_i in phi_1:
            for phi_j in phi_1:
                phi.append(matrix(np.kron(phi_i,phi_j)))

        return phi

    # measurements
    def get_M_array():
        X = matrix([[0,1],[1,0]])
        Y = matrix([[0,-1j],[1j,0]])

        M_1=[]
        M_1.append(matrix([[1,0],[0,1]]))
        M_1.append(X)
        M_1.append(Y)
        M_1.append(matrix([[1,0],[0,-1]]))

        M = []

        for M_i in M_1:
            for M_j in M_1:
                M.append(matrix(np.kron(M_i,M_j)))

        return M

    phi = get_phi_array()
    M = get_M_array()

    m_ab = []

    for a in range(0,16):
        for b in range(0,16):
            den_mat = np.dot(np.dot(gate,phi[a]),gate.getH())
            measure = np.trace(np.dot(den_mat,M[b])) #+ random.uniform(0.00,0.1)

            m_ab.append(measure)

    m_ab_real = np.array(np.real(m_ab)).reshape((16, 16))

    return m_ab_real

gate = matrix(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))

m_ab = simulate_process_tomography_experiment(gate)
#optimized_x_array = two_qubit_gate_maximum_likelihood(m_ab)
class idealprocesstomographycorrelations():

    def __init__(self,gate_num=0,xdata_p = None,ydata_p = None,sweep_meas = False, sweep_final_sb = False,phi_cnot=0,phi_final_sb = 0,phase_sweep=True):

        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
#         self.Z2 = np.array([[1,0],[0,0]])
        self.Z2 = np.array([[1,0],[0,0]])
        self.P = np.array([self.I,self.X,self.Y,self.Z])
        self.UCZ = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        self.UCX = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        self.UCY = array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]])
        self.I4 = kron(self.I,self.I)
        self.M = np.kron(self.Z2,self.I4)
        self.markerlist = ['s','o','d','^']
        self.colorlist = ['r','c','orange','b']
        self.gate_num = gate_num
        self.sweep_meas = sweep_meas
        self.sweep_final_sb = sweep_final_sb
        self.phi_cnot = phi_cnot
        self.phi_final_sb = phi_final_sb
        self.phase_sweep = phase_sweep
        if xdata_p != None:
            self.xdata_p = xdata_p
            self.ydata_p = ydata_p
        else:
            self.xdata_p = None
            self.ydata_p = None

        if gate_num == 0:
            self.gate = self.I4
        elif gate_num == 1:
            self.gate = self.UCZ
        elif gate_num == 2:
            self.gate = self.UCX
        elif gate_num == 3:
            self.gate = self.UCY

        self.PP = []
        for i in arange(4):
            for j in arange(4):
                self.PP.append(kron(self.P[i],self.P[j]))

        self.PP = array(self.PP)
        self.find_fit = []

    def cphase(self,phi):

        return array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1*exp(1j*phi)]])

    def cnot(self,phi_ef1=0,phi_ef2=0,phi_cnot=0):

        return array([[1,0,0,0],[0,1,0,0],[0,0,0,exp(1j*(phi_ef1+phi_cnot))],[0,0,exp(-1j*(phi_ef2+phi_cnot)),0]])

    def expand(self,psi_mm):

        psi_q = array([1,0])
        self.psi = np.kron(psi_q,psi_mm)
        return self.psi
    # Unitary for a ge pi sideband pulses  # state = |q,2,1> (no shift)
    def U_sb_ge(self,mode,phi):

        self.U = empty([8,8],dtype= complex)

        if mode == 1:
            n1 = 0
            n2 = 1
        else:
            n1 = 1
            n2 = 0

        self.U.fill(0)
        self.U[0][0] = 1.0
        self.U[1][1] = n1  # g01
        self.U[2][2] = n2  # g10

        # these are wromg? |e01>,|e10> -- |g02> or |g11>

        # self.U[5][5] = n1
        # self.U[6][6] = n2   should always be 0


        self.U[7][7] = 1
        self.U[mode][4] = -1j*exp(-1j*phi)
        self.U[4][mode] = -1j*exp(+1j*phi)

        # Correct
        self.U[3][7-mode] = -1j*exp(-1j*phi)
        self.U[7-mode][3] = -1j*exp(+1j*phi)

        return self.U
    # Unitary for ge qubit pulses  # state = |q,2,1> (no shift)
    def U_q_ge(self,theta,phi):

        self.U = empty([8,8],dtype= complex)
        self.U.fill(0)

        for i in arange(4):

            self.U[i][i] = cos(theta/2.0)
            self.U[i+4][i+4] = cos(theta/2.0)
            self.U[i][i+4] = -1j*sin(theta/2.0)*exp(1j*phi)
            self.U[i+4][i] = -1j*sin(theta/2.0)*exp(-1j*phi)

        return self.U

    def preparestate(self,state_num):

        U_list = [self.U_q_ge(0,0),self.U_q_ge(pi/2.0,pi/2.0),self.U_q_ge(pi/2.0,0),self.U_q_ge(pi,0)]
        psi = array([1,0,0,0,0,0,0,0])

        # order does not matter in absence of dispersive shift (Correct order per convention)

        psi = dot(U_list[state_num/4],psi)
        psi = dot(self.U_sb_ge(2,-pi/2.0),psi)
        psi = dot(U_list[state_num%4],psi)
        psi = dot(self.U_sb_ge(1,-pi/2.0),psi)

        return psi

    def actgate(self,state_num,phi):
        if self.sweep_final_sb:
            add_cphase = 0
        else:
            add_cphase = phi
        if self.gate_num == 0:
            return dot(self.cphase(add_cphase+pi),self.preparestate(state_num)[:4])
        elif self.gate_num == 1:
            return  dot(self.cphase(add_cphase),self.preparestate(state_num)[:4])
        elif self.gate_num == 2:
            return  dot(self.cnot(phi_ef1=add_cphase),self.preparestate(state_num)[:4])

    def finalstate(self,tom_num,state_num,phi):


        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse
        # within the gate (in define gate)

        # State convention : |2, 1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        # CNOT(id2, id1) = CX
        # CZ(id2, id1) = CZ



        self.psi_mm = self.actgate(state_num,phi)

        if tom_num == 0:

            # IX

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==1:

            # IY

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==2:

            # IZ

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(1,pi/2.0),self.psi)

        elif tom_num ==3:

            # XI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==4:

            # XX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==5:

            # XY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==6:

            # XZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==7:

            # YI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==8:

            # YX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==9:

            # YY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==10:

            # YZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==11:

            # ZI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(2,pi/2.0),self.psi)

        elif tom_num ==12:

            # ZX

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, pi/2.0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==13:

            # ZY

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==14:

            # ZZ

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_sb_ge(1,pi/2.0),self.psi)

    def finalstate_new(self,tom_num,state_num,phi,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):
        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse
        # within the gate (in define gate)

        # State convention : |2, 1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        # CNOT(id2, id1) = CX
        # CZ(id2, id1) = CZ

        self.psi_mm = self.actgate(state_num,phi)
        if self.sweep_meas:
            phi_ef2 = -phi + phi_add_ef_sb_cnot
        else:
            phi_ef2 = 0 + phi_add_ef_sb_cnot

        if self.sweep_final_sb:
            phi_final_sb = phi + self.phi_final_sb
        else:
            phi_final_sb = self.phi_final_sb

        if tom_num == 0:
            # IX
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==1:
            # IY
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==2:
            # IZ
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi)

        elif tom_num ==3:
            # XI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==4:
            # XX
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot = phi_add_ef_q_cnot),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))
            print self.phi_cnot

        elif tom_num ==5:
            # XY
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot= phi_add_ef_q_cnot-pi/2.0),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==6:
            # XZ
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==7:
            # YI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==8:
            # YX
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot = phi_add_ef_q_cnot),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==9:
            # YY

            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot= phi_add_ef_q_cnot-pi/2.0),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(2,pi/2.0+phi_final_sb),self.psi))

        elif tom_num ==10:
            # YZ
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==11:
            # ZI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi)

        elif tom_num ==12:
            # ZX
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, pi/2.0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==13:
            # ZY
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==14:
            # ZZ
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2),self.psi_mm))
            return dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi)
    # Calculating the transmon expectation value
    def transmon_meas(self,tom_num,state_num,phi=0,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):

        psif  = self.finalstate_new(tom_num,state_num,phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)
        rho = outer(psif,conjugate(psif))
        return trace(dot(self.M,rho))

    def corr_meas_all(self,state_num):
        ans = []
        for tom_num in arange(15):
            ans.append(2*(self.transmon_meas(tom_num,state_num) - 0.5))
        return ans
    # 16,15 matrix storing the outcome vs state #, tom #
    def theory_matrix(self):
        m_ab=[]

        for state_num in arange(16):
            for tom_num in arange(15):
                m_ab.append(2*(self.transmon_meas(tom_num, state_num)-0.5))

        m_ab = around(real(array(m_ab)),4).reshape((16,15))

        return m_ab

    def expt_matrix(self,find_optimal_phase = False,phaselist=None,corrlist=None):

        m_ab=[]

        if self.xdata_p == None:
            print "No experimental data provided"
        else:
            if self.phase_sweep:
                if find_optimal_phase:
                    find_fit_list = self.find_fit_matrix()
                    for state_num in arange(16):
                        for tom_num in arange(15):
                            xdata = self.xdata_p[tom_num][state_num]
                            ydata = self.ydata_p[tom_num][state_num]
                            find_phase = find_fit_list[state_num][tom_num] #'max' or 'min'
                            if find_phase == "mean":
                                m_ab.append(-2*(mean(self.ydata_p[tom_num][state_num])-0.5))
                            else:
                                expected_period = 360.
                                x_at_extremum = sin_phase(xdata,-2*(ydata-0.5),expected_period,find_phase,showfit = False)
                                if x_at_extremum > 300 or x_at_extremum < -300:
                                    print "check fits"
                                    m_ab.append(-2*(mean(self.ydata_p[tom_num][state_num])-0.5))
                                else:
                                    m_ab.append(-2*(ydata[argmin(abs(xdata-x_at_extremum))]-0.5))
                else:

                    for state_num in arange(16):
                        for tom_num in arange(15):
                            expt_num = 16*tom_num + state_num
                            offset = 0.0
                            if corrlist == None or phaselist == None:
                                pass
                            else:
                                if expt_num in corrlist:
                                    offset = phaselist[argmin(abs(corrlist-expt_num))]

                            xdata = self.xdata_p[tom_num][state_num]
                            ydata = self.ydata_p[tom_num][state_num]
                            m_ab.append(-2*(ydata[argmin(abs(xdata-offset))]-0.5))
                            print "Phase offset for (s,t) = (%s,%s) = %s degrees"%(state_num,tom_num,offset)
            else:

                for state_num in arange(16):
                    for tom_num in arange(15):
                        # if tom_num in array([4,5,8,9]) and state_num in array([5,6,9,10]):
                        #     if self.gate_num == 1:
                        #         m_ab.append(2*((self.ydata_p[16*tom_num + state_num])-0.5))
                        #     else:
                        #         m_ab.append(-2*((self.ydata_p[16*tom_num + state_num])-0.5))
                        # else:
                        #     m_ab.append(-2*((self.ydata_p[16*tom_num + state_num])-0.5))


                        m_ab.append(-2*((self.ydata_p[16*tom_num + state_num])-0.5))

            m_ab = around(real(array(m_ab)),4).reshape((16,15))
            return m_ab

    def contrast_matrix(self):
        c_ab=[]
        phi_pts = linspace(-pi,pi,50)
        for state_num in arange(16):
            for tom_num in arange(15):
                corr_pts = array([2*(self.transmon_meas(tom_num, state_num,phi)-0.5) for phi in phi_pts])
                # fitdata = fitsin(phi_pts,corr_pts,showfit=False)
                c_ab.append(abs(max(corr_pts)-min(corr_pts)))

        c_ab = array(c_ab).reshape((16,15))

        return c_ab

    def find_fit_matrix(self):

        find_fit = []
        t_matrix = self.theory_matrix()
        c_matrix = self.contrast_matrix()
        phi_pts = linspace(-pi,pi,50)
        for state_num in arange(16):
            for tom_num in arange(15):
                corr_pts = array([2*(self.transmon_meas(tom_num, state_num,phi)-0.5) for phi in phi_pts])
                if c_matrix[state_num][tom_num] < 0.05:
                    find_fit.append('mean')
                else:
                    if abs(max(corr_pts)-t_matrix[state_num][tom_num]) < 0.05:
                        find_fit.append('max')
                    elif abs(min(corr_pts)-t_matrix[state_num][tom_num]) < 0.05:
                        find_fit.append('min')
                    else: find_fit.append('int')

        return array(find_fit).reshape((16,15))

    def plot_contrast_matrix(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.contrast_matrix(),cmap = cm.OrRd,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('State #')
        xlabel('Tomography Correlator')

        clim(0,2)
        cb = colorbar()
        cb.set_ticks([0,1,2])

    def plot_theory_matrix(self):
        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
        matplotlib.rc('font', **font)
        plt.figure(figsize=(10,8))
        plt.imshow(self.theory_matrix(),cmap = cm.PuOr,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label2(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('Input state')
        xlabel('Tomography Correlator')

        clim(-1,1)
        cb = colorbar()
        cb.set_ticks([-1,0,1])

    def plot_expt_matrix(self,find_optimal_phase = False,phaselist=None,corrlist=None):
        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
        matplotlib.rc('font', **font)

        plt.figure(figsize=(10,8))
        plt.imshow(self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist=phaselist,corrlist=corrlist),cmap = cm.PuOr,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label2(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('Input state')
        xlabel('Tomography Correlator')

        clim(-1,1)
        cb = colorbar()
        cb.set_ticks([-1,0,1])

    def tom_label(self,tom_num):
        lab=''
        if((tom_num +1)/4 == 0):
            lab = lab + 'I'
        elif((tom_num + 1)/4 == 1):
            lab = lab + 'X'
        elif((tom_num + 1)/4 == 2):
            lab = lab + 'Y'
        elif((tom_num + 1)/4 == 3):
            lab = lab + 'Z'
        if((tom_num + 1)%4 == 0):
            lab = lab + 'I'
        elif(((tom_num + 1))%4 == 1):
            lab = lab + 'X'
        elif(((tom_num + 1))%4 == 2):
            lab = lab + 'Y'
        elif(((tom_num + 1))%4 == 3):
            lab = lab + 'Z'

        return lab

    def state_label(self,state_num):
        lab = str(state_num)
        if((state_num)/4 == 0):
            lab = lab + ' -- $I$'
        elif((state_num)/4 == 1):
            lab = lab + ' -- $Y_{\pi/2}$'
        elif((state_num)/4 == 2):
            lab = lab + ' -- $X_{\pi/2}$'
        elif((state_num)/4 == 3):
            lab = lab + ' -- $X_{\pi}$'
        lab = lab + '$\otimes$'
        if((state_num)%4 == 0):
            lab = lab + '$I$'
        elif(((state_num))%4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif(((state_num))%4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif(((state_num))%4 == 3):
            lab = lab + '$X_{\pi}$'

        return lab

    def state_label2(self,state_num):
        lab = ''
        if((state_num)/4 == 0):
            lab = lab + '$I$'
        elif((state_num)/4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif((state_num)/4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif((state_num)/4 == 3):
            lab = lab + '$X_{\pi}$'
        lab = lab + '$\otimes$'
        if((state_num)%4 == 0):
            lab = lab + '$I$'
        elif(((state_num))%4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif(((state_num))%4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif(((state_num))%4 == 3):
            lab = lab + '$X_{\pi}$'

        return lab

    def plot_corr_vs_phase(self, tom_num, state_num,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):
        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
        matplotlib.rc('font', **font)
        figure(figsize = (10,5))
        subplot(111, title='State = ' + self.tom_label(tom_num)+', Tom. corr = ' + self.state_label(state_num))
        phipts = linspace(-pi,pi,30)
        out = []
        for phi in phipts:
            out.append(self.transmon_meas(tom_num,state_num,phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot))
        plot(phipts,out)
        xlabel('Phase (deg)')
        ylabel('Probability')
        xlim(-pi,pi)

    def plot_corr_vs_two_phases(self, tom_num, state_num,add_ef_sb_cnot=0):

        figure(figsize = (10,5))
        subplot(111, title='tom = ' + self.tom_label(tom_num)+', state = ' + self.state_label(state_num))

        phipts_1 = linspace(-pi,pi,30)
        phipts_2 = linspace(-pi,pi,30)

        data = empty([30,30])
        for ii,phi_1 in enumerate(phipts_1):
            for jj,phi_2 in enumerate(phipts_2):
                data[ii][jj] = 2*(self.transmon_meas(tom_num, state_num, phi_1,add_ef_sb_cnot,phi_2)-0.5)

        pcolormesh((180.0/pi)*phipts_1,(180.0)/pi*phipts_2,data.T,cmap=cm.RdBu)
        clim(-1,1)
        colorbar()
        xlabel('$\delta\phi$ Added (Subtracted) from CZ (CNOT) Phase (deg)')
        ylabel('CNOT ef qubit phase (deg)')
    #3d plots of correlators vs phase, state
    def plot_corr_all_vs_phase3D(self,start,stop, tom_num,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):


        fig = plt.figure(figsize = (10,8))
        ax = fig.gca(projection='3d')
        ax.set_title('Expected Correlator: -- ' + self.tom_label(tom_num))

        out_list=[]
        state_num_list = arange(start,stop)
        verts = []

        for state_num in state_num_list:

            phipts = linspace(-pi,pi,50)
            out = []


            for phi in phipts:
                out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

            verts.append(list(zip(phipts,out)))

        # edgecolor = ['r','b','g','y','orange','pink','gray','c','k','r','b','g','y','orange','pink','gray']
        edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']
        poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor )
        poly.set_alpha(0.5)
        ax.add_collection3d(poly, zs=state_num_list, zdir='y')
        ax.set_xlabel('Final Sideband Phase (deg)')
        ax.set_xlim3d(-pi, pi)
        ax.set_ylabel('State #')
        ax.set_ylim3d(-1, 16)
        ax.set_zlabel('Probability')
        ax.set_zlim3d(-1,1)

        plt.show()

    # 2d and 3d plots of correlators vs ef sideband phase (first ef sideband of CZ gate, and second ef sideband of CNOT gate (XX,XY,YX,YY) )
    def plot_corr_all_vs_phase(self,start,stop, tom_num,xlimit=180,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0,sta = 0,sto =50):

        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
        matplotlib.rc('font', **font)
        state_num_list = arange(start,stop)
        N=2
        if self.xdata_p == None:
            N = 1

        for pl in arange(N):

            if pl == 0:
                if N==1:
                    fig = plt.figure(figsize = (10,5))
                    ax = fig.add_subplot(111,title= 'Expected Correlator: -- ' + self.tom_label(tom_num))
                else:
                    fig = plt.figure(figsize = (10,10))
                    ax = fig.add_subplot(211,title= 'Expected Correlator: -- ' + self.tom_label(tom_num))

            elif pl == 1:

                ax = fig.add_subplot(212,title= 'Measured Correlator: -- ' +  self.tom_label(tom_num))

            for state_num in state_num_list:

                phipts = linspace(-xlimit*pi/180.0,xlimit*pi/180.0,50)
                out = []

                for phi in phipts:
                    out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

                legend(bbox_to_anchor = (1.5,1))


                if state_num < 8:
                    if pl == 0:
                        ax.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))
                    elif pl == 1:
                        ax.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                elif state_num ==8:
                    ax = plt.gca()
                    ax2 = ax.twinx()
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = self.state_label(state_num))
                    elif pl ==1 :
                        ax2.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                else:
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = self.state_label(state_num))
                    elif pl == 1:
                        ax2.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                xlabel('ef sideband phase (deg)')
                ylabel('Correlator')

                legend(bbox_to_anchor = (1.9,1))
                tight_layout()
                # xlim(-300.0,300.0)
                xlim(-xlimit,xlimit)
                ylim(-1.1,1.1)

        fig = plt.figure(figsize = (12,5))


        verts = []
        verts2 = []

        for state_num in state_num_list:

            phipts = linspace(-xlimit*pi/180.0,xlimit*pi/180.0,50)
            out = []

            for phi in phipts:
                out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

            verts.append(list(zip(phipts*180.0/pi,out)))
            if N!=1:
                verts2.append(list(zip(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5))))

            edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']


        ax = fig.add_subplot(121,projection='3d')
        ax.set_title('Expected Correlator: ' + self.tom_label(tom_num))
        poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor = edgecolor,alpha=0.5)
        poly.set_alpha(0.25)
        ax.add_collection3d(poly, zs=state_num_list, zdir='y')
        z = linspace(-1,1,10)
        x = xlimit*ones(len(z))
        y = 1*ones(len(z))
        for i in arange(16):
            ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
        for i in arange(16):
            ax.plot(z*xlimit,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)


        ax.set_xlabel('Phase (deg)')
        ax.set_xlim3d(-180.0, 180.0)
        ax.set_ylabel('State #')
        ax.set_ylim3d(-1, 16)
        ax.set_zlabel('Correlator')
        ax.set_zlim3d(-1,1)

        if N !=1:
            ax = fig.add_subplot(122,projection='3d')
            ax.set_title('Measured Correlator: ' + self.tom_label(tom_num))

            poly = LineCollection(verts2, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor,alpha=0.5)
            poly.set_alpha(0.25)
            ax.add_collection3d(poly, zs=state_num_list, zdir='y')
            z = linspace(-1,1,10)
            x = xlimit*ones(len(z))
            y = 1*ones(len(z))
            for i in arange(16):
                ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
            for i in arange(16):
                ax.plot(z*xlimit,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)

            ax.set_xlabel('Phase (deg)')
            ax.set_xlim3d(-xlimit, xlimit)
            ax.set_ylabel('State #')
            ax.set_ylim3d(-1, 16)
            ax.set_zlabel('Correlator')
            ax.set_zlim3d(-1,1)
        tight_layout()
        plt.show()
    # Bar graph of results vs correlator # for all states
    def plot_corr_all(self,find_optimal_phase=False,phaselist=None,corrlist=None):
        state_num_list = arange(16)
        tom_num_list = arange(15)

        figure(figsize = (10,3*16))
        if self.xdata_p == None:
            out=self.theory_matrix()
        else:
            out=self.theory_matrix()
            out2=self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist = phaselist,corrlist=corrlist)
            xlab = []
        for tom_num in tom_num_list:
            xlab.append(self.tom_label(tom_num))
        for state_num in state_num_list:
            subplot(16,1,state_num+1, title='Input state ' + self.state_label(state_num)+'|0,0>')
            if self.xdata_p == None:
                bar(tom_num_list,out[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.25)
                print "Goes here"
            else:
                bar(tom_num_list,out[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.25)
                bar(tom_num_list-0.25,out2[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.75)
            xticks(tom_num_list,xlab)
            xlabel('Tomography correlator')
            ylabel('Probability')
            axhline(0,color='k')
            # ylim(-1.1,1.1)
            xlim(-1,15)
            tight_layout()

    def plot_contrast_all(self):

        state_num_list = arange(16)
        tom_num_list = arange(15)
        figure(figsize = (10,3*16))

        out=self.contrast_matrix()
        xlab = []
        for tom_num in tom_num_list:
            xlab.append(self.tom_label(tom_num))
        for state_num in state_num_list:
            subplot(16,1,state_num+1, title='Input state ' + self.state_label(state_num)+'|0,0>')
            bar(tom_num_list,out[state_num],width=0.5,color=self.colorlist[state_num/4],align="center",alpha=0.75)
            xticks(tom_num_list,xlab)
            xlabel('Tomography correlator')
            ylabel('Probability')
            axhline(0,color='k')
            ylim(-0.1,2.1)
            xlim(-1,15)
            tight_layout()
    # Tomography for a given state #
    def tomography(self,state_num):

        den_mat=0.25*self.PP[0]
        avg = self.corr_meas_all(state_num)
        for i in arange(15):
            den_mat+= 0.25*(avg[i]*self.PP[i+1])


        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121, title='Real', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(real(den_mat).flatten())))

        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Reds(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
#         plt.show()


#         fig = plt.figure(figsize=(5,5))

        ax = fig.add_subplot(122, title='Imaginary', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(imag(den_mat).flatten())))


        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Blues(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0.3,1.3,2.3,3.3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0.3,1.3,2.3,3.3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
        plt.show()

    def plot_process_matrix_theory(self,scale=1):
        m_ab = np.append(np.ones([16,1]),self.theory_matrix(),1)
        pmat = two_qubit_gate_inversion(m_ab)
        plt.figure(figsize=(10,8))
        plt.imshow(real(pmat),cmap = cm.RdBu,interpolation='none',extent=(0.5,16.5,16.5,0.5),origin='upper')
        plt.xticks(arange(1,17,1))
        plt.yticks(arange(1,17,1))
        grid(True)
        ylabel('m')
        xlabel('n')
        clim(-scale,scale)
        cb = colorbar()
        cb.set_ticks([-scale,0,scale])

    def plot_process_matrix_expt(self,find_optimal_phase=False,phaselist=None,corrlist=None,scale=1,filename = None):
        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}
        matplotlib.rc('font', **font)
        m_ab = np.append(np.ones([16,1]),self.theory_matrix(),1)
        pmattheory = two_qubit_gate_inversion(m_ab)
        m_ab = np.append(np.ones([16,1]),self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist=phaselist,corrlist=corrlist),1)
        pmat = two_qubit_gate_inversion(m_ab)

        plt.figure(figsize=(10,8))
        plt.imshow(real(pmat),cmap = cm.RdBu,interpolation='none',extent=(0.5,16.5,16.5,0.5),origin='upper')
        plt.xticks(arange(1,17,1))
        plt.yticks(arange(1,17,1))
        grid(True)
        ylabel('m')
        xlabel('n')
        clim(-scale,scale)
        cb = colorbar()
        cb.set_ticks([-scale,0,scale])
        if filename is not None:
            plt.savefig(filename,bbox_inches='tight', dpi=500)
        print "Process Fidelity = " + str(trace(dot(pmat,pmattheory)))

    def process_fid_expt(self,find_optimal_phase=False,phaselist=None,corrlist=None,scale=1,filename = None):
        font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}
        matplotlib.rc('font', **font)
        m_ab = np.append(np.ones([16,1]),self.theory_matrix(),1)
        pmattheory = two_qubit_gate_inversion(m_ab)
        m_ab = np.append(np.ones([16,1]),self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist=phaselist,corrlist=corrlist),1)
        pmat = two_qubit_gate_inversion(m_ab)

        return trace(dot(pmat,pmattheory))

class processtomographycorrelations():

    def __init__(self,gate_num=0,xdata_p = None,ydata_p = None,sweep_meas = False, sweep_final_sb = False,phi_cnot=0,phi_final_sb = 0):

        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
#         self.Z2 = np.array([[1,0],[0,0]])
        self.Z2 = np.array([[1,0],[0,0]])
        self.P = np.array([self.I,self.X,self.Y,self.Z])
        self.UCZ = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        self.UCX = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        self.UCY = array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]])
        self.I4 = kron(self.I,self.I)
        self.M = np.kron(self.Z2,self.I4)
        self.markerlist = ['s','o','d','^']
        self.colorlist = ['r','c','orange','b']
        self.gate_num = gate_num
        self.sweep_meas = sweep_meas
        self.sweep_final_sb = sweep_final_sb
        self.phi_cnot = phi_cnot
        self.phi_final_sb = phi_final_sb
        if xdata_p != None:
            self.xdata_p = xdata_p
            self.ydata_p = ydata_p
        else:
            self.xdata_p = None
            self.ydata_p = None

        if gate_num == 0:
            self.gate = self.I4
        elif gate_num == 1:
            self.gate = self.UCZ
        elif gate_num == 2:
            self.gate = self.UCX
        elif gate_num == 3:
            self.gate = self.UCY

        self.PP = []
        for i in arange(4):
            for j in arange(4):
                self.PP.append(kron(self.P[i],self.P[j]))

        self.PP = array(self.PP)
        self.find_fit = []

    def cphase(self,phi):

        return array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1*exp(1j*phi)]])

    def cnot(self,phi_ef1=0,phi_ef2=0,phi_cnot=0):

        return array([[1,0,0,0],[0,1,0,0],[0,0,0,exp(1j*(phi_ef1+phi_cnot))],[0,0,exp(-1j*(phi_ef2+phi_cnot)),0]])

    def expand(self,psi_mm):

        psi_q = array([1,0])
        self.psi = np.kron(psi_q,psi_mm)
        return self.psi
    # Unitary for a ge pi sideband pulses  # state = |q,2,1> (no shift)
    def U_sb_ge(self,mode,phi):

        self.U = empty([8,8],dtype= complex)

        if mode == 1:
            n1 = 0
            n2 = 1
        else:
            n1 = 1
            n2 = 0

        self.U.fill(0)
        self.U[0][0] = 1.0
        self.U[1][1] = n1  # g01
        self.U[2][2] = n2  # g10

        # these are wromg? |e01>,|e10> -- |g02> or |g11>

        # self.U[5][5] = n1
        # self.U[6][6] = n2   should always be 0


        self.U[7][7] = 1
        self.U[mode][4] = -1j*exp(-1j*phi)
        self.U[4][mode] = -1j*exp(+1j*phi)

        # Correct
        self.U[3][7-mode] = -1j*exp(-1j*phi)
        self.U[7-mode][3] = -1j*exp(+1j*phi)

        return self.U
    # Unitary for ge qubit pulses  # state = |q,2,1> (no shift)
    def U_q_ge(self,theta,phi):

        self.U = empty([8,8],dtype= complex)
        self.U.fill(0)

        for i in arange(4):

            self.U[i][i] = cos(theta/2.0)
            self.U[i+4][i+4] = cos(theta/2.0)
            self.U[i][i+4] = -1j*sin(theta/2.0)*exp(1j*phi)
            self.U[i+4][i] = -1j*sin(theta/2.0)*exp(-1j*phi)

        return self.U

    def preparestate(self,state_num):

        U_list = [self.U_q_ge(0,0),self.U_q_ge(pi/2.0,pi/2.0),self.U_q_ge(pi/2.0,0),self.U_q_ge(pi,0)]
        psi = array([1,0,0,0,0,0,0,0])

        # order does not matter in absence of dispersive shift (Correct order per convention)

        psi = dot(U_list[state_num/4],psi)
        psi = dot(self.U_sb_ge(2,-pi/2.0),psi)
        psi = dot(U_list[state_num%4],psi)
        psi = dot(self.U_sb_ge(1,-pi/2.0),psi)

        return psi

    def actgate(self,state_num,phi):
        if self.sweep_final_sb:
            add_cphase = 0
        else:
            add_cphase = phi
        if self.gate_num == 0:
            return dot(self.cphase(add_cphase+pi),self.preparestate(state_num)[:4])
        elif self.gate_num == 1:
            return  dot(self.cphase(add_cphase),self.preparestate(state_num)[:4])
        elif self.gate_num == 2:
            return  dot(self.cnot(phi_ef1=add_cphase),self.preparestate(state_num)[:4])

    def finalstate(self,tom_num,state_num,phi):


        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse
        # within the gate (in define gate)

        # State convention : |2, 1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        # CNOT(id2, id1) = CX
        # CZ(id2, id1) = CZ



        self.psi_mm = self.actgate(state_num,phi)

        if tom_num == 0:

            # IX

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==1:

            # IY

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==2:

            # IZ

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(1,pi/2.0),self.psi)

        elif tom_num ==3:

            # XI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==4:

            # XX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==5:

            # XY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==6:

            # XZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==7:

            # YI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==8:

            # YX

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==9:

            # YY

            self.psi = self.expand(dot(self.UCY,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==10:

            # YZ

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0.0),dot(self.U_sb_ge(2,pi/2.0),self.psi))

        elif tom_num ==11:

            # ZI

            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(2,pi/2.0),self.psi)

        elif tom_num ==12:

            # ZX

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, pi/2.0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==13:

            # ZY

            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(1,pi/2.0),self.psi))

        elif tom_num ==14:

            # ZZ

            self.psi = self.expand(dot(self.UCX,self.psi_mm))
            return dot(self.U_sb_ge(1,pi/2.0),self.psi)

    def finalstate_new(self,tom_num,state_num,phi,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):
        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse
        # within the gate (in define gate)

        # State convention : |2, 1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        # CNOT(id2, id1) = CX
        # CZ(id2, id1) = CZ

        self.psi_mm = self.actgate(state_num,phi)
        if self.sweep_meas:
            phi_ef2 = -phi + phi_add_ef_sb_cnot
        else:
            phi_ef2 = 0 + phi_add_ef_sb_cnot

        if self.sweep_final_sb:
            phi_final_sb = phi + self.phi_final_sb
        else:
            phi_final_sb = self.phi_final_sb

        if tom_num == 0:
            # IX
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==1:
            # IY
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==2:
            # IZ
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi)

        elif tom_num ==3:
            # XI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==4:
            # XX
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot = phi_add_ef_q_cnot),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))
            print self.phi_cnot

        elif tom_num ==5:
            # XY
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot= phi_add_ef_q_cnot-pi/2.0),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==6:
            # XZ
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,pi/2.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==7:
            # YI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_q_ge(pi/2.0, 0.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==8:
            # YX
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot = phi_add_ef_q_cnot),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, 0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==9:
            # YY

            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2,phi_cnot= phi_add_ef_q_cnot-pi/2.0),self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(2,pi/2.0+phi_final_sb),self.psi))

        elif tom_num ==10:
            # YZ
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0.0),dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==11:
            # ZI
            self.psi = self.expand(self.psi_mm)
            return dot(self.U_sb_ge(2,pi/2.0 + phi_final_sb),self.psi)

        elif tom_num ==12:
            # ZX
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0, pi/2.0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==13:
            # ZY
            self.psi = self.expand(dot(self.UCZ,self.psi_mm))
            return dot(self.U_q_ge(pi/2.0,0),dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi))

        elif tom_num ==14:
            # ZZ
            self.psi = self.expand(dot(self.cnot(phi_ef2=phi_ef2),self.psi_mm))
            return dot(self.U_sb_ge(1,pi/2.0 + phi_final_sb),self.psi)
    # Calculating the transmon expectation value
    def transmon_meas(self,tom_num,state_num,phi=0,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):

        psif  = self.finalstate_new(tom_num,state_num,phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)
        rho = outer(psif,conjugate(psif))
        return trace(dot(self.M,rho))

    def corr_meas_all(self,state_num):
        ans = []
        for tom_num in arange(15):
            ans.append(2*(self.transmon_meas(tom_num,state_num) - 0.5))
        return ans
    # 16,15 matrix storing the outcome vs state #, tom #
    def theory_matrix(self):
        m_ab=[]

        for state_num in arange(16):
            for tom_num in arange(15):
                m_ab.append(2*(self.transmon_meas(tom_num, state_num)-0.5))

        m_ab = around(real(array(m_ab)),4).reshape((16,15))

        return m_ab

    def expt_matrix(self,find_optimal_phase = False,phaselist=None,corrlist=None):

        m_ab=[]

        if self.xdata_p == None:
            print "No experimental data provided"
        else:
            if find_optimal_phase:
                find_fit_list = self.find_fit_matrix()
                for state_num in arange(16):
                    for tom_num in arange(15):
                        xdata = self.xdata_p[tom_num][state_num]
                        ydata = self.ydata_p[tom_num][state_num]
                        find_phase = find_fit_list[state_num][tom_num] #'max' or 'min'
                        if find_phase == "mean":
                            m_ab.append(-2*(mean(self.ydata_p[tom_num][state_num])-0.5))
                        else:
                            expected_period = 360.
                            x_at_extremum = sin_phase(xdata,-2*(ydata-0.5),expected_period,find_phase,showfit = False)
                            if x_at_extremum > 300 or x_at_extremum < -300:
                                print "check fits"
                                m_ab.append(-2*(mean(self.ydata_p[tom_num][state_num])-0.5))
                            else:
                                m_ab.append(-2*(ydata[argmin(abs(xdata-x_at_extremum))]-0.5))
            else:

                for state_num in arange(16):
                    for tom_num in arange(15):
                        expt_num = 16*tom_num + state_num
                        offset = 0.0
                        if corrlist == None or phaselist == None:
                            pass
                        else:
                            if expt_num in corrlist:
                                offset = phaselist[argmin(abs(corrlist-expt_num))]

                        xdata = self.xdata_p[tom_num][state_num]
                        ydata = self.ydata_p[tom_num][state_num]
                        m_ab.append(-2*(ydata[argmin(abs(xdata-offset))]-0.5))
                        print "Phase offset for (s,t) = (%s,%s) = %s degrees"%(state_num,tom_num,offset)

            m_ab = around(real(array(m_ab)),4).reshape((16,15))
            return m_ab

    def contrast_matrix(self):
        c_ab=[]
        phi_pts = linspace(-pi,pi,50)
        for state_num in arange(16):
            for tom_num in arange(15):
                corr_pts = array([2*(self.transmon_meas(tom_num, state_num,phi)-0.5) for phi in phi_pts])
                # fitdata = fitsin(phi_pts,corr_pts,showfit=False)
                c_ab.append(abs(max(corr_pts)-min(corr_pts)))

        c_ab = array(c_ab).reshape((16,15))

        return c_ab

    def find_fit_matrix(self):

        find_fit = []
        t_matrix = self.theory_matrix()
        c_matrix = self.contrast_matrix()
        phi_pts = linspace(-pi,pi,50)
        for state_num in arange(16):
            for tom_num in arange(15):
                corr_pts = array([2*(self.transmon_meas(tom_num, state_num,phi)-0.5) for phi in phi_pts])
                if c_matrix[state_num][tom_num] < 0.05:
                    find_fit.append('mean')
                else:
                    if abs(max(corr_pts)-t_matrix[state_num][tom_num]) < 0.05:
                        find_fit.append('max')
                    elif abs(min(corr_pts)-t_matrix[state_num][tom_num]) < 0.05:
                        find_fit.append('min')
                    else: find_fit.append('int')

        return array(find_fit).reshape((16,15))

    def plot_contrast_matrix(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.contrast_matrix(),cmap = cm.OrRd,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('State #')
        xlabel('Tomography Correlator')

        clim(0,2)
        cb = colorbar()
        cb.set_ticks([0,1,2])

    def plot_theory_matrix(self):
        plt.figure(figsize=(7,7))
        plt.imshow(self.theory_matrix(),cmap = cm.PuOr,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label2(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('Input state')
        xlabel('Tomography Correlator')

        clim(-1,1)
        cb = colorbar()
        cb.set_ticks([-1,0,1])

    def plot_expt_matrix(self,find_optimal_phase = False,phaselist=None,corrlist=None):
        plt.figure(figsize=(7,7))
        plt.imshow(self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist=phaselist,corrlist=corrlist),cmap = cm.PuOr,interpolation='none',extent=(0.5,15.5,16.5,0.5),origin='upper')
        ylab = []
        xlab = []
        for tom_num in arange(15):
            xlab.append(self.tom_label(tom_num))
        for tom_num in arange(16):
            ylab.append(self.state_label2(tom_num))
        # plt.xticks(arange(1,16,1))
        # plt.yticks(arange(1,17,1))
        plt.xticks(arange(1,16,1),xlab)
        plt.yticks(arange(1,17,1),ylab)
        # xlim(0.5,16.5)
        # ylim(0.5,15.5)
        grid(True)
        ylabel('Input state')
        xlabel('Tomography Correlator')

        clim(-1,1)
        cb = colorbar()
        cb.set_ticks([-1,0,1])

    def tom_label(self,tom_num):
        lab=''
        if((tom_num +1)/4 == 0):
            lab = lab + 'I'
        elif((tom_num + 1)/4 == 1):
            lab = lab + 'X'
        elif((tom_num + 1)/4 == 2):
            lab = lab + 'Y'
        elif((tom_num + 1)/4 == 3):
            lab = lab + 'Z'
        if((tom_num + 1)%4 == 0):
            lab = lab + 'I'
        elif(((tom_num + 1))%4 == 1):
            lab = lab + 'X'
        elif(((tom_num + 1))%4 == 2):
            lab = lab + 'Y'
        elif(((tom_num + 1))%4 == 3):
            lab = lab + 'Z'

        return lab

    def state_label(self,state_num):
        lab = str(state_num)
        if((state_num)/4 == 0):
            lab = lab + ' -- $I$'
        elif((state_num)/4 == 1):
            lab = lab + ' -- $Y_{\pi/2}$'
        elif((state_num)/4 == 2):
            lab = lab + ' -- $X_{\pi/2}$'
        elif((state_num)/4 == 3):
            lab = lab + ' -- $X_{\pi}$'
        lab = lab + '$\otimes$'
        if((state_num)%4 == 0):
            lab = lab + '$I$'
        elif(((state_num))%4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif(((state_num))%4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif(((state_num))%4 == 3):
            lab = lab + '$X_{\pi}$'

        return lab

    def state_label2(self,state_num):
        lab = ''
        if((state_num)/4 == 0):
            lab = lab + '$I$'
        elif((state_num)/4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif((state_num)/4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif((state_num)/4 == 3):
            lab = lab + '$X_{\pi}$'
        lab = lab + '$\otimes$'
        if((state_num)%4 == 0):
            lab = lab + '$I$'
        elif(((state_num))%4 == 1):
            lab = lab + '$Y_{\pi/2}$'
        elif(((state_num))%4 == 2):
            lab = lab + '$X_{\pi/2}$'
        elif(((state_num))%4 == 3):
            lab = lab + '$X_{\pi}$'

        return lab

    def plot_corr_vs_phase(self, tom_num, state_num,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):

        figure(figsize = (10,5))
        subplot(111, title='State = ' + self.tom_label(tom_num)+', Tom. corr = ' + self.state_label(state_num))
        phipts = linspace(-pi,pi,30)
        out = []
        for phi in phipts:
            out.append(self.transmon_meas(tom_num,state_num,phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot))
        plot(phipts,out)
        xlabel('Phase (deg)')
        ylabel('Probability')
        xlim(-pi,pi)

    def plot_corr_vs_two_phases(self, tom_num, state_num,add_ef_sb_cnot=0):

        figure(figsize = (10,5))
        subplot(111, title='tom = ' + self.tom_label(tom_num)+', state = ' + self.state_label(state_num))

        phipts_1 = linspace(-pi,pi,30)
        phipts_2 = linspace(-pi,pi,30)

        data = empty([30,30])
        for ii,phi_1 in enumerate(phipts_1):
            for jj,phi_2 in enumerate(phipts_2):
                data[ii][jj] = 2*(self.transmon_meas(tom_num, state_num, phi_1,add_ef_sb_cnot,phi_2)-0.5)

        pcolormesh((180.0/pi)*phipts_1,(180.0)/pi*phipts_2,data.T,cmap=cm.RdBu)
        clim(-1,1)
        colorbar()
        xlabel('$\delta\phi$ Added (Subtracted) from CZ (CNOT) Phase (deg)')
        ylabel('CNOT ef qubit phase (deg)')
    #3d plots of correlators vs phase, state
    def plot_corr_all_vs_phase3D(self,start,stop, tom_num,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0):


        fig = plt.figure(figsize = (10,8))
        ax = fig.gca(projection='3d')
        ax.set_title('Expected Correlator: -- ' + self.tom_label(tom_num))

        out_list=[]
        state_num_list = arange(start,stop)
        verts = []

        for state_num in state_num_list:

            phipts = linspace(-pi,pi,50)
            out = []


            for phi in phipts:
                out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

            verts.append(list(zip(phipts,out)))

        # edgecolor = ['r','b','g','y','orange','pink','gray','c','k','r','b','g','y','orange','pink','gray']
        edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']
        poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor )
        poly.set_alpha(0.5)
        ax.add_collection3d(poly, zs=state_num_list, zdir='y')
        ax.set_xlabel('Final Sideband Phase (deg)')
        ax.set_xlim3d(-pi, pi)
        ax.set_ylabel('State #')
        ax.set_ylim3d(-1, 16)
        ax.set_zlabel('Probability')
        ax.set_zlim3d(-1,1)

        plt.show()
    # 2d and 3d plots of correlators vs ef sideband phase (first ef sideband of CZ gate, and second ef sideband of CNOT gate (XX,XY,YX,YY) )
    def plot_corr_all_vs_phase(self,start,stop, tom_num,xlimit=180,phi_add_ef_sb_cnot=0,phi_add_ef_q_cnot=0,sta = 0,sto =50):

        state_num_list = arange(start,stop)
        N=2
        if self.xdata_p == None:
            N = 1

        for pl in arange(N):

            if pl == 0:
                if N==1:
                    fig = plt.figure(figsize = (10,5))
                    ax = fig.add_subplot(111,title= 'Expected Correlator: -- ' + self.tom_label(tom_num))
                else:
                    fig = plt.figure(figsize = (10,10))
                    ax = fig.add_subplot(211,title= 'Expected Correlator: -- ' + self.tom_label(tom_num))

            elif pl == 1:

                ax = fig.add_subplot(212,title= 'Measured Correlator: -- ' +  self.tom_label(tom_num))

            for state_num in state_num_list:

                phipts = linspace(-xlimit*pi/180.0,xlimit*pi/180.0,50)
                out = []

                for phi in phipts:
                    out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

                legend(bbox_to_anchor = (1.5,1))


                if state_num < 8:
                    if pl == 0:
                        ax.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))
                    elif pl == 1:
                        ax.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                elif state_num ==8:
                    ax = plt.gca()
                    ax2 = ax.twinx()
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = self.state_label(state_num))
                    elif pl ==1 :
                        ax2.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                else:
                    if pl == 0:
                        ax2.plot(phipts*180.0/pi,out,'--',marker = self.markerlist[state_num%4],
                                 markersize = 7.5,color=self.colorlist[state_num/4],label = self.state_label(state_num))
                    elif pl == 1:
                        ax2.plot(self.xdata_p[tom_num][state_num],-2*((self.ydata_p[tom_num][state_num])-0.5),'-',
                                 marker = self.markerlist[state_num%4],markersize = 7.5,color=self.colorlist[state_num/4],label=self.state_label(state_num))

                xlabel('ef sideband phase (deg)')
                ylabel('Correlator')

                legend(bbox_to_anchor = (1.9,1))
                tight_layout()
                # xlim(-300.0,300.0)
                xlim(-xlimit,xlimit)
                ylim(-1.1,1.1)

        fig = plt.figure(figsize = (12,5))


        verts = []
        verts2 = []

        for state_num in state_num_list:

            phipts = linspace(-xlimit*pi/180.0,xlimit*pi/180.0,50)
            out = []

            for phi in phipts:
                out.append(2*(self.transmon_meas(tom_num, state_num, phi,phi_add_ef_sb_cnot,phi_add_ef_q_cnot)-0.5))

            verts.append(list(zip(phipts*180.0/pi,out)))
            if N!=1:
                verts2.append(list(zip(self.xdata_p[tom_num][state_num][sta:sto],-2*((self.ydata_p[tom_num][state_num][sta:sto])-0.5))))

            edgecolor = ['r','r','r','r','c','c','c','c','orange','orange','orange','orange','b','b','b','b']


        ax = fig.add_subplot(121,projection='3d')
        ax.set_title('Expected Correlator: ' + self.tom_label(tom_num))
        poly = LineCollection(verts, linewidth =2.5,facecolor = edgecolor,edgecolor = edgecolor,alpha=0.5)
        poly.set_alpha(0.25)
        ax.add_collection3d(poly, zs=state_num_list, zdir='y')
        z = linspace(-1,1,10)
        x = xlimit*ones(len(z))
        y = 1*ones(len(z))
        for i in arange(16):
            ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
        for i in arange(16):
            ax.plot(z*xlimit,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)


        ax.set_xlabel('Phase (deg)')
        ax.set_xlim3d(-180.0, 180.0)
        ax.set_ylabel('State #')
        ax.set_ylim3d(-1, 16)
        ax.set_zlabel('Correlator')
        ax.set_zlim3d(-1,1)

        if N !=1:
            ax = fig.add_subplot(122,projection='3d')
            ax.set_title('Measured Correlator: ' + self.tom_label(tom_num))

            poly = LineCollection(verts2, linewidth =2.5,facecolor = edgecolor,edgecolor =edgecolor,alpha=0.5)
            poly.set_alpha(0.25)
            ax.add_collection3d(poly, zs=state_num_list, zdir='y')
            z = linspace(-1,1,10)
            x = xlimit*ones(len(z))
            y = 1*ones(len(z))
            for i in arange(16):
                ax.plot(x,i*y,z,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)
            for i in arange(16):
                ax.plot(z*xlimit,i*y,y,color = edgecolor[i],linestyle = 'dotted',alpha=0.75)

            ax.set_xlabel('Phase (deg)')
            ax.set_xlim3d(-xlimit, xlimit)
            ax.set_ylabel('State #')
            ax.set_ylim3d(-1, 16)
            ax.set_zlabel('Correlator')
            ax.set_zlim3d(-1,1)
        tight_layout()
        plt.show()
    # Bar graph of results vs correlator # for all states
    def plot_corr_all(self,find_optimal_phase=False,phaselist=None,corrlist=None):
        state_num_list = arange(16)
        tom_num_list = arange(15)

        figure(figsize = (10,3*16))
        if self.xdata_p == None:
            out=self.theory_matrix()
        else:
            out=self.theory_matrix()
            out2=self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist = phaselist,corrlist=corrlist)
            xlab = []
        for tom_num in tom_num_list:
            xlab.append(self.tom_label(tom_num))
        for state_num in state_num_list:
            subplot(16,1,state_num+1, title='Input state ' + self.state_label(state_num)+'|0,0>')
            if self.xdata_p == None:
                bar(tom_num_list,out[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.25)
                print "Goes here"
            else:
                bar(tom_num_list,out[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.25)
                bar(tom_num_list-0.25,out2[state_num],width=0.25,color=self.colorlist[state_num/4],alpha=0.75)
            xticks(tom_num_list,xlab)
            xlabel('Tomography correlator')
            ylabel('Probability')
            axhline(0,color='k')
            # ylim(-1.1,1.1)
            xlim(-1,15)
            tight_layout()

    def plot_contrast_all(self):

        state_num_list = arange(16)
        tom_num_list = arange(15)
        figure(figsize = (10,3*16))

        out=self.contrast_matrix()
        xlab = []
        for tom_num in tom_num_list:
            xlab.append(self.tom_label(tom_num))
        for state_num in state_num_list:
            subplot(16,1,state_num+1, title='Input state ' + self.state_label(state_num)+'|0,0>')
            bar(tom_num_list,out[state_num],width=0.5,color=self.colorlist[state_num/4],align="center",alpha=0.75)
            xticks(tom_num_list,xlab)
            xlabel('Tomography correlator')
            ylabel('Probability')
            axhline(0,color='k')
            ylim(-0.1,2.1)
            xlim(-1,15)
            tight_layout()

    # Tomography for a given state #
    def tomography(self,state_num):

        den_mat=0.25*self.PP[0]
        avg = self.corr_meas_all(state_num)
        for i in arange(15):
            den_mat+= 0.25*(avg[i]*self.PP[i+1])


        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121, title='Real', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(real(den_mat).flatten())))

        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Reds(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
#         plt.show()


#         fig = plt.figure(figsize=(5,5))

        ax = fig.add_subplot(122, title='Imaginary', projection='3d')

        coord= [0,1,2,3]

        x_pos=[]
        y_pos=[]
        for i in range(4):
            for j in range(4):
                x_pos.append(coord[i])
                y_pos.append(coord[j])

        xpos=np.array(x_pos)
        ypos=np.array(y_pos)
        zpos=np.array([0]*16)
        dx = [0.6]*16
        dy = dx
        dz=np.squeeze(np.asarray(np.array(imag(den_mat).flatten())))


        nrm=mpl.colors.Normalize(-1,1)
        colors=cm.Blues(nrm(dz))
        alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)

        for i in range(len(dx)):
            ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
        xticks=['00','01','10','11']
        yticks=xticks
        ax.set_xticks([0.3,1.3,2.3,3.3])
        ax.set_xticklabels(xticks)
        ax.set_yticks([0.3,1.3,2.3,3.3])
        ax.set_yticklabels(yticks)
        ax.set_zlim(-1,1)
        plt.show()

    def plot_process_matrix_theory(self,scale=1):
        m_ab = np.append(np.ones([16,1]),self.theory_matrix(),1)
        pmat = two_qubit_gate_inversion(m_ab)
        plt.figure(figsize=(7,7))
        plt.imshow(real(pmat),cmap = cm.RdBu,interpolation='none',extent=(0.5,16.5,16.5,0.5),origin='upper')
        plt.xticks(arange(1,17,1))
        plt.yticks(arange(1,17,1))
        grid(True)
        ylabel('m')
        xlabel('n')
        clim(-scale,scale)
        cb = colorbar()
        cb.set_ticks([-scale,0,scale])

    def plot_process_matrix_expt(self,find_optimal_phase=False,phaselist=None,corrlist=None,scale=1):
        m_ab = np.append(np.ones([16,1]),self.theory_matrix(),1)
        pmattheory = two_qubit_gate_inversion(m_ab)
        m_ab = np.append(np.ones([16,1]),self.expt_matrix(find_optimal_phase=find_optimal_phase,phaselist=phaselist,corrlist=corrlist),1)
        pmat = two_qubit_gate_inversion(m_ab)
        plt.figure(figsize=(7,7))
        plt.imshow(real(pmat),cmap = cm.RdBu,interpolation='none',extent=(0.5,16.5,16.5,0.5),origin='upper')
        plt.xticks(arange(1,17,1))
        plt.yticks(arange(1,17,1))
        grid(True)
        ylabel('m')
        xlabel('n')
        clim(-scale,scale)
        cb = colorbar()
        cb.set_ticks([-scale,0,scale])
        print "Process Fidelity = " + str(trace(dot(pmat,pmattheory)))

def tom_label(tom_num):
        lab=''
        if((tom_num +1)/4 == 0):
            lab = lab + 'I'
        elif((tom_num + 1)/4 == 1):
            lab = lab + 'X'
        elif((tom_num + 1)/4 == 2):
            lab = lab + 'Y'
        elif((tom_num + 1)/4 == 3):
            lab = lab + 'Z'
        if((tom_num + 1)%4 == 0):
            lab = lab + 'I'
        elif(((tom_num + 1))%4 == 1):
            lab = lab + 'X'
        elif(((tom_num + 1))%4 == 2):
            lab = lab + 'Y'
        elif(((tom_num + 1))%4 == 3):
            lab = lab + 'Z'

        return lab

def state_label(state_num):
    lab = str(state_num)
    if((state_num)/4 == 0):
        lab = lab + ' -- $I$'
    elif((state_num)/4 == 1):
        lab = lab + ' -- $Y_{\pi/2}$'
    elif((state_num)/4 == 2):
        lab = lab + ' -- $X_{\pi/2}$'
    elif((state_num)/4 == 3):
        lab = lab + ' -- $X_{\pi}$'
    lab = lab + '$\otimes$'
    if((state_num)%4 == 0):
        lab = lab + '$I$'
    elif(((state_num))%4 == 1):
        lab = lab + '$Y_{\pi/2}$'
    elif(((state_num))%4 == 2):
        lab = lab + '$X_{\pi/2}$'
    elif(((state_num))%4 == 3):
        lab = lab + '$X_{\pi}$'

    return lab