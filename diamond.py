# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:02:55 2012

@author: Dave
"""

from qutip import tensor,jmat, sigmax,sigmay,sigmaz,qeye,destroy,basis
from matplotlib.pyplot import *
from numpy import array,dot,arange,transpose,real,abs

#H=B.S + A S.I
def P1H(B):
    P1muB=0.0280427/2.*B
    A=[0.08133, 0.08133, 0.11403]
    muBH=tensor(P1muB[0]*sigmax()+P1muB[1]*sigmay()+P1muB[2]*sigmaz(),qeye(3))
    ASIH=tensor(sigmax()/2.,A[0]*jmat(1,'x'))+tensor(sigmay()/2.,A[1]*jmat(1,'y'))+tensor(sigmaz()/2.,A[2]*jmat(1,'z'))
    return muBH+ASIH

def P1_transition_weights(ekets):
    weights=[]
    for ii in xrange(len(ekets)):
        for jj in xrange (ii+1,len(ekets)):
            weights.append([ii,jj,abs((ekets[ii].dag()*tensor(sigmax(),qeye(3))*ekets[jj]).tr())+abs((ekets[ii].dag()*tensor(sigmay(),qeye(3))*ekets[jj]).tr()) +abs((ekets[ii].dag()*tensor(sigmaz(),qeye(3))*ekets[jj]).tr())])
    return weights

def P1_allowed_transitions(ekets,threshold=.001):
    weights=P1_transition_weights(ekets)
    allowed_transitions=[]
    for w in weights: 
        if w[-1]>threshold: allowed_transitions.append(w[:2])        
#    for ii in xrange(len(ekets)):
#        for jj in xrange (ii+1,len(ekets)):
#            if abs((ekets[ii].dag()*tensor(sigmax(),qeye(3))*ekets[jj]).tr())+abs((ekets[ii].dag()*tensor(sigmay(),qeye(3))*ekets[jj]).tr()) +abs((ekets[ii].dag()*tensor(sigmaz(),qeye(3))*ekets[jj]).tr())>threshold:
#                allowed_transitions.append([ii,jj])
    return allowed_transitions
    
def NVH(B):
    m0=0.0289
    D=2.873
    EE=0.0043
    return D*dot(jmat(1,'z'),jmat(1,'z'))+EE*(dot(jmat(1,'x'),jmat(1,'x'))-dot(jmat(1,'y'),jmat(1,'y')))+m0*(B[0]*jmat(1,'x')+B[1]*jmat(1,'y')+B[2]*jmat(1,'z'))

#def JC_Energy_Level(B,w,g_nv,g_p1):
#
#    P1muB=0.0280427/2.*B    
#    A=[0.08133, 0.08133, 0.11403]    
#    D=2.873
#    EE=0.0043
#    m=5
#    a=tensor(destroy(m),qeye(2),qeye(3),qeye(3))
#    Sp1x=tensor(qeye(m),sigmax(),qeye(3),qeye(3))
#    Sp1y=tensor(qeye(m),sigmay(),qeye(3),qeye(3))
#    Sp1z=tensor(qeye(m),sigmaz(),qeye(3),qeye(3))
#    Ix=tensor(qeye(m),qeye(2),jmat(1,'x'),qeye(3))
#    Iy=tensor(qeye(m),qeye(2),jmat(1,'y'),qeye(3))
#    Iz=tensor(qeye(m),qeye(2),jmat(1,'z'),qeye(3))
#    Snvx=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'x'))
#    Snvy=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'y'))
#    Snvz=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'z'))
#
#    Snv_c=Snvx+1j*Snvy
#    Snv_d=Snvx-1j*Snvy
#
#    Sp1_c=Sp1x+1j*Sp1y
#    Sp1_d=Sp1x-1j*Sp1y
#    
#    H_jc=w*(a.dag()*a)+P1muB[0]*Sp1x+P1muB[1]*Sp1y+P1muB[2]*Sp1z+A[0]*Ix*Sp1x+A[1]*Iy*Sp1y+A[2]*Iz*Sp1z+D*Snvz*Snvz+EE*(Snvx*Snvx-Snvy*Snvy)+g_nv*(a*Snv_c+a.dag()*Snv_d)+g_p1*(a*Sp1_c+a.dag()*Sp1_d)+2*P1muB[2]*Snvz
#
#    ekets, evals = H_jc.eigenstates()
#
#    allowed_transitions=[]
#    tag=0.
#    ekets=transpose(ekets)
#    b1=tensor(basis(5,1)*basis(5,1).dag(),qeye(2),qeye(3),qeye(3))
#    b2=tensor(basis(5,2)*basis(5,2).dag(),qeye(2),qeye(3),qeye(3))
#    b3=tensor(basis(5,3)*basis(5,3).dag(),qeye(2),qeye(3),qeye(3))
#    b4=tensor(basis(5,4)*basis(5,4).dag(),qeye(2),qeye(3),qeye(3))
#    b5=tensor(basis(5,5)*basis(5,5).dag(),qeye(2),qeye(3),qeye(3))
#
#        
#    for ii in xrange(len(ekets)):
#        for kk in arange(6)
#        tag=abs((ekets[ii].dag()*b*ekets[ii]).tr())
#        if tag>0.3:
#            for jj in xrange (ii+1,len(ekets)):
#                tag=abs((ekets[ii].dag()*b*ekets[ii]).tr())
#                if tag>0.3:
#                    allowed_transitions.append([ii,jj])
#    
#    tran_E=[]
#    for trans in allowed_transitions:
#        tran_E.append(evals[trans[1]]-evals[trans[0]])
#    tran_E=array(tran_E)
#    tran_E.sort()
#    return tran_E
#
#
#def JC_Energy_kets(B,w,g_nv,g_p1):
#
#    P1muB=0.0280427/2.*B    
#    A=[0.08133, 0.08133, 0.11403]    
#    D=2.873
#    EE=0.0043
#    m=2
#    a=tensor(destroy(m),qeye(2),qeye(3),qeye(3))
#    Sp1x=tensor(qeye(m),sigmax(),qeye(3),qeye(3))
#    Sp1y=tensor(qeye(m),sigmay(),qeye(3),qeye(3))
#    Sp1z=tensor(qeye(m),sigmaz(),qeye(3),qeye(3))
#    Ix=tensor(qeye(m),qeye(2),jmat(1,'x'),qeye(3))
#    Iy=tensor(qeye(m),qeye(2),jmat(1,'y'),qeye(3))
#    Iz=tensor(qeye(m),qeye(2),jmat(1,'z'),qeye(3))
#    Snvx=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'x'))
#    Snvy=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'y'))
#    Snvz=tensor(qeye(m),qeye(2),qeye(3),jmat(1,'z'))
#
#    Snv_c=Snvx+1j*Snvy
#    Snv_d=Snvx-1j*Snvy
#
#    Sp1_c=Sp1x+1j*Sp1y
#    Sp1_d=Sp1x-1j*Sp1y
#    
#    H_jc=w*(a.dag()*a)+P1muB[0]*Sp1x+P1muB[1]*Sp1y+P1muB[2]*Sp1z+A[0]*Ix*Sp1x+A[1]*Iy*Sp1y+A[2]*Iz*Sp1z+D*Snvz*Snvz+EE*(Snvx*Snvx-Snvy*Snvy)+g_nv*(a*Snv_c+a.dag()*Snv_d)+g_p1*(a*Sp1_c+a.dag()*Sp1_d)+P1muB[2]*Snvz
#    
#    ekets, evals = H_jc.eigenstates()
#    
#
#    '''
#    threshold=.001
#    allowed_transitions=[]
#    for ii in xrange(len(ekets)):
#        for jj in xrange (ii+1,len(ekets)):
#            if abs((ekets[ii].dag()*Snvx*ekets[jj]).tr())+abs((ekets[ii].dag()*Sp1x*ekets[jj]).tr())>threshold:
#                allowed_transitions.append([ii,jj])
#    freq=[]
#    for trans in allowed_transitions:
#        freq.append(evals[trans[1]]-evals[trans[0]])
#    '''
#    
#    return ekets

def NV_allowed_transitions(ekets,threshold=.001):
    allowed_transitions=[]
    for ii in xrange(len(ekets)):
        for jj in xrange (ii+1,len(ekets)):
            if abs((ekets[ii].dag()*jmat(1,'x')*ekets[jj]).tr())+abs((jmat(1,'x').dag()*jmat(1,'y')*ekets[jj]).tr()) +abs((ekets[ii].dag()*jmat(1,'z')*ekets[jj]).tr())>threshold:
                allowed_transitions.append([ii,jj])
    return allowed_transitions

def plot_transitions(xpts,enlist,label='',allowed_transitions=None):
    if allowed_transitions is None:
        allowed_transitions=[]
        for ii in xrange(len(enlist)):
            for jj in xrange(ii+1,len(enlist)):
                allowed_transitions.append([ii,jj])
    if label!='': label+="-"
    for trans in allowed_transitions:
        plot(xpts,abs(enlist[trans[0]]-enlist[trans[1]]),label=label+str(trans[0])+","+str(trans[1]))
    legend()
'''     
def Plot_JC_Model():
    bzlist=arange(-200.00001,200.,1.)
    for bz in bzlist:
'''        
    
if __name__=="__main__": 
    
    print "calculating"
    bzlist=arange(-200.00001,200.,10.)
    Evals=[]
    print JC_Energy_Level(array([0.0001,0.0001,50.]),5.8,.1,.1)
    for bz in bzlist:
        Eval= JC_Energy_Level(array([0.0001,0.0001,bz]),5.8,.1,.1)
        Evals.append(Eval)
    Evals=transpose(array(Evals))
    '''
    figure(1)
    for ii in arange(len(JC_Energy_Level(array([0.0001,0.0001,10]),5.8,0.1,0.1))):
        plot(bzlist,Evals[ii])
    show()
    
    n=10
    '''
    figure(2)
    for ii in arange(3):
        plot(bzlist,Evals[ii])
    show()
    
    
    '''
    H_p1 = P1H(array([0,0,1]))
    H_nv = NVH(array([0,0,1]))
    p1list=[]
    nvlist=[]
    bzlist=arange(-200.00001,200.,10.)
    for bz in bzlist:
        #print bz
        H_p1 = P1H(array([0.0001,0.0001,bz]))
        H_nv = NVH(array([0.001,0.001,bz]))
        
        p1_ekets, evals = H_p1.eigenstates()
        p1list.append(evals)
        nv_ekets, evals = H_nv.eigenstates()
        nvlist.append(evals)
        
    p1list=transpose(array(p1list))
    nvlist=transpose(array(nvlist))
    
    
    
    figure(1)
    plot_transitions(p1list,'P1',P1_allowed_transitions(p1_ekets,.05))
    #figure(2)
    plot_transitions(nvlist,'NV',NV_allowed_transitions(nv_ekets))
    #print P1_allowed_transitions(p1_ekets)
    
    show()
    '''