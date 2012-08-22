# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:02:55 2012

@author: Dave
"""

from qutip import tensor,jmat, sigmax,sigmay,sigmaz,qeye,destroy,basis
from matplotlib.pyplot import *
from numpy import array,dot,arange,transpose,real,abs,sqrt,cross,cos,sin,pi,zeros,ones,vstack,hstack

def spin_hanger_s21(tlist,glist,kslist,f0,kc,kci,f):
    return abs(1+kc/( complex(0,1)*(f-f0)-(kc+kci/2)+sum(glist**2/(complex(0,1)*(f-tlist)-kslist/2))))**2

def RotateAboutN(theta,n):
    
    R=[ [cos(theta)+n[0]**2*(1-cos(theta)),n[0]*n[1]*(1-cos(theta))-n[2]*sin(theta),n[0]*n[2]*(1-cos(theta))+n[1]*sin(theta)],
        [n[0]*n[1]*(1-cos(theta))+n[2]*sin(theta),cos(theta)+n[1]**2*(1-cos(theta)),n[1]*n[2]*(1-cos(theta))-n[0]*sin(theta)],
        [n[0]*n[2]*(1-cos(theta))-n[1]*sin(theta), n[1]*n[2]*(1-cos(theta))+n[0]*sin(theta),cos(theta)+n[2]**2*(1-cos(theta))]
        ]
    return array(R)

def RotateAtoB(a,b):
    """Get rotation matrix that rotates point a to point b about axis a x b"""
    a=a/sqrt(dot(a,a))
    b=b/sqrt(dot(b,b))
    axb=cross(a,b)
    cosab=dot(a,b)
    sinab=sqrt(dot(axb,axb))
    axb=axb/sqrt(dot(axb,axb))
    
    R=[ [cosab+axb[0]**2*(1-cosab),axb[0]*axb[1]*(1-cosab)-axb[2]*sinab,axb[0]*axb[2]*(1-cosab)+axb[1]*sinab],
        [axb[0]*axb[1]*(1-cosab)+axb[2]*sinab,cosab+axb[1]**2*(1-cosab),axb[1]*axb[2]*(1-cosab)-axb[0]*sinab],
        [axb[0]*axb[2]*(1-cosab)-axb[1]*sinab, axb[1]*axb[2]*(1-cosab)+axb[0]*sinab,cosab+axb[2]**2*(1-cosab)]
        ]
    
    return array(R)
    

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

def NV_allowed_transitions(ekets,threshold=.001):
    allowed_transitions=[]
    for ii in xrange(len(ekets)):
        for jj in xrange (ii+1,len(ekets)):
            if abs((ekets[ii].dag()*jmat(1,'x')*ekets[jj]).tr())+abs((jmat(1,'x').dag()*jmat(1,'y')*ekets[jj]).tr()) +abs((ekets[ii].dag()*jmat(1,'z')*ekets[jj]).tr())>threshold:
                allowed_transitions.append([ii,jj])
    return allowed_transitions

def transition_frequencies(enlist,transitions):
    return array([abs(enlist[trans[0]]-enlist[trans[1]]) for trans in transitions])
        

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

    
if __name__=="__main__": 
    
    
    H_p1 = P1H(array([0,0,1]))
    H_nv = NVH(array([0,0,1]))
    p1list_1=[]; p1list_2=[]; p1list_3=[]; p1list_4=[]
    nvlist_1=[]; nvlist_2=[]; nvlist_3=[]; nvlist_4=[]
    blist=arange(40.,60.,.1)
    theta=0.0001
    phi=0.
    Bn=array([cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)])
    
    Bn=dot(RotateAboutN(90.*pi/180.,[1.,1.,0.]),[1.,0.,0.])
    
    R1=RotateAtoB([0,0,1.],[ 1., 1., 1.])
    R2=RotateAtoB([0,0,1.],[-1.,-1., 1.])
    R3=RotateAtoB([0,0,1.],[ 1.,-1.,-1.])
    R4=RotateAtoB([0,0,1.],[-1., 1.,-1.])
    print "Calculating transitions"
    for b in blist:
        B=b*Bn
        B1=dot(R1,B)
        B2=dot(R2,B)
        B3=dot(R3,B)
        B4=dot(R4,B)
        #print bz
        H_p1_1 = P1H(B1)
        H_nv_1 = NVH(B1)
#        H_p1_2 = P1H(B2)
#        H_nv_2 = NVH(B2)
#        H_p1_3 = P1H(B3)
#        H_nv_3 = NVH(B3)
#        H_p1_4 = P1H(B4)
#        H_nv_4 = NVH(B4)
        
        p1_ekets, evals = H_p1_1.eigenstates()
        p1list_1.append(evals)
        nv_ekets, evals = H_nv_1.eigenstates()
        nvlist_1.append(evals)
#        p1_ekets, evals = H_p1_2.eigenstates()
#        p1list_2.append(evals)
#        nv_ekets, evals = H_nv_2.eigenstates()
#        nvlist_2.append(evals)
#        p1_ekets, evals = H_p1_3.eigenstates()
#        p1list_3.append(evals)
#        nv_ekets, evals = H_nv_3.eigenstates()
#        nvlist_3.append(evals)
#        p1_ekets, evals = H_p1_4.eigenstates()
#        p1list_4.append(evals)
#        nv_ekets, evals = H_nv_4.eigenstates()
#        nvlist_4.append(evals)
        
    p1list_1=transpose(array(p1list_1))
    nvlist_1=transpose(array(nvlist_1))
#    p1list_2=transpose(array(p1list_2))
#    nvlist_2=transpose(array(nvlist_2))
#    p1list_3=transpose(array(p1list_3))
#    nvlist_3=transpose(array(nvlist_3))
#    p1list_4=transpose(array(p1list_4))
#    nvlist_4=transpose(array(nvlist_4))
    
    
#    print "P1 transitions"
#    figure(1)
#    xlabel('Magnetic Field, B (mT)')
#    ylabel('Frequency, F (GHz)')
#    plot_transitions(blist,p1list_1,'P1_1',P1_allowed_transitions(p1_ekets,.05))
#    plot_transitions(blist,p1list_2,'P1_2',P1_allowed_transitions(p1_ekets,.05))
#    plot_transitions(blist,p1list_3,'P1_3',P1_allowed_transitions(p1_ekets,.05))
#    plot_transitions(blist,p1list_4,'P1_4',P1_allowed_transitions(p1_ekets,.05))
#    print "NV transitions"
#    figure(2)
#    xlabel('Magnetic Field, B (mT)')
#    ylabel('Frequency, F (GHz)')
#    plot_transitions(blist,nvlist_1,'NV_1',NV_allowed_transitions(nv_ekets))
#    plot_transitions(blist,nvlist_2,'NV_2',NV_allowed_transitions(nv_ekets))
#    plot_transitions(blist,nvlist_3,'NV_3',NV_allowed_transitions(nv_ekets))
#    plot_transitions(blist,nvlist_4,'NV_4',NV_allowed_transitions(nv_ekets))
    #print P1_allowed_transitions(p1_ekets)
    figure(3)
    print "Calculating density plot"
    f0=5.
    kc=.01
    kci=0.0001
    ks=.001
    g=.05
    fpts=arange(4.9,5.1,.001)
    spin_pts=arange(4.,5.5,.001)
    tlist=transpose(transition_frequencies(nvlist_1,NV_allowed_transitions(nv_ekets)))
    s21pts=zeros((len(fpts),len(tlist)))
    for xx,f in enumerate(fpts):
        for yy,t in enumerate(tlist):
            s21pts[xx,yy]=spin_hanger_s21(t,g,ks,f0,kc,kci,f)    
    imshow(s21pts,aspect='auto',origin='lower',interpolation='none',extent=(blist[0],blist[-1],fpts[0],fpts[-1]))
    xlabel('Magnetic Field, B (mT)')
    ylabel('Frequency, F (GHz)')
    #plot_transitions(blist,nvlist_1,'NV_1',NV_allowed_transitions(nv_ekets))
    plot(blist,tlist)
    show()
    