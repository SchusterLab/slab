# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 00:11:22 2012

@author: Nitrogen
"""
from numpy import *
from guiqwt.pyplot import *

sx=array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])/sqrt(2)
sy=array([[0.,complex(0.,-1.),0.],[complex(0,1.),0.,complex(0,-1.)],[0.,complex(0.,1.),0.]])/sqrt(2)
sz=array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]])


def Hnv(bx,by,bz):
    m0=0.00289
    D=2.873
    EE=0.0043
    return D*dot(sz,sz)+EE*(dot(sx,sx)-dot(sy,sy))+m0*(bx*sx+by*sy+bz*sz)

def Env(bx,by,bz):
    return abs(linalg.eigvalsh(Hnv(bx,by,bz)))
    
bpts=arange(0.,200.)    
NVEnergies=transpose(array([Env(0.,0.,bz) for bz in bpts]))

for env in NVEnergies[1:]:
    plot(bpts,env)
    
show()

