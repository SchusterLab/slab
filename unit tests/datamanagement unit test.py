# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:14:08 2012

@author: Dave
"""

from slab import *
from os import *
from numpy import *
import matplotlib.pyplot as plt

def test_slabfile():
    fname='slabfile_test.h5'
    try: os.remove(fname)
    except: pass

    #print "Test save and load settings"
    f=SlabFile(fname,'w')
    d={'a':1,'b':2.1,'c':'This is a string!'}    
    f.save_settings(d)
    f.close()

    f=SlabFile(fname)
    d2=f.load_settings()
    f.close()
    error=False
    for k in d:
        if d2[k]!=d[k]:
            print "Error! loaded item != to original: ", str(d2[k])," != ", str(d[k])
    if not error: print "Passed: Save/load settings."
    else: print "Failed: Rewrite settings."
    
    #print "Test rewriting settings"
    d3={'a':2,'b':2.1,'c':'This is a string!','d':'Extra string'}    
    f=SlabFile(fname)
    f.save_settings(d3)
    f.close()

    f=SlabFile(fname,'r')
    d4=f.load_settings()
    f.close()
    error=False
    for k in d:
        if d4[k]!=d3[k]:
            print "Error! loaded item != to original: ", str(d4[k])," != ", str(d3[k])
    if not error: print "Passed: Rewrite settings."
    else: print "Failed: Rewrite settings."
    os.remove(fname)
    
def test_append_data():
    fname='append_data_test.h5'
    try: os.remove(fname)
    except: pass

    error=False

    num_fpts=400
    num_tpts=1000
    freqs=linspace(1.,10.,num_fpts)

    tpts=linspace(0.,1.,num_tpts)
    data=array([exp(-tpts/0.5)*sin(2*pi*freq *tpts) for freq in freqs])
#    plt.figure(1)
#    plt.imshow(data)
#    plt.show()

    
    
    f=SlabFile(fname)
    f.create_dataset('Rabi',shape=(0,num_tpts),maxshape=(None,num_tpts),dtype=float64)
    f.create_dataset('fpts', shape=(0,),maxshape=(None,),dtype=float64)
    f.close()
    for ii,d in enumerate(data):
        f=SlabFile(fname)
        #ds=f['Rabi']
        #ds.resize((ii+1,num_tpts)) 
        #ds[ii,:]=d
        f.append_line(f['Rabi'],d)
        f.append_pt(f['fpts'],freqs[ii])
        f.close()

    f2=SlabFile(fname)
    data2=array(f['Rabi'])
    f2.close()
    plt.figure(1)
    plt.imshow(data2)
    plt.show()

    if not error: print "Passed: Append data."
    else: print "Failed: Append data."
        
    
    
    #os.remove(fname)

    
    
    
if __name__=="__main__":
    test_slabfile()
    test_append_data()