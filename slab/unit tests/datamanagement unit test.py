# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:14:08 2012

@author: Dave
"""

from slab import *
from os import *

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
    


    
    
    
if __name__=="__main__":
    test_slabfile()