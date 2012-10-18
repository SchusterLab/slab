# -*- coding: utf-8 -*-
"""
datamanagement.py
Created on Fri May 18 16:07:13 2012

@author: Phil
"""

import sys
import h5py
import inspect

class SlabFile(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        #self.attrs["_script"] = open(sys.argv[0], 'r').read()
#        if self.mode is not 'r':
#            self.attrs["_script"] = get_script()
        #if not read-only or existing then save the script into the .h5
        #Maybe should take this automatic feature out and just do it when you want to
        if (self.mode is not 'r') and ("_script" not in self.attrs):     
            self.save_script()
        self.flush()


    def set_range(self,dataset, xmin, xmax, ymin=None, ymax=None):
        if ymin is not None and ymax is not None:
            dataset.attrs["_axes"] = ((xmin, xmax), (ymin, ymax))
        else:
            dataset.attrs["_axes"] = (xmin, xmax)
    
    def set_labels(self,dataset, x_lab, y_lab, z_lab=None):
        if z_lab is not None:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
        else:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab)

    def append_line(self,dataset,line,axis=0):
        shape=list(dataset.shape)
        shape[axis]=shape[axis]+1
        dataset.resize(shape)
        if axis==0:
            dataset[-1,:]=line
        else:
            dataset[:,-1]=line
            
    def append_pt(self,dataset,pt):
        shape=list(dataset.shape)
        shape[0]=shape[0]+1
        dataset.resize(shape)
        dataset[-1]=pt
        
    def save_script(self,name="_script"):
            self.attrs[name] = get_script()

    def save_settings(self,dic,group='settings'):
        if group not in self:
            self.create_group(group)
        for k in dic.keys():
            self[group].attrs[k]=dic[k]
            
    def load_settings(self,group='settings'):
        d={}
        for k in self[group].attrs.keys():
            d[k]=self[group].attrs[k]
        return d
    

def get_script():
    """returns currently running script file as a string"""
    fname=inspect.stack()[-1][1]    
    #print fname
    f=open(fname,'r')
    s=f.read()
    f.close()
    return s  

        
def open_to_path(h5file, path, pathsep='/'):
    f = h5file
    for name in path.split(pathsep):
        if name:
            f = f[name]
    return f

def get_next_trace_number(h5file, last=0, fmt="%03d"):
    i = last
    while (fmt % i) in h5file:
        i += 1
    return i
    
def open_to_next_trace(h5file, last=0, fmt="%03d"):
    return h5file[fmt % get_next_trace_number(h5file, last, fmt)]
    