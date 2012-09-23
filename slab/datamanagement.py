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
        self.attrs["_script"] = get_script()
        self.flush()
        
    def save_settings(self,dic,group='settings'):
        if group not in self:
            self.create_group(group)
        for k in dic.keys():
            self[group].attrs[k]=dic[k]
    

def get_script():
    """returns currently running script file as a string"""
    f=open(inspect.stack()[-1][1],'r')
    s=f.read()
    f.close()
    return s  


def set_range(h5file, xmin, xmax, ymin=None, ymax=None):
    if ymin is not None and ymax is not None:
        h5file.attrs["_axes"] = ((xmin, xmax), (ymin, ymax))
    else:
        h5file.attrs["_axes"] = (xmin, xmax)

def set_labels(h5file, x_lab, y_lab, z_lab=None):
    if z_lab is not None:
        h5file.attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
    else:
        h5file.attrs["_axes_labels"] = (x_lab, y_lab)
        
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
    