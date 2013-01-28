# -*- coding: utf-8 -*-
"""
datamanagement.py
Created on Fri May 18 16:07:13 2012

@author: Phil
"""

import numpy as np
import sys
import h5py
import inspect
import Pyro4
import threading

def get_SlabFile(fname, local=False):
    """Retrieves a Pyro4 proxy object representing a SlabFile"""
    if local:
        return SlabFile(fname)
    ns = Pyro4.locateNS()
    try:
        return DSetProxy(ns.lookup(fname))
    except Pyro4.errors.NamingError:
        print "didn't find on server, creating..."
        fileserver = Pyro4.Proxy(ns.lookup('fileserver'))
        fileserver.add_file(fname)
        return DSetProxy(ns.lookup(fname))

class DSetProxy(object):
    def __init__(self, uri_or_proxy, dspath=[]):
        if isinstance(uri_or_proxy, Pyro4.Proxy):
            self.proxy = uri_or_proxy
        else: # It's a URI
            self.proxy = Pyro4.Proxy(uri_or_proxy)
        self.proxy._pyroOneway.update(['create_group', 'create_dataset'])
        self.dspath = dspath

    def __getitem__(self, ds):
        newpath = self.dspath + [ds]
        res = self.proxy._get_dset_array(newpath)
        if res == "group":
            return DSetProxy(self.proxy, newpath)
        data_arr, attrs = res
        data_arr.attrs = DSetAttrsProxy(self.dspath + [ds], self.proxy, attrs)
        return data_arr

    def items(self):
        return [ (k, self.__getitem__(k)) for k in self.proxy.keys() ]

    def values(self):
        return [ self.__getitem__(k) for k in self.proxy.keys() ]

    def create_group(self, groupname, *args, **kwargs):
        self.proxy.create_group(groupname, *args, **kwargs)
        return DSetProxy(self.proxy, self.dspath + [groupname])

    def create_dataset(self, ds, *args, **kwargs):
        self.proxy.create_group(ds, *args, **kwargs)
        return DSetProxy(self.proxy, self.dspath + [ds])

    def __setitem__(self, ds, value):
        self.proxy._my_assign_dset(self.dspath, ds, value)

    def __getattr__(self, attr):
        # Fall back to behaving like a proxy
        return lambda *args, **kwargs: self.proxy._call_with_path(self.dspath, attr, args, kwargs)

class DSetAttrsProxy(dict):
    def __init__(self, dspath, proxy, *args, **kwargs):
        self.proxy = proxy
        self.dspath = dspath
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, item, value):
        print self.dspath
        self.proxy._set_attr(self.dspath, item, value)
        dict.__setitem__(self, item, value)

class H5Array(np.ndarray):
    def __new__(cls, dset):
        obj = np.asarray(dset).view(cls)
        return obj
    def __array_finalize__(self, obj):
        self.attrs = None

class FileServer():
    def __init__(self):
        self.ns = Pyro4.locateNS()
        self.start_server()

    def add_file(self, fname):
        print 'adding', fname
        try:
            self.ns.unregister(fname)
        except Exception:
            pass
        self.ns.register(fname, self.daemon.register(SlabFile(fname)))

    def start_server(self):
        self.selfdaemon = Pyro4.Daemon()
        self.ns.register('fileserver', self.selfdaemon.register(self))
        self.self_th = threading.Thread(target=self.selfdaemon.requestLoop)
        self.self_th.daemon = True
        self.self_th.start()
        self.daemon = Pyro4.Daemon()
        self.file_th = threading.Thread(target=self.daemon.requestLoop)
        self.file_th.daemon = True
        self.file_th.start()

    def restart(self):
        self.close()
        self.start_server()

    def list(self):
        return self.daemon.objectsById

    def close(self):
        for v in self.daemon.objectsById.values():
            try:
                self.ns.remove(v.filename)
                self.daemon.unregister(v)
            except AttributeError:
                pass
        self.ns.remove('fileserver')
        self.selfdaemon.unregister(self)
        self.selfdaemon.shutdown()
        self.daemon.shutdown()

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

    # Methods for proxy use
    def _my_ds_from_path(self, dspath):
        """returns the object (dataset or group) specified by dspath"""
        branch = self
        for ds in dspath:
            branch = branch[ds]
        return branch

    def _my_assign_dset(self, dspath, ds, val):
        print 'assigning', ds, val
        branch = self._my_ds_from_path(dspath)
        branch[ds] = val

    def _get_dset_array(self, dspath):
        """returns a pickle-safe array for the branch specified by dspath"""
        branch = self._my_ds_from_path(dspath)
        if isinstance(branch, h5py.Group):
            return 'group'
        else:
            return (H5Array(branch), dict(branch.attrs))
            #return np.array(branch)

    def _get_attrs(self, dspath):
        branch = self._my_ds_from_path(dspath)
        return dict(branch.attrs)

    def _set_attr(self, dspath, item, value):
        branch = self._my_ds_from_path(dspath)
        branch.attrs[item] = value

    def _call_with_path(self, dspath, method, args, kwargs):
        branch = self._my_ds_from_path(dspath)
        return getattr(branch, method)(*args, **kwargs)


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
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                shape, maxshape = (0, len(line)), (None, len(line))
                if axis == 1:
                    shape, maxshape = (shape[1], shape[0]), (maxshape[1], maxshape[0])
                self.create_dataset(dataset, shape=shape, maxshape=maxshape, dtype='float64')
                dataset = self[dataset]
        shape=list(dataset.shape)
        shape[axis]=shape[axis]+1
        dataset.resize(shape)
        if axis==0:
            dataset[-1,:]=line
        else:
            dataset[:,-1]=line
        self.flush()
            
    def append_pt(self,dataset,pt):
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                self.create_dataset(dataset, shape=(0,), maxshape=(None,), dtype='float64')
                dataset = self[dataset]
        shape=list(dataset.shape)
        shape[0]=shape[0]+1
        dataset.resize(shape)
        dataset[-1]=pt
        self.flush()
        
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
    

class H5Server(object):
    def __init__(self):
        """Creates a server for resolving multiple access to HDF5 files"""
        pass

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
    
