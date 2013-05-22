import sys
import os

import numpy as np
import Pyro4
import traceback
import h5py

import helpers
import config

Pyro4.config.ONEWAY_THREADED = False

class DataClient(object):
    def __init__(self):
        self.proxy = Pyro4.Proxy(config.manager_uri)
        self.proxy_closed = False
        self.proxy._pyroOneway.add('load_h5file')

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.proxy._pyroRelease()

    def __getattr__(self, item):
        def try_proxy_fn(*args, **kwargs):
            if self.proxy_closed:
                return 'proxy closed'
            try:
                return getattr(self.proxy, item)(*args, **kwargs)
            except Pyro4.errors.ConnectionClosedError:
                print "PROXY CLOSED"
                self.proxy_closed = True
                return 'proxy closed'
            except:
                self.proxy.msg('failure on ' + helpers.method_string(item, args, kwargs))
                self.proxy.msg(traceback.format_exc())
                remote_tb = "".join(Pyro4.util.getPyroTraceback())
                self.proxy.msg(remote_tb)
                sys.stderr.write(remote_tb)
                sys.stderr.flush()
                raise
        return try_proxy_fn


def SlabFile(*args, **kwargs):
    if kwargs.pop('remote', False):
        return SlabFileRemote(*args, **kwargs)
    return SlabFileLocal(*args, **kwargs)


class SlabFileRemote:
    def __init__(self, filename=None, manager=None, context=(), autosave=None, **ignoredargs):
        if autosave is None:
            autosave = filename is not None
        self.autosave = autosave
        if len(ignoredargs) > 0:
            print 'Warning', ', '.join(ignoredargs.keys()), 'ignored in call to SlabFileRemote'
        if manager is None:
            self.manager = DataClient()
            if filename and os.path.exists(filename):
                self.manager.load_h5file(filename)
        else:
            self.manager = manager
        self.filename = filename
        if context == () and filename is not None:
            self.context = (filename,)
        else:
            self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.manager._pyroRelease()

    def create_dataset(self, name, shape=None, data=None, **initargs):
        if data is None:
            if shape is None:
                data = np.array([0])
            else:
                data = np.zeros(shape)
        self.manager.set_data(self.context + (name,), data, **initargs)
#        if self.autosave:
#            self.flush(self.context + (name,))

    def set_range(self, x0=0, y0=None, xscale=1, yscale=None):
        if y0 is None and yscale is None:
            rank = 1
            self.manager.set_params(self.context, rank, x0=x0, xscale=xscale)
        else:
            assert None not in (y0, yscale)
            rank = 2
            self.manager.set_params(self.context, rank, x0=x0, y0=y0, xscale=xscale, yscale=yscale)
            #if self.autosave:
            #    self.flush(self.context)

    def set_labels(self, xlabel="", ylabel="", zlabel=None):
        if zlabel is None:
            self.manager.set_params(self.context, 1, xlabel=xlabel, ylabel=ylabel)
        else:
            self.manager.set_params(self.context, 2, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            #if self.autosave:
            #    self.flush(self.context)

    # No need to create groups with remote file, as their existence is inferred,
    # but it shouldn't cause an error either
    def create_group(self, name):
        pass

    def __getitem__(self, key):
        if isinstance(key, str):
            return SlabFileRemote(self.filename, self.manager, self.context + (key,), self.autosave)
        elif isinstance(key, (int, slice, tuple)):
            return self.manager.get_data(self.context, key)
        else:
            raise ValueError('Unknown key type ' + str(key))

    def __setitem__(self, key, value):
        if isinstance(key, str):
            context = self.context + (key,)
            self.manager.set_data(context, value)
        elif isinstance(key, (int, slice, tuple)):
            self.manager.set_data(self.context, value, slice=key)
        else:
            raise ValueError('Unknown key type ' + str(key))

    def __delitem__(self, key): # TODO
        pass

    def __array__(self):
        return self.manager.get_data(self.context, None)

    def flush(self, path=None):
        if path is None:
            self.manager.save_as_file((self.filename,))
        else:
            self.manager.save_with_file(path, self.filename)

    def __getattr__(self, item):
        if item is 'attrs':
            return AttrsProxy(self.manager, self.context)
        # For some reason getslice isn't properly deprecated
        elif item is '__getslice__':
            return lambda start, stop: self.__getitem__(slice(start, stop, None))
        elif item is '__setslice__':
            return lambda start, stop, value: self.__setitem__(slice(start, stop, None), value)
        # Array creation tests for the presence of these properties, which we don't have
        elif item in ('__array_struct__', '__array_interface__'):
            raise AttributeError
        def call_with_context(*args, **kwargs):
            getattr(self.manager, item)(self.context, *args, **kwargs)
        return call_with_context


class AttrsProxy:
    def __init__(self, manager, context):
        self.manager = manager
        self.context = context

    def __setitem__(self, item, value):
        self.manager.set_attr(self.context, item, value)

    def __getitem__(self, item):
        self.manager.get_attr(self.context, item)


class SlabFileLocal(h5py.File):
    pass

def append_data(file_or_group, dataset_name, new_data):
    new_data = np.array(new_data)
    rank = len(new_data.shape) + 1
    if dataset_name in file_or_group:
        dataset = file_or_group[dataset_name]
        new_shape = list(dataset.shape)
        new_shape[0] += 1
        dataset.resize(new_shape)
        if rank > 1:
            dataset.attrs['parametric'] = new_data.shape[0] == 2
    else:
        dataset = file_or_group.create_dataset(dataset_name,
                                               shape=(1,) + new_data.shape,
                                               maxshape=(None,)*rank)
    if rank == 1:
        dataset[-1] = new_data
    elif rank == 2:
        dataset[-1,:] = new_data
    elif rank == 3:
        dataset[-1,:,:] = new_data

h5py.File.append_data = append_data
h5py.Group.append_data = append_data
