import sys
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.proxy._pyroRelease()

    def __getattr__(self, item):
        print item
        def try_proxy_fn(*args, **kwargs):
            if self.proxy_closed:
                return 'proxy closed'
            try:
                print 'calling', item
                res = getattr(self.proxy, item)(*args, **kwargs)
                print 'result', res
                return res
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
        else:
            self.manager = manager
        self.filename = filename
        if context == () and filename is not None:
            self.context = (filename,)
        else:
            self.context = context

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
        elif isinstance(key, int):
            s = slice(key, key+1, None)
            return self.manager.get_data(self.context + (key,), slice=s)[0]
        elif isinstance(key, slice):
            return self.manager.get_data(self.context, key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            context = self.context + (key,)
            self.manager.set_data(context, value)
        elif isinstance(key, int):
            context = self.context
            s = slice(key, key+1, None)
            self.manager.set_data(self.context, value, slice=s)
        elif isinstance(key, slice):
            context = self.context
            self.manager.set_data(self.context, value, slice=key)
#        if self.autosave:
#            self.flush(context)

    def __delitem__(self, key): # TODO
        pass

    def __array__(self):
        return np.array(self[:])

    def flush(self, path=None):
        if path is None:
            self.manager.save_as_file((self.filename,))
        else:
            self.manager.save_with_file(path, self.filename)

    def __getattr__(self, item):
        def call_with_context(*args, **kwargs):
            getattr(self.manager, item)(self.context, *args, **kwargs)
        return call_with_context

    #def append_data(self, new_data, show_most_recent=None):
    #    self.manager.append_data(self.context, new_data, show_most_recent=show_most_recent)
    #        if self.autosave:
    #            self.flush(self.context)

    #def set_data(self, *args, **kwargs):
    #    self.manager.set_data(self.context, *args, **kwargs)


class SlabFileLocal(h5py.File):
    pass


def create_dataset(file_or_group, name, show=True, plot=True, **kwargs):
    assert show or plot
    h5py.File.create_dataset(file_or_group, name, **kwargs)

