from PyQt4 import Qt
import h5py
import Pyro4
import numpy as np

from model import DataTree, DataTreeLeaf, DataTreeLeafReduced
import helpers
import config

Pyro4.config.COMMTIMEOUT = 3.5
Pyro4.config.DOTTEDNAMES = True
RUNNING = True

class BackgroundObject(Qt.QObject):
    def __init__(self):
        Qt.QObject.__init__(self)
        self.gui = None
        self.params = {}

    def set_param(self, name, value):
        self.params[name] = value

class DataManager(BackgroundObject):
    def __init__(self, path_delim='/'):
        BackgroundObject.__init__(self)
        self.data = None
        self.delim = path_delim

    def connect_data(self):
        self.data = DataTree(self.gui)

    def get_or_make_leaf(self, path, rank=None, data_tree_args={}, plot_args={}, reduced=False):
        group = self.data.resolve_path(path[:-1])
        if path[-1] not in group:
            assert rank is not None
            assert isinstance(group, DataTree)
            leaf = group.make_child_leaf(path[-1], rank, **data_tree_args)
            self.gui.add_tree_widget(path, data=True, save=leaf.save, plot=leaf.plot)
            if leaf.plot:
                #params = {'x0': leaf.x0, 'xscale': leaf.xscale, 'xlabel': leaf.xlabel, 'ylabel': leaf.ylabel}
                #if rank > 1:
                #    params.update({'y0': leaf.y0, 'yscale': leaf.yscale, 'zlabel': leaf.zlabel})
                self.gui.add_plot_widget(path, rank, **plot_args)
        else:
            leaf = group[path[-1]]
            assert (rank is None) or (rank == leaf.rank)
            for key, val in data_tree_args.items():
                setattr(leaf, key, val)
            if len(data_tree_args) > 0:
                self.update_plot(path, refresh_labels=True)

        if not isinstance(leaf, DataTreeLeaf):
            raise ValueError('path does not store a leaf, but rather a ' + str(type(leaf)))
        if reduced:
            return DataTreeLeafReduced(leaf)
        return leaf

    def set_params(self, path, rank, **initargs):
        data_tree_args, plot_args, curve_args = helpers.separate_init_args(initargs)
        leaf = self.get_or_make_leaf(path, rank, data_tree_args, plot_args)
        if leaf.file is not None:
            leaf.save_in_file()
        self.update_plot(path, refresh_labels=True, **curve_args)

    def set_data(self, name_or_path, data, slice=None, parametric=False, **initargs):
        path = helpers.canonicalize_path(name_or_path)

        if parametric and isinstance(data, (np.ndarray, list)):
            if data.shape[1] == 2:
                data = np.transpose(data)

        if parametric or isinstance(data, tuple):
            parametric = True
            rank = len(data) - 1
            data = np.array(data)
        else:
            data = np.array(data)
            rank = len(data.shape)


        assert rank in (1, 2)

        data_tree_args, plot_args, curve_args = helpers.separate_init_args(initargs)
        data_tree_args['parametric'] = parametric

        leaf = self.get_or_make_leaf(path, rank, data_tree_args, plot_args)

        if slice is None:
            leaf.data = data
        else:
            leaf.data[slice] = data
        if leaf.file is not None:
            leaf.save_in_file()
        #if leaf.plot:
        self.update_plot(name_or_path, refresh_labels=(len(initargs) > 0), **curve_args)

    def append_data(self, name_or_path, data, show_most_recent=None, parametric=False, **initargs):
        path = helpers.canonicalize_path(name_or_path)

        if parametric or isinstance(data, tuple):
            parametric = True
            rank = len(data) - 1
            data = np.array(data)
        else:
            data = np.array(data)
            rank = len(data.shape) + 1

        data_tree_args, plot_args, curve_args = helpers.separate_init_args(initargs)
        data_tree_args['parametric'] = parametric
        leaf = self.get_or_make_leaf(path, rank, data_tree_args, plot_args)

        if leaf.data is None:
            leaf.data = np.array([data])
        else:
            if rank is 1 and not parametric:
                leaf.data = np.hstack((leaf.data, data))
            else:
                leaf.data = np.vstack((leaf.data, data))

        if leaf.file is not None:
            leaf.save_in_file()
        #if leaf.plot:
        self.update_plot(name_or_path, refresh_labels=(len(initargs) > 0),
                         show_most_recent=show_most_recent, **curve_args)

    def get_data(self, name_or_path, slice=None):
        path = helpers.canonicalize_path(name_or_path)
        item = self.data.resolve_path(path)
        if not isinstance(item, DataTreeLeaf):
            raise ValueError('Leaf not found ' + str(path))
        if slice is not None:
            x = np.array(item.data)[slice]
            return x
        else:
            return np.array(item.data)

    def set_attr(self, name_or_path, item, value):
        path = helpers.canonicalize_path(name_or_path)
        node = self.data.resolve_path(path)
        node.attrs[item] = value

    def get_attr(self, name_or_path, item):
        path = helpers.canonicalize_path(name_or_path)
        node = self.data.resolve_path(path)
        return node.attrs[item]

    def update_plot(self, name_or_path, refresh_labels=False, show_most_recent=None, **curve_args):
        path = helpers.canonicalize_path(name_or_path)
        item = self.data.resolve_path(path)
        tree_widget = self.gui.tree_widgets[path]
        try:
            tree_widget.setText(1, str(item.data.shape))
        except AttributeError:
            tree_widget.setText(1, 'No Data')
        tree_widget.setText(2, str(item.save))
        tree_widget.setText(3, str(item.plot))
        if item.plot:
            if item.rank == 2:
                self.gui.plot_widgets[path].update_plot(item,
                        refresh_labels=refresh_labels, show_most_recent=show_most_recent, **curve_args)
            else:
                self.gui.plot_widgets[path].update_plot(item, refresh_labels=refresh_labels, **curve_args)
        self.gui.update_multiplots(path, item)

    def save_all(self):
        raise NotImplementedError

    def load_h5file(self, filename, readonly=False):
        mode = 'r' if readonly else 'a'
        f = h5py.File(filename, mode)
        if not readonly:
            self.data[filename] = DataTree(self.gui, filename, file=f)
        self.rec_load_file(f, (filename,), readonly)
        if readonly:
            f.close()

    def rec_load_file(self, file, path, readonly):
        for name, ds in file.items():
            this_path = path + (name,)
            self.msg(name, type(ds))
            if isinstance(ds, h5py.Group):
                self.rec_load_file(ds, this_path, readonly)
            else:
                parametric = ds.attrs.get('parametric', False)
                self.set_data(this_path, np.array(ds), save=(not readonly), parametric=parametric)
                self.get_or_make_leaf(this_path).load_attrs_from_ds(ds)
                self.update_plot(this_path, refresh_labels=True)

    def clear_data(self, path=None, leaf=None):
        assert path is not None or leaf is not None
        if leaf is None:
            leaf = self.get_or_make_leaf(path)
        if leaf.rank == 1:
            leaf.data = None
        elif leaf.rank == 2:
            leaf.data = None
        if leaf.plot:
            self.update_plot(path)

    def clear_all_data(self):
        for path, leaf in self.data.leaves():
            self.clear_data(leaf=leaf)

    def serve(self):
        print 'serving'
        with Pyro4.Daemon(host=config.manager_host, port=config.manager_port) as d:
            self.running = True
            d.register(self, config.manager_id)
            d.requestLoop(lambda: RUNNING)
        print 'done serving'
        self.data.close()
        print "data closed"

    def abort_daemon(self):
        global RUNNING
        RUNNING = False
        print 'Aborted!'

    def save_as_file(self, path): #TODO
        data = self.data.resolve_path(path)
        assert isinstance(data, DataTree)
        assert helpers.valid_h5file(path[-1])
        with h5py.File(path[-1], 'a') as f:
            for item in data.values():
                item.save_in_file(f)

    def save_with_file(self, path, filename): #TODO
        with h5py.File(filename, 'a') as h5file:
            data = self.data.resolve_path(path)
            f = h5file
            for p in path[:-1]:
                if p not in f:
                    f = f.create_group(p)
                else:
                    f = f[p]
            data.save_in_file(f)

    def msg(self, *args):
        self.gui.msg(*args)

    def gui_method(self, method, *args, **kwargs):
        getattr(self.gui, method)(*args, **kwargs)

if __name__ == "__main__":
    test_manager = DataManager()
