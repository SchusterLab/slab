import h5py
from collections import defaultdict
import helpers

# http://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

    def resolve_path(self, path):
        res = self
        for item in path:
            res = res[item]
        return res


class DataTree(keydefaultdict):
    def __init__(self, gui, name=None, parentpath=(), file=None):
        if (name is not None) and (file is None) and helpers.valid_h5file(name):
            file = h5py.File(name, 'a')
        self.file = file
        if name is None:
            self.path = ()
        else:
            self.path = parentpath + (name,)
            gui.add_tree_widget(self.path)
        self.gui = gui
        keydefaultdict.__init__(self, self.make_internal_child)

    def close(self):
        for v in self.values():
            if isinstance(v, DataTree):
                v.close()

        if isinstance(self.file, h5py.File):
            print 'closing', self.file
            self.file.close()

    def make_internal_child(self, key):
        if self.file is not None:
            if key in self.file:
                file = self.file[key]
            else:
                file = self.file.create_group(key)
        else:
            file = None
        return DataTree(self.gui, name=key, parentpath=self.path, file=file)

    def make_child_leaf(self, name, rank, **initargs):
        leaf = DataTreeLeaf(self.path + (name,), rank=rank, file=self.file, **initargs)
        self[name] = leaf
        return leaf

    def leaves(self):
        for k, v in self.items():
            if isinstance(v, DataTree):
                for k2, v2 in v.leaves():
                    yield (k,)+k2, v2
            else:
                yield (k,), v

    def save_in_file(self, file_or_group):
        for key, item in self.items():
            print 'Key', key, item
            if isinstance(item, DataTree):
                print 'Group', key
                if any(i.save for k, i in item.leaves()):
                    print 'saving group'
                    file_or_group.create_group(key)
                    item.save_in_file(file_or_group[key])
            elif isinstance(item, DataTreeLeaf):
                if item.save:
                    print 'Dset', key
                    item.save_in_file(file_or_group)
            else:
                raise ValueError("Can't save unknown item " + str(type(item)))


class DataTreeLeaf(object):
    def __init__(self, path, rank=None, file=None, data=None, save=None, parametric=False,
                 plot=True, x0=0, xscale=1, xlabel='x', y0=0, yscale=1, ylabel='y', zlabel='z'):
        self.path = path
        self.rank = rank
        self.file = file
        self.data = data
        if save is None:
            self.save = self.file is not None
        else:
            self.save = save
        self.parametric = parametric
        self.plot = plot
        self.x0 = x0
        self.y0 = y0
        self.xscale = xscale
        self.yscale = yscale
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        print self.path, self.rank, self.parametric #, self.data.shape

    def save_in_file(self, file_or_group=None):
        if self.data is None:
            return
        if file_or_group is None:
            assert self.file is not None
            file_or_group = self.file
        if self.save:
            if self.path[-1] in file_or_group:
                del file_or_group[self.path[-1]]
            file_or_group[self.path[-1]] = self.data
            attrs = file_or_group[self.path[-1]].attrs
            attrs['rank'] = self.rank
            attrs['parametric'] = self.parametric
            print 'saving parametric', self.path, self.parametric
            attrs['x0'] = self.x0
            attrs['xscale'] = self.xscale
            attrs['xlabel'] = self.xlabel
            attrs['ylabel'] = self.ylabel
            if self.rank > 1:
                attrs['y0'] = self.y0
                attrs['yscale'] = self.yscale
                attrs['zlabel'] = self.zlabel

    def load_attrs_from_ds(self, dataset):
        for name in ['rank', 'parametric', 'x0', 'xscale', 'xlabel', 'ylabel', 'y0', 'yscale', 'zlabel']:
            try:
                setattr(self, name, dataset.attrs[name])
            except KeyError:
                pass

    def to_rank1(self, idx=-1):
        assert self.rank > 1
        return DataTreeLeaf(path=self.path, rank=1, data=self.data[idx, :])
