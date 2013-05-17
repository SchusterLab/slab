import time
import cStringIO
import traceback

# Module imports for py2exe's benefit
import h5py._errors
import h5py._proxy
import h5py._objects
import h5py._conv

from PyQt4 import Qt
from window import PlotWindow

def excepthook(excType, excValue, tracebackobj):
    """
    Global function to catch unhandled exceptions.

    @param excType exception type
    @param excValue exception value
    @param tracebackobj traceback object
    """
    separator = '-' * 80
    logFile = "datamanagement.log"
    notice = \
        """An unhandled exception occurred. Please report the problem\n""" \
        """A log has been written to "%s".\n\nError information:\n""" % (logFile)
    versionInfo="0.0.1"
    timeString = time.strftime("%Y-%m-%d, %H:%M:%S")


    tbinfofile = cStringIO.StringIO()
    traceback.print_tb(tracebackobj, None, tbinfofile)
    tbinfofile.seek(0)
    tbinfo = tbinfofile.read()
    errmsg = '%s: \n%s' % (str(excType), str(excValue))
    sections = [separator, timeString, separator, errmsg, separator, tbinfo]
    msg = '\n'.join(sections)
    try:
        f = open(logFile, "w")
        f.write(msg)
        f.write(versionInfo)
        f.close()
    except IOError:
        pass
    errorbox = Qt.QMessageBox()
    errorbox.setText(str(notice)+str(msg)+str(versionInfo))
    errorbox.exec_()


if __name__ == "__main__":
    import sys
    sys.excepthook = excepthook
    app = Qt.QApplication([])
    win = PlotWindow()
    win.show()
    win.showMaximized()
    app.connect(app, Qt.SIGNAL("lastWindowClosed()"), win, Qt.SIGNAL("lastWindowClosed()"))
    sys.exit(app.exec_())

##import sys
#from collections import defaultdict
#import traceback
#import time
#import cStringIO
#
#from PyQt4 import Qt
#import numpy as np
#import h5py
#import Pyro4
#import pyqtgraph
#import pyqtgraph.dockarea
#
#import dashboard_gui as gui
#from plot_widgets import CrossSectionWidget
#
#manager_id = 'manager'
#manager_host = 'localhost'
#manager_port = 5555
#manager_uri = "PYRO:%s@%s:%d" % (manager_id, manager_host, manager_port)
#RUNNING = True
#Pyro4.config.HMAC_KEY = '6551d449b0564585a9d39c0bd327dcf1'
#
#h5file_directory = r'S:\_Data'
#h5file_filter = 'HDF5 Files (*.h5)'
#
#
## http://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
#class keydefaultdict(defaultdict):
#    def __missing__(self, key):
#        if self.default_factory is None:
#            raise KeyError(key)
#        else:
#            ret = self[key] = self.default_factory(key)
#            return ret
#
#    def resolve_path(self, path):
#        res = self
#        for item in path:
#            res = res[item]
#        return res
#
#
#class DataTree(keydefaultdict):
#    def __init__(self, gui, name=None, parentpath=(), file=None): # TODO, optionally integrate file object to prevent opening and closing.
#        if name is None:
#            self.path = ()
#        else:
#            self.path = parentpath + (name,)
#            gui.add_tree_widget(self.path)
#        self.gui = gui
#        keydefaultdict.__init__(self, self.make_child)
#
#    def make_child(self, key):
#        if self.file is not None:
#            if key in self.file:
#                file = self.file[key]
#            else:
#                file = self.file.create_group(key)
#        else:
#            file = None
#        return DataTree(self.gui, name=key, parentpath=self.path, file=file)
#
#    def leaves(self):
#        for k, v in self.items():
#            if isinstance(v, DataTree):
#                for k2, v2 in v.leaves():
#                    yield (k,)+k2, v2
#            else:
#                yield (k,), v
#
#    def save_in_file(self, file_or_group):
#        for key, item in self.items():
#            print 'Key', key, item
#            if isinstance(item, DataTree):
#                print 'Group', key
#                if any(i.save for k, i in item.leaves()):
#                    print 'saving group'
#                    file_or_group.create_group(key)
#                    item.save_in_file(file_or_group[key])
#            elif isinstance(item, DataTreeLeaf):
#                if item.save:
#                    print 'Dset', key
#                    item.save_in_file(file_or_group)
#            else:
#                raise ValueError("Can't save unknown item " + str(type(item)))
#
#
#class DataTreeLeaf(object):
#    def __init__(self, path, rank=None, file=None, data=(), save=True, plot=True,
#                 x0=0, xscale=1, xlabel='x', y0=0, yscale=1, ylabel='y', zlabel='z'):
#        self.path = path
#        self.rank = rank
#        self.file = file
#        self.data = data
#        self.save = save
#        self.plot = plot
#        self.x0 = x0
#        self.y0 = y0
#        self.xscale = xscale
#        self.yscale = yscale
#        self.xlabel = xlabel
#        self.ylabel = ylabel
#        self.zlabel = zlabel
#
#    def save_in_file(self, file_or_group=None):
#        if file_or_group is None:
#            file_or_group = self.file
#        if self.save:
#            if self.path[-1] in file_or_group:
#                del file_or_group[self.path[-1]]
#            file_or_group[self.path[-1]] = self.data
#            attrs = file_or_group[self.path[-1]].attrs
#            attrs['rank'] = self.rank
#            attrs['x0'] = self.x0
#            attrs['xscale'] = self.xscale
#            attrs['xlabel'] = self.xlabel
#            attrs['ylabel'] = self.ylabel
#            if self.rank > 1:
#                attrs['y0'] = self.y0
#                attrs['yscale'] = self.yscale
#                attrs['zlabel'] = self.zlabel
#
#    def load_attrs_from_ds(self, dataset):
#        for name in ['rank', 'x0', 'xscale', 'xlabel', 'ylabel', 'y0', 'yscale', 'zlabel']:
#            try:
#                setattr(self, name, dataset.attrs[name])
#            except KeyError:
#                pass
#
#    def to_rank1(self, idx=-1):
#        assert self.rank > 1
#        return DataTreeLeaf(path=self.path, rank=1, data=self.data[:,idx])
#
#
#class DataManager(gui.BackgroundObject):
#    def __init__(self, path_delim='/'):
#        gui.BackgroundObject.__init__(self)
#        self.data = None
#        self.delim = path_delim
#
#    def connect_data(self):
#        self.data = DataTree(self.gui)
#
#    def get_leaf(self, path, rank=None, **initargs):
#        group = self.data.resolve_path(path[:-1])
#        if path[-1] not in group:
#            assert rank is not None
#            leaf = DataTreeLeaf(path, rank, file=group.file, **initargs)
#            group[path[-1]] = leaf
#            self.gui.add_tree_widget(path, data=True, save=leaf.save, plot=leaf.plot)
#            if group[path[-1]].plot:
#                params = {'x0': leaf.x0, 'xscale': leaf.xscale, 'xlabel': leaf.xlabel, 'ylabel': leaf.ylabel}
#                if rank > 1:
#                    params.update({'y0': leaf.y0, 'yscale': leaf.yscale, 'zlabel': leaf.zlabel})
#                self.gui.add_plot_widget(path, rank, **params)
#        else:
#            leaf = group[path[-1]]
#            assert (rank is None) or (rank == leaf.rank)
#            for key, val in initargs.items():
#                setattr(leaf, key, val)
#
#        if not isinstance(leaf, DataTreeLeaf):
#            raise ValueError('path does not store a leaf, but rather a ' + str(type(leaf)))
#        return leaf
#
#    def set_params(self, path, rank, **initargs):
#        self.get_leaf(path, rank, **initargs)
#        self.update_plot(path, refresh_labels=True)
#
#    def set_data(self, name_or_path, data, slice=None, **initargs):
#        path = canonicalize_path(name_or_path)
#        arrdata, rank = canonicalize_data(data, slice)
#        leaf = self.get_leaf(path, rank, **initargs)
#        refresh_labels = len(initargs) > 0
#        if slice is None:
#            leaf.data = arrdata
#        else:
#            leaf.data[slice] = arrdata
#        if leaf.plot:
#            self.update_plot(name_or_path, refresh_labels=refresh_labels)
#
#    def append_data(self, name_or_path, data, show_most_recent=None, **initargs):
#        path = canonicalize_path(name_or_path)
#        data, rank = canonicalize_append_data(data)
#        leaf = self.get_leaf(path, rank, **initargs)
#        refresh_labels = len(initargs) > 0
#        if show_most_recent is not None:
#            assert isinstance(show_most_recent, bool)
#            if show_most_recent:
#                self.gui.plot_widgets[path].show_recent()
#            else:
#                self.gui.plot_widgets[path].show_accumulated()
#        if rank == 1 and data[0] is None: # Set x value if none is given
#            if leaf.data is None:
#                data = leaf.x0, data[1]
#            else:
#                data = leaf.x0 + leaf.xscale*len(leaf.data), data[1]
#        # TODO: Canonicalize appended data
#        if isinstance(leaf.data, np.ndarray):
#            leaf.data = np.vstack((leaf.data, np.array(data)))
#        else:
#            leaf.data = np.array([data])
#        if leaf.plot:
#            self.update_plot(name_or_path, refresh_labels=refresh_labels)
#
#    def get_data(self, name_or_path, slice=None):
#        path = canonicalize_path(name_or_path)
#        item = self.data.resolve_path(path)
#        if slice is not None:
#            x = np.array(item.data)[slice]
#            return x
#        else:
#            return np.array(item.data)
#
#    def update_plot(self, name_or_path, refresh_labels=False):
#        path = canonicalize_path(name_or_path)
#        item = self.data.resolve_path(path)
#        if not item.plot:
#            return
#        tree_widget = self.gui.tree_widgets[path]
#        try:
#            tree_widget.setText(1, str(item.data.shape))
#        except AttributeError:
#            tree_widget.setText(1, 'No Data')
#        tree_widget.setText(2, str(item.save))
#        tree_widget.setText(3, str(item.plot))
#        self.gui.plot_widgets[path].update_plot(item, refresh_labels=refresh_labels)
#        self.gui.update_multiplots(path, item)
#
#    def save_all(self):
#        raise NotImplementedError
#
#    def load(self, filename):
#        with h5py.File(filename, 'r') as f:
#            self.rec_load_file(f, (filename,))
#
#    def rec_load_file(self, file, path):
#        for name, ds in file.items():
#            this_path = path + (name,)
#            self.msg(name, type(ds))
#            if isinstance(ds, h5py.Group):
#                self.rec_load_file(ds, this_path)
#            else:
#                self.set_data(this_path, ds)
#                self.get_leaf(this_path).load_attrs_from_ds(ds)
#                self.update_plot(this_path, refresh_labels=True)
#
#    def clear_data(self, path=None, leaf=None):
#        assert path is not None or leaf is not None
#        if leaf is None:
#            leaf = self.get_leaf(path)
#        if leaf.rank == 1:
#            leaf.data = None
#        elif leaf.rank == 2:
#            leaf.data = None
#        if leaf.plot:
#            self.update_plot(path)
#
#    def clear_all_data(self):
#        for path, leaf in self.data.leaves():
#            self.clear_data(leaf=leaf)
#
#    def serve(self):
#        print 'serving'
#        with Pyro4.Daemon(host=manager_host, port=manager_port) as d:
#            self.running = True
#            d.register(self, manager_id)
#            d.requestLoop(lambda: RUNNING)
#        print 'done serving'
#
#    def abort_daemon(self):
#        global RUNNING
#        RUNNING = False
#
#    def save_as_file(self, path): #TODO
#        data = self.data.resolve_path(path)
#        assert isinstance(data, DataTree)
#        assert valid_h5file(path[-1])
#        with h5py.File(path[-1], 'a') as f:
#            for item in data.values():
#                item.save_in_file(f)
#
#    def save_with_file(self, path, filename): #TODO
#        with h5py.File(filename, 'a') as h5file:
#            print 'swf', path, filename
#            data = self.data.resolve_path(path)
#            f = h5file
#            for p in path[:-1]:
#                if p not in f:
#                    f = f.create_group(p)
#                else:
#                    f = f[p]
#            data.save_in_file(f)
#
#    def msg(self, *args):
#        self.gui.msg(*args)
#
#
#class DataTreeLeafItem(Qt.QTreeWidgetItem):
#    @property
#    def path(self):
#        if self.parent() is None:
#            return (str(self.text(0)),)
#        else:
#            return self.parent().path + (str(self.text(0)),)
#
#    @property
#    def strpath(self):
#        return '/'.join(map(str, self.path))
#
#
#Qt.QTreeWidgetItem.is_leaf = lambda item: item.childCount() == 0
#Qt.QTreeWidgetItem.getChildren = lambda item: [item.child(i) for i in range(item.childCount())]
#
#
#class LeafEditWidget(Qt.QFrame):
#    def __init__(self, leaf):
#        Qt.QFrame.__init__(self)
#        self.setFrameStyle(Qt.QFrame.Panel)
#        self.path = leaf.path
#        self.spin_widgets = {}
#        self.text_widgets = {}
#
#        self.setLayout(Qt.QVBoxLayout())
#        self.layout().addWidget(Qt.QLabel('Editing ' + '/'.join(self.path)))
#        self.range_area = Qt.QWidget()
#        self.range_area.setLayout(Qt.QHBoxLayout())
#        self.layout().addWidget(self.range_area)
#        self.add_spin('x0', leaf)
#        self.add_spin('xscale', leaf)
#        if leaf.rank > 1:
#            self.add_spin('y0', leaf)
#            self.add_spin('yscale', leaf)
#        self.label_area = Qt.QWidget()
#        self.label_area.setLayout(Qt.QHBoxLayout())
#        self.layout().addWidget(self.label_area)
#        self.add_edit('xlabel', leaf)
#        self.add_edit('ylabel', leaf)
#        if leaf.rank > 1:
#            self.add_edit('zlabel', leaf)
#        self.commit_button = Qt.QPushButton('Commit')
#        self.layout().addWidget(self.commit_button)
#
#    def add_spin(self, name, leaf):
#        self.range_area.layout().addWidget(Qt.QLabel(name))
#        widget = pyqtgraph.SpinBox(value=getattr(leaf, name))
#        self.spin_widgets[name] = widget
#        self.range_area.layout().addWidget(widget)
#
#    def add_edit(self, name, leaf):
#        self.label_area.layout().addWidget(Qt.QLabel(name))
#        widget = Qt.QLineEdit(getattr(leaf, name))
#        self.text_widgets[name] = widget
#        self.label_area.layout().addWidget(widget)
#
#    def to_dict(self):
#        params = {name: spin.value() for name, spin in self.spin_widgets.items()}
#        params.update({name: edit.text() for name, edit in self.text_widgets.items()})
#        return params
#
#
#class PlotWindow(gui.SlabWindow):
#    def __init__(self):
#        manager = DataManager()
#        gui.SlabWindow.__init__(self, manager)
#        manager.connect_data()
#
#        self.structure_tree = Qt.QTreeWidget()
#        self.structure_tree.setColumnCount(4)
#        self.structure_tree.setHeaderLabels(['Name', 'Shape', 'Save?', 'Plot?'])
#        self.structure_tree.itemClicked.connect(self.change_edit_widget)
#        self.structure_tree.itemDoubleClicked.connect(self.toggle_item)
#        self.structure_tree.itemSelectionChanged.connect(self.configure_tree_buttons)
#        self.structure_tree.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)
#        #structure_tree_menu = Qt.QMenu(self.structure_tree)
#        #self.structure_tree.setContextMenuPolicy(Qt.Qt.ActionsContextMenu)
#        #change_scale_action = Qt.QAction('Change Labels/Scale')
#        #change_scale_action.triggered().connect(self.)
#        #structure_tree_menu.addAction(change_scale_action)
#
#
#        self.dock_area = pyqtgraph.dockarea.DockArea()
#        self.dock_insert_location = 'bottom'
#
#        self.save_button = Qt.QPushButton('Save Selection')
#        self.save_button.clicked.connect(self.save_selection)
#        self.save_button.setEnabled(False)
#        self.multiplot_button = Qt.QPushButton('Plot Multiple Items')
#        self.multiplot_button.clicked.connect(self.add_multiplot)
#        self.multiplot_button.setEnabled(False)
#        self.remove_button = Qt.QPushButton('Remove Selection')
#        self.remove_button.clicked.connect(self.remove_selection)
#        self.remove_button.setEnabled(False)
#
#        self.setCentralWidget(Qt.QSplitter())
#        self.sidebar = sidebar = Qt.QWidget()
#        sidebar.setLayout(Qt.QVBoxLayout())
#        sidebar.layout().addWidget(self.structure_tree)
#        sidebar.layout().addWidget(self.save_button)
#        sidebar.layout().addWidget(self.multiplot_button)
#        sidebar.layout().addWidget(self.remove_button)
#        self.centralWidget().addWidget(sidebar)
#        self.centralWidget().addWidget(self.dock_area)
#        self.centralWidget().setSizes([300, 1000])
#        self.current_edit_widget = None
#
#        file_menu = self.menuBar().addMenu('File')
#        file_menu.addAction('Save').triggered.connect(lambda: self.background_client.save_all())
#        file_menu.addAction('Load').triggered.connect(self.load)
#        file_menu.addAction('Clear').triggered.connect(lambda: self.background_client.clear_all_data())
#
#        self.message_box = Qt.QTextEdit()
#        self.message_box.setReadOnly(True)
#        self.message_box.setVisible(False)
#        self.centralWidget().addWidget(self.message_box)
#        debug_menu = self.menuBar().addMenu('Debug')
#        action = debug_menu.addAction('View Debug Panel')
#        action.setCheckable(True)
#        action.setChecked(False)
#        action.triggered.connect(self.message_box.setVisible)
#
#        def fail():
#            assert False
#        debug_menu.addAction('Fail').triggered.connect(fail)
#
#        self.plot_widgets = {}
#        self.tree_widgets = {}
#        self.multiplot_widgets = {}
#        self.multiplots = defaultdict(list)
#
#        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), lambda: self.background_client.abort_daemon())
#        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), self.background_thread.wait)
#
#        self.start_thread()
#        self.background.serve()
#        self.background_client = DataClient()
#
#    # TODO: BIG ONE, implement saving
#    def save_selection(self):
#        selection = self.structure_tree.selectedItems()
#        if len(selection) == 1 and valid_h5file(selection[0].text(0)):
#            self.background_client.save_as_file(selection[0].path)
#        else:
#            filename = str(Qt.QFileDialog.getSaveFileName(self, "Destination File", h5file_directory, h5file_filter))
#            for item in selection:
#                self.background_client.save_with_file(item.path, filename)
#
#    def add_multiplot(self):
#        selection = self.structure_tree.selectedItems()
#        name = '::'.join(item.strpath for item in selection)
#        widget = MultiplotItemWidget(name)
#        widget.remove_button.clicked.connect(lambda: self.remove_multiplot(name))
#        self.multiplot_widgets[name] = widget
#        for item in selection:
#            self.multiplots[item.strpath].append(widget)
#            self.background_client.update_plot(item.path)
#        self.dock_area.addDock(widget, self.dock_insert_location)
#        self.cycle_insert_location()
#
#    def remove_multiplot(self, name):
#        names = name.split('::')
#        widget = self.multiplot_widgets[name]
#        for n in names:
#            self.multiplots[n].remove(widget)
#        widget.setParent(None)
#
#    def update_multiplots(self, path, leaf):
#        for widget in self.multiplots['/'.join(path)]:
#            widget.update_plot(path, leaf)
#
#    def configure_tree_buttons(self):
#        selection = self.structure_tree.selectedItems()
#        save = len(selection) > 0
#        multiplot = (len(selection) > 1) and all(self.plot_widgets[i.path].rank == 1 for i in selection)
#        remove = len(selection) > 0
#        self.save_button.setEnabled(save)
#        self.multiplot_button.setEnabled(multiplot)
#        self.remove_button.setEnabled(remove)
#
#    def remove_selection(self): #TODO
#        pass
#
#    def load(self):
#        filename = str(Qt.QFileDialog().getOpenFileName(self, 'Load HDF5 file', h5file_directory, h5file_filter))
#        self.background_client.load(filename)
#
#    def add_tree_widget(self, path, data=False, shape=(), save=True, plot=True):
#        if data:
#            item = DataTreeLeafItem([path[-1], str(shape), str(save), str(plot)])
#        else:
#            item = DataTreeLeafItem([path[-1]])
#
#        if len(path[:-1]) > 0:
#            parent = self.tree_widgets[path[:-1]]
#            parent.addChild(item)
#            parent.setExpanded(True)
#        else:
#            self.structure_tree.addTopLevelItem(item)
#
#        if not data:
#            item.setFirstColumnSpanned(True)
#
#        self.tree_widgets[path] = item
#
#    def toggle_item(self, item, col):
#        if item.is_leaf():# and item.plot:
#            self.plot_widgets[item.path].toggle_hide()
#        else:
#            for child in item.getChildren():
#                self.toggle_item(child, col)
#
#    def change_edit_widget(self, item, col):
#        if item.is_leaf():
#            if self.current_edit_widget is not None:
#                self.sidebar.layout().removeWidget(self.current_edit_widget)
#                self.current_edit_widget.setParent(None)
#            leaf = self.background_client.get_leaf(item.path)
#            self.current_edit_widget = LeafEditWidget(leaf)
#            self.sidebar.layout().addWidget(self.current_edit_widget)
#
#            def update_fn():
#                params = self.current_edit_widget.to_dict()
#                self.background_client.set_params(leaf.path, leaf.rank, **params)
#            self.current_edit_widget.commit_button.clicked.connect(update_fn)
#
#    def add_plot_widget(self, path, rank=1, **kwargs):
#        if path in self.plot_widgets:
#            raise ValueError('Plot %s already exists in window' % (path,))
#        strpath = "/".join(path)
#        if rank == 1:
#            item = Rank1ItemWidget(strpath, **kwargs)
#        elif rank == 2:
#            item = Rank2ItemWidget(strpath, **kwargs)
#        else:
#            raise ValueError('Rank must be either 1 or 2, not ' + str(rank))
#
#        item.clear_button.clicked.connect(lambda: self.background_client.clear_data(path))
#        self.register_param('update'+strpath, item.update_toggle)
#        self.plot_widgets[path] = item
#        self.dock_area.addDock(item, self.dock_insert_location)
#        self.cycle_insert_location()
#
#    def cycle_insert_location(self):
#        self.dock_insert_location = \
#            {'bottom': 'right', 'right': 'bottom'}[self.dock_insert_location]
#
#    def msg(self, *args):
#        self.message_box.append(', '.join(map(str, args)))
#
#
#class ItemWidget(pyqtgraph.dockarea.Dock):
#    def __init__(self, ident, **kwargs):
#        if len(ident) > 25:
#            name = ident[:5].strip() + ' ... ' + ident[-15:]
#        else:
#            name = ident
#        pyqtgraph.dockarea.Dock.__init__(self, name)
#        self.label.setFont(Qt.QFont('Helvetica', pointSize=16))
#        self.ident = ident
#        self.add_plot_widget(**kwargs)
#        self.buttons_widget = Qt.QWidget() #QHBoxWidget()
#        self.buttons_widget.setLayout(Qt.QHBoxLayout())
#        self.remove_button = Qt.QPushButton('Hide')
#        self.remove_button.clicked.connect(self.toggle_hide)
#        self.clear_button = Qt.QPushButton('Clear') # Connected to manager action
#        self.update_toggle = Qt.QCheckBox('Update')
#        self.update_toggle.setChecked(True)
#        self.autoscale_toggle = Qt.QCheckBox('AutoScale')
#        self.autoscale_toggle.setChecked(True)
#        self.buttons_widget.layout().addWidget(self.remove_button)
#        self.buttons_widget.layout().addWidget(self.clear_button)
#        self.buttons_widget.layout().addWidget(self.update_toggle)
#        self.buttons_widget.layout().addWidget(self.autoscale_toggle)
#        self.addWidget(self.buttons_widget)
#
#    def update_params(self, **kwargs):
#        self.__dict__.update(kwargs)
#
#    def toggle_hide(self):
#        self.setVisible(not(self.isVisible()))
#
#    def add_plot_widget(self, **kwargs):
#        raise NotImplementedError
#
#    def update_plot(self, leaf):
#        raise NotImplementedError
#
#    def clear_plot(self):
#        raise NotImplementedError
#
#
#class Rank1ItemWidget(ItemWidget):
#    def __init__(self, ident, **kwargs):
#        ItemWidget.__init__(self, ident, **kwargs)
#        self.rank = 1
#
#    def add_plot_widget(self, **kwargs):
#        #self.line_plt = pyqtgraph.PlotWidget(title=self.ident, **kwargs)
#        self.line_plt = pyqtgraph.PlotWidget(**kwargs)
#        self.addWidget(self.line_plt)
#        self.curve = None
#
#    def update_plot(self, leaf, refresh_labels=False):
#        if self.update_toggle.isChecked():
#            if leaf is None or leaf.data is None:
#                self.clear_plot()
#                return
#            elif refresh_labels:
#                self.line_plt.plotItem.setLabels(bottom=(leaf.xlabel,), left=(leaf.ylabel,))
#                try:
#                    assert leaf.data.shape[1] == 2
#                    _, ydata = leaf.data.T
#                    type = 'A'
#                except:
#                    ydata = leaf.data
#                    type = 'B'
#                xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)
#            else:
#                try:
#                    assert leaf.data.shape[1] == 2
#                    xdata, ydata = leaf.data.T
#                    type = 'C'
#                except:
#                    ydata = leaf.data
#                    xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)
#                    type = 'D'
#            if self.curve is None:
#                self.curve = self.line_plt.plot(xdata, ydata)
#            else:
#                try:
#                    self.curve.setData(x=xdata, y=ydata)
#                except:
#                    print 'update failed', xdata, ydata, leaf.path, type
#                    raise
#
#    def clear_plot(self):
#        if self.curve is not None:
#            self.line_plt.removeItem(self.curve)
#            self.curve = None
#
#
#class MultiplotItemWidget(Rank1ItemWidget):
#    def add_plot_widget(self, **kwargs):
#        Rank1ItemWidget.add_plot_widget(self, **kwargs)
#        self.curves = defaultdict(lambda: self.line_plt.plot([], pen=tuple(random_color())))
#
#    def update_plot(self, path, leaf):
#        self.curve = self.curves[path]
#        Rank1ItemWidget.update_plot(self, leaf)
#
#
#class Rank2ItemWidget(Rank1ItemWidget):
#    def __init__(self, ident, **kwargs):
#        Rank1ItemWidget.__init__(self, ident, **kwargs)
#        self.rank = 2
#
#        self.histogram_check = Qt.QCheckBox('Histogram')
#        self.histogram_check.stateChanged.connect(self.img_view.set_histogram)
#        self.recent_button = Qt.QPushButton('Most Recent Trace')
#        self.recent_button.clicked.connect(self.show_recent)
#        self.accum_button = Qt.QPushButton('Accumulated Traces')
#        self.accum_button.clicked.connect(self.show_accumulated)
#        self.accum_button.hide()
#        self.buttons_widget.layout().addWidget(self.histogram_check)
#        self.buttons_widget.layout().addWidget(self.recent_button)
#        self.buttons_widget.layout().addWidget(self.accum_button)
#
#    def add_plot_widget(self, **kwargs):
#        Rank1ItemWidget.add_plot_widget(self)
#        self.line_plt.hide() # By default, start out with accum view
#
#        # plt/view explanation here
#        # https://groups.google.com/d/msg/pyqtgraph/ccrYl1yyasw/fD9tLrco1PYJ
#        # TODO: Identify and separate 1d kwargs and 2d kwargs
#        #self.img_plt = pyqtgraph.PlotItem(title=self.ident)
#        #self.img_view = pyqtgraph.ImageView(view=self.img_plt)
#        self.img_view = CrossSectionWidget()
#        self.addWidget(self.img_view)
#
#        #self.line_plt = pyqtgraph.PlotWidget(parent=self, title=self.ident)
#        self.addWidget(self.line_plt)
#        self.curve = None
#
#    def update_plot(self, leaf, refresh_labels=False):
#        if leaf is None or leaf.data is None:
#            self.clear_plot()
#            return
#        if self.update_toggle.isChecked():
#            if refresh_labels:
#                self.img_view.setLabels(leaf.xlabel, leaf.ylabel, leaf.zlabel)
#            Rank1ItemWidget.update_plot(self, leaf.to_rank1())
#            self.img_view.imageItem.show()
#            self.img_view.setImage(leaf.data, pos=[leaf.x0, leaf.y0], scale=[leaf.xscale, leaf.yscale])
#
#    def clear_plot(self):
#        Rank1ItemWidget.clear_plot(self)
#        #Rank1ItemWidget.update_plot(self, None)
#        #self.img_view.setImage(np.array([[]]))
#        self.img_view.imageItem.hide()
#
#    def show_recent(self):
#        self.img_view.hide()
#        self.line_plt.show()
#        self.recent_button.hide()
#        self.accum_button.show()
#
#    def show_accumulated(self):
#        self.img_view.show()
#        self.line_plt.hide()
#        self.recent_button.show()
#        self.accum_button.hide()
#
#
#class DataClient(object):
#    def __init__(self):
#        self.proxy = Pyro4.Proxy(manager_uri)
#        self.proxy_closed = False
#
#    def __getattr__(self, item):
#        def try_proxy_fn(*args, **kwargs):
#            if self.proxy_closed:
#                return 'proxy closed'
#            try:
#                return getattr(self.proxy, item)(*args, **kwargs)
#            except Pyro4.errors.ConnectionClosedError:
#                print "PROXY CLOSED"
#                self.proxy_closed = True
#                return 'proxy closed'
#            except:
#                self.proxy.msg('failure on ' + method_string(item, args, kwargs))
#                self.proxy.msg(traceback.format_exc())
#                raise
#        return try_proxy_fn
#
#
#def SlabFile(*args, **kwargs):
#    if kwargs.pop('remote', False):
#        return SlabFileRemote(*args, **kwargs)
#    return SlabFileLocal(*args, **kwargs)
#
#
#class SlabFileRemote:
#    def __init__(self, filename=None, manager=None, context=(), autosave=False, **ignoredargs):
#        self.autosave = autosave
#        if len(ignoredargs) > 0:
#            print 'Warning', ', '.join(ignoredargs.keys()), 'ignored in call to SlabFileRemote'
#        if manager is None:
#            self.manager = DataClient()
#        else:
#            self.manager = manager
#        self.filename = filename
#        if context == () and filename is not None:
#            self.context = (filename,)
#        else:
#            self.context = context
#
#    def close(self):
#        self.manager._pyroRelease()
#
#    def create_dataset(self, name, shape=None, data=None, **initargs):
#        if data is None:
#            if shape is None:
#                data = np.array([0])
#            else:
#                data = np.zeros(shape)
#        self.manager.set_data(self.context + (name,), data, **initargs)
#        if self.autosave:
#            self.flush(self.context + (name,))
#
#    def set_range(self, x0=0, y0=None, xscale=1, yscale=None):
#        if y0 is None and yscale is None:
#            rank = 1
#            self.manager.set_params(self.context, rank, x0=x0, xscale=xscale)
#        else:
#            assert None not in (y0, yscale)
#            rank = 2
#            self.manager.set_params(self.context, rank, x0=x0, y0=y0, xscale=xscale, yscale=yscale)
#        #if self.autosave:
#        #    self.flush(self.context)
#
#    def set_labels(self, xlabel="", ylabel="", zlabel=None):
#        if zlabel is None:
#            self.manager.set_params(self.context, 1, xlabel=xlabel, ylabel=ylabel)
#        else:
#            self.manager.set_params(self.context, 2, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
#        #if self.autosave:
#        #    self.flush(self.context)
#
#    # No need to create groups with remote file, as their existence is inferred,
#    # but it shouldn't cause an error either
#    def create_group(self, name):
#        pass
#
#    def __getitem__(self, key):
#        if isinstance(key, str):
#            return SlabFileRemote(self.filename, self.manager, self.context + (key,), self.autosave)
#        elif isinstance(key, int):
#            s = slice(key, key+1, None)
#            return self.manager.get_data(self.context + (key,), slice=s)[0]
#        elif isinstance(key, slice):
#            return self.manager.get_data(self.context, key)
#
#    def __setitem__(self, key, value):
#        if isinstance(key, str):
#            context = self.context + (key,)
#            self.manager.set_data(context, value)
#        elif isinstance(key, int):
#            context = self.context
#            s = slice(key, key+1, None)
#            self.manager.set_data(self.context, value, slice=s)
#        elif isinstance(key, slice):
#            context = self.context
#            self.manager.get_data(self.context, value, slice=key)
#        if self.autosave:
#            self.flush(context)
#
#    def __delitem__(self, key): # TODO
#        pass
#
#    def __array__(self):
#        return np.array(self[:])
#
#    def append(self, new_data, show_most_recent=None):
#        self.manager.append_data(self.context, new_data, show_most_recent=show_most_recent)
#        if self.autosave:
#            self.flush(self.context)
#
#    def flush(self, path=None):
#        if path is None:
#            self.manager.save_as_file((self.filename,))
#        else:
#            self.manager.save_with_file(path, self.filename)
#
#
#class SlabFileLocal(h5py.File):
#    pass
#
#
#def create_dataset(file_or_group, name, show=True, plot=True, **kwargs):
#    assert show or plot
#    h5py.File.create_dataset(file_or_group, name, **kwargs)
#
#
#def valid_h5file(text): #TODO
#    return True
#
#
#def random_color(base=128):
#    'A whitish random color. Adjust whiteness up by increasing base'
#    return np.random.randint(base, 255, 3)
#
#
#def method_string(name, args, kwargs):
#    argstr = ",".join(map(str, args))
#    kwargstr = ",".join(k+'='+str(i) for k, i in kwargs.items())
#    if kwargstr != "":
#        argstr += ","
#    return name + "(" + argstr + kwargstr + ")"
#
#
#def add_x_data(arr, slice=None):
#    if slice is None:
#        return np.vstack((np.arange(len(arr)), arr)).T
#    a = np.vstack((np.arange(slice.start, slice.stop), arr)).T
#    return a
#
#
#def canonicalize_data(data, slice=None):
#    arr = np.array(data)
#    parametric = False
#    if len(arr.shape) == 1:
#        arr = add_x_data(arr, slice)
#        rank = 1
#    elif len(arr.shape) == 2:
#        if arr.shape[0] == 1:
#            arr = add_x_data(arr[0,:], slice)
#            rank = 1
#        elif arr.shape[1] == 1:
#            arr = add_x_data(arr[:,0], slice)
#            rank = 1
#        elif arr.shape[0] == 2:
#            arr = arr.T
#            rank = 1
#            parametric = True
#        elif arr.shape[1] == 2:
#            rank = 1
#        else:
#            rank = 2
#    else:
#        raise NotImplementedError
#    return arr, rank#, parametric
#
#
#def canonicalize_append_data(data):
#    if isinstance(data, (int, float)):
#        return np.array((None, data)), 1
#    data = np.array(data)
#    if len(data) == 2:
#        return data, 1
#    if len(data.shape) == 1:
#        return data, 2
#    if len(data.shape) == 2: # Rank 3
#        raise NotImplementedError
#    raise ValueError
#
#
#def canonicalize_path(name_or_path):
#    if isinstance(name_or_path, str):
#        path = (name_or_path,)
#    else:
#        path = tuple(name_or_path)
#    return path
#
#
## patch Dataset to have append method, which resizes and writes in one step
#def append(dataset, new_data, axis=0):
#    shape = list(dataset.shape)
#    shape[axis] += 1
#    dataset.resize(shape)
#    # TODO: This isn't very general
#    if axis == 0:
#        dataset[-1, :] = new_data
#    else:
#        dataset[:, -1] = new_data
#    #self.flush()
#h5py.Dataset.append = append
#
#
# http://www.riverbankcomputing.com/pipermail/pyqt/2009-May/022961.html
