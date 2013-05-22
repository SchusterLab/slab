import types
import time
from collections import defaultdict

from PyQt4 import Qt
#from PyQt4.QtTest import QTest
#import pyqtgraph

from controller import DataManager
from interface import DataClient
from widgets import *
import helpers
import config

class GenericProxy(object):
    """
    Turns proxy.a['b'].c[1](d, e='f') into
    emitter.emit(sig_name, [('prop','a'),'b','c',1], d, {'e':'f'})

    Warning, not capable of returning values!
    """
    def __init__(self, sig_name,  emitter, context=[]):
        self._emitter = emitter
        self._sig_name = sig_name
        self._context = context

    def __getattr__(self, attr):
        return GenericProxy(self._sig_name, self._emitter, self._context + [('attr', attr)])

    def __getitem__(self, item):
        return GenericProxy(self._sig_name, self._emitter, self._context + [('item', item)])

    def __setitem__(self, item, value):
        return GenericProxy(self._sig_name, self._emitter, self._context + [('setitem', (item, value))])

    def __call__(self, *args, **kwargs):
        self._emitter.emit(Qt.SIGNAL(self._sig_name), self._context, args, kwargs)


def proxify(obj, connector, sig_name):
    emitter_obj = Qt.QObject()
    proxy = GenericProxy(sig_name, emitter_obj)
    def run_from_proxy_fn(self, context, args, kwargs):
        current = self
        for operation, arg in context:
            if operation == 'attr':
                current = getattr(current, arg)
            elif operation == 'item':
                try:
                    current = current[arg]
                except:
                    print context, current
                    raise
            elif operation == 'setitem':
                item, value = arg
                current[item] = value
            else:
                raise ValueError('Unidentified operation ' + str(operation))
        current(*args, **kwargs)
    obj._run_from_proxy = types.MethodType(run_from_proxy_fn, obj, type(obj))
    def conn_fn():
        connector.connect(proxy._emitter, Qt.SIGNAL(sig_name),
                          obj._run_from_proxy)

    return proxy, conn_fn

class TempDataClient(object):
    def __getattr__(self, item):
        def call_on_data_client(*args, **kwargs):
            with DataClient() as c:
                return getattr(c, item)(*args, **kwargs)
        return call_on_data_client

class SlabWindow(Qt.QMainWindow):
    def __init__(self, background_obj):
        Qt.QMainWindow.__init__(self)
        self.background_obj = background_obj
        self.background_obj.gui, connect_gui = proxify(self, self, 'gui')
        connect_gui()

        if hasattr(self, 'setupUi'):
            self.setupUi(self)

        self.background_thread = Qt.QThread()
        self.background, connect_background = proxify(self.background_obj, self, 'background')
        self.background_obj.moveToThread(self.background_thread)
        connect_background()
        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), self.background_thread.exit)

    def register_param(self, name, widget):
        for klass, actions in widget_tools.items():
            if isinstance(widget, klass):
                if isinstance(actions["read"], str):
                    read_action = actions["read"]
                    read_fn = lambda: \
                        self.background_obj.set_param(name, getattr(widget, read_action)())
                else:
                    read_fn = actions["read"]
                getattr(widget, actions["change_signal"]).connect(read_fn)
                read_fn() # Call it to initialize

    def start_thread(self):
        self.background_thread.start()


class PlotWindow(SlabWindow):
    def __init__(self):
        manager = DataManager()
        SlabWindow.__init__(self, manager)
        manager._connect_data()
        self.background_client = TempDataClient() # Don't actually call this until __init__ has returned!

        self.structure_tree = Qt.QTreeWidget()
        self.structure_tree.setColumnCount(4)
        self.structure_tree.setHeaderLabels(['Name', 'Shape', 'Save?', 'Plot?'])
        self.structure_tree.itemClicked.connect(self.change_edit_widget)
        self.structure_tree.itemDoubleClicked.connect(self.toggle_item)
        self.structure_tree.itemSelectionChanged.connect(self.configure_tree_buttons)
        self.structure_tree.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)
        self.structure_tree.setColumnWidth(0, 200)
        self.structure_tree.setColumnWidth(1, 50)
        self.structure_tree.setColumnWidth(2, 50)
        self.structure_tree.setColumnWidth(3, 50)

        #structure_tree_menu = Qt.QMenu(self.structure_tree)
        #self.structure_tree.setContextMenuPolicy(Qt.Qt.ActionsContextMenu)
        #change_scale_action = Qt.QAction('Change Labels/Scale')
        #change_scale_action.triggered().connect(self.)
        #structure_tree_menu.addAction(change_scale_action)


        #self.dock_area = pyqtgraph.dockarea.DockArea()
        self.dock_area = MyDockArea()
        #self.dock_insert_location = 'bottom'

        self.max_plots_spinner = Qt.QSpinBox()
        self.max_plots_spinner.setValue(6)
        max_plots_widget = Qt.QWidget()
        max_plots_widget.setLayout(Qt.QHBoxLayout())
        max_plots_widget.layout().addWidget(Qt.QLabel('Maximum Plot Count'))
        max_plots_widget.layout().addWidget(self.max_plots_spinner)
        self.save_button = Qt.QPushButton('Save Selection')
        self.save_button.clicked.connect(self.save_selection)
        self.save_button.setEnabled(False)
        self.multiplot_button = Qt.QPushButton('Plot Multiple Items')
        self.multiplot_button.clicked.connect(self.add_multiplot)
        self.multiplot_button.setEnabled(False)
        self.remove_button = Qt.QPushButton('Remove Selection')
        self.remove_button.clicked.connect(self.remove_selection)
        self.remove_button.setEnabled(False)
        self.parametric_button = Qt.QPushButton('Plot Pair Parametrically')
        self.parametric_button.clicked.connect(lambda: self.add_multiplot(parametric=True))
        self.parametric_button.setEnabled(False)

        self.setCentralWidget(Qt.QSplitter())
        self.sidebar = sidebar = Qt.QWidget()
        sidebar.setLayout(Qt.QVBoxLayout())
        sidebar.layout().addWidget(max_plots_widget)
        sidebar.layout().addWidget(self.structure_tree)
        sidebar.layout().addWidget(self.save_button)
        sidebar.layout().addWidget(self.multiplot_button)
        sidebar.layout().addWidget(self.remove_button)
        sidebar.layout().addWidget(self.parametric_button)
        self.centralWidget().addWidget(sidebar)
        self.centralWidget().addWidget(self.dock_area)
        self.centralWidget().setSizes([300, 1000])
        self.current_edit_widget = None

        file_menu = self.menuBar().addMenu('File')
        file_menu.addAction('Save').triggered.connect(self.background_client.save_all)
        file_menu.addAction('Load').triggered.connect(self.load)
        file_menu.addAction('Load (readonly)').triggered.connect(lambda: self.load(readonly=True))
        file_menu.addAction('Clear').triggered.connect(self.background_client.clear_all)

        self.message_box = Qt.QTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setVisible(False)
        self.centralWidget().addWidget(self.message_box)
        debug_menu = self.menuBar().addMenu('Debug')
        action = debug_menu.addAction('View Debug Panel')
        action.setCheckable(True)
        action.setChecked(False)
        action.triggered.connect(self.message_box.setVisible)

        self.plot_widgets = {}
        self.plot_widgets_update_log = {}
        self.tree_widgets = {}
        self.multiplot_widgets = {}
        self.parametric_widgets = {}
        self.multiplots = defaultdict(list)

        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), lambda: self.background_client.abort_daemon())
        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), self.background_thread.wait)

        self.start_thread()
        self.background.serve()

    def save_selection(self):
        selection = self.structure_tree.selectedItems()
        if len(selection) == 1 and helpers.valid_h5file(selection[0].text(0)):
            self.background_client.save_as_file(selection[0].path)
        else:
            filename = str(Qt.QFileDialog.getSaveFileName(self, "Destination File",
                                                          config.h5file_directory, config.h5file_filter))
            if not filename:
                return
            for item in selection:
                self.background_client.save_with_file(item.path, filename)

    def add_multiplot(self, parametric=False):
        selection = self.structure_tree.selectedItems()
        paths = tuple(item.strpath for item in selection)
        if parametric:
            widget = ParametricItemWidget(selection[0].path, selection[1].path, self.dock_area)
            self.parametric_widgets[paths] = widget
        else:
            widget = MultiplotItemWidget('::'.join(paths), self.dock_area)
            self.multiplot_widgets[paths] = widget
        widget.remove_button.clicked.connect(lambda: self.remove_multiplot(paths, parametric=parametric))
        for item in selection:
            self.multiplots[item.strpath].append(widget)
            self.background_client.update_plot(item.path)
        self.regulate_plot_count()

    def remove_multiplot(self, paths, parametric=False):
        if parametric:
            widget = self.parametric_widgets.pop(paths)
        else:
            widget = self.multiplot_widgets.pop(paths)
        for n in paths:
            self.multiplots[n].remove(widget)
        widget.setParent(None)

    def update_multiplots(self, path, leaf):
        for widget in self.multiplots['/'.join(path)]:
            widget.update_plot(path, leaf)

    def configure_tree_buttons(self):
        selection = self.structure_tree.selectedItems()
        save = len(selection) > 0
        multiplot = len(selection) > 1
        multiplot = multiplot and all(i.is_leaf() for i in selection)
        multiplot = multiplot and all(self.plot_widgets[i.path].rank == 1 for i in selection)
        remove = len(selection) > 0
        parametric = len(selection) == 2
        parametric = parametric and all(i.is_leaf() for i in selection)
        parametric = parametric and all(self.plot_widgets[i.path].rank == 1 for i in selection)
        self.save_button.setEnabled(save)
        self.multiplot_button.setEnabled(multiplot)
        self.remove_button.setEnabled(remove)
        self.parametric_button.setEnabled(parametric)

    #def make_parametric_plot(self):
    #    selection = self.structure_tree.selectedItems()
    #    xpath, ypath = (i.path for i in selection)
    #    widget = ParametricItemWidget(xpath, ypath, self.dock_area)
    #    widget.remove_button.clicked.connect(lambda: self.)


    def remove_selection(self): #TODO
        pass

    def load(self, readonly=False):
        filename = str(Qt.QFileDialog().getOpenFileName(self, 'Load HDF5 file',
                                                        config.h5file_directory, config.h5file_filter))
        if not filename:
            return
        self.background_client.load_h5file(filename, readonly=readonly)

    def add_tree_widget(self, path, data=False, shape=(), save=True, plot=True):
        if data:
            item = DataTreeLeafItem([path[-1], str(shape), str(save), str(plot)])
        else:
            item = DataTreeLeafItem([path[-1]])

        if len(path[:-1]) > 0:
            parent = self.tree_widgets[path[:-1]]
            parent.addChild(item)
            parent.setExpanded(True)
        else:
            self.structure_tree.addTopLevelItem(item)

        if not data:
            item.setFirstColumnSpanned(True)

        self.tree_widgets[path] = item

    def toggle_item(self, item, col):
        if item.is_leaf():# and item.plot:
            widget = self.plot_widgets[item.path]
            widget.toggle_hide()
            self.background_client.set_params(item.path, widget.rank, plot=widget.visible)
            print 'toggled', item.path
        else:
            for child in item.getChildren():
                self.toggle_item(child, col)

    def change_edit_widget(self, item, col):
        if item.is_leaf():
            if self.current_edit_widget is not None:
                self.sidebar.layout().removeWidget(self.current_edit_widget)
                self.current_edit_widget.setParent(None)
            leaf = self.background_client.get_or_make_leaf(item.path, reduced=True)
            self.current_edit_widget = LeafEditWidget(leaf)
            self.sidebar.layout().addWidget(self.current_edit_widget)

            def update_fn():
                params = self.current_edit_widget.to_dict()
                self.background_client.set_params(leaf.path, leaf.rank, **params)
            self.current_edit_widget.commit_button.clicked.connect(update_fn)

    def add_plot_widget(self, path, rank=1, **kwargs):
        if path in self.plot_widgets:
            raise ValueError('Plot %s already exists in window' % (path,))
        strpath = "/".join(path)
        if rank == 1:
            item = Rank1ItemWidget(strpath, self.dock_area, **kwargs)
        elif rank == 2:
            item = Rank2ItemWidget(strpath, self.dock_area, **kwargs)
        else:
            raise ValueError('Rank must be either 1 or 2, not ' + str(rank))

        item.clear_button.clicked.connect(lambda: self.background_client.clear_data(path))
        item.remove_button.clicked.connect(lambda: self.background_client.set_params(path, rank, plot=False))
        self.register_param('update'+strpath, item.update_toggle)
        self.plot_widgets[path] = item
        self.plot_widgets_update_log[path] = time.time()
        self.regulate_plot_count()

    def regulate_plot_count(self):
        widgets = list(self.plot_widgets.values()) + list(self.multiplot_widgets.values()) + list(self.parametric_widgets.values())
        if len(filter(lambda w: w.visible, widgets)) > self.max_plots_spinner.value():
            for p, t in sorted(self.plot_widgets_update_log.items(), key=lambda x: x[1]):
                if self.plot_widgets[p].visible:
                    self.toggle_item(self.tree_widgets[p], 0)
                    break
                #widget = self.plot_widgets[p]
                #if widget.visible:
                #    widget.toggle_hide()
                #    break

    def _test_edit_widget(self, path):
        self.structure_tree.itemClicked.emit(self.tree_widgets[path], 0)
        self.current_edit_widget.commit_button.clicked.emit(False)

    def _test_show_hide(self, path):
        self.structure_tree.itemDoubleClicked.emit(self.tree_widgets[path], 0)
        time.sleep(1)
        self.structure_tree.itemDoubleClicked.emit(self.tree_widgets[path], 0)

    def _test_multiplot(self, paths, parametric=False):
        for p in paths:
            self.structure_tree.setItemSelected(self.tree_widgets[p], True)
        self.add_multiplot(parametric=parametric)

    def _test_save_selection(self, paths):
        for p in paths:
            self.structure_tree.setItemSelected(self.tree_widgets[p], True)
        self.save_selection()


    def msg(self, *args):
        self.message_box.append(', '.join(map(str, args)))

widget_tools = {
    Qt.QSpinBox : {
        "read" : "value",
        "change_signal" : "valueChanged"},
    Qt.QDoubleSpinBox : {
        "read" : "value",
        "change_signal" : "valueChanged"},
    Qt.QLineEdit : {
        "read" : "text",
        "change_signal" : "textEdited"},
    Qt.QButtonGroup : {
        "read" : "checkedId",
        "change_signal" : "buttonClicked"},
    Qt.QCheckBox : {
        "read" : "isChecked",
        "change_signal" : "stateChanged"},
    Qt.QSlider : {
        "read" : "value",
        "change_signal" : "valueChanged"},
    }
