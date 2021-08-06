import h5py
import PyQt4.Qt as Qt
import numpy as np
from guiqwt.image import *
from guiqwt.plot import ImageWidget, CurveWidget
from guiqwt.qtdesigner import loadui
from guiqwt.builder import make
from spyderlib.widgets.internalshell import InternalShell
from slab import gui
from collections import defaultdict, namedtuple
from copy import copy
import os, glob, sys
from . import syntax
UiClass = loadui(__file__.split(".")[0] + ".ui")

h5py_types = { h5py.File : "File",
               h5py.Group : "Group",
               h5py.Dataset: "Dataset" }

def add_dataset(h5file, dset, name,
                x_range=None, y_range=None,
                x_lab=None, y_lab=None, z_lab=None):
    h5file[name] = dset
    shape = dset.shape
    if len(shape) == 2:
        if x_range and y_range:
            h5file[name].attrs["_axes"] = (x_range, y_range)
        if x_lab and y_lab and z_lab:
            h5file[name].attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
    elif len(shape) == 1:
        if x_range:
            h5file[name].attrs["_axes"] = x_range
        if x_lab and y_lab:
            h5file[name].attrs["_axes_labels"] = (x_lab, y_lab)

class Tree(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Tree)
        self.meta = {}

class DataFileListItem(Qt.QListWidgetItem):
    def __init__(self, name, filename):
        Qt.QListWidgetItem.__init__(self)
        self.setText(name)
        self.name = name
        self.filename = filename

class DataSetTreeItem(Qt.QTreeWidgetItem):
    def __init__(self, name, _type):
        Qt.QTreeWidgetItem.__init__(self)
        self.setText(0, name)
        self.setText(1, _type)
        self.name = name
        self.path = [name]
        self.val = None

    def addChild(self, child):
        Qt.QTreeWidgetItem.addChild(self, child)
        child.path = copy(self.path)
        child.path.append(child.name)

class HDFViewThread(gui.DataThread):
    def set_datapath(self):
        self.msg("set_datapath")
        directory = self.params["datapath"]
        if not os.path.exists(directory):
            self.msg("No such directory")
        if directory[-1] != "/":
            directory += "/"
        self.h5files = sorted(glob.glob(directory + "*.h5"))
        # Write filenames to gui
        self.gui["datafiles_listWidget"].clear()
        for filename in self.h5files:
            name = os.path.basename(filename)
            item = DataFileListItem(name, filename)
            self.gui["datafiles_listWidget"].addItem(item)

    def load_file(self, item):
        self.msg("load_file", item.filename)
        filename = item.filename
        self.open_file = h5py.File(filename, 'r+')

        # Build intermediate representation of file in dsettree
        self.dsettree = Tree()
        # Visits nodes recursively, calling add_item on the key-value pair
        self.open_file.visititems(self.add_item)
        # take tree and build
        self.process_items()
        self.gui["datasets_treeWidget"].clear()
        for item in self.tree_items:
            self.gui["datasets_treeWidget"].addTopLevelItem(item)

        # Load script
        try:
            self.gui["script_textEdit"].setText(
                Qt.QString(self.open_file.attrs["_script"]))
        except:
            self.gui["script_textEdit"].setText("")

        # Load file into interpreter
        #self.gui["shell"].exit_interpreter()
        #ns = { 'f':self.open_file, 'np':np, 'h5py':h5py }
        #self.gui["shell"].start_interpreter(namespace=ns)

    def load_dset(self, item, column):
        dset = self.open_file
        for name in item.path:
            dset = dset[name]
        assert(isinstance(dset, h5py.Dataset))
        if len(dset.shape) == 1:
            self.line_plot(dset)
        elif len(dset.shape) == 2:
            if dset.shape[0] == 1:
                self.line_plot(dset[0,:])
            else:
                self.image_plot(dset)
        else:
            raise NotImplementedError
        
    def line_plot(self, dset, add_to_plot=False):
        self.msg("line plot")
        self.gui['plots_tabWidget'].setCurrentIndex(1)
        line_plot = self.gui['line_plot']
        if not add_to_plot:
            line_plot.del_all_items()
        try:
            xdata = np.array(dset.dims[0][0])
        except Exception as e:
            self.msg("Couldn't find x data")
            self.msg(e)
            xdata = np.arange(dset.shape[0])
        try:
            xlab = dset.dims[0].label
        except:
            self.msg("Couldn't find x label")
            xlab = ""
        try:
            ylab = dset.name
        except AttributeError:
            ylab = ""
        curve_item = make.curve(x=xdata, y=np.array(dset))
        line_plot.add_item(curve_item)
        line_plot.set_axis_unit(2, xlab)
        line_plot.set_axis_unit(0, ylab)
        line_plot.show()
        line_plot.do_autoscale()

    def image_plot(self, dset):
        self.msg("image plot")
        self.gui['plots_tabWidget'].setCurrentIndex(0)
        image_plot = self.gui['image_plot']
        try:
            xdata = np.array(dset.dims[0][0])
            ydata = np.array(dset.dims[1][0])
        except Exception as e:
            self.msg("Couldn't find x/y data")
            self.msg(e)
            xdata = np.arange(dset.shape[0])
            ydata = np.arange(dset.shape[1])
        try:
            xlab = dset.dims[0].label
            ylab = dset.dims[1].label
        except:
            self.msg("Couldn't find x/y label")
            xlab, ylab = "", ""
        zlab = dset.name
        image_item = make.image(data=np.array(dset), 
                                xdata=(xdata[0],xdata[-1]), ydata=(ydata[0],ydata[-1]))
        image_plot.add_item(image_item)
        image_plot.set_axis_unit(2, xlab)
        image_plot.set_axis_unit(0, ylab)
        image_plot.set_axis_unit(1, zlab)
        image_plot.show()
        image_plot.do_autoscale()

    def load_dset_deprecated(self, item, column):
        # Message attribute values
        _type = item.text(1)
        if _type == "attr":
            self.msg("Attribute", item.name, ":", item.val)
            return

        self.msg("Load Dataset")
        h5file = self.open_file
        for name in item.path:
            h5file = h5file[name]

        # TODO 1D, Axis Selection, etc
        original_h5file = h5file
        if isinstance(h5file, h5py.Dataset):
            # 2D dataset handling
            if len(h5file.shape) == 2:
                if h5file.shape[0] == 1:
                    h5file = h5file[0,:]
                else:
                    try:
                        xdata, ydata = h5file.attrs["_axes"]
                        pi = make.image(data=np.array(h5file), 
                                        xdata=tuple(xdata), ydata=tuple(ydata))
                    except KeyError:
                        self.msg("no _axes attribute")
                        self.msg("Axes scaling could not be set up.")
                        pi = make.image(data=np.array(h5file)) 
                    try:
                        xlab, ylab, zlab = h5file.attrs["_axes_labels"]
                        
                    except Exception as e:
                        xlab, ylab, zlab = "", "", ""
                        self.msg(e)
                        
                    self.gui["image_plot"].set_axis_unit(2, xlab)
                    self.gui["image_plot"].set_axis_unit(0, ylab)
                    self.gui["image_plot"].set_axis_unit(1, zlab)
                    self.gui["plots_tabWidget"].setCurrentIndex(0)
                    self.gui["image_plot"].del_all_items()
                    self.gui["image_plot"].add_item(pi)
                    self.gui["image_plot"].show()
                    self.gui["image_plot"].do_autoscale()

            # 1D dataset handling
            if len(h5file.shape) == 1:
                if hasattr(self, "x_data"):
                    x_data = self.x_data
                else:
                    try:
                        x0, x1 = original_h5file.attrs['_axes'][0]
                        x_data = np.linspace(x0, x1, h5file.shape[0])
                        self.msg('set up axes')
                    except:
                        x_data = list(range(h5file.shape[0]))
                try:
                    xlab, ylab = original_h5file.attrs["_axes_labels"]
                except Exception as e:
                    xlab, ylab = "", ""
                    self.msg("Labels could not be set up", e)
                self.gui["line_plot"].set_axis_unit(2, xlab)
                self.gui["line_plot"].set_axis_unit(0, ylab)
                    
                if len(x_data) != h5file.shape[0]:
                    self.msg("Cannot broadcast shapes: x data has length", len(x_data))
                    return

                ci = make.curve(x=x_data, y=np.array(h5file))
                self.gui["plots_tabWidget"].setCurrentIndex(1)
                self.gui["line_plot"].del_all_items()
                self.gui["line_plot"].add_item(ci)
                self.gui["line_plot"].show()
                self.gui["line_plot"].do_autoscale()

    def add_item(self, name, obj):
        path = name.split("/")
        tree = self.dsettree
        for p in path:
            tree = tree[p]
        tree.meta["type"] = h5py_types[type(obj)]
        if isinstance(obj, h5py.Dataset):
            tree.meta["type"] += str(obj.shape)
        tree.meta["attrs"] = {}
        for k, v in obj.attrs.items():
            tree.meta["attrs"][k] = v

    def process_items(self, tree=None, parent=None):
        if tree is None:
            tree = self.dsettree
            self.tree_items = []
        else:
            for k, v in tree.meta["attrs"].items():
                ti = DataSetTreeItem(k, "attr")
                ti.val = v
                parent.addChild(ti)

        if isinstance(tree, dict):
            for k, v in tree.items():
                try: ti = DataSetTreeItem(k, v.meta["type"])
                except: ti = DataSetTreeItem(k, "")
                if parent is None:
                    self.tree_items.append(ti)
                else:
                    parent.addChild(ti)
                self.process_items(v, ti)

    def set_y_data(self, _trash):
        item = self.params["dataset"]
        dset = self.open_file
        for p in item.path: dset = dset[p]
        self.y_data = np.array(dset)

    def set_x_data(self, _trash):
        item = self.params["dataset"]
        dset = self.open_file
        for p in item.path: dset = dset[p]
        self.x_data = np.array(dset)

    def clear_x_data(self, _trash):
        self.x_data = None
        
    def clear_y_data(self, _trash):
        self.y_data = None
                
class HDFViewWindow(gui.SlabWindow, UiClass):
    def __init__(self, fname=None):
        gui.SlabWindow.__init__(self, HDFViewThread)
        self.setupSlabWindow(autoparam=True)
        self.auto_register_gui()

        self.register_param(self.datasets_treeWidget, "dataset")
        self.register_param(self.datapath_lineEdit, "datapath")

        # Setup Plots -- Could be eliminated if QtDesigner plugins were setup
        self.toolbar = Qt.QToolBar()
        self.addToolBar(self.toolbar)
        self.image_widget = ImageWidget(show_xsection=True, show_ysection=True)
        self.image_widget.add_toolbar(self.toolbar)
        self.image_widget.register_all_image_tools()
        self.image_plot = self.image_widget.plot
        self.image_plot_layout.addWidget(self.image_widget)
        self.gui["image_plot"] = self.image_plot
        
        self.line_widget = CurveWidget()
        self.gui["line_plot"] = self.line_plot = self.line_widget.plot #CurvePlot()
        self.line_plot_layout.addWidget(self.line_plot)
        self.line_widget.register_all_image_tools()

        # Context Menu actions
        self.set_x_action = Qt.QAction("Set as x data", self)
        self.clear_x_action = Qt.QAction("Clear x data", self)
        self.datasets_treeWidget.customContextMenuRequested.connect(
            lambda point: self.msg("menu requested"))
        self.datasets_treeWidget.customContextMenuRequested.connect(
            self.datasets_context_menu)

        # Connect launchers
        self.datapath_browse_pushButton.clicked.connect(self.select_datapath)
        self.register_script("set_datapath", self.datapath_lineEdit)
        self.register_script("load_file", self.datafiles_listWidget)
        self.register_script("load_dset", self.datasets_treeWidget)
        self.register_script("set_x_data", self.set_x_action)
        self.register_script("clear_x_data", self.clear_x_action)

        # Syntax Highlighting
        self.highlight = syntax.PythonHighlighter(self.script_textEdit.document())


        # Setup Prompt
        """
        message = "The currently loaded file is stored as 'f'"
        self.shell = InternalShell(self, message=message)
        self.shell.set_font(Qt.QFont("Consolas"))
        self.shell_dockWidget.setWidget(self.shell)
        self.gui["shell"] = self.shell
        """

        self.start_thread()

        if fname is not None:
            self.shell.exit_interpreter()
            directory = os.path.dirname(fname)
            nameitem = namedtuple('pseudoitem', ('filename',))(fname)
            self.msg(fname)
            self.set_param("datapath", directory)
            self.start_script("set_datapath")
            self.start_script("load_file", nameitem)
        
    def closeEvent(self, event):
        self.shell.exit_interpreter()
        event.accept()

    def select_datapath(self):
        path = str(Qt.QFileDialog.getExistingDirectory(
                    self, 'Open Datapath',self.params["datapath"]))
        if path:
            self.params["datapath"] = path
            self.emit(Qt.SIGNAL("RunOnDataThread"), "set_datapath")

    def datasets_context_menu(self, point):
        menu = Qt.QMenu()
        menu.addAction(self.set_x_action)
        menu.addAction(self.clear_x_action)
        menu.exec_(self.datasets_treeWidget.mapToGlobal(point))

if __name__ == "__main__":
    #fname = "S:\\_Data\\120930 - EonHe - M005CHM3\\004_AutomatedFilling\\121017_M005CHM3-AutomatedFilling_003\\M005CHM3-AutomatedFilling.h5"
    fname = None
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    sys.exit(gui.runWin(HDFViewWindow, fname=fname))
