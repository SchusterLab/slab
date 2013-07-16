from collections import defaultdict
from copy import copy

from PyQt4 import Qt
import pyqtgraph
import pyqtgraph.dockarea
import numpy as np

import helpers
from plot_widgets import CrossSectionWidget
from model import DataTreeLeaf

class DataTreeLeafItem(Qt.QTreeWidgetItem):
    @property
    def path(self):
        if self.parent() is None:
            return (str(self.text(0)),)
        else:
            return self.parent().path + (str(self.text(0)),)

    @property
    def strpath(self):
        return '/'.join(map(str, self.path))


Qt.QTreeWidgetItem.is_leaf = lambda item: str(item.text(1)) != ""
Qt.QTreeWidgetItem.getChildren = lambda item: [item.child(i) for i in range(item.childCount())]

class MyDockArea(pyqtgraph.dockarea.DockArea):
    def __init__(self, *args, **kwargs):
        pyqtgraph.dockarea.DockArea.__init__(self, *args, **kwargs)
        self.insert_location = 'bottom'

    def add_dock_auto_location(self, dock):
        self.addDock(dock, self.insert_location)
        self.insert_location = {'bottom':'right', 'right':'bottom'}[self.insert_location]

class NodeEditWidget(Qt.QFrame):
    def __init__(self, path, attrs):
        Qt.QFrame.__init__(self)
        self.setFrameStyle(Qt.QFrame.Panel)
        self.path = path
        self.spin_widgets = {}
        self.text_widgets = {}
        from window import TempDataClient
        self.background_client = TempDataClient()

        self.setLayout(Qt.QVBoxLayout())
        self.layout().addWidget(Qt.QLabel('Editing ' + '/'.join(self.path)))

        self.attr_list = Qt.QTreeWidget()
        self.attr_list.setRootIsDecorated(False)
        self.attr_list.setColumnCount(2)
        self.attr_list.setHeaderLabels(['Name', 'Value', 'Type'])

        for attr, value in attrs.items():
            self.attr_list.addTopLevelItem(Qt.QTreeWidgetItem([attr, str(value), str(type(value))]))

        add_attr_box = Qt.QWidget()
        add_attr_box.setLayout(Qt.QHBoxLayout())
        self.attr_name_edit = Qt.QLineEdit()
        self.attr_value_edit = Qt.QLineEdit()
        self.attr_value_edit.returnPressed.connect(self.add_attribute)
        self.attr_list.itemClicked.connect(self.attr_clicked)
        self.add_attr_button = Qt.QPushButton('Add Attribute')
        self.add_attr_button = Qt.QPushButton('Add Attribute')
        self.add_attr_button.clicked.connect(self.add_attribute)
        add_attr_box.layout().addWidget(Qt.QLabel('name'))
        add_attr_box.layout().addWidget(self.attr_name_edit)
        add_attr_box.layout().addWidget(Qt.QLabel('value'))
        add_attr_box.layout().addWidget(self.attr_value_edit)
        add_attr_box.layout().addWidget(self.add_attr_button)

        self.new_attr = True
        self.attr_name_edit.textChanged.connect(self.check_attr_name)
        self.layout().addWidget(self.attr_list)
        self.layout().addWidget(add_attr_box)

    def check_attr_name(self, name):
        print self.attr_list.findItems("", Qt.Qt.MatchContains)
        if any(i.text(0) == name for i in self.attr_list.findItems("", Qt.Qt.MatchContains)):
            if self.new_attr:
                self.new_attr = False
                self.add_attr_button.setText('Update Attribute')
        else:
            if not self.new_attr:
                self.new_attr = True
                self.add_attr_button.setText('Add Attribute')


    def attr_clicked(self, item):
        self.attr_name_edit.setText(item.text(0))
        self.attr_value_edit.setText(item.text(1))

    def add_attribute(self):
        name = str(self.attr_name_edit.text())
        value = str(self.attr_value_edit.text())
        if value.lower() in ('true', 'false'):
            if value.lower() == 'false':
                value = False
            else:
                value = True
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        self.background_client.set_attr(self.path, name, value)
        if self.new_attr:
            self.attr_list.addTopLevelItem(Qt.QTreeWidgetItem([name, str(value), str(type(value))]))
        else:
            i = self.attr_list.findItems(name, Qt.Qt.MatchExactly, 0)[0]
            i.setText(1, str(value))
            i.setText(2, str(type(value)))
        self.attr_name_edit.setText("")
        self.attr_value_edit.setText("")
        return name, value


class LeafEditWidget(NodeEditWidget):
    param_attrs = ['x0', 'y0', 'xscale', 'yscale', 'xlabel', 'ylabel', 'zlabel', 'parametric']
    def __init__(self, leaf):
        NodeEditWidget.__init__(self, leaf.path, leaf.attrs)
        self.rank = leaf.rank

    def add_attribute(self):
        name, value = NodeEditWidget.add_attribute(self)
        if name in self.param_attrs:
            self.background_client.set_params(self.path, self.rank, **{name:value})
    # def __init__(self, leaf):
    #     self.rank = leaf.rank
    #     self.range_area = Qt.QWidget()
    #     self.range_area.setLayout(Qt.QHBoxLayout())
    #     self.layout().addWidget(self.range_area)
    #     self.add_spin('x0', leaf)
    #     self.add_spin('xscale', leaf)
    #     if leaf.rank > 1:
    #         self.add_spin('y0', leaf)
    #         self.add_spin('yscale', leaf)
    #     self.label_area = Qt.QWidget()
    #     self.label_area.setLayout(Qt.QHBoxLayout())
    #     self.layout().addWidget(self.label_area)
    #     self.add_edit('xlabel', leaf)
    #     self.add_edit('ylabel', leaf)
    #     if leaf.rank > 1:
    #         self.add_edit('zlabel', leaf)
    #     self.commit_button = Qt.QPushButton('Commit')
    #     self.commit_button.clicked.connect(self.commit_params)
    #     self.layout().addWidget(self.commit_button)
    #     self.add_attr_widget(leaf)

    # def add_spin(self, name, leaf):
    #     self.range_area.layout().addWidget(Qt.QLabel(name))
    #     widget = pyqtgraph.SpinBox(value=getattr(leaf, name))
    #     self.spin_widgets[name] = widget
    #     self.range_area.layout().addWidget(widget)

    # def add_edit(self, name, leaf):
    #     self.label_area.layout().addWidget(Qt.QLabel(name))
    #     widget = Qt.QLineEdit(getattr(leaf, name))
    #     self.text_widgets[name] = widget
    #     self.label_area.layout().addWidget(widget)

    # def commit_params(self):
    #     params = self.to_dict()
    #     self.background_client.set_params(self.path, self.rank, **params)

    # def to_dict(self):
    #     params = {name: spin.value() for name, spin in self.spin_widgets.items()}
    #     params.update({name: edit.text() for name, edit in self.text_widgets.items()})
    #     return params

class ItemWidget(pyqtgraph.dockarea.Dock):
    def __init__(self, ident, dock_area, **kwargs):
        if len(ident) > 25:
            name = '... ' + ident.split('/')[-1]
        else:
            name = ident
        pyqtgraph.dockarea.Dock.__init__(self, name)
        self.label.setFont(Qt.QFont('Helvetica', pointSize=14))
        self.ident = ident
        self.add_plot_widget(**kwargs)
        self.buttons_widget = Qt.QWidget() #QHBoxWidget()
        self.buttons_widget.setLayout(Qt.QHBoxLayout())
        self.remove_button = Qt.QPushButton('Hide')
        self.remove_button.clicked.connect(self.toggle_hide)
        self.clear_button = Qt.QPushButton('Clear') # Connected to manager action
        self.update_toggle = Qt.QCheckBox('Update')
        self.update_toggle.setChecked(True)
        #self.autoscale_toggle = Qt.QCheckBox('AutoScale')
        #self.autoscale_toggle.setChecked(True)
        self.buttons_widget.layout().addWidget(self.remove_button)
        self.buttons_widget.layout().addWidget(self.clear_button)
        self.buttons_widget.layout().addWidget(self.update_toggle)
        #self.buttons_widget.layout().addWidget(self.autoscale_toggle)
        self.addWidget(self.buttons_widget)
        self.dock_area = dock_area
        self.dock_area.add_dock_auto_location(self)
        self.visible = True

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def toggle_hide(self):
        #self.setVisible(not(self.isVisible()))
        if self.visible:
            self.setParent(None)
            self.label.setParent(None)
            self.visible = False
        else:
            self.dock_area.add_dock_auto_location(self)
            self.visible = True

    def add_plot_widget(self, **kwargs):
        raise NotImplementedError

    def update_plot(self, leaf):
        raise NotImplementedError

    def clear_plot(self):
        raise NotImplementedError


class Rank1ItemWidget(ItemWidget):
    def __init__(self, ident, dock_area, **kwargs):
        ItemWidget.__init__(self, ident, dock_area, **kwargs)
        self.rank = 1

    def add_plot_widget(self, **kwargs):
        #self.line_plt = pyqtgraph.PlotWidget(title=self.ident, **kwargs)
        self.line_plt = pyqtgraph.PlotWidget(**kwargs)
        self.addWidget(self.line_plt)
        self.curve = None

    def update_plot(self, leaf, refresh_labels=False, **kwargs):
        if leaf is None or leaf.data is None or leaf.data.shape[0] is 0:
            self.clear_plot()
            return

        if self.update_toggle.isChecked():
            if leaf.parametric:
                if leaf.data.shape[0] == 2:
                    xdata, ydata = leaf.data
                elif leaf.data.shape[1] == 2:
                    xdata, ydata = leaf.data.T
                else:
                    raise ValueError('Leaf claims to be parametric, but shape is ' + str(leaf.data.shape))
            else:
                ydata = leaf.data
                xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)

        if refresh_labels:
            self.line_plt.plotItem.setLabels(bottom=(leaf.xlabel,), left=(leaf.ylabel,))

        if self.curve is None:
            self.curve = self.line_plt.plot(xdata, ydata, **kwargs)
        else:
            self.curve.setData(x=xdata, y=ydata, **kwargs)

    def clear_plot(self):
        if self.curve is not None:
            self.line_plt.removeItem(self.curve)
            self.curve = None


class MultiplotItemWidget(Rank1ItemWidget):
    def add_plot_widget(self, **kwargs):
        Rank1ItemWidget.add_plot_widget(self, **kwargs)
        self.curves = defaultdict(lambda: self.line_plt.plot([], pen=tuple(helpers.random_color())))

    def update_plot(self, path, leaf):
        self.curve = self.curves[path]
        Rank1ItemWidget.update_plot(self, leaf)


class ParametricItemWidget(Rank1ItemWidget):
    def __init__(self, path1, path2, dock_area, **kwargs):
        Rank1ItemWidget.__init__(self, path1[-1]+' vs '+path2[-1], dock_area, **kwargs)
        self.transpose_toggle = Qt.QCheckBox('Transpose')
        self.transpose_toggle.stateChanged.connect(lambda s: self.update_plot())
        self.buttons_widget.layout().addWidget(self.transpose_toggle)
        self.combined_leaf = DataTreeLeaf(np.array([[]]), 1, parametric=True)
        self.leaf1_data, self.leaf2_data = [], []
        self.path1, self.path2 = path1, path2

    def update_plot(self, path=None, leaf=None):
        if path is None:
            self.combined_leaf.data = np.array(zip(self.leaf1_data, self.leaf2_data))
        elif path == self.path1:
            self.leaf1_data = leaf.data
            self.combined_leaf.data = np.array(zip(leaf.data, self.leaf2_data))
            self.combined_leaf.xlabel = leaf.path[-1]
        elif path == self.path2:
            self.leaf2_data = leaf.data
            self.combined_leaf.data = np.array(zip(self.leaf1_data, leaf.data))
            self.combined_leaf.ylabel = leaf.path[-1]
        else:
            raise ValueError(str(path) + ' is not either ' + str(self.path1) + ' or ' + str(self.path2))
        if self.transpose_toggle.isChecked():
            leaf = copy(self.combined_leaf)
            leaf.data = np.array((leaf.data[:,1], leaf.data[:,0]))
            leaf.xlabel, leaf.ylabel = leaf.ylabel, leaf.xlabel
        else:
            leaf = self.combined_leaf
        Rank1ItemWidget.update_plot(self, leaf, refresh_labels=True)

class Rank2ItemWidget(Rank1ItemWidget):
    def __init__(self, ident, dock_area, **kwargs):
        Rank1ItemWidget.__init__(self, ident, dock_area, **kwargs)
        self.rank = 2

        self.histogram_check = Qt.QCheckBox('Histogram')
        self.histogram_check.stateChanged.connect(self.img_view.set_histogram)
        self.recent_button = Qt.QPushButton('Most Recent Trace')
        self.recent_button.clicked.connect(self.show_recent)
        self.accum_button = Qt.QPushButton('Accumulated Traces')
        self.accum_button.clicked.connect(self.show_accumulated)
        self.accum_button.hide()
        self.buttons_widget.layout().addWidget(self.histogram_check)
        self.buttons_widget.layout().addWidget(self.recent_button)
        self.buttons_widget.layout().addWidget(self.accum_button)

    def add_plot_widget(self, **kwargs):
        Rank1ItemWidget.add_plot_widget(self)
        self.line_plt.hide() # By default, start out with accum view

        # plt/view explanation here
        # https://groups.google.com/d/msg/pyqtgraph/ccrYl1yyasw/fD9tLrco1PYJ
        # TODO: Identify and separate 1d kwargs and 2d kwargs
        #self.img_plt = pyqtgraph.PlotItem(title=self.ident)
        #self.img_view = pyqtgraph.ImageView(view=self.img_plt)
        self.img_view = CrossSectionWidget()
        self.addWidget(self.img_view)

        #self.line_plt = pyqtgraph.PlotWidget(parent=self, title=self.ident)
        self.addWidget(self.line_plt)
        self.curve = None

    def update_plot(self, leaf, refresh_labels=False, show_most_recent=None):
        if leaf is None or leaf.data is None:
            self.clear_plot()
            return

        if show_most_recent is not None:
            if show_most_recent:
                self.show_recent()
            else:
                self.show_accumulated()
        if self.update_toggle.isChecked():
            if refresh_labels:
                self.img_view.setLabels(leaf.xlabel, leaf.ylabel, leaf.zlabel)
            Rank1ItemWidget.update_plot(self, leaf.to_rank1())
            self.img_view.imageItem.show()
            self.img_view.setImage(leaf.data, pos=[leaf.x0, leaf.y0], scale=[leaf.xscale, leaf.yscale])
            if leaf.data.shape[0] == 1:
                self.show_recent()

    def clear_plot(self):
        Rank1ItemWidget.clear_plot(self)
        #Rank1ItemWidget.update_plot(self, None)
        #self.img_view.setImage(np.array([[]]))
        self.img_view.imageItem.hide()

    def show_recent(self):
        self.img_view.hide()
        self.line_plt.show()
        self.recent_button.hide()
        self.accum_button.show()

    def show_accumulated(self):
        self.img_view.show()
        self.line_plt.hide()
        self.recent_button.show()
        self.accum_button.hide()

