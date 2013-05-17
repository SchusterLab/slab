from collections import defaultdict

from PyQt4 import Qt
import pyqtgraph
import pyqtgraph.dockarea
import numpy as np

import helpers
from plot_widgets import CrossSectionWidget

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


Qt.QTreeWidgetItem.is_leaf = lambda item: item.childCount() == 0
Qt.QTreeWidgetItem.getChildren = lambda item: [item.child(i) for i in range(item.childCount())]


class LeafEditWidget(Qt.QFrame):
    def __init__(self, leaf):
        Qt.QFrame.__init__(self)
        self.setFrameStyle(Qt.QFrame.Panel)
        self.path = leaf.path
        self.spin_widgets = {}
        self.text_widgets = {}

        self.setLayout(Qt.QVBoxLayout())
        self.layout().addWidget(Qt.QLabel('Editing ' + '/'.join(self.path)))
        self.range_area = Qt.QWidget()
        self.range_area.setLayout(Qt.QHBoxLayout())
        self.layout().addWidget(self.range_area)
        self.add_spin('x0', leaf)
        self.add_spin('xscale', leaf)
        if leaf.rank > 1:
            self.add_spin('y0', leaf)
            self.add_spin('yscale', leaf)
        self.label_area = Qt.QWidget()
        self.label_area.setLayout(Qt.QHBoxLayout())
        self.layout().addWidget(self.label_area)
        self.add_edit('xlabel', leaf)
        self.add_edit('ylabel', leaf)
        if leaf.rank > 1:
            self.add_edit('zlabel', leaf)
        self.commit_button = Qt.QPushButton('Commit')
        self.layout().addWidget(self.commit_button)

    def add_spin(self, name, leaf):
        self.range_area.layout().addWidget(Qt.QLabel(name))
        widget = pyqtgraph.SpinBox(value=getattr(leaf, name))
        self.spin_widgets[name] = widget
        self.range_area.layout().addWidget(widget)

    def add_edit(self, name, leaf):
        self.label_area.layout().addWidget(Qt.QLabel(name))
        widget = Qt.QLineEdit(getattr(leaf, name))
        self.text_widgets[name] = widget
        self.label_area.layout().addWidget(widget)

    def to_dict(self):
        params = {name: spin.value() for name, spin in self.spin_widgets.items()}
        params.update({name: edit.text() for name, edit in self.text_widgets.items()})
        return params

class ItemWidget(pyqtgraph.dockarea.Dock):
    def __init__(self, ident, **kwargs):
        if len(ident) > 25:
            name = ident[:5].strip() + ' ... ' + ident[-15:]
        else:
            name = ident
        pyqtgraph.dockarea.Dock.__init__(self, name)
        self.label.setFont(Qt.QFont('Helvetica', pointSize=16))
        self.ident = ident
        self.add_plot_widget(**kwargs)
        self.buttons_widget = Qt.QWidget() #QHBoxWidget()
        self.buttons_widget.setLayout(Qt.QHBoxLayout())
        self.remove_button = Qt.QPushButton('Hide')
        self.remove_button.clicked.connect(self.toggle_hide)
        self.clear_button = Qt.QPushButton('Clear') # Connected to manager action
        self.update_toggle = Qt.QCheckBox('Update')
        self.update_toggle.setChecked(True)
        self.autoscale_toggle = Qt.QCheckBox('AutoScale')
        self.autoscale_toggle.setChecked(True)
        self.buttons_widget.layout().addWidget(self.remove_button)
        self.buttons_widget.layout().addWidget(self.clear_button)
        self.buttons_widget.layout().addWidget(self.update_toggle)
        self.buttons_widget.layout().addWidget(self.autoscale_toggle)
        self.addWidget(self.buttons_widget)

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def toggle_hide(self):
        self.setVisible(not(self.isVisible()))

    def add_plot_widget(self, **kwargs):
        raise NotImplementedError

    def update_plot(self, leaf):
        raise NotImplementedError

    def clear_plot(self):
        raise NotImplementedError


class Rank1ItemWidget(ItemWidget):
    def __init__(self, ident, **kwargs):
        ItemWidget.__init__(self, ident, **kwargs)
        self.rank = 1

    def add_plot_widget(self, **kwargs):
        #self.line_plt = pyqtgraph.PlotWidget(title=self.ident, **kwargs)
        self.line_plt = pyqtgraph.PlotWidget(**kwargs)
        self.addWidget(self.line_plt)
        self.curve = None

    def update_plot(self, leaf, refresh_labels=False):

        if leaf is None or leaf.data is None:
            self.clear_plot()
            return

        if self.update_toggle.isChecked():
            if leaf.parametric:
                if leaf.data.shape[0] == 2:
                    xdata, ydata = leaf.data
                else:
                    xdata, ydata = leaf.data.T
            else:
                ydata = leaf.data
                xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)

        if refresh_labels:
            self.line_plt.plotItem.setLabels(bottom=(leaf.xlabel,), left=(leaf.ylabel,))

        if self.curve is None:
            self.curve = self.line_plt.plot(xdata, ydata)
        else:
            self.curve.setData(x=xdata, y=ydata)

        #        try:
        #            assert leaf.data.shape[1] == 2
        #            _, ydata = leaf.data.T
        #            type = 'A'
        #        except:
        #            ydata = leaf.data
        #            type = 'B'
        #        xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)
        #    else:
        #        try:
        #            assert leaf.data.shape[1] == 2
        #            xdata, ydata = leaf.data.T
        #            type = 'C'
        #        except:
        #            ydata = leaf.data
        #            xdata = np.arange(leaf.x0, leaf.x0+(leaf.xscale*len(ydata)), leaf.xscale)
        #            type = 'D'

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


class Rank2ItemWidget(Rank1ItemWidget):
    def __init__(self, ident, **kwargs):
        Rank1ItemWidget.__init__(self, ident, **kwargs)
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

