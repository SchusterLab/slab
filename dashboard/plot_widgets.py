import pyqtgraph as pg
import warnings


class CrossSectionWidget(pg.ImageView):
    def __init__(self, trace_size=80, **kwargs):
        view = pg.PlotItem(labels=kwargs.pop('labels', None))
        self.trace_size = trace_size
        pg.ImageView.__init__(self, view=view, **kwargs)
        view.setAspectLocked(lock=False)
        self.cs_layout = pg.GraphicsLayout()
        self.cs_layout.addItem(view, row=1, col=0)
        self.ui.graphicsView.setCentralItem(self.cs_layout)
        self.cross_section_enabled = False
        self.add_cross_section()
        self.hide_cross_section()
        self.search_mode = False
        self.signals_connected = False
        self.set_histogram(False)
        self.ui.histogram.gradient.loadPreset('thermal')
        try:
            self.connect_signal()
        except RuntimeError:
            warnings.warn('Scene not set up, cross section signals not connected')

    def setLabels(self, xlabel="X", ylabel="Y", zlabel="Z"):
        self.view.setLabels(bottom=(xlabel,), left=(ylabel,))
        self.ui.histogram.item.axis.setLabel(text=zlabel)

    def setImage(self, *args, **kwargs):
        pg.ImageView.setImage(self, *args, **kwargs)
        self.update_cross_section()

    def toggle_cross_section(self):
        if self.cross_section_enabled:
            self.hide_cross_section()
        else:
            self.add_cross_section()

    def set_histogram(self, visible):
        self.ui.histogram.setVisible(visible)
        self.ui.roiBtn.setVisible(visible)
        self.ui.normBtn.setVisible(visible)

    def add_cross_section(self):
        self.h_cross_section_item = self.cs_layout.addPlot(row=0, col=0)
        self.v_cross_section_item = self.cs_layout.addPlot(row=1, col=1)
        self.h_cross_section = self.h_cross_section_item.plot([])
        self.v_cross_section = self.v_cross_section_item.plot([])
        self.cs_layout.layout.setRowMaximumHeight(0, self.trace_size)
        self.cs_layout.layout.setColumnMaximumWidth(1, self.trace_size+20)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.view.addItem(self.h_line, ignoreBounds=False)
        self.view.addItem(self.v_line, ignoreBounds=False)
        self.x_cross_index = 0
        self.y_cross_index = 0
        self.cross_section_enabled = True

    def hide_cross_section(self):
        self.cs_layout.layout.removeItem(self.h_cross_section_item)
        self.cs_layout.layout.removeItem(self.v_cross_section_item)
        self.h_cross_section_item.close()
        self.v_cross_section_item.close()
        self.view.removeItem(self.h_line)
        self.view.removeItem(self.v_line)
        self.cross_section_enabled = False

    def connect_signal(self):
        """This can only be run after the item has been embedded in a scene"""
        if self.signals_connected:
            warnings.warn("")
        if self.imageItem.scene() is None:
            raise RuntimeError('Signal can only be connected after it has been embedded in a scene.')
        self.imageItem.scene().sigMouseClicked.connect(self.toggle_search)
        self.imageItem.scene().sigMouseMoved.connect(self.handle_mouse_move)
        self.timeLine.sigPositionChanged.connect(self.update_cross_section)
        self.signals_connected = True

    def toggle_search(self, mouse_event):
        if mouse_event.double():
            self.toggle_cross_section()
        else:
            self.search_mode = not self.search_mode
            if self.search_mode:
                self.handle_mouse_move(mouse_event.scenePos())

    def handle_mouse_move(self, mouse_event):
        if self.cross_section_enabled and self.search_mode:
            view_coords = self.imageItem.getViewBox().mapSceneToView(mouse_event)
            self.h_line.setPos(view_coords.y())
            self.v_line.setPos(view_coords.x())
            max_x, max_y = self.imageItem.image.shape
            self.x_cross_index = max(min(int(view_coords.x()), max_x-1), 0)
            self.y_cross_index = max(min(int(view_coords.y()), max_y-1), 0)
            self.update_cross_section()

    def update_cross_section(self):
        self.h_cross_section.setData(self.imageItem.image[:, self.y_cross_index])
        self.v_cross_section.setData(self.imageItem.image[self.x_cross_index, :], range(self.imageItem.image.shape[1]))

if __name__ == "__main__":
    import sys
    import numpy as np
    import scipy.ndimage

    img = scipy.ndimage.gaussian_filter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
    img = img[np.newaxis,:,:]
    decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
    data = np.random.normal(size=(100, 200, 200))
    data += img * decay
    data += 2

    ## Add time-varying signal
    sig = np.zeros(data.shape[0])
    sig[30:] += np.exp(-np.linspace(1,10, 70))
    sig[40:] += np.exp(-np.linspace(1,10, 60))
    sig[70:] += np.exp(-np.linspace(1,10, 30))

    sig = sig[:,np.newaxis,np.newaxis] * 3
    data[:,50:60,50:60] += sig

    app = pg.QtGui.QApplication([])
    win = pg.QtGui.QMainWindow()
    view = CrossSectionWidget()
    win.setCentralWidget(view)
    win.show()
    view.setImage(data)
    sys.exit(app.exec_())

