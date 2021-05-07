# -*- coding: utf-8 -*-
"""
Template for writing gui programs

To use:
    - fill in UI_PREFIX with the name of your ui file which should be in the
      same directory as this file. E.g. if "test.ui" exists in this directory,
      write UI_PREFIX = "test"
    - Import from the compiled ui file, e.g. from test_ui import Ui_MainWindow
    - Rename MyDataThread and MyWindow to something more descriptive
    - 
"""
from slab import gui

COMPILE_UI = False
UI_PREFIX = "myWindow"
if COMPILE_UI:
    gui.compile_ui(UI_PREFIX)

from myWindow_ui import Ui_MainWindow



class MyDataThread(gui.DataThread):
    def my_script(self):
        # parameters in self.params["param_name"]
        # plots in self.plots["plot_name"]
        pass
    
class MyWindow(gui.SlabWindow, Ui_MainWindow):
    def __init__(self):
        gui.SlabWindow.__init__(self, MyDataThread)
        self.setupSlabWindow(autoparam=True)
        self.register_script("my_script", self.start_button, self.abort_button)
        self.start_thread()
        
if __name__ == "__main__":
    gui.runWin(MyWindow)    