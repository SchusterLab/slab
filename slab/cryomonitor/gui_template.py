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
import numpy as np
import time

COMPILE_UI = True
UI_PREFIX = "SlabWindow"
if COMPILE_UI:
    gui.compile_ui(UI_PREFIX)

from SlabWindow_ui import Ui_MainWindow
from guiqwt.builder import make
from wigglewiggle import wigglewiggle
M = wigglewiggle(None, None, None, scriptpath=None)
M.initiate_instrument('fridge', 'FRIDGE')
M.initiate_instrument('heman', 'heman')

class MyDataThread(gui.DataThread):
    def my_script(self):
        # parameters in self.params["param_name"]
        # plots in self.plots["plot_name"]
        print self.params.keys()
    
class MyWindow(gui.SlabWindow, Ui_MainWindow):
    def __init__(self):
        gui.SlabWindow.__init__(self, MyDataThread)
        self.setupSlabWindow(autoparam=True)
        #self.register_script("my_script")

        # Parameters
        #self.noof_points = 10000
        #self.dt = 10

        # Plotting
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        self.plot = self.curvewidget.plot
        self.plotsInitialized=False

        # Push buttons
        self.pushButton.clicked.connect(self.do_plot)
        self.pushButton_2.clicked.connect(self.clear_plot)

        self.pushButton_open_1.clicked.connect(self.open_gas)
        self.pushButton_open_2.clicked.connect(self.open_pump)
        self.pushButton_open_3.clicked.connect(self.open_cryo)
        self.pushButton_close_1.clicked.connect(self.close_gas)
        self.pushButton_close_2.clicked.connect(self.close_pump)
        self.pushButton_close_3.clicked.connect(self.close_cryo)

        self.pushButton_CleanManifold.clicked.connect(self.clean_manifold)

        # Temperature
        self.set_temperature_labels()

        # Pressure
        self.set_pressure_labels()

        # Heman
        self.set_heman_labels()

        # Cleaning
        self.label_cleaning_status.setText("<font style='color: %s'>%s</font>"%('Black', ""))

        #self.resize(800,600)
        #self.setGeometry(0, 0, 800, 600)

    def set_heman_labels(self):
        self.label_pres_manifold.setText(str(M.heman.get_pressure()))

        for idx, S in enumerate(M.heman.get_manifold_status_bits()):
            if S > 0:
                status = 'Open'
                color = 'Red'

            else:
                status = 'Closed'
                color = 'Green'

            if idx == 0:
                self.label_status_1.setText("<font style='color: %s'>%s</font>"%(color,status))
            if idx == 1:
                self.label_status_2.setText("<font style='color: %s'>%s</font>"%(color,status))
            if idx == 2:
                self.label_status_3.setText("<font style='color: %s'>%s</font>"%(color,status))

        gui.QApplication.processEvents()

    def set_temperature_labels(self):
        MC_cernox = M.fridge.get_temperature('MC cernox')
        MC_RuO2 = M.fridge.get_temperature('MC RuO2')

        if MC_cernox < 2:
            color = 'Red'
        else:
            color = 'Black'

        self.label_temp_MCCernox.setText("<font style='color: %s'>%s</font>"%(color,MC_cernox))
        self.label_temp_MCRuO2.setText(str(MC_RuO2))
        self.label_temp_50K.setText(str(M.fridge.get_temperature('PT1 Plate')))
        self.label_temp_4K.setText(str(M.fridge.get_temperature('PT2 Plate')))
        self.label_temp_Still.setText(str(M.fridge.get_temperature('Still')))
        self.label_temp_100mK.setText(str(M.fridge.get_temperature('100mK Plate')))
        gui.QApplication.processEvents()

    def set_pressure_labels(self):
        self.label_pres_condense.setText(str(M.fridge.get_pressures()['Condense']))
        self.label_pres_forepump.setText(str(M.fridge.get_pressures()['Forepump']))
        self.label_pres_tank.setText(str(M.fridge.get_pressures()['Tank']))
        gui.QApplication.processEvents()
        #self.update_plots()

    def open_cryo(self):
        M.heman.set_cryostat(True)
        time.sleep(0.1)
        self.set_heman_labels()

    def close_cryo(self):
        M.heman.set_cryostat(False)
        time.sleep(0.1)
        self.set_heman_labels()

    def open_pump(self):
        M.heman.set_pump(True)
        time.sleep(0.1)
        self.set_heman_labels()

    def close_pump(self):
        M.heman.set_pump(False)
        time.sleep(0.1)
        self.set_heman_labels()

    def open_gas(self):
        M.heman.set_gas(True)
        time.sleep(0.1)
        self.set_heman_labels()

    def close_gas(self):
        M.heman.set_gas(False)
        time.sleep(0.1)
        self.set_heman_labels()

    def clean_manifold(self):
        current_status = M.heman.get_manifold_status_bits()
        noof_cleans = np.int(self.spinBox_noof_cleans.value())
        print noof_cleans
        self.label_cleaning_status.setText("<font style='color: %s'>%s</font>"%('Red', "Cleaning..."))
        gui.QApplication.processEvents()
        M.heman.clean_manifold(noof_cleans)

        # Leave the manifold as we started
        M.heman.set_gas(np.bool(current_status[0]))
        M.heman.set_pump(np.bool(current_status[1]))
        M.heman.set_cryostat(np.bool(current_status[2]))

        self.set_heman_labels()
        self.label_cleaning_status.setText("<font style='color: %s'>%s</font>"%('Black', "Cleaning completed"))

    def do_plot(self):
        self.running=True
        self.noof_points = np.int(np.float(self.lineEdit_totaltime.text())/np.float(self.lineEdit.text()))
        self.dt = np.float(self.lineEdit.text())
        #create some fake data
        #self.freqs=np.linspace(1E9, 4E9, 100)
        #self.mags=np.sin(5*self.freqs/1E9)
        #self.update_plots()
        t0 = time.time()

        t = list()
        temps = list()

        # Fill the array
        for N in range(self.noof_points):
            T = M.heman.get_pressure()
            #T = M.fridge.get_temperature('MC RuO2')*1E3
            ts = time.time() - t0

            temps.append(T)
            t.append(ts)

            self.x = np.array(t)
            self.y = np.array(temps)
            time.sleep(0.1)
            self.update_plots(xlabel='Time (s)', ylabel='Manifold Pressure (bar)')
            #self.update_plots(xlabel='Time (s)', ylabel='MC RuO2 (mK)')
            #self.plots_initialized = False

            gui.QApplication.processEvents()
            time.sleep(self.dt)

            print self.x
            print self.y

            print N

        # Now keep the array size constant
        while self.running:

            print "Doing it!"
            T = M.heman.get_pressure()
            #T = M.fridge.get_temperature('MC RuO2')*1E3
            ts = time.time() - t0

            temps.append(T)
            t.append(ts)

            temps.pop(0)
            t.pop(0)

            self.x = np.array(t)
            self.y = np.array(temps)

            time.sleep(0.5)
            self.update_plots(xlabel='Time (s)', ylabel='Manifold Pressure (bar)')
            #self.update_plots(xlabel='Time (s)', ylabel='MC RuO2 (mK)')

            print self.x
            print self.y

            time.sleep(self.dt)
            gui.QApplication.processEvents()

    def clear_plot(self):
        self.set_temperature_labels()
        self.set_pressure_labels()
        self.set_heman_labels()
        #self.running = False
        #self.freqs = np.linspace(1E9, 4E9)
        #self.mags = np.zeros(100)
        #self.update_plots()

    def update_plots(self, xlabel="", ylabel=""):
        if self.x is None or self.y is None: return


        if not self.plotsInitialized:
            self.plotsInitialized=True
            self.ch1_plot = make.mcurve(self.x, self.y, label='Magnitude') #Make Ch1 curve
            self.plot.add_item(self.ch1_plot)
            self.plot.set_titles(title="", xlabel=xlabel, ylabel=ylabel)

        self.update_lineplot(self.plot,self.ch1_plot,(self.x, self.y))

    def update_lineplot(self,plot,plot_item,data):
        plot_item.set_data(data[0],data[1])
        plot.replot()

        
if __name__ == "__main__":
    gui.runWin(MyWindow)