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
from data_cache import dataCacheProxy
from PyQt4 import QtCore
from PyQt4 import QtGui
import PyQt4

COMPILE_UI = True
UI_PREFIX = "SlabWindow"
if COMPILE_UI:
    gui.compile_ui(UI_PREFIX)

from SlabWindow_ui import Ui_MainWindow
from guiqwt.builder import make
from wigglewiggle import wigglewiggle
from slab.instruments import Alazar, AlazarConfig
M = wigglewiggle(None, None, None, scriptpath=None)
M.initiate_instrument('fridge', 'FRIDGE')
M.initiate_instrument('heman', 'heman')

class MyDataThread(gui.DataThread):
    def my_script(self):
        # parameters in self.params["param_name"]
        # plots in self.plots["plot_name"]
        print(list(self.params.keys()))
    
class MyWindow(gui.SlabWindow, Ui_MainWindow):
    def __init__(self):
        gui.SlabWindow.__init__(self, MyDataThread)
        self.setupSlabWindow(autoparam=True)
        self.M = M
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

        # BNC RF control
        try:
            self.M.initiate_instrument('BNC_RF', 'BNC845_RF1')
            self.label_bncrf_status.setText("<font style='color: %s'>%s</font>"%("Green", "OK!"))
            self.set_bnc_labels()
        except:
            self.label_bncrf_status.setText("<font style='color: %s'>%s</font>"%("Red", "ERROR!"))

        self.pushButton_bncrf_update.clicked.connect(self.set_bnc_labels)
        self.pushButton_bncrf_output.clicked.connect(self.toggle_bncrf_output)
        self.pushButton_bncrf_sweep.clicked.connect(self.sweep_bncrf_frequency)
        self.pushButton_set_bncrf_frequency.clicked.connect(self.quick_set_frequency)

        self.pushButton_min10Hz.clicked.connect(self.increment_bncrf_frequency(-10))
        self.pushButton_plus10Hz.clicked.connect(self.increment_bncrf_frequency(+10))
        self.pushButton_min100Hz.clicked.connect(self.increment_bncrf_frequency(-100))
        self.pushButton_plus100Hz.clicked.connect(self.increment_bncrf_frequency(+100))
        self.pushButton_min1kHz.clicked.connect(self.increment_bncrf_frequency(-1E3))
        self.pushButton_plus1kHz.clicked.connect(self.increment_bncrf_frequency(+1E3))
        self.pushButton_min10kHz.clicked.connect(self.increment_bncrf_frequency(-10E3))
        self.pushButton_plus10kHz.clicked.connect(self.increment_bncrf_frequency(+10E3))
        self.pushButton_min100kHz.clicked.connect(self.increment_bncrf_frequency(-1E5))
        self.pushButton_plus100kHz.clicked.connect(self.increment_bncrf_frequency(+1E5))
        self.pushButton_min1MHz.clicked.connect(self.increment_bncrf_frequency(-1E6))
        self.pushButton_plus1MHz.clicked.connect(self.increment_bncrf_frequency(+1E6))

        # BNC RF Plotting
        self.curvewidget_bncrf_sweep.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget_bncrf_sweep.register_all_image_tools()
        self.plot = self.curvewidget_bncrf_sweep.plot
        self.plotsInitialized=False

        # BNC AWG control
        try:
            self.M.initiate_instrument('BNC_AWG', 'BNC')
            self.label_bncawg_status.setText("<font style='color: %s'>%s</font>"%("Green", "OK!"))
            self.set_bncawg_labels()
        except:
            self.label_bncawg_status.setText("<font style='color: %s'>%s</font>"%("Red", "ERROR!"))

        self.pushButton_bncawg_apply.clicked.connect(self.apply_bncawg_settings)
        self.pushButton_bncawg_update.clicked.connect(self.set_bncawg_labels)
        self.pushButton_bncawg_output.clicked.connect(self.toggle_bncawg_output)

        self.pushButton_min10Hz_2.clicked.connect(self.increment_bncawg_frequency(-10))
        self.pushButton_plus10Hz_2.clicked.connect(self.increment_bncawg_frequency(+10))
        self.pushButton_min100Hz_2.clicked.connect(self.increment_bncawg_frequency(-100))
        self.pushButton_plus100Hz_2.clicked.connect(self.increment_bncawg_frequency(+100))
        self.pushButton_min1kHz_2.clicked.connect(self.increment_bncawg_frequency(-1E3))
        self.pushButton_plus1kHz_2.clicked.connect(self.increment_bncawg_frequency(+1E3))
        self.pushButton_min10kHz_2.clicked.connect(self.increment_bncawg_frequency(-10E3))
        self.pushButton_plus10kHz_2.clicked.connect(self.increment_bncawg_frequency(+10E3))
        self.pushButton_min100kHz_2.clicked.connect(self.increment_bncawg_frequency(-1E5))
        self.pushButton_plus100kHz_2.clicked.connect(self.increment_bncawg_frequency(+1E5))
        self.pushButton_min1MHz_2.clicked.connect(self.increment_bncawg_frequency(-1E6))
        self.pushButton_plus1MHz_2.clicked.connect(self.increment_bncawg_frequency(+1E6))

        # NWA tab
        try:
            self.M.initiate_instrument('NWA', 'NWA')
            self.nwa_set_labels()
            self.label_nwa_status.setText("<font style='color: %s'>%s</font>"%("Green", "OK!"))
        except:
            self.label_nwa_status.setText("<font style='color: %s'>%s</font>"%("Red", "ERROR!"))

        self.pushButton_nwa_takesave.clicked.connect(self.nwa_take_and_save)

        # NWA Plotting
        self.curvewidget_nwa.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget_nwa.register_all_image_tools()
        self.nwaplot = self.curvewidget_nwa.plot
        self.nwaplotsInitialized=False

        itemList = ["Center & span", "Start & stop"]

        self.comboBox_nwa_mode.blockSignals(True)
        self.comboBox_nwa_mode.clear()
        self.comboBox_nwa_mode.addItems(sorted(itemList))
        self.comboBox_nwa_mode.setCurrentIndex(-1)
        self.comboBox_nwa_mode.blockSignals(False)
        self.comboBox_nwa_mode.setCurrentIndex(0)

        self.connect(self.comboBox_nwa_mode, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.nwa_set_labels)
        #self.resize(800,600)
        #self.setGeometry(0, 0, 800, 600)

        #self.plainTextEdit_nwa_notes.clicked.connect(self.clear_textbox)

    def set_image(self, filepath, extension='JPG'):
        img = QtGui.QImage(filepath, extension)
        cursor = QtGui.QTextCursor(self.textEdit_heman_logo.document())
        cursor.insertImage(img)

    def nwa_set_labels(self):
        mode = str(self.comboBox_nwa_mode.currentText())
        if 'Center' in mode:
            print(mode)
            self.label_nwa_center_or_start.setText("<font style='color: %s'><b>%s</b></font>"%("Black", "Center"))
            self.label_nwa_span_or_stop.setText("<font style='color: %s'><b>%s</b></font>"%("Black", "Span"))
        else:
            print(mode)
            self.label_nwa_center_or_start.setText("<font style='color: %s'><b>%s</b></font>"%("Black", "Start"))
            self.label_nwa_span_or_stop.setText("<font style='color: %s'><b>%s</b></font>"%("Black", "Stop"))

        gui.QApplication.processEvents()

    def nwa_take_and_save(self):
        self.M.NWA.set_output(True)

        self.label_nwa_save_name.setText("<font style='color: %s'>%s</font>"%("Black", "Taking data..."))
        self.x = None
        self.y = None
        self.update_plots(xlabel='Frequency (Hz)', ylabel='Transmission (dB)')

        gui.QApplication.processEvents()
        save_path = str(self.lineEdit_nwa_savepath.text())

        measure = str(self.comboBox_nwa_measure.currentText())
        mode = str(self.comboBox_nwa_mode.currentText())
        f1 = float(self.lineEdit_nwa_center_or_start.text())
        f2 = float(self.lineEdit_nwa_span_or_stop.text())
        power = float(self.lineEdit_nwa_power.text())
        ifbw = float(self.lineEdit_nwa_ifbw.text())
        points = float(self.lineEdit_nwa_points.text())
        avgs = float(self.lineEdit_nwa_avgs.text())

        if 'Center' in mode:
            self.M.NWA.set_center_frequency(f1*1E6)
            self.M.NWA.set_span(f2*1E6)
        else:
            self.M.NWA.set_start_frequency(f1*1E6)
            self.M.NWA.set_stop_frequency(f2*1E6)

        self.M.NWA.set_measure(measure)
        self.M.NWA.set_power(power)
        self.M.NWA.set_averages(avgs)
        self.M.NWA.set_sweep_points(points)
        self.M.NWA.set_ifbw(ifbw)

        print("Setup complete")

        self.M.scriptname = None
        date = time.strftime("%y%m%d")
        t = time.strftime("%H%M%S")
        datafoldername = self.M.create_new_datafolder(save_path, "nwa_scan", date, t)

        FileHandler = dataCacheProxy(expInst="nwa_scan", filepath=os.path.join(datafoldername, 'nwa_scan.h5'))

        print("Taking data")
        fpoints, mags, phases = self.M.NWA.take_one_averaged_trace()

        FileHandler.post('fpoints', fpoints)
        FileHandler.post('mags', mags)
        FileHandler.post('phases', phases)
        #self.M.save_nwa_config(FileHandler)

        self.label_nwa_save_name.setText("<font style='color: %s'>%s</font>"%("Black", "Saved in: ...\\"+date+"\\"+t))

        # Notes
        notes = self.plainTextEdit_nwa_notes.toPlainText()
        FileHandler.note(notes, "Experimental notes")

        # Plotting
        self.x = fpoints
        self.y = mags

        self.update_nwa_plots(xlabel='Frequency (Hz)', ylabel='Transmission (dB)')
        gui.QApplication.processEvents()

    def clear_textbox(self):
        self.plainTextEdit_nwa_notes.setPlainText("")

    def increment_bncawg_frequency(self, amount):
        def func_creator():
            self.M.BNC_AWG.set_frequency(self.M.BNC_AWG.get_frequency() + amount)
            self.set_bncawg_labels()
        return func_creator

    def toggle_bncawg_output(self):
        self.M.BNC_AWG.set_output(np.logical_not(self.M.BNC_AWG.get_output()))
        self.set_bncawg_labels()

    def apply_bncawg_settings(self):

        freq = np.float(self.lineEdit_bncawg_quickset_frequency.text())
        waveform = str(self.combobox_bncawg_waveform.currentText())
        amplitude = np.float(self.lineEdit_bncawg_quickset_amplitude.text())
        offset = np.float(self.lineEdit_bncawg_quickset_offset.text())
        pulse_width = np.float(self.lineEdit_bncawg_quickset_pulsewidth.text())

        self.M.BNC_AWG.set_function(waveform)

        if waveform == "PULSE":
            self.M.BNC_AWG.set_pulse_duty_cycle(pulse_width)

        self.M.BNC_AWG.set_frequency(freq)
        self.M.BNC_AWG.set_offset(offset)
        self.M.BNC_AWG.set_amplitude(amplitude)

        self.set_bncawg_labels()

    def set_bncawg_labels(self):
        """
        Set the Output and Frequency labels of the BNC AWG tab.
        """

        if M.BNC_AWG.get_output():
            status = 'ON'
            color = 'Green'
        else:
            status = 'OFF'
            color = 'Red'

        self.label_bncawg_frequency.setText("<font style='color: %s'>%s</font>"%("Black", self.M.BNC_AWG.get_frequency()))
        self.label_bncawg_output.setText("<font style='color: %s'>%s</font>"%(color, status))
        self.label_bncawg_amplitude.setText("<font style='color: %s'>%s</font>"%("Black", self.M.BNC_AWG.get_amplitude()))
        self.label_bncawg_offset.setText("<font style='color: %s'>%s</font>"%("Black", self.M.BNC_AWG.get_offset()))

        gui.QApplication.processEvents()

    def quick_set_frequency(self):
        """
        Set the frequency of the source according to the field Quick set frequency.
        """
        self.M.BNC_RF.set_frequency(np.float(self.lineEdit_bncrf_quickset_frequency.text())*1E9)
        self.set_bnc_labels()


    def sweep_bncrf_frequency(self):
        """
        Sweeps the frequency of the BNC RF source.
        """

        #self.plotsInitialized = False
        alazar_range_raw = str(self.combobox_alazar_range.currentText())
        alazar_ch = str(self.combobox_alazar_channel.currentText())
        sweep_center = np.float(self.lineEdit_bncrf_sweep_center.text())*1E9
        sweep_span = np.float(self.lineEdit_bncrf_sweep_span.text())*1E6
        sweep_numpoints = np.int(self.lineEdit_bncrf_noofpoints.text())

        sweep_points = np.linspace(sweep_center-sweep_span/2., sweep_center+sweep_span/2., sweep_numpoints)

        self.x = None
        self.y = None
        self.update_plots(xlabel='BNC Frequency (Hz)', ylabel='Scope ch%s (V)'%alazar_ch)

        if alazar_range_raw == '40 mV':
            alazar_range = 40E-3
        elif alazar_range_raw == '100 mV':
            alazar_range = 100E-3
        elif alazar_range_raw == '200 mV':
            alazar_range = 200E-3
        elif alazar_range_raw == '400 mV':
            alazar_range = 400E-3
        elif alazar_range_raw == '1 V':
            alazar_range = 1.0
        elif alazar_range_raw == '2 V':
            alazar_range = 2.0
        elif alazar_range_raw == '4 V':
            alazar_range = 4.0

        sample_rate = 20E3 #sample rate in Hz
        noof_samples = 1024
        noof_avgs = 1
        ch1_range = alazar_range
        ch2_range = alazar_range
        ch1_coupling = 'DC'
        ch2_coupling = 'DC'
        ch1_trigger_level = 2.5
        ch2_trigger_level = 2.5
        timeout = int(80E3)
        trigger_rate = 1e1

        print("Your trigger pulse width should be > %.3e s." % (1/sample_rate))
        print("Time out should be higher than %.3e" % (noof_samples/sample_rate*1e3 ))

        config = {'clock_edge': 'rising',
                  'clock_source': 'reference',
                  'samplesPerRecord': noof_samples,
                  'recordsPerBuffer': 1,
                  'recordsPerAcquisition': noof_avgs,
                  'bufferCount': 1,
                  'sample_rate': sample_rate/1E3, #in kHz
                  'timeout': timeout, #After this are ch1 settings
                  'ch1_enabled': True,
                  'ch1_filter': False,
                  'ch1_coupling': ch1_coupling,
                  'ch1_range': ch1_range, #After this are ch2 settings
                  'ch2_enabled': True,
                  'ch2_filter': False,
                  'ch2_coupling': ch2_coupling,
                  'ch2_range': ch2_range, #After this are trigger settings
                  'trigger_source1': 'external',
                  'trigger_level1': ch1_trigger_level,
                  'trigger_edge1': 'rising',
                  'trigger_source2': 'disabled',
                  'trigger_level2': ch2_trigger_level,
                  'trigger_edge2': 'rising',
                  'trigger_operation': 'or',
                  'trigger_coupling': 'DC',
                  'trigger_delay': 0}

        print("Total time trace signal is %.2f seconds" % (noof_samples/float(sample_rate)))

        alazar = Alazar()

        print("new configuration file")
        cfg = AlazarConfig(config)
        print("config file has to pass through the AlazarConfig middleware.")
        alazar.configure(cfg)

        self.x = sweep_points
        self.y = list()

        for idx,f in enumerate(sweep_points):
            self.M.BNC_RF.set_frequency(f)
            self.set_bnc_labels()
            t, ch1, ch2 = alazar.acquire_avg_data(excise=(0, -56))
            time.sleep(0.1)

            if alazar_ch == '1':
                self.y.append(np.mean(ch1))
            else:
                self.y.append(np.mean(ch2))

        self.update_plots(xlabel='BNC Frequency (Hz)', ylabel='Scope ch%s (V)'%alazar_ch)
        gui.QApplication.processEvents()




    def toggle_bncrf_output(self):
        self.M.BNC_RF.set_output(np.logical_not(self.M.BNC_RF.get_output()))
        self.set_bnc_labels()

    def increment_bncrf_frequency(self, amount):
        def func_creator():
            self.M.BNC_RF.set_frequency(self.M.BNC_RF.get_frequency() + amount)
            self.set_bnc_labels()
        return func_creator

    def set_bnc_labels(self):
        """
        Set the Output and Frequency labels of the BNC RF tab.
        """

        if M.BNC_RF.get_output():
            status = 'ON'
            color = 'Green'
        else:
            status = 'OFF'
            color = 'Red'

        self.label_bncrf_frequency.setText("<font style='color: %s'>%s</font>"%("Black", self.M.BNC_RF.get_frequency()/1E9))
        self.label_bncrf_output.setText("<font style='color: %s'>%s</font>"%(color, status))
        gui.QApplication.processEvents()

    def set_heman_labels(self):
        self.label_pres_manifold.setText(str(M.heman.get_pressure()))

        bits = M.heman.get_manifold_status_bits()

        for idx, S in enumerate(bits):
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

        self.textEdit_heman_logo.clear()
        self.set_image(os.path.join('bin', 'img', str(bits[0])+str(bits[1])+str(bits[2])+'.jpg'))

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
        print(noof_cleans)
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

            print(self.x)
            print(self.y)

            print(N)

        # Now keep the array size constant
        while self.running:

            print("Doing it!")
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

            print(self.x)
            print(self.y)

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

    def update_nwa_plots(self, xlabel="", ylabel=""):
        if self.x is None or self.y is None: return

        if not self.nwaplotsInitialized:
            self.nwaplotsInitialized=True
            self.nwa_curve = make.mcurve(self.x, self.y, label='Magnitude')
            self.nwaplot.add_item(self.nwa_curve)
            self.nwaplot.set_titles(title="", xlabel=xlabel, ylabel=ylabel)

        self.update_lineplot(self.nwaplot , self.nwa_curve, (self.x, self.y))

    def update_lineplot(self, plot, plot_item, data):
        plot_item.set_data(data[0],data[1])
        plot.do_autoscale()
        #plot.replot()

        
if __name__ == "__main__":
    gui.runWin(MyWindow)