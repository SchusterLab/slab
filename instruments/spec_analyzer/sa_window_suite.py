# -*- coding: utf-8 -*-
"""
name: sa_window_suite.py
description: This is the GUI of the spectrum analyzer bundled with LO (the 
choice of LO can set up through the external config file for InstrumentManager).
It provides plots of intensity spectrum and realtime intensity vs time plot at 
given frequency.

Created on Fri Sep 09 21:35:41 2011

@author: Dai
"""
from PyQt4.QtGui import QMainWindow, QApplication
from PyQt4.QtCore import QTimer, QThread
from guiqwt.builder import make
import numpy as np
import sys
import time
from threading import Thread

from instruments import E8257D
from spectrum_analyzer import SpectrumAnalyzer
from sa_calibration_manager import *

from spectrum_analyzer_sweep_ui import *

class SpectrumAnalyzerWindow(QMainWindow, Ui_SpectrumAnalyzerWindow):
    def __init__(self, sa, sacm, lo, parent = None):    
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        #instruments
        self.sa = sa
        self.sacm = sacm
        self.lo = lo
        self.lo.set_output(False)
        self.lo.set_power(self.sa.lo_power)
 
        #reading over time plot
        self.curvewidget.add_toolbar(self.addToolBar("Curve"))
        self.curvewidget.register_all_image_tools()
        self.plot = self.curvewidget.plot   
        self.plot.set_titles(title="Power vs. Time", xlabel="Time (s)", ylabel="Power")
       
        self.ch1_plot = make.mcurve(np.array([]),np.array([]),label='Magnitude') #Make Ch1 curve
        self.plot.add_item(self.ch1_plot)
        self.data=[]
        self.tl = []
        self.t0 = time.time()
        self.first_update = True
        
        #spectrum plot
        self.spec_plot = self.spectrumCurvewidget.plot
        self.spectrumCurvewidget.register_all_image_tools()
        self.spec_plot.set_titles(title='Power vs. Frequency', xlabel='Frequency (GHz)', ylabel='Power')
        self.spectrum = make.mcurve(np.array([]), np.array([]))
        self.spec_plot.add_item(self.spectrum)
        self.spec = []
        self.fd = []
        self.sweeping = [False, False]
        
        self.sweep_end = self.endSweepSpinBox.value()

        #register UI listeners
        self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
        self.ctimer.timeout.connect(self.update_ui)
        self.ctimer.start(100)
        self.dBmLabel.hide()
        self.calibratedOuputCheckBox.clicked.connect(self.cali_output_checkbox_clicked)
        self.updateRateSpinBox.setValue(10.0)
        self.updateRateSpinBox.editingFinished.connect(self.update_rate_changed)
        self.frequencySpinBox.editingFinished.connect(self.frequency_changed)
        self.endSweepSpinBox.editingFinished.connect(self.end_changed)
        self.clearButton.clicked.connect(self.clear_plot)
        self.sweepButton.clicked.connect(self.sweep)
        
        self.frequency_changed()
        
    def update_power(self):
        #self.lo.set_frequency(4.99e9+10.55e6)
        if self.first_update:
            self.first_update = False
            self.lo.set_output()      
            time.sleep(0.2)
            self.data=[]
            
        try:
            if self.instantRButton.isChecked():
                power = self.sa.get_power()
            else:
                power = self.sa.get_avg_power()
                
            if self.calibratedOuputCheckBox.checkState():
                power = self.sacm.get_rf_power(self.frequencySpinBox.value()*1e9, power)
            
        except OutputOutOfRangeError as e:
                print e
                power = self.sacm.get_rf_power(e.frequency, e.lower_bound)
        except Exception as e:
            print e
            return 
            
        self.powerLCD.display(power)
        self.data.append(power)
        self.tl.append(time.time()-self.t0)
        self.ch1_plot.set_data(np.array(self.tl), np.array(self.data))
        self.plot.do_autoscale()
        self.plot.replot()
    
    def update_ui(self):
        if not self.sweeping[0]:
        # read power during sweep might collide
            self.update_power()
            if self.sweeping[1]:
                # just finished a sweep, reset ui
                self.sweepButton.setEnabled(True)
                self.sweepButton.setText('Sweep')
                self.fd = []
                self.spec = []
                self.sweeping[1] = False
        
        # update the sweep plot if it is sweeping
        if self.sweeping[0]:
            # plot the new sweep data
            self.spectrum.set_data(np.array(self.fd), np.array(self.spec))
            self.spec_plot.do_autoscale()
            self.spec_plot.replot()
            
    
    def clear_plot(self):
        self.data = []
        self.tl = []
        self.ch1_plot.set_data(np.array(self.tl), np.array(self.data))
        self.plot.do_autoscale()
        self.plot.replot()
        self.t0 = time.time()
        self.first_update = True
    
    def update_rate_changed(self):
        self.ctimer.start(1000.0/self.updateRateSpinBox.value())
        
    def end_changed(self):
        te = self.endSweepSpinBox.value()        
        if te <= self.startSweepSpinBox.value(): 
            self.endSweepSpinBox.setValue(self.sweep_end)
        else:
            self.sweep_end = te
    
    def frequency_changed(self):
        self.ctimer.stop()        
        self.lo.set_frequency(self.frequencySpinBox.value()*1e9+10.55e6)
        time.sleep(0.1)
        self.ctimer.start(1000.0/self.updateRateSpinBox.value())
        self.clear_plot()

    def cali_output_checkbox_clicked(self):
        if self.calibratedOuputCheckBox.checkState():
            self.dBmLabel.show()
        else:
            self.dBmLabel.hide()
        self.clear_plot()

    def sweep(self):
        #self.ctimer.stop()
        
        """
        create a new thread to sweep and disable the sweep button
        to prevent multiple sweeps
        """
        self.sweepButton.setEnabled(False)
        self.sweepButton.setText('Sweeping...')
        
        data = {
                 'lo': self.lo, 
                 'sacm': self.sacm,
                 'sa': self.sa,
                 'fd': self.fd,
                 'spec': self.spec,
                 'calibrated': self.calibratedOuputCheckBox.checkState(),
                 'instant' : self.instantRButton.isChecked(),
                 'sweep_start': self.startSweepSpinBox.value(),
                 'sweep_end': self.sweep_end,
                 'sweep_interval': self.stepSweepSpinBox.value(),
                 'sweeping': self.sweeping
                }        

        st = SweepThread(data)
        st.start()
        
        #self.first_update = True
        #self.frequency_changed()
        
class SweepThread(Thread):
    """a thread subclass to handle sweep"""
    def __init__(self, data):
        self.lo = data['lo']
        self.sacm = data['sacm']
        self.sa = data['sa']
        self.fd = data['fd']
        self.spec = data['spec']
        self.calibrated = data['calibrated']
        self.sweep_start = data['sweep_start']
        self.sweep_end = data['sweep_end']
        self.sweep_interval = data['sweep_interval']
        self.sweeping = data['sweeping']
        self.instant = data['instant']
        Thread.__init__(self)
        
    def run(self):
        self.sweeping[0] = True
        original_frequency = self.lo.get_frequency()
        
        #self.lo.set_output(False)
        #self.lo.set_output()
        time.sleep(0.3)
        #self.lo.set_power(10)
        fr = [x*1e9 for x in np.arange(self.sweep_start, 
                                    self.sweep_end,
                                    self.sweep_interval)]
        for f in fr:
            self.lo.set_frequency(f+self.sa.lo_offset) 
            #self.lo.set_output()
            time.sleep(0.1)
            self.fd.append(f/1e9)
            
            try:
                if self.instant:
                    r = self.sa.get_power()
                else:
                    r = self.sa.get_avg_power()
            
                if self.calibrated:
                    self.spec.append(self.sacm.get_rf_power(f, r))
                else:
                    self.spec.append(r)
            
            except OutputOutOfRangeError as e:
                print e
                self.spec.append(self.sacm.get_rf_power(e.frequency, e.lower_bound))
            except Exception as e:
                print e
        
        """unlock sweep by enabling the sweep button"""
        self.sweeping[0] = False
        self.sweeping[1] = True
        self.lo.set_frequency(original_frequency)
        time.sleep(0.1)

if __name__ == '__main__':
    import pickle
    from slab.instruments import InstrumentManager

    cfg_path="sa_suite.cfg"
    im = InstrumentManager(cfg_path)
    
    app = QApplication(sys.argv)
    #sa = im['SA']
    sa = SpectrumAnalyzer(protocol='serial', address='2', query_sleep=0.05, lo_offset=10.57e6)
    #sacm = SACalibrationManager(pickle.load(open('10dBm_cali.data')))
    sacm = SACalibrationManager(pickle.load(open('10dBm_LMS_cali.data')))
    #lo = E8257D(address='rfgen1.circuitqed.com')
    lo = im['LO']
    rf = im['RF']
    
    rf.set_frequency(7e9)
    # +13dBm LO to drive IQB0618
    rf.set_power(13)
   
    window = SpectrumAnalyzerWindow(sa, sacm, lo)
    window.show()
    app.exec_()
   