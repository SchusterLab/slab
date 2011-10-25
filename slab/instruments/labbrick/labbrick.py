# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:16:08 2011

@author: Phil
"""

import ctypes as C
import numpy as np
import sys
from guiqwt.pyplot import *

from PyQt4.QtGui import *
from PyQt4.QtCore import QTimer

from slab.instruments import *
from labbrick_ui import *

U8 = C.c_uint8
U8P = C.POINTER(U8)
U32 = C.c_uint32
U32P = C.POINTER(U32)
U32PP = C.POINTER(U32P)
I32 = C.c_int32
CFLT = C.c_float

try:
    LMS103dllpath=r'S:\_Lib\python\scratch\labbrick\vnx_fmsynth.dll'
    LABBRICKDLL=C.CDLL(LMS103dllpath)
except:
    print "Warning could not load labbrick dll"

def LMS_get_device_info():
    dll=LABBRICKDLL
    dll.fnLMS_SetTestMode(U8(int(False)))
    device_info_array_type = U32 * int(dll.fnLMS_GetNumDevices(None))
    a=device_info_array_type()
    dll.fnLMS_GetDevInfo(a)
    devids=np.ctypeslib.as_array(a)
    devinfos=[]
    for devid in devids:
        model=C.create_string_buffer(8194)
        dll.fnLMS_GetModelName(U32(devid),model)
        serial=int(dll.fnLMS_GetSerialNumber(U32(devid)))
#            devstr="Device: %d\tModel: %s\tSerial: %d" % (devid,model,serial)
#            print devstr
        devinfos.append({"model":model,"serial":serial,"devid":U32(devid)})
    return devinfos

class LMS103(Instrument):
    def __init__(self,name="Labbrick",address=None,enabled=True):
        Instrument.__init__(self,name,address,enabled=True)
        self.dll=C.CDLL(LMS103dllpath)
        self.set_test_mode(False)
        if address is not None:
            self.init_by_serial(address)
        
    def init_by_serial(self,address):
        self.devinfo=self.get_info_by_serial(address)
        self.devid=self.devinfo['devid']
        if self.devid!= -1:
            self.init_device()
       
    def get_info_by_serial(self,serial):
        serial=int(serial)
        devinfos=LMS_get_device_info()
        for devinfo in devinfos:
            if devinfo['serial']==serial:
                return devinfo
        print "Error Labbrick serial # %d not found! returning first device found" % serial
        print devinfos
        return None

    def get_num_devices(self):
        return int(self.dll.fnLMS_GetNumDevices(None))
  
    def get_model_name(self):
        model_name=C.create_string_buffer(8194)
        self.dll.fnLMS_GetModelName(self.devid,model_name)
        return model_name.value
        
    def get_serial_number(self):
        return int(self.dll.fnLMS_GetSerialNumber(self.devid))
           
    def get_device_status(self):
        return int(self.dll.fnLMS_GetDeviceStatus(self.devid))
        
    def get_PLL_status(self):
        status=self.get_device_status()
        return bool(status & (1 <<6))
        
    def set_test_mode(self,mode=False):
        self.dll.fnLMS_SetTestMode(U8(int(mode)))
        
    def close_device(self):
        self.dll.fnLMS_CloseDevice(self.devid)
        
    def init_device(self):   
        self.dll.fnLMS_InitDevice(self.devid)
        
    def get_use_internal_reference(self):
        return bool(self.dll.fnLMS_GetUseInternalRef(self.devid))

    def get_use_internal_pulse_mod(self):
        return bool(self.dll.LMS_GetUseInternalPulseMod(self.devid))
        
    def get_power(self):
        """Get Power in dBm"""
        maxpower=float(self.dll.fnLMS_GetMaxPwr(self.devid)) / 4.0
        return maxpower + float(self.dll.fnLMS_GetPowerLevel(self.devid)) / (-4.0) 

    def set_power(self,power):
        """Set Power in dBm"""
        self.dll.fnLMS_SetPowerLevel(self.devid,I32(int(power * 4)))
        
    def get_frequency(self):
        """Get Frequency in Hz"""
        return float(self.dll.fnLMS_GetFrequency(self.devid)) * 10.0

    def set_frequency(self,frequency):
        """Set Frequency in Hz"""
        self.dll.fnLMS_SetFrequency(self.devid,U32(int(frequency/10.0)))
        
    def get_output(self):
        return bool(self.dll.fnLMS_GetRF_On(self.devid))

    def set_output(self,state=True):
        self.dll.fnLMS_SetRFOn(self.devid,U8(state))
        
    def get_mod(self):
        return bool(self.dll.fnLMS_GetPulseMode(self.devid))

    def set_mod(self,mod=True):
        self.dll.fnLMS_EnableInternalPulseMod(self.devid,U8(mod))

    def get_has_fast_pulse_mode(self):
        return bool(self.dll.fnLMS_GetHasFastPulseMode(self.devid))

    def get_pulse_on_time(self):
        """Get pulse on time (s)"""
        return float(self.dll.fnLMS_GetPulseOnTime(self.devid))

    def get_pulse_off_time(self):
        """Get pulse off time (s)"""
        return float(self.dll.fnLMS_GetPulseOffTime(self.devid))
        
    def set_pulse_off_time(self,off_time):
        """Set pulse off time (s)"""
        self.dll.fnLMS_SetPulseOffTime(self.devid,CFLT(off_time))

    def set_pulse_on_time(self,on_time):
        """Set pulse on time (s)"""
        self.dll.fnLMS_SetPulseOnTime(self.devid,CFLT(on_time))

    def set_pulse_parameters(self,width,period,mod=True):
        """Set Pulse parameters (width, period, mod) in seconds"""
        self.dll.fnLMS_SetFastPulsedOutput(self.devid,CFLT(width),CFLT(period),U8(mod))

    def get_pulse_width(self):
        return self.get_pulse_on_time()
        
    def set_pulse_width(self,width):
        period=self.get_pulse_period()
        self.set_pulse_on_time(width)
        self.set_pulse_off_time(period-width)
    
    def set_pulse_period(self,period):
        self.set_pulse_off_time(period-width)
    
    def get_pulse_period(self):
        return self.get_pulse_on_time()+self.get_pulse_off_time()
        
    def get_settings(self):
        return {'output':self.get_output(),'mod':self.get_mod(),
                'frequency':self.get_frequency(),'power':self.get_power(),
                'pulse_width':self.get_pulse_width(),'pulse_period':self.get_pulse_period(),
                'PLL':self.get_PLL_status()
                }
    
class LabbrickWindow(QMainWindow, Ui_labbrickWindow):
    def __init__(self, address=None,parent = None):
    
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.updateButton.clicked.connect(self.update_rf)
        self.autoupdateCheckBox.stateChanged.connect(self.autoupdateCheckBox_changed)

        if address is not None:
            self.rf=LMS103(address=address)
        else:
            devinfo=LMS_get_device_info()[0]
            self.rf=LMS103(address=devinfo['serial'])
            #self.rf=LMS103(address=1198)
            
        #print self.rf.get_serial_number()
        self.get_state()
        self.ctimer = QTimer()      #Setup autoupdating timer to call update_plots at 10Hz
        self.ctimer.timeout.connect(self.update_rf)
        
    def update_rf(self):
        self.ctimer.stop()
        self.rf.set_frequency(self.frequencySpinBox.value())
        self.rf.set_power(self.powerSpinBox.value())
        self.rf.set_pulse_parameters(self.pulsewidthSpinBox.value()*1e-6,self.pulserepSpinBox.value()*1e-6,self.modCheckBox.isChecked())
        self.rf.set_output(self.outputCheckBox.isChecked())
        self.get_state()
        self.ctimer.start(1000)
        #print "Update!"
        
    def get_state(self):
        settings=self.rf.get_settings()
        self.frequencySpinBox.setValue(settings['frequency'])
        self.powerSpinBox.setValue(settings['power'])
        self.generatorNumber.display(self.rf.get_serial_number())
        self.outputCheckBox.setChecked(settings['output'])
        self.modCheckBox.setChecked(settings['mod'])
        self.pulsewidthSpinBox.setValue(settings['pulse_width']*1e6)
        self.pulserepSpinBox.setValue(settings['pulse_period']*1e6)
        self.PLLBox.setChecked(settings['PLL'])
       
    def autoupdateCheckBox_changed(self):
        if self.autoupdateCheckBox.isChecked():
            self.ctimer.start(1000)
        else:
            self.ctimer.stop()

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = LabbrickWindow(address=1198)
    window.show()
    sys.exit(app.exec_())

#    rf=LMS103(address=1198)
#    print rf.get_settings()
