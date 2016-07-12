# -*- coding: utf-8 -*-
"""
Lab Brick (labbrick.labbrick.py)
================================

:Author: David Schuster
"""

import ctypes as C
import numpy as np
import sys

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
    LDA602dllpath=r'C:\_Lib\python\slab\instruments\labbrick\VNX_atten.dll'
    LDADLL=C.CDLL(LDA602dllpath)
except:
    print "Warning could not load LDA labbrick dll, check that dll located at '%s'" % LDA602dllpath

try:
    #LMS103dllpath=r'S:\_Lib\python\scratch\labbrick\vnx_fmsynth.dll'
    LMS103dllpath=r'C:\_Lib\python\slab\instruments\labbrick\vnx_fmsynth.dll'
    LMSDLL=C.CDLL(LMS103dllpath)
except:
    print "Warning could not load LMS labbrick dll, check that dll located at '%s'" % LMS103dllpath

try:
    #LMS103dllpath=r'S:\_Lib\python\scratch\labbrick\vnx_fmsynth.dll'
    LPSdllpath=r'C:\_Lib\python\slab\instruments\labbrick\VNX_dps.dll'
    LPSDLL=C.CDLL(LPSdllpath)
except:
    print "Warning could not load LPS labbrick dll, check that dll located at '%s'" % LPSdllpath


def LPS_get_device_info():
    """
    Returns a dictionary of device information

    :returns: {'model':Labbrick model, 'serial':Serial number, 'devid':Device ID}
    """
    dll=LPSDLL
    dll.fnLPS_SetTestMode(U8(int(False)))
    device_info_array_type = U32 * int(dll.fnLPS_GetNumDevices(None))
    a=device_info_array_type()
    dll.fnLPS_GetDevInfo(a)
    devids=np.ctypeslib.as_array(a)
    devinfos=[]
    for devid in devids:
        model=C.create_string_buffer(8194)
        dll.fnLPS_GetModelNameA(U32(devid),model)
        serial=int(dll.fnLPS_GetSerialNumber(U32(devid)))
#            devstr="Device: %d\tModel: %s\tSerial: %d" % (devid,model,serial)
#            print devstr
        devinfos.append({"model":model,"serial":serial,"devid":U32(devid)})
    return devinfos


def LMS_get_device_info():
    """
    Returns a dictionary of device information

    :returns: {'model':Labbrick model, 'serial':Serial number, 'devid':Device ID}
    """
    dll=LMSDLL
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


def LDA_get_device_info():
    """
    Returns a dictionary of device information

    :returns: {'model':Labbrick model, 'serial':Serial number, 'devid':Device ID}
    """
    dll=LDADLL
    dll.fnLDA_SetTestMode(U8(int(False)))
    device_info_array_type = U32 * int(dll.fnLDA_GetNumDevices(None))
    a=device_info_array_type()
    dll.fnLDA_GetDevInfo(a)
    devids=np.ctypeslib.as_array(a)
    devinfos=[]
    for devid in devids:
        model=C.create_string_buffer(8194)
        dll.fnLDA_GetModelName(U32(devid),model)
        serial=int(dll.fnLDA_GetSerialNumber(U32(devid)))
#            devstr="Device: %d\tModel: %s\tSerial: %d" % (devid,model,serial)
#            print devstr
        devinfos.append({"model":model,"serial":serial,"devid":U32(devid)})
    return devinfos


class LDA602(Instrument):
    'The interface to the Lab Brick Phase shifter'
    dll=LDADLL

    def __init__(self,name='LDA602',address=9083,enabled=True):
        Instrument.__init__(self,name,address,enabled=True)
        self.set_test_mode(False)
        if address is not None:
            self.init_by_serial(int(address))
        else:
            self.init_by_serial(0)

    def init_by_serial(self,address):
        self.devinfo=self.get_info_by_serial(address)
        self.devid=self.devinfo['devid']
        if self.devid!= -1:
            self.init_device()

    def get_info_by_serial(self,serial):
        serial=int(serial)
        devinfos=LDA_get_device_info()
        for devinfo in devinfos:
            #print devinfo
            if devinfo['serial']==serial:
                return devinfo
        print "Error Labbrick serial # %d not found! returning first device found" % serial
        print devinfos
        return None

    def get_num_devices(self):
        return int(self.dll.fnLDA_GetNumDevices(None))

    def get_model_name(self):
        model_name=C.create_string_buffer(8194)
        self.dll.fnLDA_GetModelName(self.devid,model_name)
        return model_name.value

    def set_test_mode(self,mode=False):
        self.dll.fnLDA_SetTestMode(U8(int(mode)))

    def close_device(self):
        self.dll.fnLDA_CloseDevice(self.devid)

    def init_device(self):
        self.dll.fnLDA_InitDevice(self.devid)

    def get_id(self):
        return "Labbrick Phase Shifter model: %s serial #: %d" % (self.get_model_name(),self.devinfo['serial'])

    def set_attenuation(self, attenuation):
        """
        :param attenuation: Attenuation in dB, may range from 0.5 dB to 63 dB in steps of 0.5 dB.
        :return:
        """
        if attenuation>63 or attenuation<0:
            print "%.2f dB falls outside the range (0 - 63 dB). Setting to attenuation to 63 dB" % attenuation
            attenuation = 63

        self.dll.fnLDA_SetAttenuation(self.devid, U32(int(attenuation/0.25)))

    def set_rf_on(self, status):
        """
        :param status: This function allows rapid switching of the attenuator from its set value “on” (status = TRUE) to its
        maximum attenuation (status = FALSE).
        :return:
        """
        self.dll.fnLDA_SetRFOn(self.devid, U8(int(status)))

    def get_attenuation(self):
        """
        :return: Attenuation of the device in dB
        """
        return float(self.dll.fnLDA_GetAttenuation(self.devid))*0.25

    def get_rf_on(self):
        """
        :return: Returns an integer value which is 1 when the attenuator is “on”, or 0 when the
        attenuator has been set “off” by the set_rf_on function
        """
        return float(self.dll.fnLDA_GetRF_On(self.devid))



class LPS802(Instrument):
    'The interface to the Lab Brick Phase shifter'
    dll=LPSDLL

    def __init__(self,name='LPS802',address=None,enabled=True):
        Instrument.__init__(self,name,address,enabled=True)
        self.set_test_mode(False)
        if address is not None:
            self.init_by_serial(int(address))
        else:
            self.init_by_serial(0)
        
    def init_by_serial(self,address):
        self.devinfo=self.get_info_by_serial(address)
        self.devid=self.devinfo['devid']
        if self.devid!= -1:
            self.init_device()
       
    def get_info_by_serial(self,serial):
        serial=int(serial)
        devinfos=LPS_get_device_info()
        for devinfo in devinfos:
            #print devinfo
            if devinfo['serial']==serial:
                return devinfo
        print "Error Labbrick serial # %d not found! returning first device found" % serial
        print devinfos
        return None

    def get_num_devices(self):
        return int(self.dll.fnLPS_GetNumDevices(None))
  
    def get_model_name(self):
        model_name=C.create_string_buffer(8194)
        self.dll.fnLPS_GetModelName(self.devid,model_name)
        return model_name.value
    
    def set_test_mode(self,mode=False):
        self.dll.fnLPS_SetTestMode(U8(int(mode)))
        
    def close_device(self):
        self.dll.fnLPS_CloseDevice(self.devid)
        
    def init_device(self):   
        self.dll.fnLPS_InitDevice(self.devid)
        
    def get_id(self):
        return "Labbrick Phase Shifter model: %s serial #: %d" % (self.get_model_name(),self.devinfo['serial'])

    def persist_settings(self):
        self.dll.fnLPS_SaveSettings(self.devid)
        
    def get_min_working_frequency(self):
        return float(self.dll.fnLPS_GetMinWorkingFrequency(self.devid))*1e5

    def get_max_working_frequency(self):
        return float(self.dll.fnLPS_GetMaxWorkingFrequency(self.devid))*1e5

    def get_working_frequency(self):
        return float(self.dll.fnLPS_GetWorkingFrequency(self.devid))*1e5
        
    def set_working_frequency(self,frequency):
        self.dll.fnLPS_SetWorkingFrequency(self.devid,U32(int(frequency/1e5)))
        
    def get_phase(self):
        return float(self.dll.fnLPS_GetPhaseAngle(self.devid))
        
    def set_phase(self,angle,frequency=None):
        if frequency is not None:
            self.set_working_frequency(frequency)
        self.dll.fnLPS_SetPhaseAngle(self.devid,U32(int(angle)))
        
    def get_ramp_parameters(self):
        rp={}
        rp['start']=float(self.dll.fnLPS_GetRampStart(self.devid))
        rp['stop']=float(self.dll.fnLPS_GetRampEnd(self.devid))
        rp['dwell']=float(self.dll.fnLPS_GetDwellTime(self.devid)) /1000.
        rp['step']=float(self.dll.fnLPS_GetPhaseAngleStep(self.devid))
        rp['dwell2']=float(self.dll.fnLPS_GetDwellTimeTwo(self.devid)) /1000.
        rp['step2']=float(self.dll.fnLPS_GetPhaseAngleStepTwo(self.devid))
        rp['hold']=float(self.dll.fnLPS_GetHoldTime(self.devid)) /1000.
        rp['idle']=float(self.dll.fnLPS_GetIdleTime(self.devid)) /1000.
        return rp
              
class LMS103(Instrument):
    'The interface to the Lab Brick signal generator'
    def __init__(self,name="Labbrick",address=None,enabled=True):
        Instrument.__init__(self,name,address,enabled=True)
        #self.dll=C.CDLL(LMS103dllpath)
        self.dll=LMSDLL
        self.set_test_mode(False)
        if address is not None:
            self.init_by_serial(int(address))
        
    def init_by_serial(self,address):
        self.devinfo=self.get_info_by_serial(address)
        self.devid=self.devinfo['devid']
        if self.devid!= -1:
            self.init_device()
       
    def get_info_by_serial(self,serial):
        serial=int(serial)
        devinfos=LMS_get_device_info()
        for devinfo in devinfos:
            print devinfo
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

    def get_use_internal_pulse_mod(self): # This is broken -- Phil
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
        ret = self.dll.fnLMS_SetFrequency(self.devid,U32(int(frequency/10.0)))
        if ret != 0:
            raise ValueError("Lab Brick refused frequency %.2e Hz" % frequency)
        
    def get_output(self):
        return bool(self.dll.fnLMS_GetRF_On(self.devid))

    def set_output(self,state=True):
        self.dll.fnLMS_SetRFOn(self.devid,U8(state))
        
    def get_mod(self):
        return bool(self.dll.fnLMS_GetPulseMode(self.devid))

    def set_mod(self,mod=True):
        self.dll.fnLMS_EnableInternalPulseMod(self.devid,U8(mod))

    def set_ext_pulse(self,mod=True):
        self.dll.fnLMS_SetUseExternalPulseMod(self.devid,U8(mod))
        if mod:
            self.set_mod(True)
        else:
            self.set_mod(False)

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

try:
    from guiqwt.pyplot import *
    
        
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
except:
    print "Warning could not load LabBrick Gui"

if __name__=="__main__":
    #app = QApplication(sys.argv)
    #window = LabbrickWindow(address=1200)
    #window.show()
    #sys.exit(app.exec_())
    testrf=False
    if testrf:
        rf=LMS103(address=1200)
        
    testLPS=True
    if testLPS:
        lps=LPS802(address=4823)
        print lps.get_id()
        print lps.get_min_working_frequency()
        print lps.get_max_working_frequency()
        print lps.get_working_frequency()
        print lps.get_phase()
        lps.set_phase(lps.get_phase()+10,6e9)
        print lps.get_phase()

