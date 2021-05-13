# -*- coding: utf-8 -*-
"""
Agilent E8257D (rfgenerators.py)
================================
:Author: David Schuster
"""

from slab.instruments import SocketInstrument
from slab.instruments import Instrument
# from slab.instruments import SerialInstrument
import time

class N5183B(SocketInstrument):
    """
    The interface to the Keysite N5183B RF Generator, implemented on top of
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    default_port=5025
    def __init__(self,name='N5183B',address='',enabled=True,timeout=10, recv_length=1024):
        SocketInstrument.__init__(self,name=name,address=address,enabled=enabled,timeout=timeout,recv_length=recv_length)

    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()

    def set_output(self,state=True):
        """Set Output State On/Off"""
        if state: self.write(':OUTPUT:STATE ON')
        else:     self.write(':OUTPUT:STATE OFF')

    def get_output(self):
        """Query Output State"""
        return int(self.query(':OUTPUT?')) == 1

    def set_mod(self,state=True):
        """Set Modulation State On/Off"""
        if state: self.write(':OUTPUT:MOD:STATE ON')
        else:     self.write(':OUTPUT:MOD:STATE OFF')

    def get_mod(self):
        """Query Modulation State"""
        return bool(self.query(':OUTPUT:MOD?'))

    def set_frequency(self, frequency):
        """Set CW Frequency in Hz"""
        self.write(':FREQUENCY %f' % frequency)

    def get_frequency(self):
        """Query CW Frequency"""
        return float(self.query(':FREQUENCY?'))

    def set_sweep(self,start,stop,numpts,dwell):
        """Sets up frequency sweeping parameters"""
        self.write(':FREQUENCY:START %f;:FREQUENCY:STOP %f; :SWEEP:POINTS %f; :SWEEP:DWELL %f' % (start,stop,numpts,dwell))

    def get_sweep(self):
        """Gets current frequency sweeping parameters"""
        return [float(s) for s in (self.query(':FREQUENCY:START?'),self.query(':FREQUENCY:STOP?'), self.query(':SWEEP:POINTS?'),self.query(':SWEEP:DWELL?'))]

    def set_sweep_mode(self,enabled=True):
        """Set's up source for sweep mode"""
        if enabled:
            self.write(':LIST:TYPE STEP; :LIST:TRIG:SOURCE EXT; :FREQuency:MODE SWEEP')
        else:
            self.write(':FREQ:MODE CW; :LIST:TRIG:SOURCE CONT')

    def set_cw_mode(self):
        """Set generator into CW mode"""
        self.write(':FREQ:MODE CW; :LIST:TRIG:SOURCE CONT')

    def set_phase(self,phase):
        """Set signal Phase in radians"""
        self.write(':PHASE %f' % phase)

    def get_phase(self):
        """Query signal phase in radians"""
        return float(self.query(':PHASE?'))

    def set_power(self,power):
        """Set CW power in dBm"""
        self.write(':POWER %f' % power)

    def get_power(self):
        return float(self.query(':POWER?'))

    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['frequency']=self.get_frequency()
        settings['power']=self.get_power()
        settings['phase']=self.get_phase()
        settings['mod']=self.get_mod()
        settings['output']=self.get_output()
        settings['id']=self.get_id()
        return settings

    def get_settled(self):
        """Get source settled state"""
        return bool(self.query(':OUTPut:SETTled?'))

    def wait_to_settle(self, timeout=1):
        """Block until source settled"""
        start=time.time()
        while not self.get_settled() and time.time()-start<timeout: pass

    def set_internal_pulse(self,pulse_time=10e-6):
        self.write(':SOUR:PULM:SOUR:INT FRUN')
        self.write(':SOUR:PULM:INT:PERIOD %f S' %(pulse_time))
        self.write(':SOUR:PULM:INT:PWIDTH %f S' %(pulse_time))
        self.write(":SOUR:PULM:STAT ON")
        self.set_mod()

    def set_ext_pulse(self,mod=True):
        self.write(':SOUR:PULM:SOUR EXT')
        if mod:
            self.write(":SOUR:PULM:STAT ON")
            self.set_mod(True)
        else:
            self.write(":SOUR:PULM:STAT OFF")
            self.set_mod(False)

class E8257D(SocketInstrument):
    """
    The interface to the Agilent E8257D RF Generator, implemented on top of 
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    default_port=5025
    def __init__(self,name='E8257D',address='',enabled=True,timeout=10, recv_length=1024):
        SocketInstrument.__init__(self,name,address,enabled,timeout,recv_length)
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()
    
    def set_output(self,state=True):
        """Set Output State On/Off"""
        if state: self.write(':OUTPUT:STATE ON')
        else:     self.write(':OUTPUT:STATE OFF')
        
    def get_output(self):
        """Query Output State"""
        return int(self.query(':OUTPUT?')) == 1
    
    def set_mod(self,state=True):
        """Set Modulation State On/Off"""
        if state: self.write(':OUTPUT:MOD:STATE ON')
        else:     self.write(':OUTPUT:MOD:STATE OFF')
        
    def get_mod(self):
        """Query Modulation State"""
        return bool(self.query(':OUTPUT:MOD?'))
            
    def set_frequency(self, frequency):
        """Set CW Frequency in Hz"""
        self.write(':FREQUENCY %f' % frequency)
    
    def get_frequency(self):
        """Query CW Frequency"""
        return float(self.query(':FREQUENCY?'))

    def set_sweep(self,start,stop,numpts,dwell):
        """Sets up frequency sweeping parameters"""
        self.write(':FREQUENCY:START %f;:FREQUENCY:STOP %f; :SWEEP:POINTS %f; :SWEEP:DWELL %f' % (start,stop,numpts,dwell))

    def get_sweep(self):
        """Gets current frequency sweeping parameters"""
        return [float(s) for s in (self.query(':FREQUENCY:START?'),self.query(':FREQUENCY:STOP?'), self.query(':SWEEP:POINTS?'),self.query(':SWEEP:DWELL?'))]

    def set_sweep_mode(self,enabled=True):
        """Set's up source for sweep mode"""
        if enabled:
            self.write(':LIST:TYPE STEP; :LIST:TRIG:SOURCE EXT; :FREQuency:MODE SWEEP')
        else:
            self.write(':FREQ:MODE CW; :LIST:TRIG:SOURCE CONT')

    def set_cw_mode(self):
        """Set generator into CW mode"""
        self.write(':FREQ:MODE CW; :LIST:TRIG:SOURCE CONT')

    def set_phase(self,phase):
        """Set signal Phase in radians"""
        self.write(':PHASE %f' % phase)
    
    def get_phase(self):
        """Query signal phase in radians"""
        return float(self.query(':PHASE?'))
        
    def set_power(self,power):
        """Set CW power in dBm"""
        self.write(':POWER %f' % power)
        
    def get_power(self):
        return float(self.query(':POWER?'))

    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['frequency']=self.get_frequency()
        settings['power']=self.get_power()
        settings['phase']=self.get_phase()
        settings['mod']=self.get_mod()
        settings['output']=self.get_output()
        settings['id']=self.get_id()
        return settings
        
    def get_settled(self):
        """Get source settled state"""
        return bool(self.query(':OUTPut:SETTled?'))
        
    def wait_to_settle(self, timeout=1):
        """Block until source settled"""
        start=time.time()
        while not self.get_settled() and time.time()-start<timeout: pass
    
    def set_internal_pulse(self,pulse_time=10e-6):
        self.write(':SOUR:PULM:SOUR:INT FRUN')
        self.write(':SOUR:PULM:INT:PERIOD %f S' %(pulse_time))
        self.write(':SOUR:PULM:INT:PWIDTH %f S' %(pulse_time))
        self.write(":SOUR:PULM:STAT ON")
        self.set_mod()
    
    def set_ext_pulse(self,mod=True):
        self.write(':SOUR:PULM:SOUR EXT')
        if mod:
            self.write(":SOUR:PULM:STAT ON")
            self.set_mod(True)
        else:
            self.write(":SOUR:PULM:STAT OFF")
            self.set_mod(False)
        
class BNC845(SocketInstrument):
    """
    The interface to the BNC845 RF Generator, implemented on top of 
    :py:class:`~slab.instruments.instrumenttypes.SocketInstrument`
    """
    default_port=18
    def __init__(self,name='BNC845',address='', enabled=True, recv_length=1024, timeout=40):
        #SocketInstrument.__init__(self,name,address,5025,enabled,timeout,recv_length)        
        SocketInstrument.__init__(self,name,address,enabled,recv_length,timeout)
        
        #default set to external reference
        self.set_reference_source("EXT")
    
    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()
    
    def set_output(self,state=True):
        """Set Output State On/Off"""
        if state: self.write(':OUTPUT:STATE ON')
        else:     self.write(':OUTPUT:STATE OFF')
        
    def get_output(self):
        """Query Output State"""
        return int(self.query(':OUTPUT?')) == 1
                
    def set_frequency(self, frequency):
        """Set CW Frequency in Hz"""
        self.write(':FREQUENCY %f' % frequency)
    
    def get_frequency(self):
        """Query CW Frequency"""
        return float(self.query(':FREQUENCY?'))

    def set_phase(self,phase):
        """Set signal Phase in radians"""
        self.write(':PHASE %f' % phase)
    
    def get_phase(self):
        """Query signal phase in radians"""
        return float(self.query(':PHASE?'))
    
    #NOTE: The BNC is a fixed output power...this does nothing!!    
    def set_power(self,power):
        """Set CW power in dBm"""
        self.write(':POWER %f' % power)
        print("BNC845 is fixed output power - 13dBm")
        
        
    def get_power(self):
        return float(self.query(':POWER?'))
        
    def set_reference_source(self,source='INT',ref_freq=10e6):
        """Sets reference oscillator source: 'INT' or 'EXT'"""
        if source!='INT' and source!='EXT':
            raise Exception('BNC845: Invalid reference oscillator source %s, must be either INT or EXT' % source)
        self.write(':ROSCillator:SOURce %s' % source)
        #Note that the BNC845 cannot autodetect the reference oscillator frequency
        if source=='EXT':
            self.write(':ROSCillator:EXTernal:FREQuency %f' % ref_freq)
        
    def get_reference_source(self):
        """Gets reference oscillator source: 'INT' or 'EXT'"""
        return self.query(':ROSCillator:SOURce?').strip()
        
    def get_settings(self):
        settings=SocketInstrument.get_settings(self)
        settings['frequency']=self.get_frequency()
        settings['power']=self.get_power()
        settings['phase']=self.get_phase()
        #settings['mod']=self.get_mod()
        settings['output']=self.get_output()
        settings['id']=self.get_id()
        return settings
        
    def get_settled(self):
        """Get source settled state"""
        #return bool(self.query(':OUTPut:SETTled?'))
        #no call for the BNC845 to tell if the output has settled
        #data sheet says the frequency settles in <100us
        return True        
        
    def wait_to_settle(self, timeout=1):
        """Block until source settled"""
        start=time.time()
        while not self.get_settled() and time.time()-start<timeout: pass
    
    def set_pulse_state(self,state=True):
        """Set pulse state True/False"""
        if state: self.write(":SOUR:PULM:STAT ON")
        else:     self.write(":SOUR:PULM:STAT OFF")

    def set_internal_pulse(self,width,period,state=True):
        """Set up an internally generated pulse with: width, period, and state"""
        #self.write(':SOUR:PULM:SOUR:INT FRUN')
        self.write(':SOUR:PULM:INT:PERIOD %f S' %(width))
        self.write(':SOUR:PULM:INT:PWIDTH %f S' %(period))
        self.set_pulse_state(state)
    
    def set_ext_pulse(self,state=True, mod=False):
        self.write(':SOUR:PULM:SOUR EXT')
        self.set_pulse_state(state)
        

# Meg added this from Gerbert's SignalCore.py which is also saved in this directory, Oct 2020
# Unclear what instrument classes we can run here?
class SignalCore(Instrument):
    name = 'Signal Core'
    serial_numbers = ['10001E47']
    deviceName = []

    def __init__(self, name, address='', enabled=True, timeout=1, query_sleep=0, **kwargs):
        Instrument.__init__(self, name, address, enabled, timeout, **kwargs)
        self._dll = ctypes.CDLL(r'C:\Program Files\SignalCore\SC5511A\api\c\x64\sc5511a.dll')
        self.set_signatures()
        self._handle = self.open_device()

    def open_device(self):
        return self._dll.sc5511a_open_device(self.address.encode('utf-8'))

    def close_device(self):
        return self._dll.sc5511a_close_device(self._handle)

    def get_rf_parameters(self):
        params = RFParameters()
        self._dll.sc5511a_get_rf_parameters(self._handle, params)
        return params

    def set_clock_reference(self, ext_ref=True):
        return self._dll.sc5511a_set_clock_reference(self._handle, 0, int(ext_ref))  # lock to the external 10 MHz clock.

    # Altered by MGP april 14 2021. tried to make set_output but this was bad. changed back
    def set_output_state(self, enable=True):
    # def set_output(self, enable=True):
        done = self._dll.sc5511a_set_output(self._handle, int(enable))
        if done == 0:
            print(self.name + self.address + ' : Set output state to %s' % enable)
        else:
            print(self.name + self.address + ' : Fail to set output state, please check the device status!' % freq)
        return done

    def set_frequency(self, freq, acknowledge = True):
        '''Todo: check RF1 output beyond 20 GHz'''
        freq = int(freq)
        if freq < 80000000:
            freq = 80000000
            print('Warning: Available range: 80 MHz to 20.505847837 GHz')

        if freq > 20505847837:
            freq = 20505847837
            print('Warning: Available range: 80 MHz to 20.505847837 GHz')

        done = self._dll.sc5511a_set_freq(self._handle, freq)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set frequency to %e Hz' % freq)
        else:
            print(self.name + self.address + ' : Failed to set frequency to %e, please check the device status!' % freq)
        return done

    def set_rf2_frequency(self, freq):
        done = self._dll.sc5511a_set_rf2_freq(self._handle, freq)
        if done == 0:
            print(self.name + self.address + ' : Set RF2 frequency to %e Hz' % freq)
        else:
            print(self.name + self.address + ' : Failed to set RF2 frequency to %e, please check the device status!' % freq)
        return done

    def set_power(self, pdBm = -10):
        if pdBm < -30:
            pdBm = -30
            print('Warning: Available range: -30 to +13 dBm')

        if pdBm > 13:
            pdBm = 13
            print('Warning: Available range: -30 to +13 dBm')

        done = self._dll.sc5511a_set_level(self._handle, pdBm)
        if done == 0:
            print(self.name + self.address + ' : Set power to %.2f dBm' % pdBm)
        else:
            print(self.name + self.address + ' : Failed to set power to %.2f, please check the device status!' % pdBm)
        return done

    def set_rf_mode(self, val=0, acknowledge = True):
        """ Sets RF1 to fixed tone (val=0) or sweep (val=1) """
        done = self._dll.sc5511a_set_rf_mode(self._handle, val)
        if done == 0:
            state = 'Single Tone' if val == 0 else 'List/Sweep'
            if acknowledge:
                print(self.name + self.address + ' : Set RF mode to %s (%s) ' % (val,state))
        else:
            print(self.name + self.address + ' : Failed to set RF mode, please check the device status!')
        return done

    def set_list_start_freq(self, freq, acknowledge = True):
        """ Frequency in Hz """
        freq = int(freq)
        done = self._dll.sc5511a_list_start_freq(self._handle, freq)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set start freq. to %e Hz' % freq)
        else:
            print(self.name + self.address + ' : Failed to set start freq., please check the device status!')
        return done

    def set_list_stop_freq(self, freq, acknowledge = True):
        """ Frequency in Hz """
        freq = int(freq)
        done = self._dll.sc5511a_list_stop_freq(self._handle, freq)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set stop freq. to %e Hz' % freq)
        else:
            print(self.name + self.address + ' : Failed to set stop freq., please check the device status!')
        return done

    def set_list_step_freq(self, freq, acknowledge = True):
        """ Frequency in Hz """
        freq = int(freq)
        done = self._dll.sc5511a_list_step_freq(self._handle, freq)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set step freq. to %e Hz' % freq)
        else:
            print(self.name + self.address + ' : Failed to set step freq., please check the device status!')
        return done

    def set_list_dwell_time(self, tunit, acknowledge = True):
        """ 1: 500 us, 2: 1 ms ... """
        done = self._dll.sc5511a_list_dwell_time(self._handle, tunit)
        time = tunit*500
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set dwell time to %u us' % time)
        else:
            print(self.name + self.address + ' : Failed to set dwell time, please check the device status!')
        return done

    def set_list_cycle_count(self, num, acknowledge = True):
        """ Sets the number of sweep cycles to perform before stopping.
            To repeat the sweep continuously, set the value to 0.
        """
        done = self._dll.sc5511a_list_cycle_count(self._handle, num)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Set list cycle count to %u' % num)
        else:
            print(self.name + self.address + ' : Failed to set list cycle count, please check the device status!')
        return done

    def set_list_cycle_count(self, acknowledge = True):
        """ triggers the device when it is configured for list mode and soft trigger is selected as the trigger source.
        """
        done = self._dll.sc5511a_list_soft_trigger(self._handle)
        if done == 0:
            if acknowledge:
                print(self.name + self.address + ' : Software trigger is sent')
        else:
            print(self.name + self.address + ' : Failed to send software trigger, please check the device status!')
        return done

    def set_auto_level_disable(self, disable=False):
        """ Disables the leveling compensation after the frequency is changed for channel RF1.
        """
        done = self._dll.sc5511a_set_auto_level_disable(self._handle, disable)
        if done == 0:
            print(self.name + self.address + ' : Set auto level disable to %s' % disable)
        else:
            print(self.name + self.address + ' : Failed to set auto level disable, please check the device status!')
        return done

    def set_alc_mode(self, mode=0):
        """ Sets the ALC to close (0) or open (1) mode operation for channel RF1.
        """
        done = self._dll.sc5511a_set_alc_mode(self._handle, mode)
        if done == 0:
            print(self.name + self.address + ' : Set ALC mode to %s' % mode)
        else:
            print(self.name + self.address + ' : Failed to set ALC mode, please check the device status!')
        return done

    def set_standby(self, enable=False):
        """ If enabled powers down channel RF1.
        """
        done = self._dll.sc5511a_set_standby(self._handle, enable)
        if done == 0:
            print(self.name + self.address + ' : Set RF1 standby mode to %s' % enable)
        else:
            print(self.name + self.address + ' : Failed to set standby mode, please check the device status!')
        return done

    def set_rf2_standby(self, enable=False):
        """ If enabled powers down channel RF2.
        """
        done = self._dll.sc5511a_set_rf2_standby(self._handle, enable)
        if done == 0:
            print(self.name + self.address + ' : Set RF2 standby mode to %s' % enable)
        else:
            print(self.name + self.address + ' : Failed to set RF2 standby mode, please check the device status!')
        return done

    def set_list_mode(self, sss_mode=1, sweep_dir=0, tri_waveform=0, hw_tdrigger=0, step_on_hw_trig=0,
                        return_to_start=0, trig_out_enable=1, trig_out_on_cycle=0):
        """ Configures list mode
            sss_mode = 0: List mode, 1: Sweep mode
            sweep_dir = 0: Forward, 1: Reverse
            tri_waveform = 0: Sawtooth waveform, 1: Triangular waveform
            hw_trigger = 0: Software trigger, 1: Hardware trigger
            step_on_hw_trig = 0: Start/Stop behavior, 1: Step on trigger, see manual for more details
            return_to_start = 0: Stop at end of sweep/list, 1: Return to start
            trig_out_enable = 0: No output trigger, 1: Output trigger enabled on trigger pin (#21)
            trig_out_on_cycle = 0: 0 = Puts out a trigger pulse at each frequency change,
                                1: trigger pulse at the completion of each sweep/list cycle.
        """
        lm = ListMode(sss_mode=sss_mode, sweep_dir=sweep_dir, tri_waveform=tri_waveform,hw_trigger=hw_trigger,
                    step_on_hw_trig=step_on_hw_trig, return_to_start=return_to_start, trig_out_enable=trig_out_enable,
                    trig_out_on_cycle=trig_out_on_cycle)

        done = self._dll.sc5511a_list_mode_config(self._handle, lm)
        if done == 0:
            print(self.name + self.address + ' : Successfully configured list mode')
        else:
            print(self.name + self.address + ' : Failed to set RF2 standby mode, please check the device status!')
        return done

    def get_device_status(self):
        """
        """
        ds = DeviceStatus()

        done = self._dll.sc5511a_get_device_status(self._handle, ds)
        if done == 0:
            print(self.name + self.address + ' : Successfully obtained device status')
        else:
            print(self.name + self.address + ' : Failed to get device status, please check the device status!')
        return ds

    def set_signatures(self):
        """Set the signature of the DLL functions we use.
        """
        self._dll.sc5511a_open_device.argtypes = [c_char_p]
        self._dll.sc5511a_open_device.restype = c_void_p

        self._dll.sc5511a_close_device.argtypes = [c_void_p]
        self._dll.sc5511a_close_device.restype = c_int

        self._dll.sc5511a_set_freq.argtypes = [c_void_p, c_ulonglong]
        self._dll.sc5511a_set_freq.restype = c_int

        self._dll.sc5511a_set_rf2_freq.argtypes = [c_void_p, c_ushort]
        self._dll.sc5511a_set_rf2_freq.restype = c_int

        self._dll.sc5511a_set_level.argtypes = [c_void_p, c_float]
        self._dll.sc5511a_set_level.restype = c_int

        self._dll.sc5511a_set_clock_reference.argtypes = [c_void_p, c_char, c_char]
        self._dll.sc5511a_set_clock_reference.restype = c_int

        self._dll.sc5511a_set_output.argtypes = [c_void_p, c_ubyte]
        self._dll.sc5511a_set_output.restype = c_int

        self._dll.sc5511a_set_rf_mode.argtypes = [c_void_p, c_ubyte]

        self._dll.sc5511a_list_start_freq.argtypes = [c_void_p, c_ulonglong]
        self._dll.sc5511a_list_start_freq.restype = c_int

        self._dll.sc5511a_list_stop_freq.argtypes = [c_void_p, c_ulonglong]
        self._dll.sc5511a_list_stop_freq.restype = c_int

        self._dll.sc5511a_list_step_freq.argtypes = [c_void_p, c_ulonglong]
        self._dll.sc5511a_list_step_freq.restype = c_int

        self._dll.sc5511a_list_dwell_time.argtypes = [c_void_p, c_uint]
        self._dll.sc5511a_list_dwell_time.restype = c_int

        self._dll.sc5511a_list_cycle_count.argtypes = [c_void_p, c_uint]
        self._dll.sc5511a_list_cycle_count.restype = c_int

        self._dll.sc5511a_set_auto_level_disable.argtypes = [c_void_p, c_ubyte]
        self._dll.sc5511a_set_auto_level_disable.restype = c_int

        self._dll.sc5511a_set_alc_mode.argtypes = [c_void_p, c_ubyte]
        self._dll.sc5511a_set_alc_mode.restype = c_int

        self._dll.sc5511a_set_standby.argtypes = [c_void_p, c_ubyte]
        self._dll.sc5511a_set_standby.restype = c_int

        self._dll.sc5511a_set_rf2_standby.argtypes = [c_void_p, c_ubyte]
        self._dll.sc5511a_set_rf2_standby.restype = c_int

        self._dll.sc5511a_get_rf_parameters.argtypes = [c_void_p, POINTER(RFParameters)]
        self._dll.sc5511a_get_rf_parameters.restype = c_int

        self._dll.sc5511a_get_device_status.argtypes = [c_void_p, POINTER(DeviceStatus)]
        self._dll.sc5511a_get_device_status.restype = c_int

        self._dll.sc5511a_list_mode_config.argtypes = [c_void_p, POINTER(ListMode)]
        self._dll.sc5511a_list_mode_config.restype = c_int

def test_BNC845(rf=None):
    if rf is None:
        rf=BNC845(address='192.168.14.151')
    
    print(rf.get_id())
    rf.set_output(False)
    print("Output: ", rf.get_output())
    rf.set_frequency(10e9)
    print("Frequency: %g" % rf.get_frequency())
    print("Reference source: %s" % rf.get_reference_source())
    
    
    
def test_8257D (rf=None):   
    if rf is None:
        rf=E8257D(address='192.168.14.131')
    print(rf.query('*IDN?'))
    rf.set_output(False)
    rf.set_frequency(10e9)
    print("Frequency: %f" % rf.get_frequency())
    rf.set_power(-10)
    print("Power: %f" % rf.get_power())
    rf.set_sweep(start=1e9,stop=2e9,numpts=101,dwell=10e-3)
    rf.set_sweep_mode()
    
if __name__=="__main__":
    test_8257D()
    #test_BNC845()

