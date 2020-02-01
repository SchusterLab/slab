"""
Driver Signal Core SC5511A signal generator. Python 3.7

Address is the serial number of the instrument

Authors: Dave & Tanay

Todo: Read status of the instrument
May 17, 2019.

"""



from slab.instruments import Instrument
import ctypes
from ctypes import c_char_p, c_void_p, c_int, c_ulonglong, c_float, c_ubyte, c_char, c_ushort, c_uint, POINTER, \
    Structure


class ListMode(Structure):
    """Structure represnting the list mode.
    """
    _fields_ = [(name, c_ubyte)
                for name in ('sss_mode', 'sweep_dir', 'tri_waveform',
                             'hw_trigger', 'step_on_hw_trig', 'return_to_start',
                             'trig_out_enable', 'trig_out_on_cycle')]


class OperateStatus(Structure):
    """Structure representing the operate status.
    """
    _fields_ = [(name, c_ubyte)
                for name in ('rf1_lock_mode', 'rf1_loop_gain', 'device_access',
                             'rf2_standby', 'rf1_standby', 'auto_pwr_disable',
                             'alc_mode', 'rf1_out_enable', 'ext_ref_lock_enable',
                             'ext_ref_detect', 'ref_out_select', 'list_mode_running',
                             'rf1_mode', 'over_temp', 'harmonic_ss')]


class PllStatus(Structure):
    """Structure representing the PLL status.
    """
    _fields_ = [(name, c_ubyte)
                for name in ('sum_pll_ld', 'crs_pll_ld', 'fine_pll_ld',
                             'crs_ref_pll_ld', 'crs_aux_pll_ld',
                             'ref_100_pll_ld', 'ref_10_pll_ld', 'rf2_pll_ld')]


class DeviceStatus(Structure):
    """Structure representing the device status.
    """
    _fields_ = [('list_mode', ListMode),
                ('operate_status', OperateStatus),
                ('pll_status', PllStatus)]


class RFParameters(Structure):
    """Structure representing the RF parameters.
    """
    _fields_ = [('rf1_freq', c_ulonglong),
                ('start_freq', c_ulonglong),
                ('stop_freq', c_ulonglong),
                ('step_freq', c_ulonglong),
                ('sweep_dwell_time', c_uint),
                ('sweep_cycles', c_uint),
                ('buffer_points', c_uint),
                ('rf_level', c_float),
                ('rf2_freq', c_ushort)]


class SignalCore(Instrument):
    name = 'Signal Core'
    serial_numbers = ['10001E48']
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

    def set_output_state(self, enable=True):
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
        if pdBm < -40:
            pdBm = -40
            print('Warning: Available range: -40 to +30 dBm')

        if pdBm > 30:
            pdBm = 30
            print('Warning: Available range: -40 to +30 dBm')

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

    def set_list_mode(self, sss_mode=1, sweep_dir=0, tri_waveform=0, hw_trigger=0, step_on_hw_trig=0,
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


if __name__ == '__main__':
    print('starting signalcores')
    # sc1 = SignalCore(name="SignalCore",address="1000209F")
    # sc2 = SignalCore(name="SignalCore", address="100020A0")
    # sc3 = SignalCore(name="SignalCore", address="100020A1")
    # # sc.set_clock_reference(ext_ref=False)
    # sc1.set_power(-50)
    # # sc.set_output_state(True)
    # # sc.set_rf_mode(0)
    # sc1.set_frequency(10e9)
    # sc2.set_frequency(11e9)
    # sc3.set_frequency(12e9)
    # # sc.set_list_start_freq(2000000000)
    # # sc.set_list_stop_freq(10000000000)
    # # sc.set_list_dwell_time(10)
    # # sc.set_list_cycle_count(100)
    # # sc.set_auto_level_disable(0)
    # # sc.set_alc_mode(0)
    # sc2.set_standby(0)
    # # sc.set_rf2_frequency(1000)
    # # sc.set_rf2_standby(0)
    # sc1.close_device()
    # sc2.close_device()
    # sc3.close_device()

