# -*- coding: utf-8 -*-
"""
Lock in amplifiers
===========================================
:Author: Gerwin Koolstra
"""

#from slab.instruments import VisaInstrument
from instrumenttypes import *
import time
import numpy as np

class SR844(VisaInstrument):
    
    def __init__(self, name="SR844", address='GPIB0::2::INSTR', enabled=True, **kwargs):
        VisaInstrument.__init__(self, name, address, enabled, term_chars='\r', **kwargs)
        self.query_sleep=0.05
        self.recv_length=65536
        #self.term_char=term_chars

        # Time constants and slopes
        self.tc = np.array([1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, 3E-1,
                            1.0, 3.0, 10., 30., 100., 300., 1E3, 3E3, 1E4, 3E4])
        self.slopes = np.array([0, 6, 12, 18, 24])

        self.sensitivities = np.arange(-127, 23, 10)
        self.displaych1 = ["X", "R [Volts rms]", "R [dBm]", "X noise", "AUX IN"]
        self.displaych2 = ["Y", "Theta", "Y noise [Volts]", "Y noise [dBm]", "AUX IN 2"]


    def get_id(self):
        return self.query('*IDN?')

    def find_nearest(self, array, value):
        """
        Finds the nearest value in array. Returns index of array for which this is true.
        """
        idx=(np.abs(array-value)).argmin()
        return idx

    def get_status(self, bitnum):
        """
        Nicholas B. Schade, 2015-12-16
        Returns the various aspects of the status of the lockin,
        depending on the value of "bitnum" as indicated below
        
        bitnum   = 1 when:
        0        A reference unlock is detected.
        1        The reference frequency is out of range.
        3        Data storage is triggered.
        4        The signal input overloads.
        5        The IF amplifier overloads.
        6        A time constant filter overloads.
        7        Reference frequency changed by more than 1%.
        8        Channel 1 display or output overloads.
        9        Channel 2 display or output overloads.
        10       Either Aux Input overloads.
        11       Ratio input underflows.
        """
        return int(self.query("LIAS? %d"%bitnum))

    def set_reference_mode(self, mode='ext'):
        """
        Sets the reference mode of the lock-in to external or internal.
        :param mode: "ext" or "int"
        :return:
        """
        if mode == 'ext':
            self.write("FMOD 0")
        else:
            self.write("FMOD 1")

    def get_reference_mode(self):
        table = {"0" : "ext", "1" : "int"}
        return table[self.query("FMOD?")]

    def set_reference_frequency(self, frequency):
        """
        Sets the reference frequency. Only allowed if reference mode is "int"
        :param frequency: Frequency in Hz
        :return:
        """
        if ((self.get_reference_mode() == 'int') & (self.get_mode_2f() == 0)):
            # Bug fixed by Nick Schade, 2016-01-07
            self.write("FREQ %.2f"%frequency)
        else:
            print "Reference mode is external or 2f! Set to internal, 1f first."

    def get_reference_frequency(self):
        return float(self.query("FREQ?"))
        
    def set_mode_2f(self, mode_2f):
        """
        Nicholas B. Schade, 2015-12-16
        Add function for setting the lockin to detect at 2F
        """
        if mode_2f == 0:
            self.write("HARM 0")
        elif mode_2f == 1:
            self.write("HARM 1")
        else:
            print "mode_2f must be 0 (off) or 1 (on).  Value ignored."
    
    def get_mode_2f(self):
        """
        Nicholas B. Schade, 2015-12-16
        Add function for getting the lockin 2F detection mode
        """
        return int(self.query("HARM?"))

    def get_if_frequency(self):
        """
        The FRIQ? command queries the IF frequency. The returned value is an integer with units of Hz.
        The IF frequency will be in the range of approximately 2–3 kHz, for all time constants 1 ms and longer.
        :return:
        """
        return np.float(self.query("FRIQ?"))

    def set_detection_phase(self, phi):
        """
        Sets the detection phase in degrees, relative to the reference. phi must lie between -360 < phi < +360
        :param phi:
        :return:
        """
        self.write("PHAS %.2f"%phi)

    def get_detection_phase(self):
        return self.query("PHAS?")

    def set_input_impedance(self, impedance):
        """
        Sets the input impedance.
        :param impedance: May be a float 50, or 1E6
        :return:
        """
        if impedance < 100:
            # Input impedance = 50 Ohm
            self.write("INPZ 0")
        else:
            # Input impedance = 1 MOhm
            self.write("INPZ 1")

    def get_input_impedance(self):
        """
        Returns the input impedance
        :return:
        """
        ans = 50 if np.int(self.query("INPZ?")) < 1 else 1E6
        return ans

    def set_reference_impedance(self, impedance):
        """
        Sets the reference input impedance.
        :param impedance: May be a float 50, or 1E4
        :return:
        """
        if impedance < 100:
            # Input impedance = 50 Ohm
            self.write("REFZ 0")
        else:
            # Input impedance = 10 kOhm
            self.write("REFZ 1")

    def get_reference_impedance(self):
        """
        Returns the reference input impedance
        :return:
        """
        ans = 50 if np.int(self.query("REFZ?")) < 1 else 10000
        return ans

    def set_wide_reserve_mode(self, mode):
        """
        Mode may be High reserve (mode = 0), Normal (mode = 1) or Low noise (mode = 2)
        :return:
        """
        if mode in [0, 1, 2]:
            self.write("WRSV %d"%mode)
        else:
            print "Specified mode is invalid! Specify mode as 0, 1 or 2, for high reserve, normal or low noise, resp."

    def get_wide_reserve_mode(self):
        ans = np.int(self.query("WRSV?"))
        table = ["High reserve", "Normal", "Low noise"]
        return table[ans]

    def set_close_reserve_mode(self, mode):
        """
        Nicholas B. Schade, 2016-01-13
        Mode may be High reserve (mode = 0), Normal (mode = 1) or Low noise (mode = 2)
        :return:
        """
        if mode in [0, 1, 2]:
            self.write("CRSV %d"%mode)
        else:
            print "Specified mode is invalid! Specify mode as 0, 1 or 2, for high reserve, normal or low noise, resp."

    def get_close_reserve_mode(self):
        """
        Nicholas B. Schade, 2016-01-13
        """
        ans = np.int(self.query("CRSV?"))
        table = ["High reserve", "Normal", "Low noise"]
        return table[ans]

    def set_sensitivity(self, sensitivity):
        """
        Set sensitivity of the output step. sensitivity may range from -127 dBm to +13 dBm. A value will be interpollated.
        :param sensitivity:
        :return:
        """
        i = range(0, 15)
        sens = i[self.find_nearest(sensitivity, self.sensitivities)]
        self.write("SENS %d"%sens)

    def get_sensitivity(self):
        return self.sensitivities[np.int(self.query("SENS?"))]

    def set_time_constant(self, time_constant):
        """
        Select a sensitivity from [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, 3E-1, 1.0, 3.0, 10., 30., 100., 300., 1E3, 3E3, 1E4, 3E4]
        :param time_constant:
        :return:
        """
        i = range(0, 18)
        timec = i[self.find_nearest(time_constant, self.tc)]
        self.write("OFLT %d"%timec)

    def get_time_constant(self):
        return self.tc[np.int(self.query("OFLT?"))]

    def set_filter_slope(self, slope):
        """
        Select the time constant filter slope. The parameter slope can be 6 dB/oct, 12, 18 or 24 dB/oct
        :param slope:
        :return:
        """

        if slope in self.slopes:
            i = self.find_nearest(self.slopes, slope)
            self.write("OFSL %d"%i)
        else:
            print "Slope input is invalid. Please select from 6, 12, 18 or 24."

    def get_filter_slope(self):
        """
        Queries the time constant filter slope.
        :return:
        """
        return self.slopes[np.int(self.query("OFSL?"))]

    def set_ch1_display(self, quantity):
        """
        Sets the display for the ch1 output
        :param quantity: may be 0, 1, 2, 3, 4 according to
        0 : X
        1 : R [Volts rms]
        2 : R [dBm]
        3 : X noise
        4 : AUX IN
        :return:
        """
        if quantity in range(5):
            self.write("DDEF 1, %d"%quantity)
        else:
            "Bad input for quantity ch1, must be between 0 and 4"

    def get_ch1_display(self):
        return self.displaych1[np.int(self.query("DDEF? 1"))]

    def set_ch2_display(self, quantity):
        """
        Sets the display for the ch2 output
        :param quantity: may be 0, 1, 2, 3, 4 according to
        0 : Y
        1 : theta
        2 : Y Noise [Volts]
        3 : Y Noise [dBm]
        4 : AUX IN 2
        :return:
        """
        if quantity in range(5):
            self.write("DDEF 2, %d"%quantity)
        else:
            "Bad input for quantity ch2, must be between 0 and 4"

    def get_ch2_display(self):
        return self.displaych2[np.int(self.query("DDEF? 2"))]

    def get_xy(self):
        ans = self.query("SNAP? 1,2").split(',')
        return [np.float(ans[i]) for i in range(2)]

    def get_rtheta(self):
        ans = self.query("SNAP? 3,5").split(',')
        return [np.float(ans[i]) for i in range(2)]

    def integrate_single(self, integration_time, sample_rate):
        """
        :param integration_time:
        :param sample_rate: Must be a power of 2
        :return:
        """

        # Set the sample rate
        self.write("SRAT %d"%(np.log2(sample_rate)+4))

        # Deletes the data buffers
        self.write("REST")
        # Starts or resumes the data storage
        self.write("STRT")

        t0 = time.time()

        while time.time() - t0 < integration_time:
            time.sleep(0.01)

        # Pause data storage
        self.write("PAUS")

        # Get number of points in buffer:
        noof_points = np.int(self.query("SPTS?"))

        # Read out buffer
        raw_ch1 = filter(None, self.query("TRCA? 1,0,%d"%(noof_points-1)).split(','))
        raw_ch2 = filter(None, self.query("TRCA? 2,0,%d"%(noof_points-1)).split(','))

        ch1 = [np.float(raw_ch1[i]) for i in range(len(raw_ch1))]
        ch2 = [np.float(raw_ch2[i]) for i in range(len(raw_ch2))]

        return ch1, ch2

if __name__ == '__main__':
    print "Testing SR844"

    get_func_list = [self.get_id,
                     self.get_reference_frequency,
                     self.get_x_y]

    for idx,p in enumerate(get_func_list):
        try:
            p()
        except:
            print "%d. Error!"%idx


'''
Nicholas B. Schade
January 29, 2016
Create similar class for SR830 lock-in amplifier
'''
class SR830(VisaInstrument):
    
    def __init__(self, name="SR830", address='GPIB0::2::INSTR', enabled=True, **kwargs):
        VisaInstrument.__init__(self, name, address, enabled, term_chars='\r', **kwargs)
        self.query_sleep=0.05
        self.recv_length=65536
        #self.term_char=term_chars

        # Time constants and slopes
        self.tc = np.array([1E-5, 3E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, 3E-1, 1E0, 3E0, 1E1, 3E1, 1E2, 3E2, 1E3, 3E3, 1E4, 3E4])
        self.slopes = np.array([6, 12, 18, 24])

        self.displaych1 = ["X", "R", "X noise", "AUX IN 1", "AUX IN 2"]
        self.displaych2 = ["Y", "Theta", "Y noise", "AUX IN 3", "AUX IN 4"]


    def get_id(self):
        return self.query('*IDN?')

    def find_nearest(self, array, value):
        """
        Finds the nearest value in array. Returns index of array for which this is true.
        """
        idx=(np.abs(array-value)).argmin()
        return idx

    def get_status(self, bitnum):
        """
        Returns the various aspects of the status of the lockin,
        depending on the value of "bitnum" as indicated below
        
        bitnum   = 1 when:
        0        An Input or Amplifier overload is detected.
        1        A Time Constant filter overload is detected.
        2        An Output overload is detected.
        3        A reference unlock is detected.
        4        Detection frequency has switched range (above or below ~200Hz).
        5        The time constant has changed indirectly, due to change in 
                 frequency range, dynamic reserve, filter slope, or expand.
        6        Data storage is triggered, only if samples or scans are in 
                 externally triggered mode.
       """
        return int(self.query("LIAS? %d"%bitnum))

    def set_reference_mode(self, mode='ext'):
        """
        Sets the reference mode of the lock-in to external or internal.
        :param mode: "ext" or "int"
        :return:
        """
        if mode == 'ext':
            self.write("FMOD 0")
        else:
            self.write("FMOD 1")

    def get_reference_mode(self):
        table = {"0" : "ext", "1" : "int"}
        return table[self.query("FMOD?")]

    def set_reference_frequency(self, frequency):
        """
        Sets the reference frequency. Only allowed if reference mode is "int"
        :param frequency: Frequency in Hz
        :return:
        """
        if (self.get_reference_mode() == 'int'):
            self.write("FREQ %.2f"%frequency)
        else:
            print "Warning: Reference mode is external.  Set to internal first."

    def get_reference_frequency(self):
        return float(self.query("FREQ?"))
        
    def set_reference_amplitude(self, amplitude):
        """
        Nicholas B. Schade, 2016-01-29
        Set the amplitude of the internal reference output
        Amplitude must be between 0.004 and 5.000 V
        :param amplitude: Voltage in V:
        :return:
        """
        if amplitude < 0.004:
            print "Reference amplitude must be at least 4 mV.  Value ignored."
        elif amplitude > 5.000:
            print "Reference amplitude must be less than 5 V.  Value ignored."
        else:
            self.write("SLVL %.2f"%amplitude)

    def get_reference_amplitude(self):
        """
        Nicholas B. Schade, 2016-01-29
        Get the amplitude of the internal reference output
        """
        return float(self.query("SLVL?"))

    def set_detection_harmonic(self, harmonic):
        """
        Nicholas B. Schade, 2016-01-29
        Add function for setting the detection harmonic
        """
        newfreq = self.get_reference_frequency() * harmonic
        if harmonic <= 0:
            print "Warning: Detection harmonic must be positive.  Value ignored."
        elif harmonic > 19999:
            print "Warning: Detection harmonic must be < 20000.  Value ignored."

        elif newfreq < 102000:
            self.write("HARM %d"%harmonic)
        else:
            print "Warning: Detection frequency too high.  Value ignored."
    
    def get_detection_harmonic(self):
        """
        Nicholas B. Schade, 2016-01-29
        Add function for getting the detection harmonic
        """
        return int(self.query("HARM?"))

    def set_detection_phase(self, phi):
        """
        Sets the detection phase in degrees, relative to the reference. phi must lie between -360 < phi < +729.99
        :param phi:
        :return:
        """
        self.write("PHAS %.2f"%phi)

    def get_detection_phase(self):
        return self.query("PHAS?")

    def set_input_config(self, config):
        """
        Nicholas B. Schade, 2016-01-29
        Config  Meaning
        0       A
        1       A-B
        2       1 megaohm
        3       100 megaohm
        :param config:
        :return:
        """
        if (config >=0 and config <= 3):
            self.write("ISRC %d"%config)
        else:
            print "Warning: input config must be 0, 1, 2, or 3.  Value ignored."

    def get_input_config(self):
        """
        Nicholas B. Schade, 2016-01-29
        """
        return self.query("ISRC?")

    def set_input_shield_ground(self, mode):
        """
        Nicholas B. Schade, 2016-01-29
        value  meaning
        0      float
        1      ground
        :param mode:
        :return:
        """
        if (mode==0 or mode==1):
            self.write("IGND %d"%mode)
        else:
            print "Input shield grounding must be 0 or 1.  Value ignored."

    def get_input_shield_ground(self):
        """
        Nicholas B. Schade, 2016-01-29
        """
        return self.query("IGND?")

    def set_input_coupling(self, mode):
        """
        Nicholas B. Schade, 2016-01-29
        value  meaning
        0      AC
        1      DC
        :param mode:
        :return:
        """
        if (mode==0 or mode==1):
            self.write("ICPL %d"%mode)
        else:
            print "Warning: Input coupling must be 0 or 1.  Value ignored."

    def get_input_coupling(self):
        """
        Nicholas B. Schade, 2016-01-29
        """
        return self.query("ICPL?")

    def set_input_notch(self, mode):
        """
        Nicholas B. Schade, 2016-01-29
        Set input line notch filter status as follows:
        value  meaning
        0      Out, or no filters
        1      Line notch in
        2      2x Line notch in
        3      Both notch filters in
        :param mode:
        :return:
        """
        if (mode >= 0 and mode <= 3):
            self.write("ILIN %d"%mode)
        else:
            print "Input notch filter must be 0, 1, 2, or 3.  Value ignored."

    def get_input_notch(self):
        """
        Nicholas B. Schade, 2016-01-29
        """
        return self.query("ILIN?")

    def set_reserve_mode(self, mode):
        """
        Dynamic reserve mode may be High reserve (mode = 0), Normal (mode = 1) or Low noise (mode = 2)
        :return:
        """
        if mode in [0, 1, 2]:
            self.write("RMOD %d"%mode)
        else:
            print "Specified mode is invalid! Specify mode as 0, 1 or 2, for high reserve, normal or low noise, resp."

    def get_reserve_mode(self):
        ans = np.int(self.query("RMOD?"))
        return ans
        
        # table = ["High reserve", "Normal", "Low noise"]
        # return table[ans]

    def set_sensitivity(self, sensitivity):
        """
        Set sensitivity of the output step:
        value  setting
         0       2 nV/fA
         1       5 nV/fA
         2      10 nV/fA
         3      20 nV/fA
         4      50 nV/fA
         5     100 nV/fA
         6     200 nV/fA
         7     500 nV/fA
         8       1 uV/pA
         9       2 uV/pA
        10       5 uV/pA
        11      10 uV/pA
        12      20 uV/pA
        13      50 uV/pA
        14     100 uV/pA
        15     200 uV/pA
        16     500 uV/pA
        17       1 mV/nA
        18       2 mV/nA
        19       5 mV/nA
        20      10 mV/nA
        21      20 mV/nA
        22      50 mV/nA
        23     100 mV/nA
        24     200 mV/nA
        25     500 mV/nA
        26       1  V/uA
        :param sensitivity:
        :return:
        """
        # i = range(0, 27)
        # sens = i[self.find_nearest(sensitivity, self.sensitivities)]
        # self.write("SENS %d"%sens)
        self.write("SENS %d"%sensitivity)

    def get_sensitivity(self):
        return self.query("SENS?")

    def auto_gain(self, timec):
        '''
        Nicholas B. Schade, 2016-02-01
        Sets the sensitivity of the lockin automatically.
        :param timec:
        '''
        self.write("AGAN")

        '''
        # Let the auto-gain function continue until the lockin reports
        # that no command execution is in progress.
        in_progress = 1
        while in_progress==1:
            time.sleep(timec)
            in_progress = 1 - self.query("STB? 1")
        '''
        
    def auto_reserve(self, timec):
        '''
        Nicholas B. Schade, 2016-02-01
        Sets the dynamic reserve mode of the lockin automatically.
        :param timec:
        '''
        self.write("ARSV")

        '''
        # Let the auto-reserve function continue until the lockin reports
        # that no command execution is in progress.
        in_progress = 1
        while in_progress==1:
            time.sleep(timec)
            in_progress = 1 - self.query("STB? 1")
        '''
        
    def set_time_constant(self, time_constant):
        """
        Select a sensitivity from [1E-5, 3E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, 3E-1, 1E0, 3E0, 1E1, 3E1, 1E2, 3E2, 1E3, 3E3, 1E4, 3E4]

        :param time_constant:
        :return:
        """
        i = range(0, 20)
        timec = i[self.find_nearest(time_constant, self.tc)]
        self.write("OFLT %d"%timec)

    def get_time_constant(self):
        return self.tc[np.int(self.query("OFLT?"))]

    def set_filter_slope(self, slope):
        """
        Select the time constant filter slope. The parameter slope can be 6 dB/oct, 12, 18 or 24 dB/oct
        :param slope:
        :return:
        """

        if slope in self.slopes:
            i = self.find_nearest(self.slopes, slope)
            self.write("OFSL %d"%i)
        else:
            print "Slope input is invalid. Please select from 6, 12, 18 or 24."

    def get_filter_slope(self):
        """
        Queries the time constant filter slope.

        :return:
        """
        return self.slopes[np.int(self.query("OFSL?"))]

    def set_ch1_display(self, quantity):
        """
        Sets the display for the ch1 output
        :param quantity: may be 0, 1, 2, 3, 4 according to
        0 : X
        1 : R
        2 : X noise
        3 : AUX IN 1
        4 : AUX IN 2
        :return:
        """
        if quantity in range(5):
            self.write("DDEF 1, %d"%quantity)
        else:
            "Bad input for quantity ch1, must be between 0 and 4"

    def get_ch1_display(self):
        return self.displaych1[np.int(self.query("DDEF? 1"))]

    def set_ch2_display(self, quantity):
        """
        Sets the display for the ch2 output
        :param quantity: may be 0, 1, 2, 3, 4 according to
        0 : Y
        1 : theta
        2 : Y Noise
        3 : AUX IN 3
        4 : AUX IN 4

        :return:
        """
        if quantity in range(5):
            self.write("DDEF 2, %d"%quantity)
        else:
            "Bad input for quantity ch2, must be between 0 and 4"

    def get_ch2_display(self):
        return self.displaych2[np.int(self.query("DDEF? 2"))]

    def get_xy(self):
        ans = self.query("SNAP? 1,2").split(',')
        return [np.float(ans[i]) for i in range(2)]

    def get_rtheta(self):
        ans = self.query("SNAP? 3,4").split(',')
        return [np.float(ans[i]) for i in range(2)]

    def integrate_single(self, integration_time, sample_rate):
        """
        :param integration_time:
        :param sample_rate: Must be a power of 2
        :return:
        """

        # Set the sample rate
        self.write("SRAT %d"%(np.log2(sample_rate)+4))

        # Deletes the data buffers
        self.write("REST")
        # Starts or resumes the data storage
        self.write("STRT")

        t0 = time.time()

        while time.time() - t0 < integration_time:
            time.sleep(0.01)

        # Pause data storage
        self.write("PAUS")

        # Get number of points in buffer:
        noof_points = np.int(self.query("SPTS?"))

        # Read out buffer
        raw_ch1 = filter(None, self.query("TRCA? 1,0,%d"%(noof_points-1)).split(','))
        raw_ch2 = filter(None, self.query("TRCA? 2,0,%d"%(noof_points-1)).split(','))

        ch1 = [np.float(raw_ch1[i]) for i in range(len(raw_ch1))]
        ch2 = [np.float(raw_ch2[i]) for i in range(len(raw_ch2))]

        return ch1, ch2

