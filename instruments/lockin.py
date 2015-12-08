# -*- coding: utf-8 -*-
"""
Lock in amplifiers
===========================================
:Author: Gerwin Koolstra
"""

from slab.instruments import VisaInstrument
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
        self.slopes = [6, 12, 18, 24]

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
        if self.get_reference_frequency() == 'int':
            self.write("FREQ %.0f"%(frequency))
        else:
            print "Reference mode is external! Set to internal first"

    def get_reference_frequency(self):
        return float(self.query("FREQ?"))

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

    def set_ch2_display(self):
        """
        Sets the display for the ch2 output
        :param quantity: may be 0, 1, 2, 3, 4 according to
        0 : X
        1 : R [Volts rms]
        2 : R [dBm]
        3 : X noise
        4 : AUX IN
        :return:
        """
        if quantity in range(5):
            self.write("DDEF 2, %d"%quantity)
        else:
            "Bad input for quantity ch1, must be between 0 and 4"

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

