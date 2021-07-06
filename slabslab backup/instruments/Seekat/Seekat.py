"""
For additional information see the web page:
http://opendacs.com/seekat-homepage/
"""

import numpy as np
import time, math
from slab.instruments.instrumenttypes import SerialInstrument


class Seekat(SerialInstrument):
    def __init__(self, name="", address='/dev/cu.usbmodem1411', enabled=True, baudrate=115200, timeout=10):
        SerialInstrument.__init__(self, name=name, address=address, enabled=enabled, timeout=timeout, recv_length=1024,
                                  baudrate=baudrate)
        self.term_char = ''

    def get_channel_bits(self, channel):
        """
        When setting or getting the voltage, there's four digits that determine the channel. This function
        returns those digits, for internal use only.
        :param channel: channel number
        :return: Tuple of the four digits
        """
        if channel == 1:
            n1 = 19
            n2 = 0
            m1 = 1
            m2 = 0
        elif channel == 2:
            n1 = 18
            n2 = 0
            m1 = 1
            m2 = 0
        elif channel == 3:
            n1 = 17
            n2 = 0
            m1 = 1
            m2 = 0
        elif channel == 4:
            n1 = 16
            n2 = 0
            m1 = 1
            m2 = 0
        elif channel == 5:
            n1 = 0
            n2 = 19
            m1 = 0
            m2 = 1
        elif channel == 6:
            n1 = 0
            n2 = 18
            m1 = 0
            m2 = 1
        elif channel == 7:
            n1 = 0
            n2 = 17
            m1 = 0
            m2 = 1
        elif channel == 8:
            n1 = 0
            n2 = 16
            m1 = 0
            m2 = 1
        else:
            raise ValueError('Invalid channel')
        return (n1, n2, m1, m2)

    def get_voltage_bits(self, voltage):
        """
        When setting or getting the voltage, there's two digits that represent the voltage. This function
        returns those digits, for internal use only.
        :param voltage: voltage
        :return: two digits
        """
        gbin = lambda x, n: format(x, 'b').zfill(n)
        if voltage > 10:  # limit voltage between -10 and +10
            voltage = 10.0
        elif voltage < -10:
            voltage = -10
        if voltage >= 0:
            dec16 = round((2 ** 15 - 1) * voltage / 10)
        else:
            dec16 = round(2 ** 16 - abs(voltage) * (2 ** 15) / 10)
        dec16 = np.uint16(dec16)
        bin16 = gbin(dec16, 16)
        d1 = int(bin16[0:8], 2)
        d2 = int(bin16[8:16], 2)

        return d1, d2

    def set_voltage(self, channel, voltage, verbose=True):
        """
        Set the voltage of a specific channel
        :param channel: channel number (1-8)
        :param voltage: voltage (-10 to 10V)
        :return: None
        """
        if verbose:
            print('Channel = ' + str(channel) + ',  Voltage = ' + str(round(voltage, 3)), end=' ')
        n1, n2, m1, m2 = self.get_channel_bits(channel)
        d1, d2 = self.get_voltage_bits(voltage)

        SS = '255,254,253,' + str(n1) + ',' + str(d1 * m1) + ',' + str(d2 * m1) + ',' + str(n2) + ',' + str(
            d1 * m2) + ',' + str(d2 * m2)

        self.ser.flushInput()
        time.sleep(0.02)
        self.write(SS)
        self.ser.flush()

    def get_voltage(self, channel, ndigits=4):
        """
        Get the voltage of a specific channel.
        :param channel: Channel number (1-8)
        :param ndigits: Number of digits for the output, determines precision.
        :return: Voltage (float)
        """
        n1, n2, m1, m2 = self.get_channel_bits(channel)
        if n1 == 0:  # channels 5-8
            n2 += 128
        if n2 == 0:  # channels 1-4
            n1 += 128

        S1 = '255,254,253,' + str(n1) + ',0,0,' + str(n2) + ',0,0'

        time.sleep(0.02)
        self.write(S1)
        self.ser.flushInput()
        time.sleep(0.02)
        self.write(S1)
        self.ser.flushInput()
        time.sleep(0.02)

        self.write('255,254,253,0,0,0,0,0,0')

        time.sleep(0.02)

        bdata = np.zeros(6)
        for i in range(0, 6):

            R = self.ser.readline()
            try:
                bdata[i] = int(R)
            except:
                bdata[i] = 0

        BD1 = bdata[1] * 2 ** 8 + bdata[2]
        BD2 = bdata[4] * 2 ** 8 + bdata[5]
        bdata2 = max(BD1, BD2)

        if bdata2 < 2 ** 15:
            volt = 10 * float(bdata2) / (2 ** 15 - 1)
        else:
            volt = -10 * (2 ** 16 - float(bdata2)) / 2 ** 15
        return round(volt, ndigits)

    def calibrate(self, channel):
        """
        The OpenDAC can store an offset and gain. This routine sets first sets the offset, then sets the gain
        for a single channel. It does the first by setting the voltage to 0 V and then measuring the voltage.
        The difference from 0 V is the offset. The gain calibration is done by setting the voltage to -10 V.
        The ratio between the measured value and -10 V is used to know the real gain of the Seekat.
        :param channel: Channel number (1-8)
        :return: None
        """
        n1, n2, m1, m2 = self.get_channel_bits(channel)

        # Calibration offset voltage
        offset_voltage = 0
        self.set_voltage(channel, offset_voltage)
        blah = self.get_voltage(channel, 9)
        offset = -blah
        print('Offset calibration:\nSet to 0 V volt\nMeasure %.6fV' % (blah))

        offsetsteps = round(offset / (38.14 * math.exp(-6)))

        gbin = lambda x, n: format(x, 'b').zfill(n)
        offest8 = gbin(int(offsetsteps) % (2 ** 8), 8)
        d1 = 0
        d2 = int(offest8)
        time.sleep(0.005)
        SS = '255,254,253,' + str(n1) + ',' + str(d1 * m1) + ',' + str(d2 * m1) + ',' + str(n2) + ',' + str(
            d1 * m2) + ',' + str(d2 * m2)
        self.write(SS)
        while self.ser.inWaiting():
            self.ser.readline()
        time.sleep(1)

        # Calibrate -10V
        min_voltage = -10
        self.set_voltage(channel, min_voltage)
        time.sleep(2)
        blah = self.get_voltage(channel, 9)
        offset = blah - min_voltage
        print('Gain calibration:\nSet to -10 V volt\nMeasure %.6fV' % (blah))
        offsetsteps = round(offset / (152.59 * math.exp(-6)))
        offest8 = gbin(int(offsetsteps) % (2 ** 8), 8)
        d1 = 0
        d2 = int(offest8)
        time.sleep(0.005)
        SS = '255,254,253,' + str(n1 + 16) + ',' + str(d1 * m1) + ',' + str(d2 * m1) + ',' + str(n2 + 16) + ',' + str(
            d1 * m2) + ',' + str(d2 * m2)
        self.write(SS)
        while self.ser.inWaiting():
            self.ser.readline()
        time.sleep(1)

        # Back to 0 V
        offset_voltage = 0
        self.set_voltage(channel, offset_voltage)

        print("Calibration Complete !!!")

    def ramp(self, channel=1, period=0, start=-10, stop=10, step=40):
        """
        Voltage ramp. Note: needs to be reviewed.
        :param channel:
        :param period:
        :param start:
        :param stop:
        :param step:
        :return:
        """

        if period * step <= 0.05:
            raise ValueError("period * step must > 0.05")
        code = np.zeros((step, 9), dtype=int)
        # generate code
        n1, n2, m1, m2 = self.get_channel_bits(channel)

        rd = 0
        for i in np.linspace(start, stop, step):
            d1, d2 = self.get_voltage_bits(i)
            code[rd] = [255, 254, 253, n1, d1 * m1, d2 * m1, n2, d1 * m2, d2 * m2]
            print(code[rd])
            rd += 1

        delay = 0
        while True:
            btic = time.time()
            for rd in range(0, step):
                SC = '255,254,253,' + str(code[rd, 3]) + ',' + str(code[rd, 4]) + ',' + str(code[rd, 5]) + ',' + str(
                    code[rd, 6]) + ',' + str(code[rd, 7]) + ',' + str(code[rd, 8]) + ','
                self.write(SC)
                self.ser.flush()
                if delay >= 0:
                    time.sleep(delay)
            for rd in range(0, step):
                SC = '255,254,253,' + str(code[step - rd - 1, 3]) + ',' + str(code[step - rd - 1, 4]) + ',' + str(
                    code[step - rd - 1, 5]) + ',' + str(
                    code[step - rd - 1, 6]) + ',' + str(code[step - rd - 1, 7]) + ',' + str(
                    code[step - rd - 1, 8]) + ','
                self.write(SC)
                self.ser.flush()
                if delay >= 0:
                    time.sleep(delay)
            btoc = time.time()
            delay -= ((btoc - btic) - float(period)) / (step * 2)
            print(btoc - btic)


if __name__ == "__main__":
    o = Seekat(address='COM11', baudrate=115200, timeout=10)
    time.sleep(3)

    for k in range(8):
        print("ch%d: Measured voltage is %.2f V" % (k + 1, o.get_voltage(k + 1)))
        time.sleep(1)
