"""
Written by Gerwin Koolstra - August 2017
For programming information visit the following sites
> Information on the DATA block, specific for the CP2800
http://lmu.web.psi.ch/docu/manuals/bulk_manuals/BlueFors/Cryomech%20PT-410/Control%20Panel%20Computer%20Interface%20Package/data_dictionary(V1.8).pdf
> Information on the Sycon Multi Drop Protocol that's used to format the bitstream
http://lmu.web.psi.ch/docu/manuals/bulk_manuals/BlueFors/Cryomech%20PT-410/Control%20Panel%20Computer%20Interface%20Package/Sycon%20Multi%20Drop%20Protocol%20II.pdf
"""

import numpy as np
import time, serial, struct
from slab.instruments.instrumenttypes import SerialInstrument

class CP2800(SerialInstrument):
    def __init__(self, name="", address='COM6', enabled=True, baudrate=115200, timeout=0.25):
        SerialInstrument.__init__(self, name=name, address=address, enabled=enabled, timeout=timeout, recv_length=16,
                                  baudrate=baudrate)
        self.term_char = ''
        self._cr = ['0x0D']
        self._preamble = ['0x02', '0x10', hex(128)]
        self.ser._bytesize = serial.EIGHTBITS
        self.ser._parity = serial.PARITY_NONE
        self.ser._stopbits = serial.STOPBITS_ONE

        # self.ser = serial.Serial(port="COM6", baudrate=115200, timeout=0.1,
        #                          bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
        #                          stopbits=serial.STOPBITS_ONE)

    def get_checksum(self, preamble, data):
        """
        Returns the checksum corresponding to the preamble and the data.
        The first part of the preamble (STX) isn't used to calculate the checksum.
        """
        stx, addr, cmd_rsp = preamble
        check = preamble[1:] + data
        checksum = np.sum([int(check[i], 16) for i in range(len(check))])%256
        checksum1 = int("{0:8b}".format(checksum)[:4], 2) + 48
        checksum2 = int("{0:8b}".format(checksum)[4:], 2) + 48
        return hex(checksum1), hex(checksum2)

    def translation(self, k):
        """
        In the byte array 0x07 is an escape character. It is always followed by
        an ascii 0, 1 or 2 which is 0x30 (48), 0x31 (49) or 0x32 (50) to denote
        the character 0x02 (STX), 0x0D (\r) or 0x07 (escape char) respectively.
        """
        if k==48:
            return 2
        elif k==50:
            return 7
        elif k==49:
            return 13

    def parse_message(self, bitstring):
        """
        Parses the bytearray sent by the CP2800 into a dictionary.
        """
        preamble_len=7
        intlist = [struct.unpack('B', bitstring[i])[0] for i in range(len(bitstring))]
        new_intlist = list()
        was_seven=False
        for i, integ in enumerate(intlist):
            if integ==7:
                was_seven=True
            else:
                if was_seven:
                    new_intlist.append(self.translation(integ))
                    was_seven=False
                else:
                    new_intlist.append(integ)
        # print(intlist)
        # print(new_intlist)
        # The final 3 bits of the third byte are the RSP from the compressor
        error_code = int("{0:8b}".format(new_intlist[2])[-3:], 2)
        error_strings = {"1" : "OK",
                         "2" : "Illegal command (CMD code not valid).",
                         "3" : "Syntax error. (too many bytes in data field, not enough bytes, etc).",
                         "4" : "Data range error",
                         "5" : "Inhibited",
                         "6" : "Obsolete command. No action taken, but not really an error.",
                         "7" : "Reserved for future protocol stack use"}

        error_string = error_strings["%d"%error_code]
        data_hexs = ''.join('{:02X}'.format(a) for a in new_intlist[preamble_len:preamble_len+4])
        data = int(data_hexs, 16)
        parsed_message = {"smdp_address" : new_intlist[1],
                          "dictionary_entry" : hex(new_intlist[4])+hex(new_intlist[5]),
                          "dictionary_data" : data,
                          "error_code" : error_code}

        if error_code != 1:
            raise ValueError("Error code %d: %s"%(error_code, error_string))

        return parsed_message

    def get_compressor_status(self):
        """
        Returns the status of the compressor (on/off) as a boolean
        """
        data = ['0x63', '0x5F', '0x95', '0x00']
        checksum1, checksum2 = self.get_checksum(self._preamble, data)
        hexs = self._preamble + data + [checksum1, checksum2] + self._cr
        # print hexs
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        time.sleep(self.timeout)
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]

    def get_compressor_runtime(self):
        """
        Returns the runtime of the compressor in minutes
        """
        data = ['0x63', '0x45', '0x4C', '0x00']
        checksum1, checksum2 = self.get_checksum(self._preamble, data)
        hexs = self._preamble + data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]

    def get_input_water_temperature(self):
        """
        Returns the input water temperature in degrees Celcius
        Trips at 29 C and resets at 27 C
        """
        actual_data = ['0x63', '0x0D', '0x8F', '0x00']
        send_data = ['0x63', '0x07', '0x31', '0x8F', '0x00']
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]*0.1

    def get_output_water_temperature(self):
        """
        Returns the output water temperature in degrees Celcius
        Trips at 52 C and resets at 49 C
        """
        actual_data = ['0x63', '0x0D', '0x8F', '0x01']
        send_data = ['0x63', '0x07', '0x31', '0x8F', '0x01']
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]*0.1

    def get_helium_temperature(self):
        """
        Returns the helium gas temperature in degrees Celcius.
        He temperature trips at 88 C and resets at 49 C
        """
        actual_data = ['0x63', '0x0D', '0x8F', '0x02']
        send_data = ['0x63', '0x07', '0x31', '0x8F', '0x07', '0x30']
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]*0.1

    def get_oil_temperature(self):
        """
        Returns the oil temperature in degrees Celcius.
        Oil temperature trips at 49 C and resets at 38 C.
        """
        actual_data = ['0x63', '0x0D', '0x8F', '0x03']
        send_data = ['0x63', '0x07', '0x31', '0x8F', '0x03']
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        answer = self.parse_message(self.ser.readline())
        return answer["dictionary_data"]*0.1

    def get_compressor_temperatures(self):
        """
        Returns the relevant temperatures of the compressor in the following order
        Input water temperature
        Output water temperature
        Helium temperature
        Oil temperature
        """
        temperatures = list()
        temperatures.append(get_input_water_temperature())
        temperatures.append(get_output_water_temperature())
        temperatures.append(get_helium_temperature())
        temperatures.append(get_oil_temperature())
        return temperatures

    def start_compressor(self):
        """
        Starts the compressor. There is a ~5-10 second delay between execution of
        the command and the start of the pulse tube.
        """
        actual_data = ['0x61', '0xD5', '0x01', '0x00',
                       '0x00', '0x00', '0x00', '0x01']
        send_data = actual_data
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        time.sleep(5)
        self.ser.flushInput()

    def stop_compressor(self):
        """
        Stops the compressor.
        """
        actual_data = ['0x61', '0xC5', '0x98', '0x00',
                       '0x00', '0x00', '0x00', '0x00']
        send_data = actual_data
        checksum1, checksum2 = self.get_checksum(self._preamble, actual_data)
        hexs = self._preamble + send_data + [checksum1, checksum2] + self._cr
        decs = [int(h, 16) for h in hexs]
        self.ser.write(bytearray(decs))
        time.sleep(0.1)
        self.ser.flushInput()
        
if __name__ == "__main__":
    cp = CP2800(address='COM10', baudrate=115200, timeout=0.5)

    print(("The compressor status is %d"%cp.get_compressor_status()))
    print(("The compressor runtime is %d minutes"%cp.get_compressor_runtime()))
    print(("The input water temperature is %.1fC"%cp.get_input_water_temperature()))
    print(("The output water temperature is %.1fC"%cp.get_output_water_temperature()))
    print(("The helium temperature is %.1fC"%cp.get_helium_temperature()))
    print(("The oil temperature is %.1fC"%cp.get_oil_temperature()))
