# -*- coding: utf-8 -*-
"""
Oxford Triton Cryostat Controller (cryostat.py)
===============================================
:Author: David Schuster, Ge Yang
"""
__author__ = 'Ge Yang'
from slab.instruments import SocketInstrument
import re
import time

class Triton_old(SocketInstrument):
    default_port = 22518

    def __init__(self, name="Triton", address='slab-fridge1.uchicago.edu', enabled=True, timeout=1.0):
        # if ':' not in address: address+=':22518'
        SocketInstrument.__init__(self, name, address, enabled, timeout)
        self.recv_length = 65536

    def get_status(self):
        self.write('status')
        if self.query_sleep is not None:
            time.sleep(self.query_sleep)
        status_string = ''.join(self.read_line('<end>'))
        while status_string.strip() == '':
            status_string = ''.join(self.read_line('<end>'))
        statuses = {'compressor' : re.findall('compressor is (.*)', status_string)[0],
                    'forepump' : re.findall('forepump is (.*)', status_string)[0],
                    'V1' : re.findall('V1 is (.*)', status_string)[0],
                    'V2': re.findall('V2 is (.*)', status_string)[0],
                    'V3': re.findall('V3 is (.*)', status_string)[0],
                    'V7': re.findall('V7 is (.*)', status_string)[0],
                    'V8': re.findall('V8 is (.*)', status_string)[0]}
        return statuses

    def get_help(self):
        self.write('help')
        print(self.read().strip('\n'))

    def get_pressures(self):
        """get_pressure returns the pressures detected by the fridge as a dictionary"""
        #s = self.query('pressures')
        self.write('pressures')
        if self.query_sleep is not None:
            time.sleep(self.query_sleep)
        pres_string = ''.join(self.read_line('<end>'))
        while pres_string.strip() == '':
            pres_string = ''.join(self.read_line('<end>'))
        matches = re.findall(r'.*?(?P<name>.*?): (?P<pressure>.*?)bar.*', pres_string)
        pressures = {}
        for match in matches:
            pressures[match[0]] = float(match[1])
        return pressures

    def get_pressure(self, channel='Condense'):
        """Get pressure returns the pressure of the desired channel.  Valid options are:
            Condense, Tank, and Forepump"""
        return self.get_pressures()[channel]

    def get_temperatures(self):
        self.write('temperatures')
        if self.query_sleep is not None:
            time.sleep(self.query_sleep)
        temp_string = ''.join(self.read_line('<end>'))
        while temp_string.strip() == '':
            temp_string = ''.join(self.read_line('<end>'))

        matches = re.findall(r'.*?name: (?P<name>.*?);.*?temperature: (?P<temperature>.*?);.*', temp_string)
        temps = {}
        for match in matches:
            temps[match[0]] = float(match[1])
        return temps

    def get_temperature(self, channel='MC RuO2'):
        # use buffered temperature data.
        now = time.time()
        if not hasattr(self, 'temperatures') or \
                        'time' not in list(self.temperatures.keys()) or \
                        (now - self.temperatures['time']) > 6.0:
            self.temperatures = self.get_temperatures()
            if self.temperatures is not None:
                self.temperatures['time'] = time.time()

        if channel in list(self.temperatures.keys()):
            return self.temperatures[channel]
        else:
            return None

    def get_mc_temperature(self):
        temperature = self.get_temperature('MC RuO2')
        if temperature > 10:
            temperature = self.get_temperature('MC cernox')
        return temperature

    def get_settings(self):
        settings = SocketInstrument.get_settings(self)
        settings.update(self.get_temperatures())
        settings.update(self.get_pressures())
        return settings


if __name__ == '__main__':
    fridge = Triton_old(address='192.168.14.129')
    # print fridge.get_status()
    d = fridge.get_temperatures()
    # print fridge.get_temperatures()
    print(fridge.get_settings())
