# -*- coding: utf-8 -*-
"""
BNC Function Generator (function_generator.py)
==============================================
:Author: David Schuster
"""
from slab.instruments import SocketInstrument
import time


class BNCAWG(SocketInstrument):
    'Interface to the BNC function generator'
    default_port = 5025

    def __init__(self, name='BNCAWG', address='', enabled=True, timeout=0.01, recv_length=1024):
        SocketInstrument.__init__(self, name, address, enabled, timeout, recv_length)

    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?')

    def set_output(self, state=True):
        """Set Output State On/Off"""
        if state:
            self.write('OUTPUT ON')
        else:
            self.write('OUTPUT OFF')

    def get_output(self):
        """Query Output State"""
        return int(self.query('OUTPUT?')) == 1

    def set_termination(self, load=None):
        """Set Output State On/Off"""
        if load:  self.write('OUTPUT:LOAD %s' % load)

    def get_termination(self):
        """Set Output State On/Off"""
        return float(self.query('OUTPUT:LOAD?'))


    def set_function(self, ftype="sine"):
        ftypes = {"SINE": "SIN", "SQUARE": "SQU", "SQU": "SQU", "RAMP": "RAMP", "PULSE": "PULSE", "NOISE": "NOISE",
                  "DC": "DC", "USER": "USER"}
        ftype_str = ftypes[ftype.upper()]
        self.write('FUNCtion %s' % ftype_str)

    def set_square(self, dutycycle=50):
        """
        There are limitations to the duty cycle you can set. 
        for frequency lower than 10MHz, it is limited to 20% and 80%."""
        self.write('FUNC:SQU')
        return self.write('FUNC:SQU:DCYC %s' % str(dutycycle))

    def get_function(self):
        return self.query('FUNCtion?')

    def set_frequency(self, frequency):
        self.write('FREQ %f' % (float(frequency)))

    def get_frequency(self):
        return float(self.query('FREQ?'))

    def set_period(self, frequency):
        self.write('PULS:PER %f' % (float(frequency)))

    def get_period(self):
        return float(self.query('PULS:PER?'))

    def set_pulse_width(self, frequency):
        self.write('FUNC:PULS:WIDT %.9f' % (float(frequency)))

    def get_pulse_width(self):
        return float(self.query('FUNC:PULS:WIDT?'))

    def set_pulse_duty_cycle(self, frequency):
        self.write('FUNC:PULS:DCYC %f' % (float(frequency)))

    def get_pulse_duty_cycle(self):
        return float(self.query('FUNC:PULS:DCYC?'))

    def set_pulse_transition(self, frequency):
        self.write('FUNC:PULS:TRAN %f' % (float(frequency)))

    def get_pulse_transition(self):
        return float(self.query('FUNC:PULS:TRAN?'))

    def set_amplitude(self, voltage):
        self.write('VOLT %f' % voltage)

    def get_amplitude(self):
        return float(self.query('VOLT?'))

    def set_autorange(self, range):
        """OFF,ON,ONCE"""
        self.write('VOLT:RANGE:AUTO %s' % range.upper())

    def get_autorange(self):
        return self.query('VOLT:RANGE:AUTO?').split('\n')

    def set_offset(self, offset):
        self.write("VOLT:OFFSET %f" % offset)

    def get_offset(self):
        return float(self.query("VOLT:OFFSET?"))

    def set_trigger_source(self, source="INT"):
        trig_types = {'INT': 'IMM', 'INTERNAL': 'IMM', 'EXTERNAL': 'EXT', 'EXT': 'EXT', 'BUS': 'BUS', 'MAN': 'MAN'}
        trig_type_str = trig_types[source.upper()]
        self.write('TRIG:SOURCE %s' % trig_type_str)

    def get_trigger_source(self):
        return self.query('TRIG:SOURCE?')

    def set_trigger_out(self, state):
        if state:
            self.write('OutPut:TRIGger %s' % "ON")
        else:
            self.write('OutPut:TRIGger %s' % "OFF")

    def get_trigger_out(self):
        if self.query('OutPut:TRIGger?') == '1\n':
            return True
        else:
            return False

    def set_trigger_slope(self, edge):
        edge = edge.upper();
        if edge == 'POS' or edge == 'POSITIVE':
            self.write('OutPut:TRIGger:SLOPe %s' % "POSitive")
        elif edge == 'NEG' or edge == 'NEGATIVE':
            self.write('OutPut:TRIGger:SLOPe %s' % "NEGative")

    def get_trigger_slope(self):
        return self.query('OutPut:TRIGger:SLOPe?')[:-1]

    def trigger(self):
        self.write('TRIGGER')

    def set_burst_cycles(self, cycles=1):
        self.write('BURST:NCYCLES %d' % cycles)

    def get_burst_cycles(self, ):
        return int(self.query('BURST:NCYCLES?'))

    def set_burst_period(self, period):
        self.write('BURSt:INTernal:PERiod %f' % period)

    def get_burst_period(self):
        return float(self.query('BURSt:INTernal:PERiod?'))

    def set_burst_phase(self, phase):
        """
        phase is between -360 and 360"""
        self.write('BURSt:PHASe %d' % phase)

    def get_burst_phase(self):
        return float(self.query('BURSt:PHASe?'))

    def set_burst_state(self, state=True):
        if state:
            self.write('BURst:STATe ON')
        else:
            self.write('BURst:STATe OFF')

    def get_burst_state(self):
        return int(self.query('BURST:STATE?')) == 1

    def set_burst_mode(self, mode):
        if mode == 'triggered':
            self.write('BURSt:MODE TRIGgered')
        elif mode == 'gated':
            self.write('BURSt:MODE GATed')

    def get_burst_mode(self):
        return int(self.query('BURSt:MODE?')) == 1

    def set_burst_trigger_slope(self, edge):
        edge = edge.upper();
        if edge == 'POS' or edge == 'POSITIVE':
            self.write('TRIGger:SLOPe %s' % "POSitive")
        elif edge == 'NEG' or edge == 'NEGATIVE':
            self.write('TRIGger:SLOPe %s' % "NEGative")

    def get_burst_trigger_slope(self):
        return self.query('TRIGger:SLOPe?')[:-1]

    def set_symmetry(self, percent):
        self.write('FUNCtion:RAMP:SYMMetry %d' % percent)

    def get_symmetry(self):
        return int(self.query('FUNCtion:RAMP:SYMMetry?'))

    def get_settings(self):
        settings = SocketInstrument.get_settings(self)
        settings['id'] = self.get_id()
        settings['output'] = self.get_output()
        settings['frequency'] = self.get_frequency()
        settings['function'] = self.get_function()
        settings['amplitude'] = self.get_amplitude()
        settings['offset'] = self.get_offset()
        return settings
        
    #setup the BNC as the master trigger for the 
    #qubit experiment
    def set_exp_trigger(self):
        
        self.set_output(False)
        self.set_function('PULSE')
        self.set_frequency(5e4)
        self.set_offset(0.)
        self.set_amplitude(1.5)
        self.write('FUNCtion:PULSe:TRANsition 5e-9')
        self.write('FUNCtion:PULSe:WIDTh 50e-9')
        self.set_output(True)


class FilamentDriver(BNCAWG):
    def setup_driver(self, amplitude, offset, frequency, pulse_length):
        self.set_output(False)
        self.set_amplitude(amplitude)
        self.set_offset(offset)
        self.set_frequency(frequency)

        self.set_burst_state(True)
        self.set_burst_cycles(round(pulse_length * frequency))
        self.set_trigger_source('bus')

        self.set_output(True)

    def fire_filament(self, pulses=1, delay=0):
        for ii in range(pulses):
            self.trigger()
            time.sleep(delay)


class BiasDriver(BNCAWG):
    """
    this class is designed to allow use of the BNCAWG as a DC voltage supply.
    internal amplifier range changes causes large spikes in the output.
    Hence to use as DC supply, auto-range-scalling has to be turned off.
    We want a sticky response. Don't want voltage to change no matter what we do. """

    def setup_volt_source(self, pulse_length=None, pulse_voltage=None, rest_voltage=None, autorange='off'):
        #set the duty cycle to 40/60, 
        #set the starting phase to be         
        #        self.set_output(False)
        self.set_autorange(autorange)
        self.set_termination(load='INFinity')
        self.set_function('square')
        if pulse_voltage != None and rest_voltage != None:
            self.pulse_voltage = pulse_voltage;
            self.rest_voltage = rest_voltage;
        if hasattr(self, 'pulse_voltage') and hasattr(self, "rest_voltage"):
            amp = abs(pulse_voltage - rest_voltage)
            offset = (pulse_voltage + rest_voltage) / 2.
            if pulse_voltage < rest_voltage:
                phase = 180
            else:
                phase = 0
            self.set_offset(offset)
            self.set_amplitude(amp)
            self.set_burst_phase(phase)
        if pulse_length != None:
            self.pulse_length = pulse_length;
        if hasattr(self, 'pulse_length'):
            freq = 1 / (pulse_length * 2)  #Need Integer
            self.set_frequency(freq)
        self.set_burst_state(True)
        self.set_burst_cycles(1)
        # this is crucial. Has to set to manual trigger(bus)
        # then trigger through the bus to make sure that the
        # internal phase is correct. Otherwise get random
        # phases.
        self.set_trigger_source('bus')
        self.trigger();
        self.set_output(True)
        #self.set_autorange(autorange)

    def pulse_voltage(self, pulses=1, delay=0):
        for ii in range(pulses):
            self.trigger()
            time.sleep(delay)

    def set_voltage(self, volt):
        phase = self.get_burst_phase()
        amp = self.get_amplitude()
        offset = self.get_offset()
        if phase == 180:
            self.set_offset(volt - amp / 2.)
        elif phase == 0:
            self.set_offset(volt + amp / 2.)
        else:
            self.set_burst_phase(0);
            self.set_voltage(volt);

    def set_volt(self, volt):
        self.set_voltage(volt)


if __name__ == "__main__":
    #bnc=BNCAWG(address='192.168.14.133')
    filament = FilamentDriver(address='192.168.14.133')
    print filament.query('*IDN?')
