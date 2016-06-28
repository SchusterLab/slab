# -*- coding: utf-8 -*-
"""
BNC Function Generator (function_generator.py)
==============================================
:Author: David Schuster

PLEASE COMMENT ANY ADDITIONAL CODE YOU WRITE

"""
from slab.instruments import SocketInstrument
import time
import numpy as np


class BNCAWG(SocketInstrument):
    'Interface to the BNC function generator'
    default_port = 5025

    def __init__(self, name='BNCAWG', address='', enabled=True, query_timeout=1000, recv_length=1024):
        SocketInstrument.__init__(self, name, address, enabled, query_timeout, recv_length)

    def get_id(self):
        """
        Returns the ID of the machine
        :return: string
        """
        return self.query('*IDN?')

    def set_output(self, state=True):
        """
        Disable or enable the Output connector on the front panel. The default is ON. The Output key is lit when enabled.
        :param state: bool
        :return: None
        """
        if state:
            self.write('OUTPUT ON')
        else:
            self.write('OUTPUT OFF')

    def get_output(self):
        """
        “0” or “1” indicating the on/off state of the Output connector on the front panel is returned.
        :return: bool
        """
        return int(self.query('OUTPUT?')) == 1

    def set_output_polarity(self, state='normal'):
        """
        Invert the waveform relative to the offset voltage. The default is NORM, in which
        the waveform goes positive during the first part of the cycle and in INV mode the
        waveform goes negative during the first part of the cycle. The offset remains the
        same when the waveform is inverted and the Sync signal is not inverted.
        :param state: string, 'normal' or 'inverted'
        :return: None
        """
        if state.lower() in ['normal', 'inverted']:
            do_set = 'NORM' if state.lower() == 'normal' else 'INV'
            self.write('OUTP:POL %s'%do_set)
        else:
            print "State must be 'normal' or 'inverted' for inverted output."

    def get_output_polarity(self):
        """
        Query the polarity of the waveform. “NORM” or “INV” indicating the polarity will be returned.
        :return: string
        """
        answer = self.query('OUTP:POL?')
        return answer.strip()

    def set_termination(self, load=None):
        """
        Select the desired output termination. It can be any value (in ohms) between 1Ω
        and 10kΩ. INF sets the output termination to “high impedance” (>10 kΩ). The
        default is 50Ω. The specified value is used for amplitude, offset, and high/low
        level settings.
        :param load: float or string
        :return: None
        """
        if load: self.write('OUTPUT:LOAD %s' % load)

    def get_termination(self):
        """
        Query the current load setting in ohms. Return the current load setting or “9.9E+37” meaning “high impedance”.
        :return: float
        """
        return float(self.query('OUTPUT:LOAD?'))

    def set_function(self, ftype="sine"):
        """
        Sets the function. ftype must be one of the strings “SIN”, “SQU”, “RAMP”, “PULS”, “NOIS”, “DC”, and “USER”.
        """
        ftypes = {"SINE": "SIN", "SQUARE": "SQU", "SQU": "SQU", "RAMP": "RAMP", "PULSE": "PULSE", "NOISE": "NOISE",
                  "DC": "DC", "USER": "USER"}
        ftype_str = ftypes[ftype.upper()]
        self.write('FUNCtion %s' % ftype_str)

    def set_square(self, dutycycle=50):
        """
        There are limitations to the duty cycle you can set. 
        for frequency lower than 10MHz, it is limited to 20% and 80%.
        """
        self.write('FUNC:SQU')
        return self.write('FUNC:SQU:DCYC %s' % str(dutycycle))

    def get_function(self):
        """
        Query the selection made by FUNCtion USER command. One of the strings “SIN”,
        “SQU”, “RAMP”, “PULS”, “NOIS”, “DC”, and “USER” will be returned.
        :return: string
        """
        return self.query('FUNCtion?')

    def set_frequency(self, frequency):
        """
        Set the frequency in Hz
        :param frequency: float
        :return: None
        """
        self.write('FREQ %f' % (float(frequency)))

    def get_frequency(self):
        """
        Returns the frequency in Hz
        :return: float
        """
        return float(self.query('FREQ?'))

    def set_period(self, period):
        """
        Specify the pulse period. The range is from 200 ns to 2000 seconds. The default is 1 ms.
        :param period: float
        :return: None
        """
        self.write('PULS:PER %f' % (float(period)))

    def get_period(self):
        """
        The period of the pulse waveform will be returned in seconds.
        :return: float
        """
        return float(self.query('PULS:PER?'))

    def set_pulse_width(self, width):
        """
        Specify the pulse width in seconds. The range is from 20 ns to 2000 seconds. The default is 100μs.
        :param frequency: float
        :return: None
        """
        self.write('FUNC:PULS:WIDT %.9f' % (float(width)))

    def get_pulse_width(self):
        """
        Query the pulse width. The pulse width in seconds will be returned.
        :return: float
        """
        return float(self.query('FUNC:PULS:WIDT?'))

    def set_pulse_duty_cycle(self, percent):
        """
        Specify the pulse duty cycle in percent. The range is from 0 percent to 100
        percent. The default is 10 percent. The minimum value is approximately 0
        percent and the maximum value is approximately 100 percent.
        :param percent: float
        :return: None
        """
        self.write('FUNC:PULS:DCYC %f' % (float(percent)))

    def get_pulse_duty_cycle(self):
        """
        Query the pulse duty cycle. The duty cycle in percent will be returned.
        :return: float
        """
        return float(self.query('FUNC:PULS:DCYC?'))

    def set_pulse_transition(self, frequency):
        self.write('FUNC:PULS:TRAN %f' % (float(frequency)))

    def get_pulse_transition(self):
        return float(self.query('FUNC:PULS:TRAN?'))

    def set_amplitude(self, voltage):
        """
        Specify the output amplitude. The minimum value is 10 mVpp into 50Ω and the
        maximum value is the largest amplitude for the chosen function (at most 10 Vpp
        into 50Ω depending on the chosen function and the offset voltage)
        :param voltage: float
        :return: None
        """
        self.write('VOLT %f' % voltage)

    def get_amplitude(self):
        """
        Query the output amplitude for the current function. The value is returned in the
        unit chosen by the VOLT:UNIT command.
        :return: float
        """
        return float(self.query('VOLT?'))

    def set_autorange(self, range):
        """
        Disable or enable the voltage auto-ranging. The default is “On” where the
        waveform generator selects an optimal setting for the output amplifier and
        attenuators.
        :param range: 'On' or 'Off'
        :return: None
        """
        self.write('VOLT:RANGE:AUTO %s' % range.upper())

    def get_autorange(self):
        """
        “0” (off) or “1” (on) indicating the auto-ranging enable state is returned.
        :return: bool
        """
        return int(self.query('VOLT:RANGE:AUTO?').split('\n')) == 1

    def set_offset(self, offset):
        """
        Specify the dc offset voltage. The default is 0 volts. The minimum value is the
        most negative dc offset for the chosen function and amplitude and the maximum
        value is the largest dc offset for the chosen function and amplitude.
        :param offset: float
        :return: None
        """
        self.write("VOLT:OFFSET %f" % offset)

    def get_offset(self):
        """
        Query the dc offset voltage for the current function.
        :return: float
        """
        return float(self.query("VOLT:OFFSET?"))

    def set_voltage_high(self, high):
        """
        Specify the high voltage level. The default high level for all functions is +50 mV.
        :param high: float
        :return: None
        """
        self.write('VOLT:HIGH %.5f' % high)

    def get_voltage_high(self):
        """
        Query the high voltage level.
        :return: float
        """
        answer = self.query('VOLT:HIGH?')
        return float(answer.strip())

    def set_voltage_low(self, low):
        """
        Specify the low voltage level. The default low level for all functions is -50 mV.
        :param high: float
        :return: None
        """
        self.write('VOLT:LOW %.5f' % low)

    def get_voltage_low(self):
        """
        Query the low voltage level.
        :return: float
        """
        answer = self.query('VOLT:LOW?')
        return float(answer.strip())

    def set_trigger_source(self, source="INT"):
        """
        Specify a trigger source for the triggered burst mode only. The waveform
        generator accepts a software (BUS) trigger, an immediate (internal) trigger, or
        a hardware trigger from the rear-panel EXT TRIG connector. The default is IMM.
        :param source:
        :return: None
        """
        trig_types = {'INT': 'IMM', 'INTERNAL': 'IMM', 'EXTERNAL': 'EXT', 'EXT': 'EXT', 'BUS': 'BUS', 'MAN': 'MAN'}
        trig_type_str = trig_types[source.upper()]
        self.write('TRIG:SOURCE %s' % trig_type_str)

    def get_trigger_source(self):
        """
        Query the trigger source. “IMM” or “BUS” or “EXT” string indicating the trigger
        source will be returned.
        :return: string
        """
        return self.query('TRIG:SOURCE?').strip()

    def set_trigger_out(self, state):
        """
        Disable or enable the trigger out signal. The default is OFF. When the trigger out
        signal is enabled, a TTL-compatible square waveform with the specified edge is
        output from the Ext Trig connector on the rear panel at the beginning of the
        sweep or burst.
        :param state: bool
        :return: None
        """
        if state:
            self.write('OutPut:TRIGger %s' % "ON")
        else:
            self.write('OutPut:TRIGger %s' % "OFF")

    def get_trigger_out(self):
        """
        “0” or “1” indicating the trigger out signal state will be returned.
        :return: bool
        """
        if self.query('OutPut:TRIGger?') == '1\n':
            return True
        else:
            return False

    def set_trigger_slope(self, edge):
        """
        Specify an edge for the “trigger out” signal.
        :param edge: string ('POS' or 'NEG')
        :return: None
        """
        edge = edge.upper();
        if edge == 'POS' or edge == 'POSITIVE':
            self.write('OutPut:TRIGger:SLOPe %s' % "POSitive")
        elif edge == 'NEG' or edge == 'NEGATIVE':
            self.write('OutPut:TRIGger:SLOPe %s' % "NEGative")

    def get_trigger_slope(self):
        """
        “POS” or “NEG” string indicating the edge for the “trigger out” signal will be returned.
        :return: 'POS' or 'NEG'
        """
        return self.query('OutPut:TRIGger:SLOPe?')[:-1]

    def trigger(self):
        """
        Issue an immediate trigger from the remote interface. This command can
        trigger a sweep or burst with any available trigger source (TRIG:SOUR
        command).
        :return: None
        """
        self.write('TRIGGER')

    def set_burst_cycles(self, cycles=1):
        """
        Specify the number of cycles to be output in each burst (triggered burst mode
        only). The range is from 1 to 50,000 cycles in 1 cycle increments and the default
        is 1 cycle.
        :param cycles: integer
        :return: None
        """
        self.write('BURST:NCYCLES %d' % cycles)

    def get_burst_cycles(self):
        """
        The burst count will be returned. The range is from 1 to 50,000, and 9.9E+37 is
        returned if Infinite is specified.
        :return: integer
        """
        answer = self.query('BURST:NCYCLES?').strip()
        return int(float(answer))

    def set_burst_period(self, period):
        """
        Specify the burst period for bursts with internal (immediate) trigger source. The
        burst period is ignored when external or manual trigger source is enabled (or
        when the gated burst mode is chosen).
        :param period: period in seconds
        :return: None
        """
        self.write('BURSt:INTernal:PERiod %f' % period)

    def get_burst_period(self):
        """
        The burst period in seconds will be returned.
        :return: float
        """
        return float(self.query('BURSt:INTernal:PERiod?'))

    def set_burst_phase(self, phase):
        """
        Specify the starting phase in degrees or radians according to UNIT:ANGL
        command. The range is from -360 degrees to +360 degrees (or from -2Π to
        +2Π radians) and the default is 0 degree (0 radians).
        """
        self.write('BURSt:PHASe %d' % phase)

    def get_burst_phase(self):
        """
        The starting phase in degree or radians will be returned.
        :return: float
        """
        return float(self.query('BURSt:PHASe?'))

    def set_burst_state(self, state=True):
        """
        Disable or enable the burst mode.
        :param state: bool
        :return: None
        """
        if state:
            self.write('BURst:STATe ON')
        else:
            self.write('BURst:STATe OFF')

    def get_burst_state(self):
        """
        “0” (OFF) or ”1” (ON) will be returned.
        :return: integer
        """
        return int(self.query('BURST:STATE?')) == 1

    def set_burst_mode(self, mode):
        if mode == 'triggered':
            self.write('BURSt:MODE TRIGgered')
        elif mode == 'gated':
            self.write('BURSt:MODE GATed')

    def get_burst_mode(self):
        return self.query('BURSt:MODE?').strip()

    def set_burst_trigger_slope(self, edge):
        edge = edge.upper();
        if edge == 'POS' or edge == 'POSITIVE':
            self.write('TRIGger:SLOPe %s' % "POSitive")
        elif edge == 'NEG' or edge == 'NEGATIVE':
            self.write('TRIGger:SLOPe %s' % "NEGative")

    def get_burst_trigger_slope(self):
        return self.query('TRIGger:SLOPe?')[:-1]

    def set_symmetry(self, percent):
        """
        Specify the symmetry percentage for ramp waves. Symmetry represents the
        amount of time per cycle that the ramp wave is rising (supposing the waveform
        polarity is not inverted). The range is from 0% to 100% and the default is 100%.
        :param percent: float
        :return: None
        """
        self.write('FUNCtion:RAMP:SYMMetry %d' % percent)

    def get_symmetry(self):
        """
        Query the current symmetry setting in percent.
        :return: integer
        """
        return int(self.query('FUNCtion:RAMP:SYMMetry?'))

    def send_waveform(self, vData, Vpp=2):
        """Rescale and send waveform data to the Tek"""
        # get range and scale to U16
        vI16 = self.scale_waveform_to_I16(vData, Vpp)
        length = len(vI16)
        # create data as string with header
        sLen = '%d' % (2 * length)
        sHead = ':DATA:DAC VOLATILE, #%d%s' % (len(sLen), sLen)
        # write header + data
        self.write(sHead + vI16.tostring())
        # select volatile waveform
        self.write(':FUNC:USER VOLATILE')

    def scale_waveform_to_I16(self, vData, dVpp):
        """Scales the waveform and returns data in a string of I16"""
        # clip waveform and store in-place
        np.clip(vData, -dVpp / 2., dVpp / 2., vData)
        vI16 = np.array(2047 * vData / (dVpp / 2.), dtype=np.int16)
        return vI16

    def get_settings(self):
        settings = SocketInstrument.get_settings(self)
        settings['id'] = self.get_id()
        settings['output'] = self.get_output()
        settings['frequency'] = self.get_frequency()
        settings['function'] = self.get_function()
        settings['amplitude'] = self.get_amplitude()
        settings['offset'] = self.get_offset()
        return settings

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
    def setup_driver(self, amplitude, offset, frequency, duration):
        self.set_output(False)
        self.set_amplitude(amplitude)
        self.set_offset(offset)
        self.set_frequency(frequency)

        self.set_burst_state(True)
        self.set_burst_cycles(round(duration * frequency))
        self.set_trigger_source('bus')

        self.set_output(True)

    def fire_filament(self, pulses=1, delay=0):
        # print "firing filament\r"
        for ii in range(pulses):
            self.trigger()
            time.sleep(delay)


class BiasDriver(BNCAWG):
    """
    this class is designed to allow use of the BNCAWG as a DC voltage supply.
    internal amplifier range changes causes large spikes in the output.
    Hence to use as DC supply, auto-range-scalling has to be turned off.
    We want a sticky response. Don't want voltage to change no matter what we do.
    """

    def setup_volt_source(self, duration=None, pulse_voltage=None, rest_voltage=None, autorange='off'):
        # set the duty cycle to 40/60,
        # set the starting phase to be
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
        if duration != None:
            self.duration = duration;
        if hasattr(self, 'duration'):
            freq = 1 / (duration * 2)  # Need Integer
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
        # self.set_autorange(autorange)

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

        self._volt = volt

    def set_volt(self, volt):
        self.set_voltage(volt)

    def get_volt(self):
        if hasattr(self, '_volt'):
            return self._volt
        else:
            return None

if __name__ == "__main__":
    # bnc=BNCAWG(address='192.168.14.133')
    filament = FilamentDriver(address='192.168.14.133')
    print filament.query('*IDN?')
