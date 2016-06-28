#
# ======================================================
#
# reference the programming manual here: http://na.support.keysight.com/pna/help/index.html?id=1000001808-1:epsg:man
# and specifically here: New Programming Commands:
# http://na.support.keysight.com/pna/help/latest/help.htm
# help: 1 800 829-4444
#
# ======================================================
#
__author__ = 'Ge Yang'
from slab.instruments import SocketInstrument
import time
import numpy as np
import glob
import os.path


def polar2mag(xs, ys):
    return np.sqrt(xs ** 2 + ys ** 2), np.arctan(ys / xs)


class N5242A(SocketInstrument):
    MAXSWEEPPTS = 1601
    default_port = 5025

    def __init__(self, name="E5071", address=None, enabled=True, **kwargs):
        SocketInstrument.__init__(self, name, address, enabled=enabled, recv_length=2 ** 20, **kwargs)
        self.query_sleep = 0.05

    def get_id(self):
        return self.query('*IDN?')

    def get_query_sleep(self):
        return self.query_sleep

    #### Frequency setup
    def set_start_frequency(self, freq, channel=1):
        self.write(":SENS%d:FREQ:START %f" % (channel, freq))

    def get_start_frequency(self, channel=1):
        return float(self.query(":SENS%d:FREQ:START?" % channel))

    def set_stop_frequency(self, freq, channel=1):
        self.write(":SENS%d:FREQ:STOP %f" % (channel, freq))

    def get_stop_frequency(self, channel=1):
        return float(self.query(":SENS%d:FREQ:STOP?" % channel))

    def set_center_frequency(self, freq, channel=1):
        self.write(":SENS%d:FREQ:CENTer %f" % (channel, freq))

    def get_center_frequency(self, channel=1):
        return float(self.query(":SENS%d:FREQ:CENTer?" % channel))

    def set_span(self, span, channel=1):
        return self.write(":SENS%d:FREQ:SPAN %f" % (channel, span))

    def get_span(self, channel=1):
        return float(self.query(":SENS%d:FREQ:SPAN?" % channel))

    def set_sweep_points(self, numpts=1600, channel=1):
        query = ":sense%d:sweep:points %d" % (channel, numpts)
        self.write(query)

    def get_sweep_points(self, channel=1):
        data = self.query(":sense%d:sweep:points?" % (channel))
        return int(data)

    def set_sweep_mode(self, mode='CONT'):
        """
        Sets the number of trigger signals the specified channel will ACCEPT

        HOLD - channel will not trigger
        CONTinuous - channel triggers indefinitely
        GROups - channel accepts the number of triggers specified with the last SENS:SWE:GRO:COUN <num>.
        SINGle - channel accepts ONE trigger, then goes to HOLD.
        """

        allowed_modes = ['cont', 'continuous', 'hold', "groups", 'gro', 'sing', 'single']
        if mode.lower() not in allowed_modes:
            return
        else:
            self.write("SENS:SWE:MODE %s" % mode)

    def get_sweep_mode(self):
        """
        Returns the sweep mode.
        """
        data = self.query("SENS:SWE:MODE?")
        if data is None:
            return None
        else:
            return data.strip()

    def get_sweep_time(self, channel=1):
        """
        :return: The sweep time in seconds
        """
        answer = self.query('SENS%d:SWE:TIME?'%channel)
        return float(answer.strip())

    def set_sweep_group_count(self, count=None):
        if count is None:
            return
        else:
            query = "SENS:SWE:GRO:COUN %d" % count
            self.write(query)

    def set_frequency_offset_mode_state(self, state):
        s = '1' if state else '0'
        self.write('SENSE:FOM:STATE ' + s)

    def setup_two_tone_measurement(self, read_frequency=None, read_power=None, probe_start=None, probe_stop=None, probe_power=None, two_tone=True ):
        if two_tone:
            print "TWO TONE ON"
            self.write('SENSE:FOM:RANGE4:COUPLED 1')
            if probe_start is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % probe_start)
            if probe_stop is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % probe_stop)

            self.write('SENSE:FOM:RANGE2:COUPLED 0')
            self.write('SENSE:FOM:RANGE3:COUPLED 0')
            if read_frequency is not None:
                self.write('SENSE:FOM:RANGE2:FREQUENCY:START %f' % read_frequency)
                self.write('SENSE:FOM:RANGE2:FREQUENCY:STOP %f' % read_frequency)
                self.write('SENSE:FOM:RANGE3:FREQUENCY:START %f' % read_frequency)
                self.write('SENSE:FOM:RANGE3:FREQUENCY:STOP %f' % read_frequency)

            self.set_frequency_offset_mode_state(True)

            if read_power is not None:
                self.set_power(read_power, channel=1, port=1)
            if probe_power is not None:
                self.set_power(probe_power, channel=1, port=3)
        else:
            print "TWO TONE OFF"
            self.write('SENSE:FOM:RANGE2:COUPLED 1')
            self.write('SENSE:FOM:RANGE3:COUPLED 1')
            self.write('SENSE:FOM:RANGE4:COUPLED 0')
            if probe_start is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % probe_start)
            if probe_stop is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % probe_stop)



            self.set_frequency_offset_mode_state(False)
            self.set_power(probe_power, channel=1, port=3, state=0)
            if read_power is not None:
                self.set_power(read_power, channel=1, port=1)

    #### Averaging
    def set_averages(self, averages, channel=1):
        self.write(":SENS%d:AVERage:COUNt %d" % (channel, averages))

    def get_averages(self, channel=1):
        return int(self.query(":SENS%d:average:count?" % channel))

    def set_average_state(self, state=True, channel=1):
        if state:
            s = "ON"
        else:
            s = "OFF"
        self.write(":SENS%d:AVERage:state %s" % (channel, s))

    def get_average_state(self, channel=1):
        return bool(self.query(":SENS%d:average:state?" % channel))

    def clear_averages(self, channel=1):
        self.write(":SENS%d:average:clear" % channel)

    def set_ifbw(self, bw, channel=1):
        self.write("sens%d:bwid %f" % (channel, bw))

    def get_ifbw(self, channel=1):
        return float(self.query("SENS%d:bwid?" % (channel)))

    def get_operation_completion(self):
        data = self.query("*OPC?")
        if data is None:
            return False
        else:
            return bool(int(data.strip()))

    def set_trigger_continuous(self, state=True):
        if state:
            _state = "on"
        else:
            _state = "off"
        self.write('initiate:continuous ' + _state)

    def trigger_single(self, channel=1):
        self.write('initiate%d:immediate' % channel)

    def set_trigger_average_mode (self, mode=None):
        allowed_modes = ['poin', 'point', 'sweep']
        if mode is None:
            return
        elif mode.lower() in allowed_modes:
            self.write('sense:AVER:mode ' + mode)
        else:
            print "trigger average mode needs to be one of " + ', '.join(allowed_modes)


    def get_trigger_average_mode (self):
        data = self.query('sense:AVER:mode?')
        if data is None: return None
        else: return data.strip()

    def set_trigger_average(self, state=True):
        if state:
            self.write('sense:AVER ON')
        else:
            self.write('sense:AVER OFF')

    def get_trigger_average(self):
        return bool(self.query('sense:AVER?'))

    def set_trigger_source(self, source="immediate"):  # IMMEDIATE, MANUAL, EXTERNAL
        allowed_sources = ['ext', 'imm', 'man', 'immediate', 'external', 'manual']
        if source.lower() not in allowed_sources:
            print "source need to be one of " + ', '.join(allowed_sources)
        self.write('TRIG:SEQ:SOUR ' + source)

    def get_trigger_source(self):  # INTERNAL, MANUAL, EXTERNAL,BUS
        return self.query(':TRIG:SEQ:SOUR?').strip()

    def set_external_trigger_mode(self, trigger_type, slope=1):
        """
        Specifies the type of EXTERNAL trigger input detection used to listen for signals on the Meas Trig IN connectors.
        Edge triggers are most commonly used.

        slope specifies the polarity expected by the external trigger input circuitry.
        slope = 0 : POSitive --> rising Edge (trigger_type='EDGE') or High Level (trigger_type='LEVEL')
        slope = 1 : NEGative --> falling Edge (trigger_type='EDGE') or Low Level (trigger_type='LEVEl')
        """
        if trigger_type.upper() in ["EDGE", "LEVEL"]:
            self.write("TRIG:TYPE %s"%trigger_type)
            if slope > 0:
                self.write("TRIG:SLOP POS")
            else:
                self.write("TRIG:SLOP NEG")
        else:
            raise ValueError("Input not understood!")

    #### Source

    def set_power(self, power, channel=1, port=1,state=1):
        # print ":SOURCE:POWER%d %f" % (channel, power)
        if state:
            self.write(":SOURCE%d:POWER%d:MODE ON" % (channel, port))
            self.write(":SOURCE%d:POWER%d %f" % (channel, port, power))
        else:
            print "Turning off the port %d" %(port)
            self.write(":SOURCE%d:POWER%d:MODE OFF" % (channel, port))
            # self.write(":SOURCE%d:POWER%d %f" % (channel, port, power))

    def get_power(self, channel=1, port=1):
        return float(self.query(":SOURCE%d:POWER%d?" % (channel, port)))

    def set_output(self, state=True):
        if state or str(state).upper() == 'ON':
            self.write(":OUTPUT ON")
        elif state == False or str(state).upper() == 'OFF':
            self.write(":OUTPUT OFF")

    def get_output(self):
        return bool(self.query(":OUTPUT?"))

    def delete_trace(self, trace=None):
        if trace is None:
            query = "disp:wind:trac:del"
        else:
            query = "disp:wind:trac%d:del" % trace
        self.write(query)

    def delete_measurement(self, name=None):
        if name is None:
            # delete all traces
            self.write(r'calc:par:del:all')
        else:
            self.write("calc:par:del '%s'" % name)

    def define_measurement(self, name, channel, mode):
        query = "calc%d:par:def:ext '%s','%s'" % (channel, name, mode)
        self.write(query)

    def get_measurements(self):
        data = self.query('calc:par:cat?')
        if data == '"NO CATALOG"\n':
            return None
        else:
            data_list = data.strip().split(',')
            return zip(*[iter(data_list)] * 2)

    def select_measurement(self, name=None, channel=1):
        query = "calc%d:par:sel '%s'" % (channel, name)
        self.write(query)

    def display_measurement(self, name=None, trace=1):
        query = "disp:wind:trace%d:feed '%s'" % (trace, name)
        self.write(query)

    def auto_scale(self, trace=None):
        """
        Performs an Autoscale on the specified trace in the specified window, providing the best fit display.
        Autoscale is performed only when the command is sent; it does NOT keep the trace autoscaled indefinitely.
        """
        if trace is None:
            query = "DISP:WIND:TRAC:Y:AUTO"
        else:
            query = "disp:wind:trac%d:Y:AUTO" % trace
        self.write(query)

    def set_measure(self, mode='S21', channel=1):
        pass

    #### Trace Operations
    def set_active_trace(self, channel=1, trace=1, fast=False):
        """
        set the active trace, which is required by the
        following commands [get_format, set_format]

        The fast option is OPTIONAL. The PNA display is
        NOT updated. Therefore, do not use this argument when
        an operator is using the PNA display. Otherwise, sending
        this argument results in much faster sweep speeds. There
        is NO other reason to NOT send this argument.
        """
        query_string = "CALC%d:PAR:MNUM %d" % (channel, trace)
        if fast:
            query_string += ",fast"
        # print query_string
        self.write(query_string)

    def get_active_trace(self, channel=1):
        """set the active trace, need to run after the active trace is set."""
        query_string = "calculate%d:parameter:mnumber:select?" % channel
        data = self.query(query_string)
        if data is None:
            return data;
        else:
            return int(data)

    def set_format(self, trace_format='MLOG', trace=1):
        """set_format: need to run after the active trace is set.
        valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        self.write("CALC%d:FORM %s" % (trace, trace_format))

    def get_format(self, trace=1):
        """set_format: need to run after active trace is set.
        valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        data = self.query("CALC%d:FORM?" % (trace))
        if data is None:
            return data
        else:
            return data.strip()

    def set_electrical_delay(self, seconds, channel=1):
        query = "calc%d:corr:edel:time %e" % (channel, seconds)
        self.write(query)

    def get_electrical_delay(self, channel=1):
        query = "calc%d:corr:edel:time?" % channel
        data = self.query(query)
        if data is None:
            return None
        else:
            return float(data.strip())

    #### File Operations
    def save_file(self, fname):
        self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')

    def read_line(self, eof_char='\n', timeout=None):
        if timeout is None:
            timeout = self.query_timeout
        done = False
        while done is False:
            buffer_str = self.read(timeout)
            # print "buffer_str", buffer_str
            yield buffer_str
            if buffer_str[-1] == eof_char:
                done = True

    def read_data(self, sweep_points=None, channel=1, timeout=None):
        """Read current NWA Data, return fpts,mags,phases"""
        if sweep_points is None:
            sweep_points = self.get_sweep_points()

        if timeout is None:
            timeout = self.query_timeout
        self.get_operation_completion()
        self.write("CALC%d:DATA? FDATA" % channel)
        data_str = ''.join(self.read_line(timeout=timeout))
        data = np.fromstring(data_str, dtype=float, sep=',')
        fpts = np.linspace(self.get_start_frequency(), self.get_stop_frequency(), sweep_points)
        if len(data) == 2 * sweep_points:
            data = data.reshape((-1, 2))
            data = data.transpose()
            return np.vstack((fpts, data))
        else:
            return np.vstack((fpts, data))

    #### Meta
    
    def take(self, sweep_points=None):
        """
        Important:
            the PNA-X need to be in the following mode
                trigger source:IMMediate,
                format:POLar,
                trigger:CONTinuous ON
        :param sweep_points:
            by taking in a sweep_points parameter, we do not need to query the PNA-X for this
            parameter. This way we can save some time.
        :return:
            either fpts, xs, ys,
            or     fpts, mags.
        """
        self.clear_averages()
        # this is the command that triggers the averaging. Execute right before read data.
        self.set_sweep_mode('gro')
        data = self.read_data(sweep_points)
        return data

    def take_in_mag_phase(self, sweep_points=None):
        fpts, xs, ys = self.take(sweep_points)
        mags, phases = polar2mag(xs, ys)
        return fpts, mags, phases

    def take_one_in_mag_phase(self, sweep_points=None):
        _trig_source = self.get_trigger_source()
        _format = self.get_format()
        self.setup_take()
        fpts, xs, ys = self.take(sweep_points)
        mags, phases = polar2mag(xs, ys)
        self.set_trigger_source(_trig_source)
        self.set_format(_format)
        return fpts, mags, phases

    def setup_take(self, averages=None, averages_state=None):
        self.set_trigger_source("imm")
        self.set_format('polar')
        self.set_trigger_continuous()
        
        if averages is not None:
            self.set_averages_and_group_count(averages, True)
        elif averages_state is not None:
            self.set_average_state(averages_state)

    def set_averages_and_group_count(self, averages, state=None):
        self.set_averages(averages)
        self.set_sweep_group_count(averages)
        if state is not None:
            self.set_average_state(state)

    def clear_traces(self):
        self.delete_trace()
        self.delete_measurement()

    def setup_measurement(self, name, mode=None):
        if mode is None:
            mode = name
        self.define_measurement(name, 1, mode)
        self.display_measurement(name)
        self.select_measurement(name)
        self.set_active_trace(1, 1, True)

    def get_settings(self):
        settings = {"start": self.get_start_frequency(), "stop": self.get_stop_frequency(),
                    "power": self.get_power(), "ifbw": self.get_ifbw(),
                    "sweep_points": self.get_sweep_points(),
                    "averaging": self.get_average_state(), "averages": self.get_averages()
                    }
        return settings

    def configure(self, start=None, stop=None, center=None, span=None,
                  power=None, ifbw=None, sweep_points=None, averages=None):
        if start is not None:      self.set_start_frequency(start)
        if stop is not None:       self.set_stop_frequency(stop)
        if center is not None:     self.set_center_frequency(center)
        if span is not None:       self.set_span(span)
        if power is not None:      self.set_power(power)
        if ifbw is not None:       self.set_ifbw(ifbw)
        if sweep_points is not None:  self.set_sweep_points(sweep_points)
        if averages is not None:
            self.set_averages_and_group_count(averages)


if __name__ == '__main__':
    na = N5242A("N5242A", address="192.168.14.242")
    print na.get_id()
