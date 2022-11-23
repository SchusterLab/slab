# -*- coding: utf-8 -*-
"""
SRS900 Voltage Source (voltsource.py)
=====================================
:Author: David Schuster
"""

from slab.instruments import SocketInstrument, SerialInstrument, VisaInstrument, Instrument
import re, time
from numpy import linspace
import numpy as np

class VoltageSource:
    def ramp_volt(self, v, sweeprate=1, channel=1):
        # channel is only for SRS, YOKO ignore channel
        start = self.get_volt(channel=channel)
        print(start)
        stop = v
        if stop == start: return
        start_t = time.time()
        self.set_volt(start, channel=channel)
        time.sleep(self.query_sleep)
        step_t = time.time() - start_t
        #print start,stop, start_t,step_t
        total_t = abs(stop - start) / sweeprate
        steps = max(total_t / step_t,2)
        #print start,stop,start_t,step_t, total_t, steps

        for ii in linspace(start, stop, steps)[1:]:
            self.set_volt(ii, channel=channel)
            #print ii
            time.sleep(self.query_sleep)

    def ramp_current(self, current, sweeprate=0.001, channel=1):
        start = self.get_current()
        stop = current
        if stop == start: return
        start_t = time.time()
        self.set_current(start, channel=channel)
        time.sleep(self.query_sleep)
        step_t = time.time() - start_t
        #print start,stop, start_t,step_t
        total_t = abs(stop - start) / sweeprate
        steps = int(max(total_t / step_t,2))
        #print start,stop,start_t,step_t, total_t, steps

        for ii in linspace(start, stop, steps)[1:]:
            self.set_current(ii, channel=channel)
            #print ii
            time.sleep(self.query_sleep)


class SRS900(SerialInstrument, VisaInstrument, VoltageSource):
    'Interface to the SRS900 voltage source'

    def __init__(self, name="", address='COM5', enabled=True, timeout=1.0):
        # if ':' not in address: address+=':22518'
        if address[:3].upper() == 'COM':
            SerialInstrument.__init__(self, name, address, enabled, timeout)
        else:
            VisaInstrument.__init__(self, name, address, enabled)
        self.query_sleep = 0.05
        self.recv_length = 65535
        self.escapekey = 'XXYYXX'
        #self.term_char='\r'


    def read(self, port=None):
        if port is not None:
            self.write("CONN %x,'%s'\n" % (port, self.escapekey))
            self.read()
            self.write(self.escapekey)
        if self.protocol == 'serial':
            return SerialInstrument.read(self)
        if self.protocol == 'GPIB':
            return VisaInstrument.read(self)

    def write(self, s, port=None):
        if port is not None:
            self.write("SNDT %x,#3%03d%s\n" % (port, len(s), s))
        if self.protocol == 'serial':
            SerialInstrument.write(self, s)
        if self.protocol == 'socket':
            VisaInstrument.write(self, s)

    def query(self, s, port=None):
        if port is not None:
            self.write("CONN %x, '%s'\n" % (port, self.escapekey))
            self.write(s)
            time.sleep(self.query_sleep)
            ans = self.read()
            self.write(self.escapekey)
        else:
            self.write(s)
            time.sleep(self.query_sleep)
            ans = self.read()

        return ans


    def __del__(self):
        return
        if self.protocol == 'serial':
            SerialInstrument.__del__(self)
        if self.protocol == 'visa':
            VisaInstrument.__del__(self)

    def get_id(self):
        return self.query("*IDN?")

    def set_volt(self, voltage, channel=1):
        self.write('SNDT %d,\"VOLT %.3f\"' % (channel, voltage))

    def get_volt(self, channel=1):
        return float(self.query("VOLT?", channel))

    def set_output(self, channel=1, state=True):
        if state:
            self.write('SNDT %d,\"OPON\"' % (channel))
        else:
            self.write('SNDT %d,\"OPOF\"' % (channel))

    def get_output(self, channel=1):
        return bool(int(self.query('EXON?', channel)))


class YokogawaGS200(SocketInstrument, VoltageSource):
    default_port = 7655

    def __init__(self, name='YOKO', address='', enabled=True,  recv_length=1024,timeout=10):
        SocketInstrument.__init__(self, name=name, address=address, enabled=enabled, recv_length=recv_length,timeout=timeout)
        self.query_sleep=0.01

    def get_id(self):
        """Get Instrument ID String"""
        return self.query('*IDN?').strip()

    def set_output(self, state=True):
        """Set output mode default state=True"""
        self.write(':OUTPUT:STATE %d' % (int(state)))

    def get_output(self):
        """Get output mode return result as bool"""
        return bool(self.query(':OUTPUT:STATE?').strip())

    def set_mode(self, mode):
        """Set yoko mode, valid inputs are mode='VOLTage' or mode='CURRent' """
        self.write(':SOURCE:FUNCTION %s' % mode)

    def get_mode(self):
        """Get yoko mode, returns either 'CURR' or 'VOLT'"""
        return self.query(':SOURCE:FUNCTION?').strip()

    def set_level(self, level):
        """Set yoko level"""
        self.write(':SOURCE:LEVEL %s' % level)

    def get_level(self):
        """Get level return as float"""
        return float(self.query(':SOURCE:LEVEL?').strip())

    def set_range(self, r):
        """set range of current/voltage"""
        self.write(':SOURCE:RANGE %f' % r)

    def get_range(self):
        """set range of current/voltage"""
        return float(self.query(':SOURCE:RANGE?'))

    def set_current_limit(self, lim):
        """set current limit"""
        self.write(':SOURCE:PROTECTION:CURRENT %f' % lim)

    def get_current_limit(self):
        """get current limit"""
        return float(self.query(':SOURCE:PROTECTION:CURRENT?'))

    def set_voltage_limit(self, lim):
        """set voltage limit"""
        self.write(':SOURCE:PROTECTION:VOLT %f' % lim)

    def get_voltage_limit(self):
        """get voltage limit"""
        return float(self.query(':SOURCE:PROTECTION:VOLT?'))


    def set_current(self, current, channel=0, safety_level=None):
        # channel does nothing...for compatibility with the SRS
        """Set yoko current (in Amps!)"""

        if self.get_mode() == "CURR":
            if safety_level is not None and current > safety_level:
                raise Exception("ERROR: Current too high (above %f mA)" % safety_level)
            else:
                curr_str = '%smA' % (current * 1e3)
                self.set_level(curr_str)
        else:
            raise Exception("ERROR: Need to set Yoko current in current mode")

    def get_current(self):
        """Get yoko current (in Amps!)"""
        if self.get_mode() == "CURR":
            return self.get_level()
        else:
            raise Exception("ERROR: Need to set Yoko voltage in current mode")

    def set_volt(self, voltage, channel=0):
        # channel does nothing...for compatibility with the SRS
        """Set yoko voltage"""
        if self.get_mode() == "VOLT":
            self.set_level(voltage)
        else:
            raise Exception("ERROR: Need to set Yoko voltage in voltage mode")

    def get_volt(self, channel=0):
        # channel does nothing...for compatibility with the SRS - alex
        """Get yoko voltage"""
        if self.get_mode() == "VOLT":
            return self.get_level()
        else:
            raise Exception("ERROR: Need to set Yoko voltage in voltage mode")

    def set_measure_state(self, state=True):
        """Set measurement state of instrument"""
        if state:
            self.write(':SENSE:STATE ON')
        else:
            self.write(':SENSE:STATE OFF')

    def get_measure(self):
        """Get measured value"""
        return float(self.query(':MEASURE?').strip())

    def get_programs(self):
        """Returns a list of programs defined in the memory"""
        answer = self.query('PROG:CAT?').strip()
        # the program names are between quotes in "answer". This returns them as a list.
        program_list = re.findall('\"(.+?)\"', answer)
        return program_list

    def set_program_slope_time(self, ts):
        """Sets the slope time between two successive steps in a program"""
        self.write(':PROG:SLOP %.6f'%ts)

    def get_program_slope_time(self):
        """Returns the slope time between two successive steps in a program"""
        answer = self.query(':PROG:SLOP?').strip()
        return float(answer)

    def set_program_interval_time(self, ti):
        """Sets the interval time between two successive steps in a program"""
        self.write(':PROG:INT %.6f'%ti)

    def get_program_interval_time(self):
        """Returns the interval time between two successive steps in a program"""
        answer = self.query(':PROG:INT?').strip()
        return float(answer)

    def define_program(self, name, points, instrument_range=None, mode='VOLT'):
        """Defines a program with filename 'name' and steps 'points'"""
        x3 = 'V' if mode == 'VOLT' else 'A'

        if instrument_range is not None:
            if isinstance(instrument_range, float):
                x2 = instrument_range
            elif isinstance(instrument_range, ndarray) or isinstance(instrument_range, list):
                x2 = instrument_range
            else:
                raise ValueError("Specified data type for variable range not understood.")
        else:
            x2 = max(abs(points))

        formatted_points = ''
        for k, p in enumerate(points):
            formatted_points += '%.6f, %f, %s\r\n' % (p, x2, x3)

        self.write(':PROG:DEF "%s", "%s"' % (name, formatted_points))

    def load_program(self, name):
        """Load an existing program from the memory"""
        if name in self.get_programs():
            self.write('PROG:LOAD "%s"'%name)
        else:
            raise ValueError("Specified program name does not exist.")

    def run_program(self):
        """Run the program. For this to work, the output needs to be set to True"""
        self.write('PROG:RUN')

    def step_program(self):
        """Move forward one step in a program"""
        self.write('PROG:STEP')

    def set_program_repeat(self, state=True):
        """Set to False if program needs to execute only once"""
        self.write(":PROG:REP %d" % int(state))

    def get_program_repeat(self):
        """True if the program repeats forever"""
        answer = int(self.query(':PROG:REP?').strip())
        return bool(answer)

    def delete_program(self, name):
        """Note: you can only delete a program when no other programs are running"""
        if name in self.get_programs():
            self.write('PROG:DEL "%s"'%name)
        else:
            raise ValueError("Specified program name does not exist.")

    def flush_program_memory(self):
        """Deletes all programs from the memory"""
        print("Deleting programs from memory:")
        for program in self.get_programs():
            print("\t%s"%program)
            self.delete_program(program)

    def linear_ramp(self, Vi, Vf, dt=0.1):
        program_name='simple_ramp.csv'
        #self.set_output(False)
        self.flush_program_memory()
        self.define_program(program_name, np.array([Vi, Vf]), instrument_range=None, mode='VOLT')
        self.load_program(program_name)
        self.set_program_repeat(False)
        self.set_program_interval_time(dt)  # dV/sweep_speed)
        self.set_program_slope_time(dt)  # dV/sweep_speed)
        self.set_volt(Vi)
        self.set_output(True)
        self.run_program()


# class SRS928(Instrument):
#    
#    def __init__(self,mainframe,name="",address=None,enabled=True,timeout=1):
#        """Initialized with link to mainframe and the address should be the port # on the mainframe"""
#        self.mainframe=mainframe
#        Instrument.__init__(self,name,address,enabled,timeout)
#        self.escape='xZZxy'
#        
#    def write(self,s):
#        self.mainframe.write('')

def test_yoko(yoko=None):
    if yoko is None:
        yoko = YokogawaGS200(address='10.120.35.219')

    print(yoko.get_id())
    yoko.set_mode('current')
    print(yoko.get_mode())
    print(yoko.get_volt())
    yoko.set_measure_state()
    print(yoko.get_measure())


if __name__ == "__main__":
    #srs=SRS900(address="COM17")
    #print srs.get_id()
    #srs.set_volt(.5,2)
    yoko = YokogawaGS200(address='10.120.35.219')
    test_yoko(yoko)
