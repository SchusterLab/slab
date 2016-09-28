# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 21:26:31 2011
@author: Dave

For the programming manual see http://ena.support.keysight.com/e5071c/manuals/webhelp/eng/
"""
from slab.instruments import SocketInstrument, VisaInstrument
import time, glob, re
import numpy as np
import os.path
from matplotlib import pyplot as plt


class HP3577A(VisaInstrument):
    MAXSWEEPPTS = 1601
    default_port = 5025

    def __init__(self, name="HP3577A", address='GPIB0::9::INSTR', enabled=True, timeout=None):
        VisaInstrument.__init__(self, name, address, enabled, term_chars='\r')
        self.query_sleep = 0.5
        self.recv_length = 65536
        self.term_char = '\r'

        # set up continuous sweep, linear sweep
        # self.write('SM1')
        # time.sleep(0.1)
        # self.write('ST1')

    def set_start_frequency(self, start_frequency):
        """
        :param start_frequency: Start frequency in Hz
        """
        return self.write('FRA %.6f MHZ' % (start_frequency / 1E6))

    def set_stop_frequency(self, stop_frequency):
        """
        :param stop_frequency: Stop frequency in Hz
        """
        return self.write('FRB %.6f MHZ' % (stop_frequency / 1E6))

    def set_center_frequency(self, center_frequency):
        """
        :param center_frequency: Center frequency in Hz
        """
        return self.write('FRC %.6f MHZ' % (center_frequency / 1E6))

    def set_span(self, span):
        """
        :param span: Span in Hz
        """
        return self.write('FRS %.6f MHZ' % (span / 1E6))

    def set_power(self, power):
        """
        :param power: Power in dBm
        """
        return self.write('SAM %.2f DBM' % power)

    def set_input(self, input):
        """
        :param input: one of the following: R, A, B, AR, BR, D1, D2, D3, D4
        """
        if input in ['R', 'A', 'B']:
            return self.write('IN%s' % input)
        elif input in ['AR', 'BR', 'D1', 'D2', 'D3', 'D4']:
            return self.write('I%s' % input)

    def set_resolution_bw(self, bw):
        """
        :param bw:  bandwidth in Hz (1, 10, 100, 1000)
        """
        return self.write('BW%d' % int(1 + np.log10(bw)))

    def set_averaging(self, N):
        """
        :param N: Number of averages (2**2 -- 2**8)
        """
        if N == 1:
            return self.write('AV0')
        else:
            return self.write('AV%d' % (int(np.log10(N) / np.log10(2)) - 1))

    def take_one(self, number_of_averages, verbose=False):
        """
        Take a spectrum
        :return:
        """
        # start with a blank sheet
        self.set_averaging(1)
        self.set_averaging(number_of_averages)

        self.write('DF7; FM1')
        N = 0
        N_old = 0
        while N <= number_of_averages:
            self.write('DAN')
            time.sleep(0.1)
            N = float(self.read())
            time.sleep(1)
            if N > N_old and verbose:
                print "%d/%d" % (N, number_of_averages)
            N_old = N

        self.write('DT1')
        time.sleep(3)
        print "reading magnitude..."
        rawlogmag = self.read()
        logmag = map(float, rawlogmag.split(','))

        self.write('DF5')
        N = 0
        N_old = 0
        while N <= number_of_averages:
            self.write('DAN')
            time.sleep(0.1)
            N = float(self.read())
            if N > N_old and verbose:
                print "%d/%d" % (N, number_of_averages)
            time.sleep(1)
            N_old = N

        self.write('DT1')
        time.sleep(3)
        print "reading phase..."
        rawphase = self.read()
        phase = map(float, rawphase.split(','))

        fminch1, fmaxch1, fminch2, fmaxch2, amp = self.get_status()
        freq = np.linspace(fminch1, fmaxch1, len(phase))

        return freq, logmag, phase

    def get_status(self):
        """
        :return: power in dBm
        """
        self.write('DCH')
        time.sleep(0.5)

        refch1, amplvlch1, refch2, amplvlch2, mrkrfch1, \
        mrkrampch1, mrkrfch2, mrkrampch2, startfch1, stopfch1, \
        startfch2, stopfch2, sourceamp, delayap, entryblock = self.read().split(',')

        time.sleep(0.2)
        # print startfch1, stopfch1, startfch2, stopfch2

        startf1 = float(re.sub("\D", "", startfch1)) / 1E3
        stopf1 = float(re.sub("\D", "", stopfch1)) / 1E3
        startf2 = float(re.sub("\D", "", startfch2)) / 1e3
        stopf2 = float(re.sub("\D", "", stopfch2)) / 1e3
        amp = re.sub("\D", "", sourceamp)

        return startf1, stopf1, startf2, stopf2, amp


class E5071(SocketInstrument):
    MAXSWEEPPTS = 1601
    default_port = 5025

    def __init__(self, name="E5071", address=None, enabled=True):
        SocketInstrument.__init__(self, name, address, enabled=enabled, timeout=10, recv_length=2 ** 20)
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

    #### Averaging

    def get_operation_completion(self):
        data = self.query("*OPC?")
        if data is None:
            return False
        else:
            return bool(int(data.strip()))

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
        self.write(":SENS%d:BANDwidth:RESolution %f" % (channel, bw))

    def get_ifbw(self, channel=1):
        return float(self.query(":SENS%d:BANDwidth:RESolution?" % (channel)))

    def averaging_complete(self):
        self.write("*OPC?")
        self.read()

    #### Trigger

    def trigger_single(self):
        """
        Send a single trigger.
        :return: None
        """
        self.write(':TRIG:SING')

    def set_trigger_average_mode(self, state=True):
        """
        If state=True, the machine initiates averaging when it receives a single trigger. It keeps
        averaging until the averaging is complete.
        :param state: bool
        :return: None
        """
        if state:
            self.write(':TRIG:AVER ON')
        else:
            self.write(':TRIG:AVER OFF')

    def get_trigger_average_mode(self):
        """
        Returns the trigger averaging mode. If True, the machine operates in a way where it keeps averaging
        a trace until averaging is complete, with only a SINGLE trigger necessary.
        :return: bool
        """
        return bool(self.query(':TRIG:AVER?'))

    def set_trigger_source(self, source="INTERNAL"):
        """
        Sets the trigger source to one of the following:
        INTERNAL, MANUAL, EXTERNAL,BUS
        :param source: string
        :return: None
        """
        self.write(':TRIG:SEQ:SOURCE ' + source)

    def get_trigger_source(self):
        """
        Returns the trigger source.
        :return: string
        """
        answer = self.query(':TRIG:SEQ:SOURCE?')
        return answer.strip()

    def set_trigger_continuous(self, state=True):
        """
        Set the trigger mode to continuous (if state = True) or set the state to Hold (if state = False)
        :param state: bool
        :return: None
        """
        set = 'ON' if state else 'OFF'
        self.write(':INIT:CONT %s'%set)

    def get_trigger_continuous(self):
        """
        Returns True if the trigger mode is set to Continuous, or False if the trigger mode is set to Hold.
        :return: bool
        """
        answer = self.query(':INIT:CONT?')
        return bool(answer.strip())

    def set_trigger_in_polarity(self, polarity=1):
        """
        Set the external input trigger polarity. If polarity = 1, the external trigger is set to the positive edge,
        if polarity = 0, the machine triggers on the negative edge.
        :param polarity:
        :return: None
        """
        set = 'POS' if polarity else 'NEG'
        self.write(':TRIG:SEQ:EXT:SLOP %s' % set)

    def get_trigger_in_polarity(self):
        """
        Returns the trigger slope for the external trigger. Returns 1 for triggering on positive edge, or
        0 for triggering on the negative edge.
        :return: integer
        """
        answer = self.query(':TRIG:SEQ:EXT:SLOP?')
        ret = 1 if answer.strip() == 'POS' else 0
        return ret

    def set_trigger_low_latency(self, state=True):
        """
        This command turns ON/OFF or returns the status of the low-latency external trigger feature.
        When turning on the low-latency external trigger feature, the point trigger feature must be set
        to on and the trigger source must be set to external trigger.
        :param state: bool
        :return: None
        """
        set = 'ON' if state else 'OFF'
        self.write(':TRIG:EXT:LLAT %s'%set)

    def get_trigger_low_latency(self):
        """
        Returns the low latency external trigger status
        :return: bool
        """
        answer = self.query(':TRIG:EXT:LLAT?')
        return bool(answer.strip())

    def set_trigger_event(self, state='sweep'):
        """
        This command turns ON/OFF the status of the point trigger feature.
        :param state: string ('sweep' or 'point')
        :return: None
        """
        if state in ['sweep', 'point']:
            do_set = 'ON' if state == 'point' else 'OFF'
            self.write(':TRIG:POIN %s'%do_set)
        else:
            print "keyword state should be either 'sweep' or 'point'"

    def get_trigger_event(self):
        """
        This command returns the status of the point trigger feature.
        :param state: string ('sweep' or 'point')
        :return: bool
        """
        answer = self.query(':TRIG:POIN?')
        return bool(answer.strip())

    def set_trigger_out_polarity(self, polarity=1):
        """
        Sets the external output trigger polarity. If polarity = 1, the external trigger is a positive voltage
        pulse. If polarity = 0, the external trigger is a negative voltage pulse.
        :param polarity: integer
        :return: None
        """
        set = 'POS' if polarity else 'NEG'
        self.write('TRIG:OUTP:POL %s' % set)

    #### Source

    def set_power(self, power, channel=1):
        self.write(":SOURCE%d:POWER %f" % (channel, power))

    def get_power(self, channel=1):
        return float(self.query(":SOURCE%d:POWER?" % channel))

    def set_output(self, state=True):
        if state or str(state).upper() == 'ON':
            self.write(":OUTPUT ON")
        elif state == False or str(state).upper() == 'OFF':
            self.write(":OUTPUT OFF")

    def get_output(self):
        return bool(self.query(":OUTPUT?"))

    def set_measure(self, mode='S21', channel=1):
        self.write(":CALC%d:PAR1:DEF %s" % (channel, mode))

    #### Sweep

    def get_sweep_time(self, channel=1):
        """
        Returns the sweep time in seconds.
        :param channel: channel number
        :return: float
        """
        answer = self.query(":SENS%d:SWE:TIME?"%channel)
        return float(answer.strip())

    def set_sweep_time(self, sweep_time, channel=1):
        """
        Sets the sweep time in seconds. If the sweep time is set to 'AUTO', this function first sets the sweep time
        to manual. Then it sets the sweep time to "sweep_time". This value cannot be lower than the value when the
        sweep time is set to auto.
        :param sweep_time: sweep time in seconds
        :param channel: channel number
        :return: None
        """
        self.set_sweep_time_auto(state=False, channel=channel)
        self.write(":SENS%d:SWE:TIME %.3e"%(channel, sweep_time))

    def set_sweep_time_auto(self, state=True, channel=1):
        """
        Sets the sweep time to automatic (the fastest option).
        :param state: True/False
        :param channel: channel number
        :return: None
        """
        set = 'ON' if state else 'OFF'
        self.write(":SENS%d:SWE:TIME:AUTO %s"%(channel, set))

    def get_sweep_time_auto(self, channel=1):
        """
        Returns True if the sweep time is automatically set, or False if the sweep time is set manually.
        :param channel: channel number
        :return: bool
        """
        answer = self.query(":SENS%d:SWE:TIME:AUTO?"%channel)
        return bool(answer.strip())

    def set_sweep_points(self, numpts=1600, channel=1):
        """
        Sets the number of sweep points
        :param numpts: integer
        :param channel: channel number
        :return: None
        """
        self.write(":SENSe%d:SWEep:POINts %f" % (channel, numpts))

    def get_sweep_points(self, channel=1):
        """
        Returns the number of points in the current sweep.
        :param channel: channel number
        :return: integer
        """
        return float(self.query(":SENSe%d:SWEep:POINts?" % (channel)))

    #### Scale

    def get_electrical_delay(self, channel=1):
        """
        Returns the electrical delay in seconds
        :param channel: channel number
        :return: float
        """
        answer = self.query(":CALC%d:CORR:EDEL:TIME?" % (channel))
        return float(answer.strip())

    def set_electrical_delay(self, electrical_delay, channel=1):
        """
        Sets the electrical delay of the trace. The number should be between -10s and 10s.
        :param electrical_delay: float
        :param channel: channel number
        :return: None
        """
        self.write(":CALC%d:CORR:EDEL:TIME %.6e"%(channel, np.float64(electrical_delay)))

    def auto_scale(self, channel=1, trace_number=1):
        """
        Auto-scales the y-axis of the trace with trace_number.
        :param channel: channel number
        :param trace_number: integer
        :return: None
        """
        self.write(":DISP:WIND%d:TRAC%d:Y:AUTO"%(channel, trace_number))

    #### File Operations

    def save_file(self, fname):
        self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')

    def set_format(self, trace_format='MLOG', channel=1):
        """
        set_format: valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        self.write(":CALC:FORMAT " + trace_format)

    def get_format(self, channel=1):
        """
        get_format: valid options are
        {MLOGarithmic|PHASe|GDELay| SLINear|SLOGarithmic|SCOMplex|SMITh|SADMittance|PLINear|PLOGarithmic|POLar|MLINear|SWR|REAL| IMAGinary|UPHase|PPHase}
        """
        return self.query(":CALC:FORMAT?")

    def read_data(self, channel=1, timeout=None):
        """
        Read current NWA Data, output depends on format, for mag phase, use set_format('SLOG')
        :param channel: channel number
        :param timeout: optional, query timeout in ms.
        :return: np.vstack((fpts, data))
        """
        #self.get_operation_completion()
        self.write(":FORM:DATA ASC")
        self.write(":CALC%d:DATA:FDAT?"%channel)

        if timeout is None:
            timeout = self.timeout

        data_str = ''.join(self.read_line(timeout=timeout))
        data = np.fromstring(data_str, dtype=float, sep=',')
        sweep_points = int(self.get_sweep_points())
        fpts = np.linspace(self.get_start_frequency(), self.get_stop_frequency(), sweep_points)
        if len(data) == 2 * sweep_points:
            data = data.reshape((-1, 2))
            data = data.transpose()
            return np.vstack((fpts, data))
        else:
            return np.vstack((fpts, data))

        return np.vstack((fpts, data))

    #### Meta

    def take_one_averaged_trace(self, fname=None):
        """Setup Network Analyzer to take a single averaged trace and grab data, either saving it to fname or returning it"""
        # print "Acquiring single trace"
        self.set_trigger_source('BUS')
        time.sleep(self.query_sleep * 2)
        old_timeout = self.get_timeout()
        self.set_timeout(100.)
        self.set_format()
        time.sleep(self.query_sleep)
        old_avg_mode = self.get_trigger_average_mode()
        self.set_trigger_average_mode(True)
        self.clear_averages()
        self.trigger_single()
        time.sleep(self.query_sleep)
        self.averaging_complete()  # Blocks!
        self.set_format('SLOG ')
        if fname is not None:
            self.save_file(fname)
        ans = self.read_data()
        time.sleep(self.query_sleep)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')
        self.set_format()
        return ans

    def segmented_sweep(self, start, stop, step):
        """Take a segmented sweep to achieve higher resolution"""
        span = stop - start
        total_sweep_pts = span / step
        if total_sweep_pts <= 1601:
            print "Segmented sweep unnecessary"
        segments = np.ceil(total_sweep_pts / 1600.)
        segspan = span / segments
        starts = start + segspan * np.arange(0, segments)
        stops = starts + segspan

        print span
        print segments
        print segspan

        # Set Save old settings and set up for automated data taking
        time.sleep(self.query_sleep)
        old_format = self.get_format()
        old_timeout = self.get_timeout()
        old_avg_mode = self.get_trigger_average_mode()

        self.set_timeout(10000)
        self.set_trigger_average_mode(True)
        self.set_trigger_source('BUS')
        self.set_format('slog')

        self.set_span(segspan)
        segs = []
        for start, stop in zip(starts, stops):
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)

            self.clear_averages()
            self.trigger_single()
            time.sleep(self.query_sleep)
            self.averaging_complete()  # Blocks!

            seg_data = self.read_data()

            seg_data = seg_data.transpose()
            last = seg_data[-1]
            seg_data = seg_data[:-1].transpose()
            segs.append(seg_data)
        segs.append(np.array([last]).transpose())
        time.sleep(self.query_sleep)
        self.set_format(old_format)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')

        return np.hstack(segs)

    def segmented_sweep2(self, start, stop, step, sweep_pts=None, fname=None, save_segments=True):
        """Take a segmented sweep to achieve higher resolution"""
        span = stop - start
        total_sweep_pts = span / step
        if total_sweep_pts < 1600:
            print "Segmented sweep unnecessary"
            self.set_sweep_points(max(sweep_pts, total_sweep_pts))
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)
            return self.take_one_averaged_trace(fname)
        segments = np.ceil(total_sweep_pts / 1600.)
        segspan = span / segments
        starts = start + segspan * np.arange(0, segments)
        stops = starts + segspan

        #        print span
        #        print segments
        #        print segspan

        # Set Save old settings and set up for automated data taking
        time.sleep(self.query_sleep)
        old_format = self.get_format()
        old_timeout = self.get_timeout()
        old_avg_mode = self.get_trigger_average_mode()

        self.set_timeout(10000)
        self.set_trigger_average_mode(True)
        self.set_trigger_source('BUS')
        self.set_format('mlog')

        self.set_span(segspan)
        segs = []
        for start, stop in zip(starts, stops):
            self.set_start_frequency(start)
            self.set_stop_frequency(stop)

            self.clear_averages()
            self.trigger_single()
            time.sleep(self.query_sleep)
            self.averaging_complete()  # Blocks!
            self.set_format('slog')
            seg_data = self.read_data()
            self.set_format('mlog')
            seg_data = seg_data.transpose()
            last = seg_data[-1]
            seg_data = seg_data[:-1].transpose()
            segs.append(seg_data)
            if (fname is not None) and save_segments:
                np.savetxt(fname, np.transpose(segs), delimiter=',')
        segs.append(np.array([last]).transpose())
        time.sleep(self.query_sleep)
        self.set_format(old_format)
        self.set_timeout(old_timeout)
        self.set_trigger_average_mode(old_avg_mode)
        self.set_trigger_source('INTERNAL')
        ans = np.hstack(segs)
        if fname is not None:
            np.savetxt(fname, np.transpose(ans), delimiter=',')
        return ans

    def take_one(self, fname=None):
        """Tell Network Analyzer to take a single averaged trace and grab data, 
        either saving it to fname or returning it.  This function does not set up
        the format or anything else it just starts, blocks, and grabs the next trace."""
        # print "Acquiring single trace"
        # time.sleep(na.query_sleep*2)
        # na.set_format()
        # time.sleep(na.query_sleep)
        self.clear_averages()
        self.trigger_single()
        time.sleep(self.get_query_sleep())
        self.averaging_complete()
        # na.set_format('slog')
        if fname is not None:
            self.save_file(fname)
        ans = self.read_data()
        # time.sleep(na.query_sleep)
        # na.set_format()
        return ans

    def get_settings(self):
        settings = {"start": self.get_start_frequency(), "stop": self.get_stop_frequency(),
                    "power": self.get_power(), "ifbw": self.get_ifbw(),
                    "sweep_points": self.get_sweep_points(),
                    "averaging": self.get_average_state(), "averages": self.get_averages()
                    }
        return settings

    def configure(self, start=None, stop=None, center=None, span=None, power=None, ifbw=None, sweep_points=None, averages=None,
                  defaults=False, remote=False):
        if defaults: self.set_default_state()
        if remote: self.set_remote_state()
        if start is not None: self.set_start_frequency(start)
        if stop is not None: self.set_stop_frequency(stop)
        if center is not None: self.set_center_frequency(center)
        if span is not None: self.set_span(span)
        if power is not None: self.set_power(power)
        if ifbw is not None: self.set_ifbw(ifbw)
        if sweep_points is not None: self.set_sweep_points(sweep_points)
        if averages is not None: self.set_averages(averages)

    def set_remote_state(self):
        self.set_trigger_source('BUS')
        self.set_trigger_average_mode(True)
        self.set_timeout(10000)
        self.set_format('slog')

    def set_default_state(self):
        self.set_sweep_points()
        self.set_format()
        self.set_trigger_source()
        self.set_trigger_average_mode(False)
        self.write(":INIT1:CONT ON")


def condense_nwa_files(datapath, prefix):
    prefixes, data = load_nwa_dir(datapath)
    np.save(prefix, np.array(data))


def load_nwa_file(filename):
    """return three arrays: frequency magnitude and phase"""
    return np.transpose(np.loadtxt(filename, skiprows=3, delimiter=','))


def load_nwa_dir(datapath):
    fnames = glob.glob(os.path.join(datapath, "*.CSV"))
    fnames.sort()
    prefixes = [os.path.split(fname)[-1] for fname in fnames]
    data = [load_nwa_file(fname) for fname in fnames]
    return prefixes, data


def nwa_watch_temperature_sweep(na, fridge, datapath, fileprefix, windows, powers, ifbws, avgs, timeout=10000, delay=0):
    """nwa_watch_temperature_sweep monitors the temperature (via fridge) and tells the network analyzer (na) to watch certain windows
    at the specified powers, ifbws, and avgs specified
    windows= [(center1,span1),(center2,span2), ...]
    powers= [power1,power2,...]
    ifbws=[ifbw1,ifbw2,...]
    avgs=[avgs1,avgs2,...]"""
    f = open(datapath + fileprefix + ".cfg", 'w')
    f.write('datapath: %s\nfileprefix: %s\n\Window #\tCenter\tSpan\tPower\tIFBW\tavgs' % (datapath, fileprefix))
    for ii, w in enumerate(windows):
        f.write('%d\t%f\t%f\t%f\t%f\t%d' % (windows[ii][0], windows[ii][1], powers[ii], ifbws[ii], avgs[ii]))
    f.close()
    na.set_trigger_average_mode(True)
    na.set_sweep_points()
    na.set_average_state(True)
    na.set_timeout(timeout)
    na.set_format('slog')
    na.set_trigger_source('BUS')
    count = 0
    while (True):
        for ii, w in enumerate(windows):
            Temperature = fridge.get_temperature('MC RuO2')
            if not Temperature > 0:
                Temperature = fridge.get_temperature('MC cernox')
            print "Trace: %d\t\tWindow: %d\tTemperature: %3.3f" % (count, ii, Temperature)
            na.set_center_frequency(w[0])
            na.set_span(w[1])
            na.set_power(powers[ii])
            na.set_ifbw(ifbws[ii])
            na.set_averages(avgs)
            na.trigger_single()
            na.averaging_complete()  # Blocks!
            na.save_file("%s%04d-%3d-%s-%3.3f.csv" % (datapath, count, ii, fileprefix, Temperature))
            time.sleep(delay)


def nwa_test1(na):
    """Read NWA data and plot results"""
    fpts, mags, phases = na.read_data()

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission, S21 (dB)")
    plt.plot(fpts / 1e9, mags)
    plt.subplot(2, 1, 2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmitted Phase (deg)")
    plt.plot(fpts / 1e9, phases)
    plt.show()


def nwa_test2(na):
    """Test segmented Sweep"""
    na.set_default_state()
    na.set_power(-20)
    na.set_ifbw(1e3)
    na.set_averages(1)

    freqs, mags, phases = na.segmented_sweep(9e9, 15e9, 50e3)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission, S21 (dB)")
    plt.plot(freqs / 1e9, mags)
    plt.subplot(2, 1, 2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmitted Phase (deg)")
    plt.plot(freqs / 1e9, phases)
    plt.show()


def nwa_test3(na):
    na.set_trigger_average_mode(False)
    na.set_power(-20)
    na.set_ifbw(1e3)
    na.set_sweep_points()
    na.set_averages(10)
    na.set_average_state(True)

    print na.get_settings()
    na.clear_averages()
    na.take_one_averaged_trace("test.csv")


def convert_nwa_files_to_hdf(nwa_file_dir, h5file, sweep_min, sweep_max, sweep_label="B Field", ext=".CSV"):
    import glob, h5py, csv
    hfile = h5py.File(h5file)
    files = glob.glob(nwa_file_dir + "*" + ext)
    n_files = len(files)
    for j, fn in enumerate(files):
        f = open(fn, 'r')
        # Skip Header
        for i in range(3):
            f.readline()
        rows = np.array((csv.reader(f)))
        if j is 0:
            n_points = len(rows)
            for t in ["mag", "phase"]:
                hfile[t] = np.zeros((n_points, n_files))
                hfile[t].attrs["_axes"] = ((rows[0][0], rows[-1][0]), (sweep_min, sweep_max))
                hfile[t].attrs["_axes_labels"] = ("Frequency", sweep_label, "S21")
        hfile["mag"][:, j] = rows[:, 1]
        hfile["phase"][:, j] = rows[:, 2]


if __name__ == '__main__':
    #    condense_nwa_files(r'C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data','C:\\Users\\dave\\Documents\\My Dropbox\\UofC\\code\\004 - test temperature sweep\\sweep data\\test')
    na = E5071("E0571", address="192.168.14.130")
    print na.get_id()


# print "Setting window"

# from guiqwt.pyplot import *
# nwa_test2(na)
#    nwa_test2(na)
# nwa_test2(na)

