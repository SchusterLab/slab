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
__author__ = 'Christopher Nolan'

from slab.instruments import SocketInstrument
from time import *
import numpy as np
import glob
import os.path


def polar2mag(xs, ys):
    return 20 * np.log10(np.sqrt(xs ** 2 + ys ** 2)), np.arctan2(ys, xs) * 180 / np.pi

class ZVB8(SocketInstrument):

	MAXSWEEPPTS = 1601
	default_port = 5025
	def __init__(self, name="BatMouse", address=None, enabled=True, reset = True, **kwargs):
		SocketInstrument.__init__(self, name, address, enabled=enabled, recv_length=2 ** 20, **kwargs)
		self.query_sleep = 0.05
		self.timeout = 100
		if reset:
			self.reset()

	def get_id(self):
		print('I\'m Bat Mouse!')
		return self.query('*IDN?')

	def reset(self):
		# Reset and clear all errors
		self.write('*RST; *CLS')
		self.write('OUTP OFF')
		self.write('ROSCillator EXT') # external 10MHz reference clock

	def trans_default_settings(self):
		# Transmission measurement settings
		settings={}

		settings['channel']      = 1
		settings['avg_time']     = 10
		settings['measurement']  = 'S21'
		settings['start_freq']   = 6e9
		settings['stop_freq']    = 8e9
		settings['freq_points']  = 1001
		settings['RFpower']      = -20
		settings['ifBW']         = 500
		settings['mode']         = 'MOV'

		return settings
	
	def trans_meas(self, settings):

		channel      = settings['channel']
		time         = settings['avg_time']
		measurement  = settings['measurement']
		start        = settings['start_freq']
		stop         = settings['stop_freq']
		sweep_points = settings['freq_points']
		rf_power     = settings['RFpower']
		ifBW         = settings['ifBW']
		mode         = settings['mode']

		#Turn off output and switch to single sweep mode instead of continuous sweeps
		self.write('OUTP OFF')
		self.write('INIT:CONT OFF')

		#Configure averaging
		self.configure_averages(channel, 1e4, mode) #high number so that VNA keeps averaging regardless of user averaging time
		self.configure_measurement(channel, measurement)

		#Configure frequency sweep and RF power
		self.configure_frequency(channel, start, stop, sweep_points)
		self.write("SOURce:POWer:MODE ON")
		self.write("SOURce:POWer %f" % (rf_power))
		self.write("sens%d:bwid %f" % (channel, ifBW))

		# start averaging and wait
		self.avg_time(channel, time)
		self.autoscale(window=1)

		# sleep(2.0)
		# self.configure_trace(channel, 'mag', measurement, 'MLOG')
		sleep(1.0)
		# data_mag = self.read_data()
		# data_mag = self.get_trace(channel,'mag') # attempt to grab data referring to trace only

		# self.configure_trace(channel, 'phase', measurement, 'PHAS')
		sleep(1.0)
		# data_phase = self.read_data()
		# data_phase = self.get_trace(channel,'phase')
		data = self.get_channel_data(channel)

		# data={}
		# data['freq'] = datastore[2]
		# data['mag'] = datastore[0]
		# data['phase'] = datastore[1]

		self.write('OUTP OFF')
		return data

	def spec_default_settings(self):
		'''Return default settings for spec measurement'''
		settings = {}

		settings['channel']      = 1
		settings['avg_time']     = 10
		settings['measurement']  = 'S21'
		settings['start_freq']   = 3e9
		settings['stop_freq']    = 5e9
		settings['freq_points'] = 1001
		settings['RFpower']      = -10
		settings['RFport']       = 3
		settings['Mport']        = 2
		settings['CAVport']      = 1
		settings['CAVpower']     = -5
		settings['CAVfreq']      = 7e9
		settings['ifBW']         = 500
		settings['mode']         = 'MOV'

		return settings

	def spec_meas(self, settings):
		'''
		Perform spec measurement using settings dict.
		Measurement consists of:
		    -setting up an cavity port (fixed frequency/power), CAVport
		    -setting up a probe port (swept freq), RFport
		    -configuring the measurement port (Mport) to measure
		        S(Mport-CAVport) as function of RFfreq
		Currently just plots data on VNA
		'''
		channel      = settings['channel']
		time         = settings['avg_time']
		measurement  = settings['measurement']
		start        = settings['start_freq']
		stop         = settings['stop_freq']
		sweep_points = settings['freq_points']
		rf_power     = settings['RFpower']
		rf_port      = settings['RFport']
		m_port       = settings['Mport']
		cav_port     = settings['CAVport']
		cav_power    = settings['CAVpower']
		cav_freq     = settings['CAVfreq']
		ifBW         = settings['ifBW']
		mode         = settings['mode']

		#Turn off output and switch to single sweep mode instead of continuous sweeps
		self.write('OUTP OFF')
		self.write('INIT:CONT OFF')

		#Configure averaging
		self.configure_averages(channel, 1e4, mode)

		#Clear old traces on channels and define a new trace to measure 'S21'
		self.configure_measurement(channel, measurement)

		#Configure frequency sweep and RF power
		# self.configure_frequency(channel, start, stop, sweep_points)
		self.write("SOURce:POWer:MODE ON")
		self.write("SOURce:POWer %f" % (rf_power))
		self.write("sens%d:bwid %f" % (channel, ifBW))

		#Configure ports
		#RF
		##########################################
		#It seems like the data comes out more cleanly when only the rf port is 
		#configured in generator mode. Not sure why this is the case but this
		#cleared out a lot of noise spikes in the spec data
		##########################################
		#configures rf_port to always be on during measurements
		self.write('SOUR{}:POW{}:PERM ON'.format(channel, rf_port))
		self.write('SENS{}:FREQ{}:STAR {}; STOP {}'.format(channel, rf_port, start, stop))
		# self.write('SENS{}:SWE{}:POIN {}'.format(channel, rf_port, sweep_points))
		self.write('SENS{}:SWE:POIN {}'.format(channel, sweep_points))
		#Measure
		#configure to only receive
		# self.write('SOUR{}:POW{}:STATE OFF'.format(channel, m_port))
		#only look at what is happening at cav_freq
		self.write('SOUR{}:FREQ{}:CONV:ARB:IFR 1, 1, {}, FIX'.format(channel, m_port, cav_freq))
		#LO
		#always keep cav_port on during measurements
		# self.write('SOUR{}:POW{}:PERM ON'.format(channel, cav_port))
		#only output at 60MHz
		self.write('SOUR{}:FREQ{}:CONV:ARB:IFR 1, 1, {}, FIX'.format(channel, cav_port, cav_freq))
		#fixed power of cav_power (ignores the power level of Ch1)
		self.write('SOUR{}:POW{}:OFFS {}, ONLY'.format(channel, cav_port, cav_power))
		self.query('*OPC?')

		self.avg_time(channel, time)
		# self.autoscale(window=1)

		# data = self.get_channel_data(channel)
		# sleep(2.0)
		# self.configure_trace(channel, 'mag', measurement, 'MLOG')

		sleep(1)
		# data_mag = self.read_data()

		# self.configure_trace(channel, 'phase', measurement, 'PHAS')
		# sleep(1.0)
		# data_phase = self.read_data()
		# data={}
		# data['freq'] = data_mag[0]
		# data['mag'] = data_mag[1]
		# data['phase'] = data_mag[1]#data_phase[1]
		data = self.get_channel_data(channel)
		self.write('OUTP OFF')
		return data


	### Helpers and utility functions
	def autoscale(self, window=1, ref_trace=None):
		'''Autoscale window to match ref_trace range'''
		tracestr = ''
		if ref_trace is not None:
			tracestr = ",'{}'".format(ref_trace)
			self.write("DISP:WIND{}:TRAC:Y:AUTO ONCE{}".format(window, tracestr))
		if ref_trace is None:
			traces = self.get_channel_traces(1)[::2]
			for trace in traces:
				tracestr = ",'{}'".format(trace)
				self.write("DISP:WIND{}:TRAC:Y:AUTO ONCE{}".format(window, tracestr))

	def avg_time(self, channel, time):
		'''
		Set the instrument to measure for 'time' seconds. The instrument will
		autoscale after 20% of the time has elapsed so that the traces can be
		examined on the VNA
		Arguments:
		    channel (int): channel being used for the measurement
		    time (int): time in seconds that the instrument will measure
		'''
		self.clear_averages(channel)
		self.write('OUTP ON')
		self.write('INIT{}:IMM'.format(channel))
		sleep(time/5)
		self.autoscale()
		sleep(4*time/5)

	def clear_all_traces(self):
		'''Clear all traces defined on instrument'''
		self.write('CALC:PAR:DEL:ALL')
		self.query('*OPC?')

	def clear_averages(self, channel):
		'''Clear averages on channel'''
		self.write('SENS{}:AVER:CLE'.format(channel))

	def clear_channel_traces(self, channel):
		'''Clear all traces defined in channel'''
		self.write('CALC{}:PAR:DEL:CALL'.format(channel))
		self.query('*OPC?')


	def configure_averages(self, channel, averages, mode='MOV'):
		'''Set up channel to measure averages traces'''
		self.write('SENS{}:AVER ON'.format(channel))
		self.write('SENS{}:AVER:MODE {}'.format(channel, mode))
		self.write('SENS{}:AVER:COUN {}'.format(channel, averages))
		self.write('SENS{}:SWE:COUN {}'.format(channel, averages))
		self.clear_averages(channel)
		self.query('*OPC?')

	def configure_frequency(self, channel, start=None, stop=None, sweep_points=None, center=None, span=None):
		'''
		Set up frequency axis
		Arguments:
		    channel(int): channel number, channel must have been created beforehand
		    if specified:
		        start(float/str): frequency of sweep start ('10 MHz' or 10e6)
		        stop(float/str): frequency of sweep end ('100 MHz' or 100e6)
		    if specified:
		        center(float/str): center of frequency sweep
		        span(float/str): span of frequency sweep
		    sweep_points(int): number of points in sweep
		'''
		if sweep_points is None:
			sweep_points = 501
		if start is not None and stop is not None:
			self.write('SENS{}:FREQ:STAR {}; STOP {}'.format(channel, start, stop))
			self.write('SENS{}:SWE:POIN {}'.format(channel, sweep_points))
		if center is not None and span is not None:
			self.write('SENS{}:FREQ:CENT {}; SPAN {}'.format(channel, center, span))
			self.write('SENS{}:SWE:POIN {}'.format(channel, sweep_points))
		self.query('*OPC?')

	def configure_measurement(self, channel, measurement, window = 1):
		'''
		Set up a basic 'Sij' measurement configuring traces for magnitude and 
		phase. If the traces already exist, the code will ignore the clearing
		commands and return without doing anything
		Arguments:
		    channel (int): channel number for measurement
		    measurement (string): measurement to be performed (e.g 'S21')
		    window (int): window to display the traces in (default 1)
		'''
		traces = self.get_channel_traces(channel)
		reinit = True
		if 'mag' in traces and 'phase' in traces:
			reinit = False
		if reinit:
			self.clear_channel_traces(channel)
			self.configure_trace(channel, 'mag', measurement, 'MLOG')
			self.configure_trace(channel, 'phase', measurement, 'PHAS')
			self.display_trace('mag',tracenum=1, window=window)
			self.display_trace('phase',tracenum=2, window=window)

	def configure_trace(self, channel, name, meastype, measformat):
		'''
		Configure trace to measure a specific parameter in a given format
		Argument:
		    channel(int): channel number
		    name(str): name to assign to trace
		    meastype(str): parameter being measured (e.g. 'S21')
		    measformat(str): format to measure parameter (e.g. 'MLOG' for dB scale)
		'''
		self.write("CALC{}:PAR:SDEF '{}', '{}'".format(channel, name, meastype))
		self.write("CALC{}:FORM {}".format(channel, measformat))
		self.query('*OPC?')

	def display_trace(self, trace, tracenum, window):
		'''
		Display trace in window assigning it the tracenum number
		Arguments:
		    trace (str): name of trace to be displayed
		    tracenum (int): number to assign to trace in window
		    window (int): window to display trace in (will create a new one if needed)
		'''
		self.write('DISP:WIND{}:STAT ON'.format(window))
		self.query('*OPC?')
		self.write('DISP:WIND{}:TRAC{}:FEED "{}"'.format(window, tracenum, trace))
		self.query('*OPC?')

	def get_channel_axis(self, channel):
		'''Return numpy array of channel x-axis'''
		# xaxis = self.query_ascii_values("CALC{}:DATA:STIM?".format(channel))
		xaxis = self.query("CALC{}:DATA:STIM?".format(channel))
		return np.asarray(xaxis)

	def get_channel_data(self, channel):
		'''Return dictionary with all the data from a channel'''
		traces = self.get_channel_traces(channel)[::2]
		data = {}
		for trace in traces:
			data[trace] = self.get_trace(channel, trace)
		data['xaxis'] = self.get_channel_axis(channel)
		return data

	def get_channel_traces(self, channel):
		'''Return list of traces defined in channel'''
		instresp  = self.query('CALC{}:PAR:CAT?'.format(channel))
		tracelist = instresp.strip("\n'").split(',')
		return tracelist

	def get_trace(self, channel, name):
		'''
		Return measurement data for trace with name 'name'
		Arguments:
		    channel(int): channel number
		    name(str): trace name, same as name given at trace creation
		'''
		tracelist = self.get_channel_traces(channel)[::2]
		if name not in tracelist:
			print('Trace not found, possible options are: {}'.format(tracelist))
			return np.zeros(1000)
		# data = self.query_ascii_values("CALC{}:DATA:TRAC? '{}',FDAT".format(channel, name))
		data = self.query("CALC{}:DATA:TRAC? '{}',FDAT".format(channel, name))
		return data#np.asarray(data)

	def read_data(self, sweep_points=None, channel=1, timeout=None, data_format='ascii'):
		"""
		Read current NWA Data that is displayed on the screen. Returns TWO numbers per data point for Polar ('POL')
		and Smith Chart ('SMIT') format, see set_format.
		:param sweep_points: number of sweep points (optional, saves time)
		:param channel: measurement channel (optional, saves time)
		:param timeout: timeout in seconds (optional)
		:param data_format: 'binary' or 'ascii' (optional, saves time). If specificied, this must be equal to get_data_transfer_format()
		:return: 2 or 3 column data containing the frequency and data (1 or 2 column).
		"""
		if data_format is None or not (data_format in ['binary', 'ascii']):
			data_format = self.get_data_transfer_format()

		if sweep_points is None:
			sweep_points = self.get_sweep_points()

		if timeout is None:
			timeout = self.timeout
		# self.get_operation_completion()
		self.read(timeout=1)
		self.write("CALC%d:DATA? FDATA" % channel)
		data_str = b''.join(self.read_lineb(timeout=timeout))

		if data_format == 'binary':
			len_data_dig = np.int(data_str[1:2])
			len_data_expected = int(data_str[2: 2 + len_data_dig])
			len_data_actual = len(data_str[2 + len_data_dig:-1])
			# It may happen that only part of the message is received. We know that this is the case by checking
			# the checksum. If the received data is too short, just read out again.
			while len_data_actual != len_data_expected:

				data_str += b''.join(self.read_lineb(timeout=timeout))
				len_data_actual = len(data_str[2 + len_data_dig:-1])

		data = np.fromstring(data_str, dtype=float, sep=',') if data_format == 'ascii' else np.fromstring(
			data_str[2 + len_data_dig:-1], dtype=np.float32)
		fpts = np.linspace(self.get_start_frequency(), self.get_stop_frequency(), sweep_points)
		if len(data) == 2 * sweep_points:
			data = data.reshape((-1, 2))
			data = data.transpose()
			return np.vstack((fpts, data))
		else:
			return np.vstack((fpts, data))

	def get_data_transfer_format(self):
		"""
		Returns the data format for transferring measurement data and frequency data.
		:return: 'ascii' or 'binary'
		"""
		answer = self.query('FORM:DATA?')
		ret = 'ascii' if 'ASC' in answer else 'binary'
		return ret

	def get_sweep_points(self, channel=1):
		data = self.query(":sense%d:sweep:points?" % (channel))
		return int(data)

	def get_operation_completion(self):
		data = self.query("*OPC?")
		if data is None:
			return False
		else:
			return bool(int(data.strip()))

	def get_start_frequency(self, channel=1):
		return float(self.query(":SENS%d:FREQ:START?" % channel))

	def get_stop_frequency(self, channel=1):
		return float(self.query(":SENS%d:FREQ:STOP?" % channel))





class RhodeSchwarz(SocketInstrument):
    MAXSWEEPPTS = 1601
    default_port = 5025

    def __init__(self, name="BatMouse", address=None, enabled=True, **kwargs):
        SocketInstrument.__init__(self, name, address, enabled=enabled, recv_length=2 ** 20, **kwargs)
        self.query_sleep = 0.05
        self.timeout = 100

    def get_id(self):
        return self.query('*IDN?')

    def get_query_sleep(self):
        return self.query_sleep

    def set_display_state(self, state=True):
        """
        Specifies whether to disable or enable all analyzer display information in all windows in the analyzer application.
        Marker data is not updated. More CPU time is spent making measurements instead of updating the display.
        http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/Display.htm
        :param state: True/False
        :return: None
        """
        enable = 1 if state else 0
        self.write("DISP:ENAB %d" % enable)

    def get_display_state(self):
        """
        Specifies whether to disable or enable all analyzer display information in all windows in the analyzer application.
        Marker data is not updated. More CPU time is spent making measurements instead of updating the display.
        :return:
        """
        return self.query("DISP:ENAB?")

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

        allowed_modes = ['cont', 'continuous', 'hold', 'groups', 'gro', 'sing', 'single']
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
        answer = self.query('SENS%d:SWE:TIME?' % channel)
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

    def set_port_powers_coupled(self, state=True):
        self.write("SOUR:POW:COUP %d" % int(state))

    def setup_two_tone_measurement(self, read_frequency=None, read_power=None, probe_start=None, probe_stop=None,
                                   probe_power=None, two_tone=True):
        if two_tone:
            print("TWO TONE ON")
            self.write('SENSE:FOM:RANGE4:COUPLED 1')
            self.write('SENSE:FOM:RANGE2:COUPLED 0')
            self.write('SENSE:FOM:RANGE3:COUPLED 0')

            if probe_start is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % probe_start)
            if probe_stop is not None:
                self.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % probe_stop)

            # self.write('SENSE:FOM:RANGE2:COUPLED 0')
            # self.write('SENSE:FOM:RANGE3:COUPLED 0')
            if read_frequency is not None:
                self.write('SENSE:FOM:RANGE2:FREQUENCY:START %f' % read_frequency)
                self.write('SENSE:FOM:RANGE2:FREQUENCY:STOP %f' % read_frequency)
                self.write('SENSE:FOM:RANGE3:FREQUENCY:START %f' % read_frequency)
                self.write('SENSE:FOM:RANGE3:FREQUENCY:STOP %f' % read_frequency)

            self.set_frequency_offset_mode_state(True)
            self.set_port_powers_coupled(False)

            if read_power is not None:
                self.set_power(read_power, channel=1, port=1)
            if probe_power is not None:
                self.set_power(probe_power, channel=1, port=3)
        else:
            print("TWO TONE OFF")
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

    def setup_rf_flux_measurement(self, read_power=None, probe_power=None, read_start=None, read_stop=None,
                                  probe_frequency=None):

        self.write('SENSE:FOM:RANGE2:COUPLED 1')
        self.write('SENSE:FOM:RANGE3:COUPLED 1')
        self.write('SENSE:FOM:RANGE4:COUPLED 0')
        if read_start is not None:
            self.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % read_start)
        if read_stop is not None:
            self.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % read_stop)

        if probe_frequency is not None:
            self.write('SENSE:FOM:RANGE4:FREQUENCY:START %f' % probe_frequency)
            self.write('SENSE:FOM:RANGE4:FREQUENCY:STOP %f' % probe_frequency)

        self.set_frequency_offset_mode_state(True)
        if read_power is not None:
            self.set_power(read_power, channel=1, port=1)
        if probe_power is not None:
            self.set_power(probe_power, channel=1, port=3)

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
        """
        This command sets the trigger mode to continuous (internal) or manual
        NB: to refresh the display, use set_sweep_mode("CONT") in combination
        with this command.
        """
        if state:
            _state = "on"
        else:
            _state = "off"
        self.write('initiate:continuous ' + _state)

    def trigger_single(self, channel=1):
        self.write('initiate%d:immediate' % channel)

    def set_trigger_average_mode(self, mode=None):
        allowed_modes = ['poin', 'point', 'sweep']
        if mode is None:
            return
        elif mode.lower() in allowed_modes:
            self.write('sense:AVER:mode ' + mode)
        else:
            print("trigger average mode needs to be one of " + ', '.join(allowed_modes))

    def get_trigger_average_mode(self):
        data = self.query('sense:AVER:mode?')
        if data is None:
            return None
        else:
            return data.strip()

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
            print("source need to be one of " + ', '.join(allowed_sources))
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
            self.write("TRIG:TYPE %s" % trigger_type)
            if slope > 0:
                self.write("TRIG:SLOP POS")
            else:
                self.write("TRIG:SLOP NEG")
        else:
            raise ValueError("Input not understood!")

    def set_external_trigger_connector(self, connector="MAIN"):
        """
        Specifies the connector to use for the external trigger input.
        Meas Trig In BNC || material handler I/O Pin 18 || Internal routing of  pulse 3 output to the MEAS TRIG IN on the rear panel.
        :param trigger_connector: string; one of the following: ["MAIN", "MATH", "PULSE3"]
        :return: None
        """
        if connector.upper() in ["MAIN", "MATH", "PULSE3"]:
            self.write("TRIG:ROUTE:INP %s" % connector.upper())

    def get_external_trigger_connector(self):
        """
            Gets the connector to use for the external trigger input.
            Meas Trig In BNC || material handler I/O Pin 18 || Internal routing of  pulse 3 output to the MEAS TRIG IN on the rear panel.
            :param trigger_connector: string; one of the following: ["MAIN", "MATH", "PULSE3"]
            :return: None
            """
        return self.query("TRIG:ROUTE:INP?")

    def set_channel_trigger_state(self, state):
        """
        Trigger mode. choose from:

        CHANnel - Each trigger signal causes ALL traces in that channel to be swept.
        SWEep - Each Manual or External trigger signal causes ALL traces that share a source port to be swept.
        POINt -- Each Manual or External trigger signal causes one data point to be measured.
        TRACe - Allowed ONLY when SENS:SWE:GEN:POIN is enabled. Each trigger signal causes two identical measurements to be
        triggered separately - one trigger signal is required for each measurement. Other trigger mode settings cause two
        identical parameters to be measured simultaneously.
        :param state: 'CHAN', 'POIN', 'SWE', 'TRAC'
        :return: None
        """
        if state.upper() in ['CHAN', 'POIN', 'SWE', 'TRAC']:
            self.write("SENS:SWE:TRIG:MODE %s" % state)

    def get_channel_trigger_state(self):
        """
        CHANnel - Each trigger signal causes ALL traces in that channel to be swept.
        SWEep - Each Manual or External trigger signal causes ALL traces that share a source port to be swept.
        POINt -- Each Manual or External trigger signal causes one data point to be measured.
        TRACe - Allowed ONLY when SENS:SWE:GEN:POIN is enabled. Each trigger signal causes two identical measurements to be
        triggered separately - one trigger signal is required for each measurement. Other trigger mode settings cause two
        identical parameters to be measured simultaneously.
        :return:
        """
        return self.query("SENS:SWE:TRIG:MODE?")

    #### Source
    def set_power(self, power, channel=1, port=1, state=1):
        if state:
            self.write(":SOURCE%d:POWER%d:MODE ON" % (channel, port))
            self.write(":SOURCE%d:POWER%d %f" % (channel, port, power))
        else:
            print("Turning off the port %d" % (port))
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
            return list(zip(*[iter(data_list)] * 2))

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

    def set_data_transfer_format(self, format='ascii'):
        """
        Sets the data format for transferring measurement data and frequency data.
        See the Format Commands help section for more help on this topic.
        :param format: Either 'ascii' or 'binary'
        :return:
        """
        send_data = 'ASC,0' if format.lower() == 'ascii' else 'REAL,32'
        self.write("FORM %s" % send_data)
        if send_data == 'REAL,32':
            self._set_byte_order('SWAP')

    def get_data_transfer_format(self):
        """
        Returns the data format for transferring measurement data and frequency data.
        :return: 'ascii' or 'binary'
        """
        answer = self.query('FORM:DATA?')
        ret = 'ascii' if 'ASC' in answer else 'binary'
        return ret

    def _set_byte_order(self, order='SWAP'):
        """
        #NOTE for Plutonium, the byte order needs to be swapped!
        Set the byte order used for GPIB data transfer. Some computers read data from the analyzer in the reverse order.
        This command is only implemented if FORMAT:DATA is set to :REAL. If FORMAT:DATA is set to :ASCII, the swapped command is ignored.
        :param order: 'swap' for swapped or 'norm' normal order
        :return: None
        """
        if order.upper() in ['SWAP', 'NORM']:
            self.write("FORM:BORD %s" % order.upper())

    def _get_byte_order(self):
        """
        Returns the byte order used for GPIB data transfer.
        :return: 'SWAP' (swapped) or 'NORM' (normal order)
        """
        return self.query("FORM:BORD?").strip()

    def set_format(self, trace_format='MLOG', trace=1):
        """
        Sets the display format for the measurement.
        This needs to be run after the active trace is set. The following options are available:
        MLINear, MLOGarithmic, PHASe, UPHase (Unwrapped phase), IMAGinary, REAL, POLar, SMITh, SADMittance (Smith Admittance)
        SWR, GDELay (Group Delay), KELVin, FAHRenheit, CELSius
        """
        allowed = ['MLIN', 'MLOG', 'PHAS', 'UPH', 'IMAG', 'REAL', 'POL', 'SMIT', 'SADM', 'SWR', 'GDEL', 'KEL', 'FAHR',
                   'CEL']
        if trace_format.upper() in allowed:
            self.write("CALC%d:FORM %s" % (trace, trace_format.upper()))
        else:
            raise ValueError("Specified trace format not allowed. Use %s" % allowed)

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
        """
        Sets the electrical delay in seconds
        :param seconds: Electrical delay in seconds
        :param channel: Measurement channel
        :return: None
        """
        query = "calc%d:corr:edel:time %e" % (channel, seconds)
        self.write(query)

    def get_electrical_delay(self, channel=1):
        """
        Returns the electrical delay in seconds
        :param channel: Measurement channel
        :return: Electrical delay in seconds
        """
        query = "calc%d:corr:edel:time?" % channel
        data = self.query(query)
        if data is None:
            return None
        else:
            return float(data.strip())

    def set_phase_offset(self, degrees, channel=1):
        """
        Sets the phase offset for the selected measurement
        :param degrees: Phase offset in degrees. Choose any number between -360 and 360.
        :param channel: Measurement channel
        :return:
        """
        query = "CALC%d:OFFS:PHAS %.3f" % (channel, degrees)
        self.write(query)

    def get_phase_offset(self, channel=1):
        """
        Returns the phase offset for the selected measurement
        :param channel: Measurement channel
        :return: Numeric, returned value always in degrees
        """
        query = "CALC%d:OFFS:PHAS?" % channel
        data = self.query(query)
        if data is None:
            return None
        else:
            return float(data.strip())

    #### File Operations
    def save_file(self, fname):
        self.write('MMEMORY:STORE:FDATA \"' + fname + '\"')

    def read_data(self, sweep_points=None, channel=1, timeout=None, data_format=None):
        """
        Read current NWA Data that is displayed on the screen. Returns TWO numbers per data point for Polar ('POL')
        and Smith Chart ('SMIT') format, see set_format.
        :param sweep_points: number of sweep points (optional, saves time)
        :param channel: measurement channel (optional, saves time)
        :param timeout: timeout in seconds (optional)
        :param data_format: 'binary' or 'ascii' (optional, saves time). If specificied, this must be equal to get_data_transfer_format()
        :return: 2 or 3 column data containing the frequency and data (1 or 2 column).
        """
        if data_format is None or not (data_format in ['binary', 'ascii']):
            data_format = self.get_data_transfer_format()

        if sweep_points is None:
            sweep_points = self.get_sweep_points()

        if timeout is None:
            timeout = self.timeout
        self.get_operation_completion()
        self.read(timeout=0.1)
        self.write("CALC%d:DATA? FDATA" % channel)
        data_str = b''.join(self.read_lineb(timeout=timeout))

        if data_format == 'binary':
            len_data_dig = np.int(data_str[1:2])
            len_data_expected = int(data_str[2: 2 + len_data_dig])
            len_data_actual = len(data_str[2 + len_data_dig:-1])
            # It may happen that only part of the message is received. We know that this is the case by checking
            # the checksum. If the received data is too short, just read out again.
            while len_data_actual != len_data_expected:
                data_str += b''.join(self.read_lineb(timeout=timeout))
                len_data_actual = len(data_str[2 + len_data_dig:-1])

        data = np.fromstring(data_str, dtype=float, sep=',') if data_format == 'ascii' else np.fromstring(
            data_str[2 + len_data_dig:-1], dtype=np.float32)
        fpts = np.linspace(self.get_start_frequency(), self.get_stop_frequency(), sweep_points)
        if len(data) == 2 * sweep_points:
            data = data.reshape((-1, 2))
            data = data.transpose()
            return np.vstack((fpts, data))
        else:
            return np.vstack((fpts, data))

    #### Meta
    def take(self, sweep_points=None, data_format=None):
        """
        Important:
            the PNA-X needs to be in the following mode
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
        data = self.read_data(sweep_points=sweep_points, data_format=data_format)
        return data

    def take_in_mag_phase(self, sweep_points=None, data_format=None):
        fpts, xs, ys = self.take(sweep_points=sweep_points, data_format=data_format)
        mags, phases = polar2mag(xs, ys)
        return fpts, mags, phases

    def take_one_in_mag_phase(self, sweep_points=None, data_format=None):
        """
        Takes one averaged trace and return fpts, magnitudes and phases
        :param sweep_points: Sweep points (optional, saves time)
        :param data_format: 'ascii' or 'binary' (optional, saves time)
        :return: fpts, mags, phases
        """
        _trig_source = self.get_trigger_source()
        _format = self.get_format()
        self.setup_take()
        fpts, xs, ys = self.take(sweep_points=sweep_points, data_format=data_format)
        mags, phases = polar2mag(xs, ys)
        self.set_trigger_source(_trig_source)
        self.set_format(_format)
        return fpts, mags, phases

    def setup_take(self, averages=None, averages_state=None):
        self.set_trigger_source("imm")
        self.set_format('POL')
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

    ##########################
    ### Tested R&S scripts ###
    ##########################

if __name__ == '__main__':
	print("hello")
    # na = N5242A("N5242A", address="192.168.14.242")
    # print(na.get_id())
