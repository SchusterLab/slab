# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:59:04 2013

@author: Dave
"""
from slab.instruments import VisaInstrument, InstrumentManager
import numpy as np
from numpy import array, floor, zeros
from collections import namedtuple
from .TekPattern import write_Tek_file
import hashlib
import struct
import io
import io
import time

#comment out if not debugging
#from liveplot import LivePlotClient
#lp=LivePlotClient()


class Tek5014(VisaInstrument):
    """Tektronix 5014 Arbitrary Waveform Class"""
    # default_port=4000
    def __init__(self, name='Tek5014', address='', enabled=True,timeout = 50):
        address = address.upper()

        if address[:5] != 'TCPIP':
            address = 'TCPIP::' + address + '::INSTR'
        VisaInstrument.__init__(self, name, address, enabled, timeout)
        self._loaded_waveforms = []
        self.current_sequence_hash = ''

    def get_id(self):
        return self.query("*IDN?")

    def set_amplitude(self, channel, amp):
        self.write('SOURce%d:VOLTage:AMPLitude %f' % (channel, amp))

    def get_amplitude(self, channel):
        return float(self.query('SOURce%d:VOLTage:AMPLitude?' % (channel)))

    def set_analogHigh(self, channel, value):
        self.write('SOURce%d:VOLTage:HIGH %f' % (channel, value))

    def get_analogHigh(self, channel):
        return float(self.query('SOURce%d:VOLTage:HIGH?' % (channel)))

    def set_analogLow(self, channel, value):
        self.write('SOURce%d:VOLTage:LOW %f' % (channel, value))

    def get_analogLow(self, channel):
        return float(self.query('SOURce%d:VOLTage:LOW?' % (channel)))

    def set_DACResolution(self, channel, value):
        checkset = [8, 10, 14]
        if value not in checkset: raise Exception('Error: invalid DAC Resolution in Tek5014')
        self.write('SOURce%d:DAC:RESolution %d' % (channel, value))

    def get_DACResolution(self, channel):
        return int(self.query('SOURce%d:DAC:RESolution %d' % (channel)))

    def set_delay(self, channel, value):
        self.write('SOURce%d:DELay %f' % (channel, value))

    def get_delay(self, channel):
        return float(self.query('SOURce%d:DELay?' % (channel)))

    def set_enabled(self, channel, state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write('OUTPut%d:STATe ON' % channel)
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write('OUTPut%d:STATe OFF' % channel)
        else:
            raise Exception('Tek5014: Invalid enabled state')

    def get_enabled(self, channel):
        return bool(self.query('OUTPut%d:STATe?' % channel))

    def set_lowpassFilterFrequency(self, channel, value):
        self.write('OUTPut%d:FILTer:LPASs:FREQuency %f' % (channel, value))

    def get_lowpassFilterFrequency(self, channel):
        return float(self.query('OUTPut%d:FILTer:LPASs:FREQuency %f' % (channel)))

    def set_markerHigh(self, channel, marker_index, value):
        self.write('SOURce%d:MARKER%d:VOLTage:HIGH %f' % (channel, marker_index, value))

    def get_markerHigh(self, channel, marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:HIGH?' % (channel, marker_index)))

    def set_markerLow(self, channel, marker_index, value):
        self.write('SOURce%d:MARKER%d:VOLTage:LOW %f' % (channel, marker_index, value))

    def get_markerLow(self, channel, marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:LOW?' % (channel, marker_index)))

    def set_markerOffset(self, channel, marker_index, value):
        self.write('SOURce%d:MARKER%d:VOLTage:OFFSet %f' % (channel, marker_index, value))

    def get_markerOffset(self, channel, marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:OFFSet?' % (channel, marker_index)))

    def set_markerAmplitude(self, channel, marker_index, value):
        self.write('SOURce%d:MARKER%d:VOLTage:AMPLitude %f' % (channel, marker_index, value))

    def get_markerAmplitude(self, channel, marker_index):
        return float(self.query('SOURce%d:MARKER%d:VOLTage:AMPLitude?' % (channel, marker_index)))

    def set_marker1High(self, channel, value):
        self.set_markerHigh(channel, 1, value)

    def get_marker1High(self, channel):
        return self.get_markerHigh(channel, 1)

    def set_marker2High(self, channel, value):
        self.set_markerHigh(channel, 2, value)

    def get_marker2High(self, channel):
        return self.get_markerHigh(channel, 2)

    def set_marker1Low(self, channel, value):
        self.set_markerHigh(channel, 1, value)

    def get_marker1Low(self, channel):
        return self.get_markerHigh(channel, 1)

    def set_marker2Low(self, channel, value):
        self.set_markerHigh(channel, 2, value)

    def get_marker2Low(self, channel):
        return self.get_markerHigh(channel, 2)

    def set_marker1Offset(self, channel, value):
        self.set_markerHigh(channel, 1, value)

    def get_marker1Offset(self, channel):
        return self.get_markerHigh(channel, 1)

    def set_marker2Offset(self, channel, value):
        self.set_markerHigh(channel, 2, value)

    def get_marker2Offset(self, channel):
        return self.get_markerHigh(channel, 2)

    def set_marker1Amplitude(self, channel, value):
        self.set_markerHigh(channel, 1, value)

    def get_marker1Amplitude(self, channel):
        return self.get_markerHigh(channel, 1)

    def set_marker2Amplitude(self, channel, value):
        self.set_markerHigh(channel, 2, value)

    def get_marker2Amplitude(self, channel):
        return self.get_markerHigh(channel, 2)

    def set_outputWaveformName(self, channel, value):
        self.write('SOURce%d:WAVeform "%s"' % (channel, value))

    def get_outputWaveformName(self, channel):
        return self.query('SOURce%d:WAVeform?' % (channel))

    def set_offset(self, channel, value):
        self.write('SOURce%d:VOLTage:OFFSet %f' % (channel, value))

    def get_offset(self, channel):
        return float(self.query('SOURce%d:VOLTage:OFFSet?' % (channel)))

    def set_skew(self, channel, value):
        self.write('SOURce%d:SKEW %f' % (channel, value))

    def get_skew(self, channel):
        return float(self.query('SOURce%d:SKEW?' % (channel)))


    def get_setup_filename(self):
        return self.query('AWGControl:SNAMe?')

    def save_config(self, name):
        self.write('AWGControl:SSAVe "%s"' % name)

    def open_config(self, name):
        self.write('AWGControl:SREStore "%s"' % name)

    def import_waveform(self, waveform_name, filename, file_type):
        self.write('MMEMory:IMPort "%s", "%s", "%s"' % (waveform_name, filename, file_type))

    def make_directory(self, name):
        self.write('MMEMory:MDIRectory "%s"' % name)

    def reset(self):
        self.write('*RST')

    def clear_reg_and_queue(self):
        self.write('*CLS')

    def calibrate(self):
        return self.query('*CAL?')

    def operation_complete(self):
        return self.query('*OPC?')

    def run(self):
        self.write('AWGControl:RUN')
        self.operation_complete()

    def stop(self):
        self.write('AWGControl:STOP')
        self.operation_complete()

    def sync(self):
        self.write('*OPC')

    def create_waveform(self, name, size, waveform_type):
        self.write(':WLISt:WAVeform:NEW "%s", %d, %s' % (name, size, waveform_type))

    def delete_waveform(self, name):
        self.write(':WLISt:WAVeform:DEL "%s"' % name)

    def set_current_working_directory(self, value):
        return self.query('MMEMory:CDIRectory "%s"' % value)

    def get_current_working_directory(self):
        return self.query('MMEMory:CDIRectory?')

    def set_impedance(self, value):
        """value can be '50' or '1k'"""
        d = {'50': '50', 50: '50', '1k': '1k', 1000: '1k', '1000': '1k'}
        self.write('TRIGger:IMPedance %s' % d[value])

    def get_impedance(self):
        return self.query('TRIGger:IMPedance?')

    def set_trigger_interval(self, value):
        self.write('TRIGger:TIMer %f' % value)

    def get_trigger_interval(self):
        return float(self.query('TRIGger:Timer?'))

    def set_trigger_level(self, value):
        self.write('TRIGger:LEVel %f' % value)

    def get_trigger_level(self):
        return float(self.query('TRIGger:LEVel?'))

    def set_trigger_slope(self, value):
        """Sets trigger slope, value = 'POS' or 'NEG'"""
        self.write('TRIGger:Polarity %s' % value)

    def get_trigger_slope(self):
        return self.query('TRIGger:Polarity?')

    def set_trigger_source(self, value):
        """valid values = 'INT', 'EXT' """
        self.write('TRIGger:SOURce %s' % value)

    def get_trigger_source(self):
        return self.query('TRIGger:SOURce?')

    def set_clock_source(self, value):
        """valid values = 'INT' , 'EXT' """
        self.write('AWGControl:CLOCk:SOURce %s' % value)

    def get_clock_source(self):
        return self.query('AWGControl:CLOCk:SOURce?')

    def set_run_mode(self, value):
        """valid values = 'TRIG','SEQ','GAT','CONT' """
        self.write('AWGControl:RMODe %s' % value)

    def get_run_mode(self, value):
        return self.query('AWGControl:RMODe?')

    def set_reference_source(self, value):
        """valid values = 'INT', 'EXT' """
        self.write('SOURce1:ROSCillator:SOURce %s' % value)

    def get_reference_source(self):
        return self.query('SOURce1:ROSCillator:SOURce?')

    def set_repetition_rate(self, value):
        self.write('AWGControl:RRATe %f' % value)

    def get_repetition_rate(self):
        return float(self.query('AWGControl:RRATe?'))

    def set_sampling_rate(self, value):
        self.write('SOURCe1:FREQuency %f' % value)

    def get_sampling_rate(self):
        return float(self.query('SOURCe1:FREQuency?'))

    def set_sequence_length(self, value):
        self.write('SEQuence:LENGth %d' % value)

    def get_sequence_length(self):
        return int(self.query('SEQuence:LENGth?'))

    def set_loop_count(self, value):
        self.write('SEQuence:ELEMent:LOOP:COUNt %d' % value)

    def get_loop_count(self):
        return int(self.query('SEQuence:ELEMent:LOOP:COUNt?'))

    def set_waveform_name(self, index, name):
        self.write('SEQuence:ELEMent1:WAVeform%d "%s"' % (index, name))

    def get_waveform_name(self, index):
        return self.query('SEQuence:ELEMent1:WAVeform%d?' % (index))

    def set_divider_rate(self, value):
        """valid options 1,2,4,8"""
        self.write('AWGControl:CLOCk:DRATe %d' % value)

    def get_divider_rate(self):
        return int(self.query('AWGControl:CLOCk:DRATe?'))

    def set_reference_oscillator_type(self, value):
        """valid values = 'FIX', 'VAR' """
        self.write('SOURce1:ROSCillator:TYPE %s' % value)

    def get_reference_oscillator_type(self):
        return self.query('SOURce1:ROSCillator:TYPE?')

    def set_reference_oscillator_frequency(self, value):
        """valid values '10MHz', '20MHz', '100MHz' """
        self.write('SOURce1:ROSCillator:FREQuency %s' % value)

    def get_reference_oscillator_frequency(self):
        return self.query('SOURce1:ROSCillator:FREQuency?')

    def get_error_log(self):
        done = False
        log = ''
        while not done:
            s = self.query('SYSTEM:ERR?')
            done = s[0] == '0'
            log += s
        return log


    def pre_load(self):
        self.stop()
        self.reset()

    def load_sequence_file(self, filename, force_reload=False):

        sequence_hash = hashlib.md5(open(filename).read()).hexdigest()
        if (self.current_sequence_hash != sequence_hash) or force_reload:
            self.current_sequence_hash = sequence_hash
            self.write("AWGControl:SREStore '%s' \n" % (filename))

    def prep_experiment(self):
        self.write("SEQuence:JUMP 1")
        for i in range(1, 5):
            self.set_enabled(i, True)

    def stop_and_prep(self):
        self.stop()
        self.prep_experiment()
        #self.run()


    def set_amps_offsets(self, channel_amps=[1.0, 1.0, 1.0, 1.0], channel_offsets=[0.0, 0.0, 0.0, 0.0],
                         marker_amps=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):

        for i in range(1, 5):
            self.set_amplitude(i, channel_amps[i - 1])
            self.set_offset(i, channel_offsets[i - 1])
            for j in range(2):
                self.set_markerHigh(i, j + 1, marker_amps[2 * (i - 1) + j])
                self.set_markerLow(i, j + 1, 0.0)

    # ##############################################
    # Waveform loading functions  --- from phil
    ###############################################

    def get_bindata(self, data, m1=None, m2=None):
        '''
        Convert floating point data into 14 bit integers.
        '''
        absmax = np.max(np.abs(data))
        if absmax > 1:
            raise ValueError('Unable to convert data with absolute value larger than 1')

        # 0 corresponds to minus full-scale + (1 / 2**14)
        # 2**13-1 = 8191 corresponds to zero
        # 2**14-1 = 16383 corresponds to plus full-scale
        bytemem = np.round(data * (2 ** 13 - 2)) + (2 ** 13 - 1)
        bytemem = bytemem.astype(np.uint16)

        if m1 is not None:
            if len(data) != len(m1):
                raise ValueError('Data and marker1 should have same length')
            bytemem |= 1 << 14 * m1.astype(np.bool)
        if m2 is not None:
            if len(data) != len(m2):
                raise ValueError('Data and marker2 should have same length')
            bytemem |= 1 << 15 * m2.astype(np.bool)

        return bytemem

    # add custom waveform as file, not correct
    def add_file(self, fn, data):
        bindata = self.get_bindata(data)
        cmd = ('MMEM:DATA "%s",#6%06d' % (fn, 2 * len(data))) + bindata.tostring() + '\n'
        self.write(cmd)

    def add_waveform(self, wname, data, m1=None, m2=None, replace=True, return_cmd=False):
        '''
        Add waveform <wname> to AWG with content <data> and marker content
        <m1> and <m2>.
        '''
        if not replace and wname in self._loaded_waveforms:
            return None
        # logging.info('Adding waveform %s (%d bytes)', wname, len(data))
        self._loaded_waveforms.append(wname)

        bindata = self.get_bindata(data, m1, m2)
        cmd = 'WLIST:WAV:DEL "%s";' % wname
        cmd += ':WLIST:WAV:NEW "%s",%d,INT;' % (wname, len(data))
        cmd += ':WLIST:WAV:DATA "%s",0,%d,#6%06d' % (wname, len(bindata), 2 * len(bindata))
        cmd += bindata.tostring() + '\n'
        # logging.info(self.get_error())

        if return_cmd:
            return cmd

        cmd += ':OUTP?'
        old_timeout = self.get_timeout()
        self.set_timeout(60)
        # self.query(cmd)
        self.instrument.write_raw(cmd)
        self.read()
        self.set_timeout(old_timeout)

    ###############################################
    # Sequence functions
    ###############################################

    def do_get_seq_pos(self):
        return int(self.query('AWGC:SEQ:POS?'))

    def clear_sequence(self):
        '''
        Clear the sequence memory.
        '''
        self.write('SEQ:LENG 0\n')

    def setup_sequence(self, n_el, reset=True, loop=True):
        self.set_run_mode('SEQ')
        old_timeout = self.get_timeout()
        self.set_timeout(60000)
        self.get_enabled()
        if reset:
            self.write('SEQ:LENG 0\n')
            self.get_enabled()
        self.write('SEQ:LENG %d\n' % n_el)
        self.get_enabled()
        self.set_timeout(old_timeout)

        if loop:
            self.write('SEQ:ELEM%d:GOTO:STATE ON' % n_el)
            self.write('SEQ:ELEM%d:GOTO:INDEX 1' % n_el)

    def set_seq_element(self, ch, el, wname, repeat=1, trig=False):
        self.write('SEQ:ELEM%d:WAV%d "%s"' % (el, ch, wname))
        if repeat > 1:
            self.write('SEQ:ELEM%d:LOOP:COUNT %d' % (el, repeat))
        if trig:
            self.write('SEQ:ELEM%d:TWAIT 1' % (el,))


MAX_WAVEFORM_VALUE = 2 ** 13 - 1  # maximum waveform value i.e. 14bit DAC


def write_field(FID, fieldName, data, dataType):
    typeSizes = {'int16': 2, 'int32': 4, 'double': 8, 'uint128': 16}
    formatChars = {'int16': '<h', 'int32': '<i', 'double': '<d'}

    if dataType == 'char':
        dataSize = len(data) + 1
        data = data + chr(0)
    else:
        dataSize = typeSizes[dataType]

    FID.write(struct.pack('<II', len(fieldName) + 1, dataSize))
    FID.write(fieldName + chr(0))
    if dataType == 'char':
        FID.write(data)
    elif dataType == 'uint128':
        # struct doesn't support uint128 so write two 64bits
        # there are smarter ways but we really only need this for the fake timestamp
        FID.write(struct.pack('<QQ', 0, data))
    else:
        FID.write(struct.pack(formatChars[dataType], data))


def read_field(FID):
    fieldNameLen, dataSize = struct.unpack('<II', FID.read(8))
    fieldName = FID.read(fieldNameLen)[:-1]
    data = FID.read(dataSize)

    return fieldName, data


def pack_waveform(analog, marker1, marker2):
    """
    Helper function to convert a floating point analog channel and two logical marker channel to a sequence of 16bit integers.
    AWG 5000 series binary data format
    m2 m1 d14 d13 d12 d11 d10 d9 d8 d7 d6 d5 d4 d3 d2 d1
    16-bit format with markers occupying left 2 bits followed by the 14 bit
    analog channel value
    """

    # Convert decimal shape on [-1,1] to binary on [0,2^14 (16383)]
    # AWG actually makes 111,111,111,111,10 the 100% output, and
    # 111,111,111,111,11 is one step larger than 100% output so we
    # ignore the one extra positive number and scale from [0,16382]
    analog[analog > 1] = 1.0
    analog[analog < -1] = -1.0

    max_length = max(analog.size, marker1.size, marker2.size)

    if marker1.size < max_length:
        marker1 = np.append(marker1, np.zeros(max_length - marker1.size, dtype=np.bool))
    if marker2.size < max_length:
        marker2 = np.append(marker2, np.zeros(max_length - marker2.size, dtype=np.bool))
    if analog.size < max_length:
        analog = np.append(analog, np.zeros(max_length - analog.size, dtype=np.float64))

    bin_data = np.uint16(MAX_WAVEFORM_VALUE * analog + MAX_WAVEFORM_VALUE);
    bin_data += 2 ** 14 * np.uint16(marker1) + 2 ** 15 * np.uint16(marker2)

    return bin_data


def write_waveform(FID, WFname, WFnumber, data):
    """
    Helper function to write a waveform
    """
    numString = str(WFnumber)

    write_field(FID, 'WAVEFORM_NAME_' + numString, WFname, 'char')

    # Set integer format
    write_field(FID, 'WAVEFORM_TYPE_' + numString, 1, 'int16')

    write_field(FID, 'WAVEFORM_LENGTH_' + numString, data.size, 'int32')

    write_field(FID, 'WAVEFORM_TIMESTAMP_' + numString, 0, 'uint128')
    tmpString = 'WAVEFORM_DATA_' + numString + chr(0)
    dataSize = 2 * data.size
    FID.write(struct.pack('<II', len(tmpString), dataSize))
    FID.write(tmpString)
    FID.write(data.tostring())


def write_Tek5014_file(waveforms, markers, filename, seq_name, options=None, do_string_io=False):
    """
    Main function for writing a AWG format file.
    waveforms are (4,num_seqs,max_length) arrays of floats
    markers are (2*4,num_seqs,max_length) arrays of (0,1) values
    do_string_IO is for testing
    """

    # Set the default options
    # Marker levels default to 1V.
    if options is None:
        options = {'markerLevels': {}}
    for chanct in range(1, 5):
        for markerct in range(1, 3):
            tmpStr = 'ch{0}m{1}'.format(chanct, markerct)
            if tmpStr not in options['markerLevels']:
                options['markerLevels'][tmpStr] = {}
                options['markerLevels'][tmpStr]['low'] = 0.0
                options['markerLevels'][tmpStr]['high'] = 1.0

    num_seqs = max(len(waveforms[0]), len(waveforms[1]), len(waveforms[2]), len(waveforms[3]))

    # Open the file
    if do_string_io:
        FID = io.StringIO()
    else:
        FID = io.open(filename, 'wb')

    # Write the necessary MAGIC and VERSION fields
    write_field(FID, 'MAGIC', 5000, 'int16')
    write_field(FID, 'VERSION', 1, 'int16')

    # Default to the fastest sampling rate
    if 'clock_speed' not in options: options['clock_speed']=1.2e9
    write_field(FID, 'SAMPLING_RATE', options['clock_speed'], 'double')

    # Run mode (1 = continuous, 2 = triggered, 3 = gated, 4 = sequence)
    # If we only have one step then there is no sequence
    runMode = 2 if num_seqs == 1 else 4
    write_field(FID, 'RUN_MODE', runMode, 'int16')

    # Default to off state
    write_field(FID, 'RUN_STATE', 0, 'int16')

    # Set the reference source (1: internal; 2: external)
    write_field(FID, 'REFERENCE_SOURCE', 2, 'int16')

    # Trigger threshold
    write_field(FID, 'TRIGGER_INPUT_THRESHOLD', 1.0, 'double')

    # Marker's to high/low (1 = amp/offset, 2 = high/low)
    for chanct in range(1, 5):
        chanStr = str(chanct)
        write_field(FID, 'CHANNEL_STATE_' + chanStr, 1, 'int16')
        write_field(FID, 'MARKER1_METHOD_' + chanStr, 2, 'int16')
        write_field(FID, 'MARKER1_LOW_' + chanStr, options['markerLevels']['ch' + chanStr + 'm1']['low'], 'double')
        write_field(FID, 'MARKER1_HIGH_' + chanStr, options['markerLevels']['ch' + chanStr + 'm1']['high'], 'double')
        write_field(FID, 'MARKER2_METHOD_' + chanStr, 2, 'int16')
        write_field(FID, 'MARKER2_LOW_' + chanStr, options['markerLevels']['ch' + chanStr + 'm2']['low'], 'double')
        write_field(FID, 'MARKER2_HIGH_' + chanStr, options['markerLevels']['ch' + chanStr + 'm2']['high'], 'double')

    # If we have only one step then we specify the waveform names
    if num_seqs == 1:
        for chanct in range(1, 5):
            write_field(FID, 'OUTPUT_WAVEFORM_NAME_' + str(chanct), seq_name + 'Ch' + str(chanct) + '001', 'char')

    # Now write the waveforms (i.e. extract out the waveform data from the dictionaries)
    for seqct in range(num_seqs):
        # On the Tek, all four channels need to have the same length
        print("x", end=' ')
        for wfct in range(4):
            data = pack_waveform(waveforms[wfct][seqct], markers[2 * wfct][seqct], markers[2 * wfct + 1][seqct])
            write_waveform(FID, '{0}Ch{1}{2:03d}'.format(seq_name, wfct + 1, seqct + 1), 4 * seqct + 1 + wfct, data)

    # Write the sequence table
    for seqct in range(1, num_seqs + 1):
        ctStr = str(seqct)
        # We wait for a trigger at every sequence
        write_field(FID, 'SEQUENCE_WAIT_' + ctStr, 1, 'int16')
        write_field(FID, 'SEQUENCE_JUMP_' + ctStr, 0, 'int16')
        write_field(FID, 'SEQUENCE_LOOP_' + ctStr, 1, 'int32')

        # If we are on the final one then set the goto back to the beginning
        goto = 1 if seqct == num_seqs else 0
        write_field(FID, 'SEQUENCE_GOTO_' + ctStr, goto, 'int16')

        for chanct in range(1, 5):
            WFname = '{0}Ch{1}{2:03d}'.format(seq_name, chanct, seqct)
            write_field(FID, 'SEQUENCE_WAVEFORM_NAME_CH_' + str(chanct) + '_' + ctStr, WFname, 'char')

    if do_string_io:
        return FID.getvalue()
    FID.close()
    print("\nFinished writing sequence with %d steps to %s" % (num_seqs,filename))


#### Creating pattern files
class Tek5014Sequence:
    def __init__(self, waveform_length, sequence_length):
        self.waveforms = zeros((4, sequence_length, waveform_length))
        self.markers = zeros((4, 2, sequence_length, waveform_length))
        self.sequence_length = sequence_length
        self.waveform_length = waveform_length

    # this loads into the TEK
    def load_into_awg(self, filename, awg_name=None):

        # filename is where to build the sequence file

        # create AWG data file
        awgdata = dict()
        key_names = ['ch12', 'ch34']
        for i in range(1, 5):
            for j in range(1, 3):
                key_names.append('ch{0}m{1}'.format(i, j))

        for i in range(len(key_names)):
            awgdata[key_names[i]] = dict()
            awgdata[key_names[i]]['wfLib'] = dict()
            awgdata[key_names[i]]['linkList'] = list()



        # add wf's to awgdata

        # analog
        # This goes through all the waveforms and combines channels 1 and 2 and 3 and 4
        for i in range(2):
            data_key = 'ch{0}{1}'.format(2 * i + 1, 2 * i + 2)
            for j in range(self.sequence_length):
                awgdata[data_key]['wfLib'][str(j)] = self.waveforms[2 * i][j] + (self.waveforms[2 * i + 1][j]) * 1j

            #create sequence
            for j in range(self.sequence_length):
                awgdata[data_key]['linkList'].append(list())
                awgdata[data_key]['linkList'][j].append(namedtuple('a', 'key isTimeAmp length repeat'))
                awgdata[data_key]['linkList'][j][0].key = "{0:g}".format(
                    (j + 1) % self.sequence_length)  #go to next waveform (or beginning)
                awgdata[data_key]['linkList'][j][0].key = "{0:g}".format((j ) % self.sequence_length)  #go to next waveform (or beginning)
                awgdata[data_key]['linkList'][j][0].isTimeAmp = False

        for i in range(4):
            for j in range(2):
                marker_key = 'ch{0}m{1}'.format(i + 1, j + 1)
                for k in range(self.sequence_length):
                    awgdata[marker_key]['wfLib'][str(k)] = self.markers[i][j][k]

                #create sequence
                for k in range(self.sequence_length):
                    awgdata[marker_key]['linkList'].append(list())
                    awgdata[marker_key]['linkList'][k].append(namedtuple('a', 'key isTimeAmp length repeat'))
                    awgdata[marker_key]['linkList'][k][0].key = "{0:g}".format((k) % self.sequence_length)
                    awgdata[marker_key]['linkList'][k][0].isTimeAmp = False

        #lp.plot_y('debug',array(awgdata['ch12']['wfLib']['100'],dtype=float))
        write_Tek_file(awgdata, filename, 'seq1', None, False)

        #load the file into the TEK
        if awg_name is not None:
            im = InstrumentManager()
            awg = im[awg_name]
            awg.pre_load()
            awg.load_sequence_file(filename)


if __name__ == "__main__":
    awg = Tek5014(address='192.168.14.136')
    print(awg.get_id())