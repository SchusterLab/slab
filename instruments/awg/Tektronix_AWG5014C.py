import time
import types
import visa
import numpy as np
import hashlib
from visainstrument import VisaInstrument, Instrument
import logging
import copy

def chars_in_str(chars, s):
    for ch in chars:
        if ch in s:
            return True
    return False

class Tektronix_AWG5014C(VisaInstrument):

    def __init__(self, name, **kwargs):
        super(Tektronix_AWG5014C, self).__init__(name, term_chars='\n', **kwargs)
        self.set_timeout(60000)

        self.add_parameter('id', type=types.StringType,
            flags=Instrument.FLAG_GET)
        self.add_visa_parameter('clock',
            'SOUR:FREQ?', 'SOUR:FREQ %f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            minval=1e6, maxval=1.2e9, units='Hz')
        self.add_visa_parameter('refsrc',
            'SOUR:ROSC:SOUR?', 'SOUR:ROSC:SOUR %s',
            type=types.StringType,
            flags=Instrument.FLAG_GETSET,
            option_list=('INT', 'EXT'))
        self.add_visa_parameter('reffreq',
            'SOUR:ROSC:FREQ?', 'SOUR:ROSC:FREQ %f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            option_list=(10e6, 20e6, 100e6), units='Hz')
        self.add_visa_parameter('mode',
            'AWGC:RMOD?', 'AWGC:RMOD %s',
            type=types.StringType,
            option_list=(
                'CONT', 'TRIG', 'GAT', 'SEQ', 'ENH'
            ))
        self.add_visa_parameter('trig_impedance',
            'TRIG:IMP?', 'TRIG:IMP %f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            option_list=(1000, 50),
            units='Ohm')
        self.add_visa_parameter('trig_level',
            'TRIG:LEV?', 'TRIG:LEV %.03f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            units='V')
        self.add_visa_parameter('trig_slope',
            'TRIG:SLOP?', 'TRIG:SLOP %s',
            type=types.StringType,
            flags=Instrument.FLAG_GETSET,
            option_list=('POS', 'NEG'))

        # Channel options
        self.add_parameter('amplitude', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=0, maxval=4.5, units='V')
        self.add_parameter('offset', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-2, maxval=2, units='V')
        self.add_parameter('skew', type=types.IntType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-5000, maxval=5000, units='ps',
            gui_group='channels')
        self.add_parameter('m1_low', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-2, maxval=2, units='V',
            gui_group='channels')
        self.add_parameter('m1_high', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-2, maxval=2, units='V',
            gui_group='channels')
        self.add_parameter('m2_low', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-2, maxval=2, units='V',
            gui_group='channels')
        self.add_parameter('m2_high', type=types.FloatType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            minval=-2, maxval=2, units='V',
            gui_group='channels')
        self.add_parameter('output', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET,
            channels=(1, 4), channel_prefix='ch%d_',
            gui_group='channels')
        self.add_visa_parameter('error',
            'SYST:ERR?', '',
            type=types.StringType,
            flags=Instrument.FLAG_GET)

        self._loaded_waveforms = []

        self.add_function('all_on')
        self.add_function('save_settings')
        self.add_function('load_settings')
        self.add_function('output_zeros')
        self.add_function('sideband_modulate')
        self.add_function('output_sqwave')

        if kwargs.pop('reset', False):
            self.reset()
        else:
            self.get_all()
        self.set(kwargs)

    def do_get_id(self):
        ret = self.ask('*IDN?')
        return ret.replace('TEKTRONIX,', '')

    ###############################################
    # Status checks / controls
    ###############################################

    def wait_getID(self, delay=60000):
        timeout = self.get_timeout()
        self.set_timeout(delay)
        self.ask('*IDN?')
        self.set_timeout(timeout)
        return

    def wait_done(self, delay=60000):
        timeout = self.get_timeout()
        self.set_timeout(delay)
        self.do_get_output(1)
        self.set_timeout(timeout)
        return

    def get_runstate(self):
        val = self.ask('AWGC:RST?')
        return int(val)

    def wait_until_run(self):
        i = 0
        while i < 100:
            try:
                if self.get_runstate() == 1:
                    return True
            except:
                pass
            time.sleep(0.5)
            i += 1
        print 'Timed out waiting for AWG to enter run state'
        return False

    def load_settings(self, fn):
        self.write('AWGC:SLOAD "%s"' % fn)

    def save_settings(self, fn):
        self.write('AWGC:SSAVE "%s"' % fn)

    ###############################################
    # Sequence management
    ###############################################

    def delete_all_waveforms(self, wait=60000):
        self._loaded_waveforms = []
        self.write('WLIST:WAV:DEL ALL')
        if wait:
            self.wait_done(delay = wait)

    def run(self):
        self.write('AWGC:RUN')
        time.sleep(1)
        self.all_on()

    def stop(self):
        self.all_off()
        time.sleep(0.1)
        self.write('AWGC:STOP')

    ###############################################
    # Channel options
    ###############################################

    def do_get_output(self, channel):
        outs = self.ask('OUTP%d?'%channel)
        return int(float(outs))

    def do_set_output(self, enable, channel):
        cmd = 'OUTP%d %d' % (channel, enable)
        self.write(cmd)

    def all_on(self):
        for ch in (1, 2, 3, 4):
            self.set('ch%d_output'%ch, True)

    def all_off(self):
        for ch in (1, 2, 3, 4):
            self.set('ch%d_output'%ch, False)

    def do_set_amplitude(self, amp, channel):
        self.write('SOURCE%d:VOLT:IMM:AMPL %.05f' % (channel, amp))

    def do_get_amplitude(self, channel):
        val = self.ask('SOURCE%d:VOLT:AMPL?' % (channel,))
        return val

    def do_set_offset(self, ofs, channel):
        self.write('SOUR%d:VOLT:LEV:IMM:OFFS %.04f' % (channel, ofs))

    def do_get_offset(self, channel):
        val = self.ask('SOUR%d:VOLT:LEV:IMM:OFFS?' % (channel,))
        return val

    def do_set_skew(self, skew, channel):
        '''Set channel skew in ps.'''
        self.write('SOURCE%d:SKEW %dPS' % (channel, skew))

    def do_get_skew(self, channel):
        '''Get channel skew in ps.'''
        val = self.ask('SOURCE%d:SKEW?' % (channel,))
        return float(val) * 1e12

    def do_set_m1_low(self, val, channel):
        self.write('SOUR%d:MARKER1:VOLT:LEV:IMM:LOW %.04f' % (channel, val))

    def do_get_m1_low(self, channel):
        val = self.ask('SOUR%d:MARKER1:VOLT:LEV:IMM:LOW?' % (channel,))
        return val

    def do_set_m1_high(self, val, channel):
        self.write('SOUR%d:MARKER1:VOLT:LEV:IMM:HIGH %.04f' % (channel, val))

    def do_get_m1_high(self, channel):
        val = self.ask('SOUR%d:MARKER1:VOLT:LEV:IMM:HIGH?' % (channel,))
        return val

    def do_set_m2_low(self, val, channel):
        self.write('SOUR%d:MARKER2:VOLT:LEV:IMM:LOW %.04f' % (channel, val))

    def do_get_m2_low(self, channel):
        val = self.ask('SOUR%d:MARKER2:VOLT:LEV:IMM:LOW?' % (channel,))
        return val

    def do_set_m2_high(self, val, channel):
        self.write('SOUR%d:MARKER2:VOLT:LEV:IMM:HIGH %.04f' % (channel, val))

    def do_get_m2_high(self, channel):
        val = self.ask('SOUR%d:MARKER2:VOLT:LEV:IMM:HIGH?' % (channel,))
        return val

    ###############################################
    # Waveform loading functions
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
        bytemem = np.round(data * (2**13-2)) + (2**13-1)
        bytemem = bytemem.astype(np.uint16)

        if m1 is not None:
            if len(data) != len(m1):
                raise ValueError('Data and marker1 should have same length')
            bytemem |= 1<<14 * m1.astype(np.bool)
        if m2 is not None:
            if len(data) != len(m2):
                raise ValueError('Data and marker2 should have same length')
            bytemem |= 1<<15 * m2.astype(np.bool)

        return bytemem

    # add custom waveform as file, not correct
    def add_file(self, fn, data):
        bindata = self.get_bindata(data)
        cmd = ('MMEM:DATA "%s",#6%06d'%(fn,2*len(data))) + bindata.tostring() + '\n'
        self.write(cmd)

    def add_waveform(self, wname, data, m1=None, m2=None, replace=True, return_cmd=False):
        '''
        Add waveform <wname> to AWG with content <data> and marker content
        <m1> and <m2>.
        '''
        if not replace and wname in self._loaded_waveforms:
            return None
        logging.info('Adding waveform %s (%d bytes)', wname, len(data))
        self._loaded_waveforms.append(wname)

        bindata = self.get_bindata(data, m1, m2)
        cmd = 'WLIST:WAV:DEL "%s";' % wname
        cmd += ':WLIST:WAV:NEW "%s",%d,INT;' % (wname, len(data))
        cmd += ':WLIST:WAV:DATA "%s",0,%d,#6%06d' % (wname, len(bindata), 2*len(bindata))
        cmd += bindata.tostring() + '\n'
        logging.info(self.get_error())

        if return_cmd:
            return cmd

        cmd += ':OUTP?'
        self.ask(cmd)

    ###############################################
    # Sequence functions
    ###############################################

    def do_get_seq_pos(self):
        return int(self.ask('AWGC:SEQ:POS?'))

    def clear_sequence(self):
        '''
        Clear the sequence memory.
        '''
        self.write('SEQ:LENG 0\n')

    def setup_sequence(self, n_el, reset=True, loop=True):
        self.set_mode('SEQ')
        self.wait_done()
        if reset:
            self.write('SEQ:LENG 0\n')
            self.wait_done()
        self.write('SEQ:LENG %d\n' % n_el)
        self.wait_done()

        if loop:
            self.write('SEQ:ELEM%d:GOTO:STATE ON' % n_el)
            self.write('SEQ:ELEM%d:GOTO:INDEX 1' % n_el)

    def set_seq_element(self, ch, el, wname, repeat=1, trig=False):
        self.write('SEQ:ELEM%d:WAV%d "%s"' % (el, ch, wname))
        if repeat > 1:
            self.write('SEQ:ELEM%d:LOOP:COUNT %d' % (el, repeat))
        if trig:
            self.write('SEQ:ELEM%d:TWAIT 1' % (el,))

    ###############################################
    # Convenience functions to play simple waveforms
    ###############################################

    def play_waveforms(self, chan_wform, run=True):
        '''
        Play simple waveforms on channels.
        chan_wform is a list of <chan>, <waveform name> tuples.
        '''
        self.get_runstate()
        self.setup_sequence(1)
        for chan, wform in chan_wform:
            cmd = 'SEQ:ELEM1:WAV%d "%s"\n' % (chan, wform)
            self.write(cmd)
        if run:
            self.run()

    def sideband_modulate(self, period=None, freq=None, dphi=0, amp=1.0, chans=(1,2), run=True):
        '''
        Sideband modulate using period <period> or frequency <freq>.
        (period is negative for negative frequencies).

        The first channel will be setup as I (cosine) and the second as
        Q (sine). The phase adjustment <dphi> is added to the Q channel,
        resulting in outputs:

        chan[0] = cos(2pi*x/period)
        chan[1] = sin(2pi*x/period + dphi)
        '''

        if freq is not None:
            period = 1. / freq
        if np.abs(period - np.round(period)) < 1e-2:
            N = 1
            period = np.round(period)
        else:
            N = 20

        # Make sure we have a waveform longer than 250 ns
        if round(N*period) < 250:
            N *= np.ceil(250.0 / (N * period))
        npoints = round(N * period)
        print 'npoints: %s, period: %s, N: %s' % (npoints, period, N)

        xs = np.arange(npoints)
        phase = 2 * np.pi * xs / period
        self.add_waveform('cosine', amp*np.cos(phase), replace=True)
        self.add_waveform('sine', amp*np.sin(phase + dphi), replace=True)
        self.play_waveforms((
            (chans[0], 'cosine'),
            (chans[1], 'sine')), run=run)

    def output_zeros(self, chans=(1,2), run=True):
        self.add_waveform('zeros1', np.zeros([1,]))
        chan_wform = []
        for chan in chans:
            chan_wform.append((chan, 'zeros1'))
        self.play_waveforms(chan_wform, run=run)

    def output_sqwave(self, period, chans=(1,2), amp=0.1, dcycle=0.5, run=True):
        sqwave = np.zeros([period,])
        sqwave[:int(round(period*dcycle))] = amp
        name = 'sqwave%d'%period
        self.add_waveform(name, sqwave)
        chan_wform = []
        for chan in chans:
            chan_wform.append((chan, name))
        self.play_waveforms(chan_wform, run=run)

    ###############################################
    # Bulk loading routines
    ###############################################

    def bulk_waveform_load(self, wforms, maxlen=100000, replace=False):
        '''
        Bulk load of waveforms.
        <wforms> should be a list of (name, data, m1, m2) tuples.
        <maxlen> is the maximum command string length at which the command
        should be sent to the AWG.
        '''

        cmd = ''
        for (name, data, m1, m2) in wforms:
            cmd2 = self.add_waveform(name, data, m1, m2, replace=replace, return_cmd=True)
            if cmd2 is not None:
                cmd += ':' + cmd2 + ';'

            # Send command if longer than <maxlen>
            if len(cmd) > maxlen:
                cmd += ':OUTP?'
                self.ask(cmd[1:])
                cmd = ''

        # Send remaining command
        if len(cmd) > 0:
            cmd += ':OUTP?'
            ret = self.ask(cmd[1:])

    def bulk_sequence_load(self, chan, seq):
        '''
        Bulk load of sequence elements.
        <seq> should be a list of (index, waveform name, loop count, trigger).
        '''

        cmd = ''
        for idx, wname, repeat, trigger in seq:
            cmd += ':SEQ:ELEM%d:WAV%d "%s";' % (idx, chan, wname)
            if repeat > 1:
                cmd += ':SEQ:ELEM%d:LOOP:COUNT %d;' % (idx, repeat)
            if trigger:
                cmd += ':SEQ:ELEM%d:TWAIT 1;' % (idx,)

        if len(cmd) > 0:
            cmd += ':OUTP?'
            ret = self.ask(cmd[1:])


    def pull_dot_awg(self, path):
        '''
        Save the AWG state to a .awg file.
        '''
        self.write('AWGCONTROL:SSAVE "%s"\n' % path )

    def load_dot_awg(self, path):
        '''
        Load the AWG state from a .awg file.
        '''
        self.write('AWGCONTROL:SRESTORE "%s"\n' % path )

if __name__ == '__main__':
    with VisaInstrument.test(Tektronix_AWG5014C) as ins:
#        ins.sideband_modulate(20)
#        ins.output_sqwave(50)
        ins.close()
