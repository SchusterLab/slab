# -*- coding: utf-8 -*-
"""
Created on 5 Jul 2015

@author: Nelson Leung
"""

from slab.instruments import SocketInstrument


class M8195A(SocketInstrument):
    """Keysight M8195A Arbitrary Waveform Class"""
    # default_port=5025
    def __init__(self, name='M8195A', address='', enabled=True,timeout = 1000):
        address = address.upper()

        SocketInstrument.__init__(self, name, address, enabled, timeout)
        self._loaded_waveforms = []

    def get_id(self):
        return self.query("*IDN?")

    ## 6.9 TRIGger - Trigger Input

    def set_advancement_event_source(self,value):
        if value in ['TRIG','EVEN','INT']:
            self.write(':TRIG:SOUR:ADV %s' %value)
        else:
            raise Exception('M8195A: Invalid advancement event source')

    def get_advancement_event_source(self):
        return self.query(':TRIG:SOUR:ADV?')

    def send_trigger_enable_event(self):
        self.write(':TRIG:ENAB')

    def send_trigger_begin_event(self):
        self.write(':TRIG:BEG')

    def send_trigger_gate(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':TRIG:BEG:GATE ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':TRIG:BEG:GATE OFF')
        else:
            raise Exception('M8195A: Invalid trigger gate state')

    def get_trigger_gate(self):
        return self.query(':TRIG:BEG:GATE?')

    def send_trigger_advancement_event(self):
        self.write(':TRIG:ADV')

    ## 6.10 :FORMat Subsystem
    def set_byte_order(self,value):
        if value in ['NORM','SWAP']:
            self.write(':FORM:BORD %s' %value)
        else:
            raise Exception('M8195A: Invalid Byte Order')

    def get_byte_order(self):
        return self.query(':FORM:BORD?')

    ## 6.11 :INSTrument Subsystem
    def get_slot_number(self):
        return self.query(':INST:SLOT?')

    def flash_access_led(self, seconds):
        self.write(':INST:IDEN %d' %seconds)

    def stop_flash_access_led(self):
        self.write(':INST:IDEN:STOP')

    def get_hwardware_revision_number(self):
        return self.query(':INST:HWR?')

    def set_dac_mode(self,value):
        if value in ['SING','DUAL','FOUR','MARK','DCD','DCM']:
            self.write(':INST:DACM %s' %value)
        else:
            raise Exception('M8195A: Invalid DAC mode')

    def get_dac_mode(self):
        return self.query(':INST:DACM?')

    def set_dac_sample_rate_divider(self,value):
        if value in [1,2,4]:
            self.write(':INST:MEM:EXT:RDIV DIV%d' %value)
        else:
            raise Exception('M8195A: Invalid DAC sample rate divider')

    def get_dac_sample_rate_divider(self):
        return self.query(':INST:MEM:EXT:RDIV?')

    def get_multi_module_configuration(self):
        return self.query(':INST:MMOD:CONF?')

    def get_multi_module_mode(self):
        return self.query(':INST:MMOD:MODE?')

    ## 6.12 :MMEMory Subsystem

    def get_disk_usage_information(self, value):
        return self.query(':MMEM:CAT? %s' %value)

    def set_default_directory(self,value):
        self.write('MMEM:CDIR %s' %value)

    def get_default_directory(self):
        return self.query(':MMEM:CDIR?')

    def file_copy(self,file,new_file):
        self.write(':MMEM:COPY %s, %s' %(file,new_file))

    def file_delete(self,value):
        self.write(':MMEM:DEL %s' %(value))

    def set_file_data(self,file,data):
        # <data> is in 488.2 block format
        self.write(':MMEM:DATA %s, %s' %(file,data))

    def get_file_data(self,file):
        return self.query(':MMEM:DATA? %s' %file)

    def create_directory(self,value):
        self.write(':MMEM:MDIR %s' %value)

    def move_path(self,old_path,new_path):
        self.write(':MMEM:MOVE %s, %s' %(old_path,new_path))

    def remove_directory(self,value):
        self.write(':MMEM:RDIR %s' %value)

    def load_state_from_file(self,value):
        self.write(':MMEM:LOAD:CST %s' %value)

    def store_state_to_file(self,value):
        self.write(':MMEM:STOR:CST %s' %value)

    ## 6.13 Output subsystem

    def set_enabled(self, channel, state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':OUTP%d ON' % channel)
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':OUTP%d OFF' % channel)
        else:
            raise Exception('M8195A: Invalid enabled state')

    def get_enabled(self, channel):
        return self.query(':OUTP%d?' % (channel))

    def set_output_clock_source(self,value):
        if value in ['INT','EXT','SCLK1','SCLK2']:
            self.write(':OUTP:ROSC:SOUR %s' %value)
        else:
            raise Exception('M8195A: Invalid reference source')

    def get_output_clock_source(self):
        return self.query(':OUTP:ROSC:SOUR?')

    def set_sample_clock_divider(self,value):
        self.write(':OUTP:ROSC:SCD %d' %value)

    def get_sample_clock_divider(self):
        return self.query('OUTP:ROSC:SCD?')

    def set_reference_clock_divider_1(self,value):
        self.write(':OUTP:ROSC:RCD1 %d' %value)

    def get_freference_clock_divider_1(self):
        return self.query(':OUTP:ROSC:RCD1?')

    def set_reference_clock_divider_2(self,value):
        self.write(':OUTP:ROSC:RCD2 %d' %value)

    def get_freference_clock_divider_2(self):
        return self.query(':OUTP:ROSC:RCD2?')

    def set_differential_offset(self,channel,value):
        self.write(':OUTP%d:DIOF %f' %(channel,value))

    def get_differential_offset(self,channel):
        return self.query(':OUTP%d:DIOF?' %channel)

    def rate_divider_codename(self, rate_divider):
        if rate_divider == 1:
            codename = 'FRAT'
        elif rate_divider == 2:
            codename = 'HRAT'
        elif rate_divider == 4:
            codename = 'QRAT'
        else:
            raise Exception('M8195A: Invalid rate divider')
        return codename

    def set_fir_coefficients(self,channel,rate_divider,value):
        # value is comma-separated values

        codename = self.rate_divider_codename(rate_divider)

        self.write(':OUTP%d:FILT:%s %s' %(channel,codename,value))

    def get_fir_coefficients(self,channel,rate_divider):
        # value is comma-separated values

        codename = self.rate_divider_codename(rate_divider)

        return self.query(':OUTP%d:FILT:%s?' %(channel,codename))

    def set_fir_type(self,channel,rate_divider,value):
        codename = self.rate_divider_codename(rate_divider)

        if rate_divider == 1:
            if value in ['LOWP','ZOH','USER']:
                self.write(':OUTP%d:FILT:%s:TYPE %s' %(channel,codename,value))
            else:
                raise Exception('M8195A: Invalid FIR type')
        elif rate_divider == 2 or rate_divider == 4:
            if value in ['NYQ','LIN','ZOH','USER']:
                self.write(':OUTP%d:FILT:%s:TYPE %s' %(channel,codename,value))
            else:
                raise Exception('M8195A: Invalid FIR type')

    def get_fir_type(self,channel,rate_divider):

        codename = self.rate_divider_codename(rate_divider)

        return self.query(':OUTP%d:FILT:%s:TYPE?' %(channel,codename))

    def set_fir_scale(self,channel,rate_divider,value):
        codename = self.rate_divider_codename(rate_divider)

        self.write(':OUTP%d:FILT:%s:SCAL %s' %(channel,codename,value))

    def get_fir_scale(self,channel,rate_divider):

        codename = self.rate_divider_codename(rate_divider)

        return self.query(':OUTP%d:FILT:%s:SCAL?' %(channel,codename))

    def set_fir_delay(self,channel,rate_divider,value):
        codename = self.rate_divider_codename(rate_divider)

        if rate_divider == 1:
            if abs(value) > 50:
                raise Exception('M8195A: Invalid FIR delay')
        elif rate_divider == 2:
            if abs(value) > 100:
                raise Exception('M8195A: Invalid FIR delay')
        elif rate_divider == 4:
            if abs(value) > 200:
                raise Exception('M8195A: Invalid FIR delay')

        self.write(':OUTP%d:FILT:%s:DEL %fps' %(channel,codename,value))

    def get_fir_delay(self,channel,rate_divider):

        codename = self.rate_divider_codename(rate_divider)

        return self.query(':OUTP%d:FILT:%s:DEL?' %(channel,codename))

    ## 6.14 Sampling Frequency Commands

    def set_sample_frequency(self,value):
        self.write(':FREQ:RAST %f' %value)

    def get_sample_frequency(self):
        return self.query(':FREQ:RAST?')

    ## 6.15 Reference Oscillator Commands

    def set_reference_source(self,value):
        if value in ['EXT','AXI','INT']:
            self.write(':ROSC:SOUR %s' %value)
        else:
            raise Exception('M8195A: Invalid reference source')

    def get_reference_source(self):
        return self.query(':ROSC:SOUR?')

    def get_reference_source_availability(self,value):
        if value in ['EXT','AXI','INT']:
            return self.query(':ROSC:SOUR:CHEC? ' %value)
        else:
            raise Exception('M8195A: Invalid reference source')

    def set_reference_clock_frequency(self,value):
        if self.get_reference_source() == 'EXT':
            self.write(':ROSC:FREQ %f' %value)
        else:
            raise Exception('M8195A: Not in external reference source')

    def get_reference_clock_frequency(self):
        if self.get_reference_source() == 'EXT':
            return self.query(':ROSC:FREQ?')
        else:
            raise Exception('M8195A: Not in external reference source')

    def set_reference_clock_range(self,value):
        if self.get_reference_source() == 'EXT':
            if value in ['RANG1','RANG2']:
                self.write(':ROSC:RANG %s' %value)
            else:
                raise Exception('M8195A: Not in valid reference source frequency range')
        else:
            raise Exception('M8195A: Not in external reference source')

    def get_reference_clock_range(self):
        if self.get_reference_source() == 'EXT':
            return self.query(':ROSC:RANG?')
        else:
            raise Exception('M8195A: Not in external reference source')

    def set_reference_clock_range_frequency(self,range,value):
        if self.get_reference_source() == 'EXT':
            if range in ['RNG1','RNG2']:
                self.write(':ROSC:%s:FREQ %f' %(range,value))
            else:
                raise Exception('M8195A: Not in valid reference source frequency range')
        else:
            raise Exception('M8195A: Not in external reference source')

    def get_reference_clock_range_frequency(self,range):
        if self.get_reference_source() == 'EXT':
            if range in ['RNG1','RNG2']:
                return self.query(':ROSC:%s:FREQ?' %range)
        else:
            raise Exception('M8195A: Not in external reference source')


    ## 6.16 :VOLTage Subsystem

    def set_amplitude(self,channel,value):
        self.write(':VOLT%d %f' % (channel, value))

    def get_amplitude(self,channel):
        return self.query(':VOLT%d?' % (channel))

    def set_analog_high(self, channel, value):
        self.write(':VOLT%d:HIGH %f' % (channel, value))

    def get_analog_high(self, channel):
        return float(self.query(':VOLT%d:HIGH?' % (channel)))

    def set_analog_low(self, channel, value):
        self.write(':VOLT%d:LOW %f' % (channel, value))

    def get_analog_low(self, channel):
        return float(self.query(':VOLT%d:LOW?' % (channel)))

    def set_offset(self,channel,value):
        self.write(':VOLT%d:OFFS %f' % (channel, value))

    def get_offset(self,channel):
        return self.query(':VOLT%d:OFFS?' % (channel))

    def set_termination(self,channel,value):
        self.write(':VOLT%d:TERM %f' % (channel, value))

    def get_termination(self,channel):
        return self.query(':VOLT%d:TERM?' % (channel))

    ## 6.17 Function mode setting
    def set_mode(self,mode):
        if mode in ['ARB', 'STS','STSC']:
            self.write(':FUNC:MODE %s' %mode)
        else:
            raise Exception('M8195A: Invalid enabled mode')

    def get_mode(self):
        return self.query(':FUNC:MODE?')
