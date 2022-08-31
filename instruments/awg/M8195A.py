# -*- coding: utf-8 -*-
"""
Created on 5 Jul 2015

@author: Nelson Leung
"""

from slab.instruments import SocketInstrument

import numpy as np
import sys
from tqdm import tqdm
import os


class M8195A(SocketInstrument):
    """Keysight M8195A Arbitrary Waveform Class"""
    # default_port=5025
    def __init__(self, name='M8195A', address='', enabled=True, timeout = 1000):
        address = address.upper()

        SocketInstrument.__init__(self, name, address, enabled=enabled, timeout=timeout)
        self._loaded_waveforms = []
        self.socket.setblocking(1)

    ## 6.5 System Related Commands

    def set_event_in_mode(self,value):
        if value in ['EIN','TOUT']:
            self.write(':SYST:EIN:MODE %s' %value)
        else:
            raise Exception('M8195A: Invalid event in mode')

    def get_event_in_mode(self):
        return self.query(':SYST:EIN:MODE?')

    def get_error(self):
        return self.query(':SYST:ERR?')

    def get_all_scpi_commands(self):
        return self.query(':SYST:HELP:HEAD?')

    def get_licenses_installed(self):
        return self.query(':SYST:LIC:EXT:LIST?')

    def set_system_setup(self,value):
        self.write(':SYST:SET %s' %value)

    def get_system_setup(self):
        return self.query(':SYST:SET?')

    def get_system_version(self):
        return self.query(':SYST:VERS?')

    def get_system_communication_availability(self):
        return self.query(':SYST:COMM:*?')

    def get_vxi_11_instrument_number(self):
        return self.query(':SYST:COMM:INST?')

    def get_hislip_number(self):
        return self.query(':SYST:COMM:HISL?')

    def get_socket_port(self):
        return self.query(':SYST:COMM:SOCK?')

    def get_telnet_port(self):
        return self.query(':SYST:COMM:TELN?')

    def get_tcp_port(self):
        return self.query(':SYST:COMM:TCP:CONT?')

    ## 6.6 Common Command List
    def get_id(self):
        return self.query("*IDN?")

    def clear_event_register(self):
        self.write('*CLS')

    def set_enable_bits_status_register(self,value):
        self.write('*ESE %d' %value)

    def get_enable_bits_status_register(self):
        return self.query('*ESE?')

    def get_event_status_register(self):
        return self.query('ESR?')

    def set_operation_complete(self):
        self.write('*OPC')

    def get_operation_complete(self):
        return self.query('*OPC?')

    def get_installed_options(self):
        return self.query('*OPT?')

    def set_factory_default(self):
        self.write('*RST')

    def set_enable_bits_service_request(self,value):
        self.write('*SRE %d' %value)

    def get_enable_bits_service_request(self):
        return self.query('*SRE?')

    def get_summary_register(self):
        return self.query('*STB?')

    def execute_self_test(self):
        return self.query('*TST?')

    def get_instrument_data(self):
        return self.query('*LRN?')

    def wait_current_execution(self):
        return self.query('*WAI?')

    ## 6.8 :ARM/TRIGger Subsystem
    def stop_output(self):
        self.write(':ABOR')

    def set_module_delay(self,seconds):
        self.write(':ARM:MDEL %f' %seconds)

    def get_module_delay(self):
        return self.query(':ARM:MDEL?')

    def set_sample_delay(self,value):
        self.write(':ARM:SDEL %d' %value)

    def get_sample_delay(self):
        return self.query(':ARM:SDEL?')

    def set_arming_mode(self, value):
        if value in ['SELF','ARM']:
            self.write(':INIT:CONT:ENAB %s' %value)
        else:
            raise Exception('M8195A: Invalid arming mode')

    def get_arming_mode(self):
        return self.query(':INIT:CONT:ENAB?')

    def set_continuous_mode(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':INIT:CONT:STAT ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':INIT:CONT:STAT OFF')
        else:
            raise Exception('M8195A: Invalid continuous mode command')

    def get_continuous_mode(self):
        return self.query(':INIT:CONT:STAT?')

    def set_gate_mode(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':INIT:GATE:STAT ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':INIT:GATE:STAT OFF')
        else:
            raise Exception('M8195A: Invalid continuous mode command')

    def get_gate_mode(self):
        return self.query(':INIT:GATE:STAT?')

    def start_all_output(self):
        self.write(':INIT:IMM')

    def set_trigger_level(self,value):
        self.write(':ARM:TRIG:LEV %f' %value)

    def get_trigger_level(self):
        return self.query(':ARM:TRIG:LEV?')

    def set_trigger_input_slope(self,value):
        if value in ['POS','NEG','EITH']:
            self.write(':ARM:TRIG:SLOP %s' %value)
        else:
            raise Exception('M8195A: Invalid trigger slope')

    def get_trigger_input_slope(self):
        return self.query(':ARM:TRIP:SLOP?')

    def set_trigger_source(self,value):
        if value in ['TRIG','EVEN','INT']:
            self.write(':ARM:TRIG:SOUR %s' %value)
        else:
            raise Exception('M8195A: Invalid trigger source')

    def get_trigger_source(self):
        return self.query(':ARM:TRIG:SOUR?')

    def set_internal_trigger_frequency(self,value):
        self.write(':ARM:TRIG:FREQ %f' %value)

    def get_internal_trigger_frequency(self):
        return self.query(':ARM:TRIG:FREQ?')

    def set_trigger_operation_mode(self,value):
        if value in ['ASYN','SYNC']:
            self.write(':ARM:TRIG:OPER %s' %value)
        else:
            raise Exception('M8195A: Invalid trigger operation mode')

    def get_trigger_operation_mode(self):
        return self.query(':ARM:TRIG:OPER?')

    def set_event_level(self,value):
        self.write(':ARM:EVEN:LEV %f' %value)

    def get_event_level(self):
        return self.query(':ARM:EVEN:LEV?')

    def set_event_input_slope(self,value):
        if value in ['POS','NEG','EITH']:
            self.write(':ARM:EVEN:SLOP %s' %value)
        else:
            raise Exception('M8195A: Invalid trigger slope')

    def get_event_input_slope(self):
        return self.query(':ARM:EVEN:SLOP?')

    def set_enable_event_source(self,value):
        if value in ['TRIG','EVEN']:
            self.write(':TRIG:SOUR:ENAB %s' %value)
        else:
            raise Exception('M8195A: Invalid trigger source')

    def get_enable_event_source(self):
        return self.query(':TRIG:SOUR:ENAB?')

    def set_enable_hardware_input_disable_state(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':TRIG:ENAB:HWD ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':TRIG:ENAB:HWD OFF')
        else:
            raise Exception('M8195A: Invalid continuous mode command')

    def get_enable_hardware_input_disable_state(self):
        return self.query(':TRIG:ENAB:HWD?')

    def set_trigger_hardware_input_disable_state(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':TRIG:BEG:HWD ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':TRIG:BEG:HWD OFF')
        else:
            raise Exception('M8195A: Invalid continuous mode command')

    def get_trigger_hardware_input_disable_state(self):
        return self.query(':TRIG:BEG:HWD?')

    def set_advancement_hardware_input_disable_state(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':TRIG:ADV:HWD ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':TRIG:ADV:HWD OFF')
        else:
            raise Exception('M8195A: Invalid continuous mode command')

    def get_advancement_hardware_input_disable_state(self):
        return self.query(':TRIG:ADV:HWD?')

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

    def set_fir_delay(self,channel,rate_divider,ps):
        codename = self.rate_divider_codename(rate_divider)

        if rate_divider == 1:
            if abs(ps) > 50:
                raise Exception('M8195A: Invalid FIR delay')
        elif rate_divider == 2:
            if abs(ps) > 100:
                raise Exception('M8195A: Invalid FIR delay')
        elif rate_divider == 4:
            if abs(ps) > 200:
                raise Exception('M8195A: Invalid FIR delay')

        self.write(':OUTP%d:FILT:%s:DEL %fps' %(channel,codename,ps))

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
        self.write(':ROSC:FREQ %f' %value)

    def get_reference_clock_frequency(self):
        return self.query(':ROSC:FREQ?')

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

    ## 6.18 :STABle Subsystem
    def reset_sequence(self):
        self.write(':STAB:RES')

    def write_sequence_data(self,sequence_table_index,segment_id,segment_advancement_mode='SING',sequence_advancement_mode='SING',start=False,end=False,sequence_loop=1,segment_loop=1,start_address='0',end_address='4294967295'):
        control = 0

        if start == True:
            control += 2**28

        if end == True:
            control += 2**30

        if sequence_advancement_mode in ['AUTO', 'COND', 'REP', 'SING']:
            if sequence_advancement_mode == 'AUTO':
                control = control
            elif sequence_advancement_mode == 'COND':
                control += 1*(2**20)
            elif sequence_advancement_mode == 'REP':
                control += 2*(2**20)
            elif sequence_advancement_mode == 'SING':
                control += 3*(2**20)
        else:
            raise Exception('M8195A: Invalid sequence advancement mode')

        if segment_advancement_mode in ['AUTO', 'COND', 'REP', 'SING']:
            if segment_advancement_mode == 'AUTO':
                control = control
            elif segment_advancement_mode == 'COND':
                control += 1*(2**16)
            elif segment_advancement_mode == 'REP':
                control += 2*(2**16)
            elif segment_advancement_mode == 'SING':
                control += 3*(2**16)
        else:
            raise Exception('M8195A: Invalid segment advancement mode')

        self.write(':STAB:DATA %d, %d, %d, %d, %d, %s, %s' %(sequence_table_index,control,sequence_loop,segment_loop,segment_id,start_address,end_address))

    def write_sequence_idle(self,sequence_table_index,idle_delay,sequence_loop=1,idle_sample='0'):
        self.write(':STAB:DATA %d, 2147483648,%d,0,%s,%f,0' %(sequence_table_index,sequence_loop,idle_sample,idle_delay))

    def read_sequence_data(self,sequence_table_index,length):
        return self.query(':STAB:DATA? %d, %d' %(sequence_table_index,length))

    def read_sequence_data_block(self,sequence_table_index,length):
        return self.query(':STAB:DATA:BLOC? %d, %d' %(sequence_table_index,length))

    def set_sequence_starting_id(self,sequence_table_index):
        self.write(':STAB:SEQ:SEL %d' %sequence_table_index)

    def get_sequence_starting_id(self):
        return self.query(':STAB:SEQ:SEL?')

    def get_sequence_execution_state(self):
        return self.query(':STAB:SEQ:STAT?')

    def set_dynamic_mode(self,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':STAB:DYN ON')
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':STAB:DYN OFF')
        else:
            raise Exception('M8195A: Invalid dynamic mode command')

    def get_dynamic_mode(self):
        return self.query(':STAB:DYN?')

    def set_dynamic_starting_id(self,sequence_table_index):
        self.write(':STAB:DYN:SEL %d' %sequence_table_index)

    def set_scenario_starting_id(self,sequence_table_index):
        self.write(':STAB:SCEN:SEL %d' %sequence_table_index)

    def get_scenario_starting_id(self):
        return self.query(':STAB:SCEN:SEL?')

    def set_scenario_advancement_mode(self,value):
        if value in ['AUTO','COND','REP','SING']:
            self.write(':STAB:SCEN:ADV %s' %value)
        else:
            raise Exception('M8195A: Invalid scenario advancement mode')

    def get_scenario_advancement_mode(self):
        return self.query(':STAB:SCEN:ADV?')

    def set_scenario_loop(self,value):
        self.write(':STAB:SCEN:COUN %d' %value)

    def get_scenario_loop(self):
        return self.query(':STAB:SCEN:COUN?')

    ## 6.19 Frequency and Phase Response Data Access

    def get_frequency_phase_response_data(self,channel):
        return self.query(':CHAR%d?' %channel)

    ## 6.20 :TRACe Subsystem

    def set_waveform_sample_source(self,channel,value):
        if value in ['INT','EXT']:
            self.write(':TRAC%d:MMOD %s' %(channel,value))
        else:
            raise Exception('M8195A: Invalid waveform sample source')

    def get_waveform_sample_source(self,channel):
        return self.query('TRAC%d:MMOD?' %channel)

    def set_segment_size(self,channel,segment_id,length,init_value=0,write_only = False):
        if write_only:
            self.write(':TRAC%d:DEF:WONL %d,%d,%f' %(channel,segment_id,length,init_value))
        else:
            self.write(':TRAC%d:DEF %d,%d,%f' %(channel,segment_id,length,init_value))

    def set_new_segment_size(self,channel,length,init_value=0, write_only = False):
        if write_only:
            return self.query(':TRAC%d:DEF:WONL:NEW? %d,%f' %(channel,length,init_value))
        else:
            return self.query(':TRAC%d:DEF:NEW? %d,%f' %(channel,length,init_value))

    def set_segment_data(self,channel,segment_id,offset,data):
        #data in comma-separated list
        self.write(':TRAC%d:DATA %d,%d,%s' %(channel,segment_id,offset,data))

    def get_segment_data(self,channel,segment_id,offset,length):
        return self.query(':TRAC%d:DATA? %d,%d,%d' %(channel,segment_id,offset,length))

    def set_segment_data_from_file(self,channel,segment_id,file_name,data_type,marker_flag,padding,init_value,ignore_header_parameters):
        self.write(':TRAC%d:IMP %d,%s,%s,%s,%s,%d,%s' %(channel,segment_id,file_name,data_type,marker_flag,padding,init_value,ignore_header_parameters))

    def set_segment_data_from_bin_file(self,channel,segment_id,file_name):
        self.write(':TRAC%d:IMP %d,%s,%s, IONLY, ON, ALEN' %(channel,segment_id,file_name,'BIN8'))

    def delete_segment(self,channel,segment_id):
        self.write(':TRAC%d:DEL %d' %(channel,segment_id))

    def delete_all_segment(self,channel):
        self.write(':TRAC%d:DEL:ALL' %channel)

    def get_segment_catalog(self,channel):
        return self.query(':TRAC%d:CAT?' %channel)

    def get_memory_space_amount(self,channel):
        return self.query(':TRAC%d:FREE?' %channel)

    def set_segment_name(self,channel,segment_id,name):
        self.write(':TRAC%d:NAME %d, %s' %(channel,segment_id,name))

    def get_segment_name(self,channel,segment_id):
        return self.query(':TRAC%d:NAME? %d' %(channel,segment_id))

    def set_segment_comment(self,channel,segment_id,comment):
        self.write(':TRAC%d:COMM %d, %s' %(channel,segment_id,comment))

    def get_segment_comment(self,channel,segment_id):
        return self.query(':TRAC%d:COMM? %d' %(channel,segment_id))

    def set_select_segment(self,channel,segment_id):
        self.write(':TRAC%d:SEL %d' %(channel,segment_id))

    def get_select_segment(self,channel):
        return self.query(':TRAC%d:SEL?' %channel)

    def set_selected_segment_advancement_mode(self,channel,value):
        if value in ['AUTO','COND','REP','SING']:
            self.write(':TRAC%d:ADV %s' %(channel,value))
        else:
            raise Exception('M8195A: Invalid segment advancement mode')

    def get_selected_segment_advancement_mode(self,channel):
        return self.query(':TRAC%d:ADV?' %channel)

    def set_selected_segment_loop(self,channel,value):
        self.write(':TRAC%d:COUN %d' %(channel,value))

    def get_selected_segment_loop(self,channel):
        return self.query(':TRAC%d:COUN?' %channel)

    def set_selected_segment_marker_enable(self,channel,state):
        if state in ['on', 'ON', True, 1, '1']:
            self.write(':TRAC:MARK%d ON' %channel)
        elif state in ['off', 'OFF', False, 0, '0']:
            self.write(':TRAC:MARK%d OFF' %channel)
        else:
            raise Exception('M8195A: Invalid selected segment marker enable command')

    def get_selected_segment_marker_enable(self,channel):
        return self.query(':TRAC%d:MARK?' %channel)

    ## 6.21 :TEST Subsystem
    def get_power_on_self_tests_result(self):
        return self.query(':TEST:PON?')

    def get_power_on_self_tests_results_with_test_message(self):
        return self.query(':TEST:TST?')


    ## Setup AWG

    def setup_awg(m8195a,num_channels,amplitudes=[1.,1.,1.,1.]):

        m8195a.stop_output()
        m8195a.set_factory_default()
        # num_channels = 1  # currently hard coded for testing stuff with Sasha
        if num_channels == 1:
            m8195a.set_dac_mode('SING')

        elif num_channels == 2:
            m8195a.set_dac_mode('DUAL')

        elif num_channels == 4:
            m8195a.set_dac_mode('FOUR')

        m8195a.set_dac_sample_rate_divider(num_channels)

        for ii in range(1,num_channels+1):
        # for ii in range(3,4):  # changed to test stuff with Sasha
            m8195a.set_waveform_sample_source(ii,'EXT')
            m8195a.set_amplitude(ii,amplitudes[ii-1])


        m8195a.set_reference_clock_frequency(10e6)
        m8195a.set_reference_source('EXT')


    def define_segments(m8195a,waveform_matrix):

        waveform_shape = waveform_matrix.shape

        num_channels = waveform_shape[0]
        sequence_length = waveform_shape[1]
        segment_length = waveform_shape[2]

        for sequence_id in range(1,sequence_length+1):
            sys.stdout.write('x')

            m8195a.set_segment_size(1,sequence_id,segment_length)

            for channel in range(1,num_channels+1):

                segment_data_array = 127*waveform_matrix[channel-1][sequence_id-1]
                segment_data_csv = ','.join(['%d' %num for num in segment_data_array])
                m8195a.set_segment_data(channel,sequence_id,0,segment_data_csv)

        print('\n')


    def define_segments_binary(m8195a,waveform_matrix, path):

        # waveform_shape = waveform_matrix.shape
        #
        # num_channels = waveform_shape[0]
        # sequence_length = waveform_shape[1]
        # segment_length = waveform_shape[2]

        #waveform_shape = waveform_matrix.shape

        num_channels = len(waveform_matrix)#[0]
        sequence_length = waveform_matrix[0].shape[0]
        segment_length = waveform_matrix[0].shape[1]

        sys.stdout.write('Writing and uploading M8195A sequences...')
        for sequence_id in tqdm(list(range(1,sequence_length+1))):

            m8195a.set_segment_size(1,sequence_id,segment_length)

            for channel in range(1,num_channels+1):

                #sys.stdout.write('writing seq id=' + str(sequence_id) + '..')

                # todo:
                # hack: M8195A sequence start from second sequence - '(sequence_id-2)'
                segment_data_array = 127*waveform_matrix[channel-1][(sequence_id-2)%sequence_length]
                # filename = r'S:\_Data\160714 - M8195A Test\sequences\m8195a_%d_%d.bin8' %(sequence_id,channel)
                filename = os.path.join(path, r'sequences\m8195a_%d_%d.bin8' %(sequence_id,channel))
                # filename = r'C:\M8195_sequences\m8195a_%d_%d.bin8' % (sequence_id, channel)
                with open(filename, 'wb')  as f:
                    segment_data_array.astype('int8').tofile(f)

                # name = r'C:\slab_data_temp\m8195test\m8195a_ramsey_long_%d_%d.npy' % (sequence_id, channel)
                # np.save(name, segment_data_array)

                #sys.stdout.write('uploading..\n')
                # filename = '\"' + r'S:\_Data\160714 - M8195A Test\sequences\m8195a_%d_%d.bin8' %(sequence_id,channel) + '\"'
                # filename = '\"' + r'\\THORIUM-PC\M8195_sequences\m8195a_%d_%d.bin8' %(sequence_id,channel) + '\"'
                filename = '\"' + os.path.join(path, r'sequences\m8195a_%d_%d.bin8' %(sequence_id,channel)) + '\"'
                #m8195a.set_segment_data_from_bin_file(channel,sequence_id,filename)

                m8195a.set_segment_data_from_bin_file(channel, sequence_id, filename)


    def define_segments_test(m8195a,segment_length,sequence_length,dt):

        time_array = np.arange(0,segment_length)*dt

        for ii in range(1,sequence_length+1):
            sys.stdout.write('x')

            m8195a.set_segment_size(1,ii,segment_length)

            freq1 = 0.05 #GHz
            freq3 = 0.05  #GHz

            segment_data_array = 127*np.cos(2*np.pi*freq1*time_array)
            segment_data_csv = ','.join(['%d' %num for num in segment_data_array])
            m8195a.set_segment_data(1,ii,0,segment_data_csv)

            segment_data_array = (ii/float(sequence_length)) * 127*np.cos(2*np.pi*freq3*time_array)
            segment_data_csv = ','.join(['%d' %num for num in segment_data_array])
            m8195a.set_segment_data(3,ii,0,segment_data_csv)

        print('\n')


    def define_sequence(m8195a,sequence_length):

        m8195a.write_sequence_data(0,1,start=True)
        for ii in range(2,sequence_length):
            m8195a.write_sequence_data(ii-1,ii)
        m8195a.write_sequence_data(sequence_length-1,sequence_length,end=True)


    def start_output(m8195a):

        for ii in [1,2,3,4]:
            m8195a.set_enabled(ii, True)
        m8195a.start_all_output()

    def get_sample_sequence(m8195a,num_channels,segment_length,sequence_length,dt):

        time_array = np.arange(0,segment_length)*dt
        freq = 0.05 #GHz

        waveform_channel_list = []

        for channels in range(1,num_channels+1):
            waveform_sequence_list = []
            for sequence_id in range(sequence_length):
                if channels == 1:
                    waveform_sequence_list.append(np.cos(2*np.pi*freq*time_array))
                else:
                    waveform_sequence_list.append((sequence_id/float(sequence_length))*np.cos(2*np.pi*freq*time_array))
            waveform_sequence_array = np.array(waveform_sequence_list)
            waveform_channel_list.append(waveform_sequence_array)

        waveform_channel_array = np.array(waveform_channel_list)

        return waveform_channel_array


def upload_M8195A_sequence(m8195a, waveform_matrix, awg, path):

    amplitudes = awg['amplitudes']

    # m8195a = M8195A(address ='192.168.14.234:5025')
    # m8195a.socket.setblocking(1)

    # waveform_shape = waveform_matrix.shape
    #
    # num_channels = waveform_shape[0]
    # sequence_length = waveform_shape[1]
    # segment_length = waveform_shape[2]

    num_channels = len(waveform_matrix)  # [0]
    sequence_length = waveform_matrix[0].shape[0]
    segment_length = waveform_matrix[0].shape[1]

    m8195a.setup_awg(num_channels=num_channels,amplitudes=amplitudes)

    dt = float(num_channels)/64. #ns

    # waveform_matrix = get_sample_sequence(4,segment_length,sequence_length,dt)

    # define_segments_test(m8195a,segment_length,sequence_length,dt)

    # define_segments(m8195a,waveform_matrix)

    m8195a.define_segments_binary(waveform_matrix, path)

    m8195a.set_mode('STS')
    m8195a.define_sequence(sequence_length)

    m8195a.set_advancement_event_source('TRIG')
    m8195a.set_sequence_starting_id(0)

    # m8195a.set_internal_trigger_frequency(1./period)

    # confirm upload complete by waiting for response
    print('Wait for upload to finish...')
    print(m8195a.get_id())
    print('Upload to M8195A finished.')

    m8195a.start_output()


if __name__ == "__main__":
    m8195a = M8195A(address ='192.168.14.234:5025')
    m8195a.socket.setblocking(1)

    num_channels = 4

    m8195a.setup_awg(num_channels=num_channels)



    segment_length = 25600
    sequence_length = 5

    dt = float(num_channels)/64. #ns

    waveform_matrix = m8195a.get_sample_sequence(4,segment_length,sequence_length,dt)

    period = 1./50 #s

    # define_segments_test(m8195a,segment_length,sequence_length,dt)

    m8195a.define_segments_binary(waveform_matrix)

    # define_segments_binary(m8195a,waveform_matrix)

    m8195a.set_mode('STS')
    m8195a.define_sequence(sequence_length)

    m8195a.set_advancement_event_source('INT')
    m8195a.set_sequence_starting_id(0)

    m8195a.set_internal_trigger_frequency(1./period)

    m8195a.start_output()

