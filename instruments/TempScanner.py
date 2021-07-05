"""
This is a driver for the AG/HP 34970 Scanner. This scanner mainframe is designed for three independant and configurable
cards. The driver is written with base functionality using a 16 channel temperature/voltage measurement card.

Author: Andrew Oriani

"""

import slab
from slab import *
import numpy as np
import time
from slab.instruments import SerialInstrument


class HP34970A(SerialInstrument):
    def __init__(self, name="", address='COM23', enabled=True, timeout=.5):
        self.query_sleep=.01
        self.timeout=timeout
        self.recv_length=1024
        self.term_char='\r\n'
        self.therm_type=['TC', 'K']
        SerialInstrument.__init__(self, name, address, enabled, timeout, baudrate=115200, query_sleep=self.query_sleep)
        
    def c_query(self, cmd, bit_len=1024):
        self.ser.write(str(cmd+self.term_char).encode())
        time.sleep(self.query_sleep)
        mes=self.ser.read(bit_len)
        return mes.decode().split('\r\n')[0]
    
    def get_id(self):
        return self.c_query('*IDN?', bit_len=33)
    
    def ch_list(self, ch, card=1):
        if type(ch)==list:
            ch_list="@"
            for chs in ch:
                ch_val=str(card)+'{:>02d}'.format(chs)[0::]+','
                ch_list+=ch_val
            return ch_list[0:-1]
        elif type(ch)==int:
            ch_list="@%s"%(str(card)+'{:>02d}'.format(ch)[0::])
            return ch_list
        
    def val_unpack(self, val):
        return list(map(float, val.split(',')))
        
    def get_temp(self, ch, sens="TC", t_type='K', card=1):
        if type(ch)!=list:
            ch=[ch]
        else:
            pass
        val=self.c_query("MEAS:TEMP? %s, %s, (%s)"%(sens, t_type, self.ch_list(ch, card)), bit_len=len(ch)*16+1)
        return self.val_unpack(val)
    
    def conf_temp_tcouple(self, ch, t_type='K', card=1):
        type_list=['B', 'E', 'K', 'J', 'N', 'R', 'S', 'T']
        if type(ch)!=list:
            ch=[ch]
        else:
            pass
        if t_type in type_list:
            self.write('CONF:TEMP TC,%s, (%s)'%(t_type, self.ch_list(ch, card)))
        else:
            raise Exception('Invalid thermocouple type')

    def read_chs(self, trig=None):
        if trig==None:
            trig='IMM'
        self.write('TRIG:SOUR %s'%trig)
        val=self.c_query('READ?')
        return self.val_unpack(val)
            
    def get_ch_config(self, ch, card=1):
        return self.c_query("CONF? (%s)"%(self.ch_list(ch, card))).split("\"")[1::2]
    
    def conf_temp_unit(self, ch, unit='C', card=1):
        unit_type=['C', 'F', 'K']
        if unit in unit_type:
            self.write("UNIT:TEMP %s, (%s)"%(unit, self.ch_list(ch, card)))
        else:
            raise Exception("Invalid temperature unit, must be C (default), F, or K")
    
    def get_temp_unit(self, ch, card=1):
        return self.c_query("UNIT:TEMP? (%s)"%(self.ch_list(ch)), bit_len=3).split(',')

    def abort(self):
        self.write('ABOR')
    

