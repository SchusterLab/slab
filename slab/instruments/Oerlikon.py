'''
This is a driver for the Cnter Three (Inficon VGC403) three channel pressure gauge readout.

Author: Andrew Oriani

'''
import slab
from slab import *
import numpy as np
import time
import warnings
from slab.instruments import SerialInstrument
import yaml

class Center_Three(SerialInstrument):
    def __init__(self, name="", address='COM7', enabled=True, timeout=.5, config_path=None):
        self.query_sleep = .05
        self.timeout = timeout
        self.rec_length = 1024
        self.s_handle = SerialInstrument.__init__(self, name, address, enabled, timeout, baudrate=38400,
                                                  query_sleep=self.query_sleep)
        self.term_char = bytes([13])
        self.c_char = {'EXT': bytes([3]), 'ENQ': bytes([5])}
        self.config_path=config_path
        if config_path == None:
            pass
        else:
            try:
                self.load_config(config_path=self.config)
            except:
                print('Unable to load pressure gauge configuration')

    def load_config(self, config_path=None):
        self.config = config_path
        if config_path == None:
            print('No config present')
            self.config_path = config_path
        else:
            with open(config_path, 'r') as f:
                self.P_config = yaml.full_load(f)['Pressure_Monitor']
            self.set_units(self.P_config['UNITS'])
            self.SP_configs = self.P_config['SET_POINTS']
            for SP in self.SP_configs:
               self.set_point(SP, self.SP_configs[SP]['CHANNEL'], self.SP_configs[SP]['SP_H'], self.SP_configs[SP]['SP_L'])
            print('______________________')
            print('Pressure Gauge Parameters Set')
            print('______________________')

    def get_gauges(self):
        val=self.p_query('TID')
        return val
    
    def p_query(self, cmd, bit_len=1024):
        self.write(cmd)
        ack=self.ser.read(3)
        self.write(self.c_char['ENQ'])
        mes=self.ser.read(bit_len)
        return mes.decode().split('\r\n')[0]
    
    def p_read(self, ch):
        val=self.p_query('PR'+str(ch), bit_len=15).split(',')
        status = val[0]
        pressure = float(val[1])
        return pressure, status

    def units(self):
        unit=int(self.p_query('UNI'))
        if unit==0:
            return 'mbar'
        elif unit==1:
            return 'torr'
        elif unit==2:
            return 'Pa'
        elif unit==3:
            return 'micron'
        else:
            pass
        
    def set_units(self, unit):
        if unit=='mbar':
            a=0
        elif unit=='torr':
            a=1
        elif unit=='pascal':
            a=2
        elif unit=='micron':
            a=3
        else:
            raise Exception('Not valid unit, must be mbar, torr, pascal or micron')
        self.p_query('UNI, '+str(a), bit_len=6)
        return
    
    def set_point(self, SP, ch, high_val, low_val):
        val=self.p_query('SP{},{},{:.4E},{:.4E}'.format(SP,ch-1, float(high_val), float(low_val))).split(',')
        output=[]
        for vals in val:
            output.append(vals)
        return output
    
    def get_set_point(self, SP):
        if SP==None:
            raise Exception('Invalid SP input')
        else:
            pass
        val=self.p_query('SP{}'.format(SP), bit_len=25).split(',')
        output=[]
        for vals in val:
            output.append(vals)
        return output
    
    def get_trans_ID(self):
        val=self.p_query('TID').split(',')
        output=[]
        for vals in val:
            output.append(vals)
        return output
    
    def cal_CTR(self):
        ctr_loc=self.get_trans_ID().index('CTR')
        cal_loc=[0,0,0]
        cal_loc[ctr_loc]=3
        print(cal_loc)
        val=self.p_query('OFC,{},{},{}'.format(cal_loc[0], cal_loc[1], cal_loc[2]))
        return val.split(',')

    def set_point_status(self):
        return list(map(int, self.p_query('SPS', bit_len=13).split(',')))