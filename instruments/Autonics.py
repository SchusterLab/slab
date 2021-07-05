'''
Driver for Autonics TM4 PID heater control modules. This is a modbus RTU instrument using RS485 to
multiplex across multiple units.

Author: Andrew Oriani

'''

import slab
from slab import *
import crcmod
from crcmod.predefined import *
import time
from slab.instruments import SerialInstrument
import serial
import yaml


class TM4(SerialInstrument):
    def __init__(self, name="", address='COM15', enabled=True, timeout=.5, modbus_address=1, config_path=None):
        self.query_sleep = .1
        self.timeout = timeout
        self.recv_length = 1024
        self.modbus_address = modbus_address
        self.crc16 = crcmod.predefined.mkCrcFun('modbus')
        self.s_handle = serial.Serial(port=address, baudrate=9600, stopbits=serial.STOPBITS_TWO, timeout=timeout)
        self.set_dec()
        self.config = config_path
        if config_path == None:
            pass
        else:
            try:
                self.load_config(config_path=self.config)
            except:
                print('Unable to load PID control configuration')

    def load_config(self, config_path=None):
        self.config = config_path
        if config_path == None:
            print('No config present')
            self.config_path = config_path
        else:
            with open(config_path, 'r') as f:
                self.PID_config = yaml.full_load(f)['PID']

            self.CH_configs = self.PID_config['CHANNELS']
            for ch in self.CH_configs:
                CH_props = self.CH_configs[ch]
                self.run(ch, state=False)
                self.set_ramp_up_rate(ch, self.PID_config['Ramp_Up_Rate'], self.PID_config['Ramp_Time_Unit'])
                self.set_ramp_down_rate(ch, self.PID_config['Ramp_Down_Rate'], self.PID_config['Ramp_Time_Unit'])
                self.set_units(ch, unit=self.PID_config['UNITS'])
                self.set_con_interval(ch, self.PID_config['Con_Interval'])
                self.set_control_mode(ch, CH_props['Con_Mode'])
                self.set_PID(ch, CH_props['Set_Point'], P=CH_props['P'], I=int(CH_props['I']), D=int(CH_props['D']))
            print('______________________')
            print('PID Heater Control Set')
            print('______________________')

    def set_dec(self, point=True):
        if point == True:
            self.data_point = 1
        else:
            self.data_point = 0

    def hex_to_dec(self, value, signed=True, point=1):
        hex_val = ''
        for byte in value:
            hex_val += hex(byte).split('x')[1]

        dec = 0
        power = len(hex_val) - 1
        for num in hex_val:
            if num.capitalize() == 'A':
                num = 10
            elif num.capitalize() == 'B':
                num = 11
            elif num.capitalize() == 'C':
                num = 12
            elif num.capitalize() == 'D':
                num = 13
            elif num.capitalize() == 'E':
                num = 14
            elif num.capitalize() == 'F':
                num = 15
            dec += int(num) * 16 ** power
            power -= 1
        if signed == True:
            dec = -(dec & 0x8000) | (dec & 0x7fff)
        else:
            pass
        return dec / (10 ** point)

    def m_query(self, cmd_str):
        barray, dtype, wr = self.cmds_table(cmd_str)
        if wr == 'write':
            raise Exception('Write only command')
        else:
            pass
        cmd = self.cmd_gen(barray)
        self.s_handle.write(bytes(cmd))
        time.sleep(self.query_sleep)
        trans_array = self.s_handle.read(self.recv_length)
        err_check = list(trans_array)[1]
        if err_check % 128 <= 15 and err_check > 128:
            err_val = list(trans_array)[2]
            if err_val == 1:
                raise Exception('Write Error: ILLEGAL FUNCTION')
            elif err_val == 2:
                raise Exception('Write Error: ILLEGAL DATA ADDRESS')
            elif err_val == 3:
                raise Exception('Write Error: ILLEGAL DATA VALUE')
            elif err_val == 4:
                raise Exception('Write ErrorL SLAVE DEVICE ERROR')
        else:
            pass
        trans_data = list(trans_array)[0:-2]
        data_bytes = trans_data[2]
        value = trans_data[3:3 + data_bytes]
        output = []
        if dtype == 'numeric':
            for i, k in zip(value[0::2], value[1::2]):
                output.append(self.hex_to_dec([i, k], point=self.data_point))
        elif dtype == 'bool' or dtype == 'int':
            for i, k in zip(value[0::2], value[1::2]):
                output.append(int(self.hex_to_dec([i, k], point=0)))
        return output

    def m_write(self, cmd_str):
        barray, dtype, wr = self.cmds_table(cmd_str)
        if wr == 'read':
            raise Exception('Read only command')
        else:
            pass
        cmd = self.cmd_gen(barray)
        self.s_handle.write(bytes(cmd))
        time.sleep(self.query_sleep)
        trans_array = self.s_handle.read(self.recv_length)
        trans_data = list(trans_array)
        err_check = trans_data[1]
        if err_check % 128 <= 15 and err_check > 128:
            err_val = trans_data[2]
            if err_val == 1:
                raise Exception('Write Error: ILLEGAL FUNCTION')
            elif err_val == 2:
                raise Exception('Write Error: ILLEGAL DATA ADDRESS')
            elif err_val == 3:
                raise Exception('Write Error: ILLEGAL DATA VALUE')
            elif err_val == 4:
                raise Exception('Write Error: SLAVE DEVICE ERROR')
        else:
            pass
        return

    def cmd_gen(self, barray):
        crc = self.crc16(bytes(barray))
        crc_low_hi = [(crc & 0xFF), (crc >> 8)]
        data_array = barray + crc_low_hi
        return data_array

    def float_to_hex(self, val):
        int_val = int(val * 10)
        high, low = int_val.to_bytes(2, 'big', signed=True)
        return high, low

    def int_to_hex(self, val):
        high, low = int(val).to_bytes(2, 'big', signed=True)
        return high, low

    def cmds_table(self, keyword):
        cmd_str = keyword.split(':')
        if len(cmd_str) > 1:
            ch_val = cmd_str[1].split('CH')
            if ch_val[0] == '':
                ch = int(ch_val[1])
                if ch % 4 == 0:
                    self.modbus_address = int(ch / 4)
                else:
                    self.modbus_address = int(1 + (ch - ch % 4) / 4)

                if ch > 4:
                    ch = ch % 4
                    if ch == 0:
                        ch = 4
                    else:
                        pass
                else:
                    pass
            else:
                raise Exception('Missing channel value')
        else:
            ch = 1

        cmd_table = {
            'ReadTemp': {'func': 4, 'reg': int(1000 + 6 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'numeric'},
            'GetDot': {'func': 4, 'reg': int(1001 + 6 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'bool'},
            'GetUnit': {'func': 4, 'reg': int(1002 + 6 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'bool'},
            'GetSV': {'func': 4, 'reg': int(1003 + 6 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'numeric'},
            'GetMV': {'func': 4, 'reg': int(1004 + 6 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'numeric'},
            'SetMV': {'func': 6, 'reg': int(1 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'SetSV': {'func': 6, 'reg': int(0 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'SetCon': {'func': 6, 'reg': int(3 + 1000 * (ch - 1)), 'val': {'AUTO': 0, 'MANUAL': 1}, 'WR': 'write',
                       'dtype': 'int'},
            'GetCon': {'func': 3, 'reg': int(3 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'bool'},
            'Run': {'func': 5, 'reg': int(0 + (ch - 1) * 2), 'val': {'START': 0, 'STOP': 1}, 'WR': 'write',
                    'dtype': 'bool'},
            'AutoTune': {'func': 6, 'reg': int(100 + (ch - 1) * 1000), 'val': {'START': 0, 'STOP': 1}, 'WR': 'write',
                         'dtype': 'bool'},
            'GetAutoTune': {'func': 3, 'reg': int(100 + (ch - 1) * 1000), 'val': {'START': 0, 'STOP': 1}, 'WR': 'read',
                            'dtype': 'bool'},
            'SetHPBand': {'func': 6, 'reg': int(101 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'SetCPBand': {'func': 6, 'reg': int(102 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetHPBand': {'func': 3, 'reg': int(101 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                          'dtype': 'numeric'},
            'GetCPBand': {'func': 3, 'reg': int(102 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                          'dtype': 'numeric'},
            'SetHIBand': {'func': 6, 'reg': int(103 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'int'},
            'SetCIBand': {'func': 6, 'reg': int(104 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'int'},
            'GetHIBand': {'func': 3, 'reg': int(103 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'GetCIBand': {'func': 3, 'reg': int(104 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'SetHDBand': {'func': 6, 'reg': int(105 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'int'},
            'SetCDBand': {'func': 6, 'reg': int(106 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'int'},
            'GetHDBand': {'func': 3, 'reg': int(105 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'GetCDBand': {'func': 3, 'reg': int(106 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'PModeManReset': {'func': 6, 'reg': int(108 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'HeatHyst': {'func': 6, 'reg': int(109 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetHeatHyst': {'func': 3, 'reg': int(109 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                            'dtype': 'numeric'},
            'HeatOffset': {'func': 6, 'reg': int(110 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetHeatOffset': {'func': 3, 'reg': int(110 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                              'dtype': 'numeric'},
            'CoolHyst': {'func': 6, 'reg': int(111 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetCoolHyst': {'func': 3, 'reg': int(111 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                            'dtype': 'numeric'},
            'CoolOffset': {'func': 6, 'reg': int(112 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetCoolOffset': {'func': 3, 'reg': int(112 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                              'dtype': 'numeric'},
            'SetMVLowLimit': {'func': 6, 'reg': int(113 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetMVLowLimit': {'func': 3, 'reg': int(113 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                              'dtype': 'numeric'},
            'SetMVHighLimit': {'func': 6, 'reg': int(114 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetMVHighLimit': {'func': 3, 'reg': int(114 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                               'dtype': 'numeric'},
            'SetRampUpRate': {'func': 6, 'reg': int(115 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetRampUpRate': {'func': 3, 'reg': int(115 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                              'dtype': 'numeric'},
            'SetRampDownRate': {'func': 6, 'reg': int(116 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetRampDownRate': {'func': 3, 'reg': int(116 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                                'dtype': 'numeric'},
            'SetRampTimeUnit': {'func': 6, 'reg': int(117 + 1000 * (ch - 1)), 'WR': 'write',
                                'val': {'SEC': 0, 'MIN': 1, 'HOUR': 2}, 'dtype': 'int'},
            'GetRampTimeUnit': {'func': 3, 'reg': int(117 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                                'dtype': 'int'},
            'SetTherm': {'func': 6, 'reg': int(150 + 1000 * (ch - 1)), 'WR': 'write',
                         'val': {'KH': 0, 'KL': 1, 'JH': 2, 'JL': 3, 'EH': 4, 'EL': 5, 'TH': 6, 'TL': 7, 'B': 8, 'R': 9,
                                 'S': 10,
                                 'N': 11, 'C': 12, 'G': 13, 'LH': 14, 'LL': 15, 'UH': 16, 'UL': 17, 'Plat': 18,
                                 'JPt100H': 19, 'JPt100L': 20, 'DPt100H': 21, 'DPt100L': 22},
                         'dtype': 'int'},
            'GetTherm': {'func': 3, 'reg': int(150 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'SetUnit': {'func': 6, 'reg': int(151 + 1000 * (ch - 1)), 'WR': 'write', 'val': {'C': 0, 'F': 1},
                        'dtype': 'int'},
            'SetInputBias': {'func': 6, 'reg': int(152 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetInputBias': {'func': 3, 'reg': int(152 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                             'dtype': 'numeric'},
            'SetDigFilt': {'func': 6, 'reg': int(153 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetDigFilt': {'func': 3, 'reg': int(153 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                           'dtype': 'numeric'},
            'SetSVLowLimit': {'func': 6, 'reg': int(154 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetSVLowLimit': {'func': 3, 'reg': int(154 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                              'dtype': 'numeric'},
            'SetSVHighLimit': {'func': 6, 'reg': int(155 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetSVHighLimit': {'func': 3, 'reg': int(155 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                               'dtype': 'numeric'},
            'SetOpMode': {'func': 6, 'reg': int(156 + 1000 * (ch - 1)), 'WR': 'write',
                          'val': {'HEAT': 0, 'COOL': 1, 'BOTH': 2}, 'dtype': 'int'},
            'GetOpMode': {'func': 3, 'reg': int(156 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'SetConType': {'func': 6, 'reg': int(157 + 1000 * (ch - 1)), 'WR': 'write', 'val': {'PID': 0, 'ONOFF': 1},
                           'dtype': 'int'},
            'GetConType': {'func': 3, 'reg': int(157 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'SetHCConType': {'func': 6, 'reg': int(157 + 1000 * (ch - 1)), 'WR': 'write',
                             'val': {'PID-PID': 0, 'PID-ONOFF': 1, 'ONOFF-PID': 2, 'ONOFF-ONOFF': 3},
                             'dtype': 'int'},
            'GetHCConType': {'func': 3, 'reg': int(157 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1, 'dtype': 'int'},
            'SetAutoTuneMode': {'func': 6, 'reg': int(158 + 1000 * (ch - 1)), 'WR': 'write',
                                'val': {'TUN1': 0, 'TUN2': 1}, 'dtype': 'int'},
            'GetAutoTuneMode': {'func': 3, 'reg': int(158 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                                'dtype': 'int'},
            'SetHConTime': {'func': 6, 'reg': int(159 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetHConTime': {'func': 3, 'reg': int(159 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                            'dtype': 'numeric'},
            'SetCConTime': {'func': 6, 'reg': int(160 + 1000 * (ch - 1)), 'WR': 'write', 'dtype': 'numeric'},
            'GetCConTime': {'func': 3, 'reg': int(160 + 1000 * (ch - 1)), 'WR': 'read', 'byte_len': 1,
                            'dtype': 'numeric'},
            }

        cmd = cmd_str[0]

        func = cmd_table[cmd]['func']
        reg_high, reg_low = cmd_table[cmd]['reg'].to_bytes(2, 'big')
        dtype = cmd_table[cmd]['dtype']

        if len(cmd_str) == 2:
            val = None
            cmd_table[cmd]['WR'] = 'read'
        else:
            val_in = cmd_str[2]
            if dtype == 'bool':
                val = cmd_table[cmd]['val'][val_in]
                val_high = 0
                val_low = 0
                if val == 1:
                    val_high = 255
                else:
                    pass
            elif dtype == 'numeric':
                val_high, val_low = self.float_to_hex(float(val_in))
            elif dtype == 'int':
                if 'val' in cmd_table[cmd]:
                    val = cmd_table[cmd]['val'][val_in]
                else:
                    val = val_in
                val_high, val_low = self.int_to_hex(val)

        wr = cmd_table[cmd]['WR']
        if wr == 'read':
            n_bytes_high, n_bytes_low = cmd_table[cmd]['byte_len'].to_bytes(2, 'big')
            barray = [self.modbus_address, func, reg_high, reg_low, n_bytes_high, n_bytes_low]
        else:
            barray = [self.modbus_address, func, reg_high, reg_low, val_high, val_low]
        return [barray, dtype, wr]

    def set_mode(self, ch, mode='heating'):
        m_val = {'heating': 'HEAT', 'cooling': 'COOl', 'both': 'BOTH'}
        self.m_write('SetOpMode:CH' + str(int(ch)) + ':' + m_val[mode])

    def get_mode(self, ch):
        val = self.m_query('GetOpMode:CH' + str(int(ch)))[0]
        modes = ['HEAT', 'COOL', 'BOTH']
        return modes[val]

    def set_control_type(self, ch, val, mode='heating'):
        if mode == 'heating':
            val_tbl = ['PID', 'ONOFF']
            m_val = ''
        elif mode == 'cooling':
            val_tbl = ['PID', 'ONOFF']
            m_val = ''
        if mode == 'both':
            val_tbl = ['PID-PID', 'PID-ONOFF', 'ONOFF-PID', 'ONOFF-ONOFF']
            m_val = 'HC'
        if val in val_tbl:
            self.m_write('Set' + m_val + 'ConType:CH' + str(int(ch)) + ':' + val)
        else:
            raise Exception('Not valid control mode, must be ONOFF or PID')

    def get_control_type(self, ch):
        val = self.m_query('GetConType:CH' + str(int(ch)))[0]
        mode = self.get_mode(ch)
        if mode == 'HEAT' or mode == 'COOL':
            val_tbl = ['PID', 'ONOFF']
        elif mode == 'BOTH':
            val_tbl = ['PID-PID', 'PID-ONOFF', 'ONOFF-PID', 'ONOFF-ONOFF']
        return val_tbl[val]

    def set_control_mode(self, ch, val='AUTO'):
        self.m_write('SetCon:CH' + str(int(ch)) + ':' + val)

    def get_control_mode(self, ch):
        modes = ['AUTO', 'MANUAL']
        val = self.m_query('GetCon:CH' + str(int(ch)))[0]
        return modes[val]

    def set_MV(self, ch, val):
        self.m_write('SetMV:CH' + str(int(ch)) + ':' + str(val))

    def set_P(self, ch, val, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        self.m_write('Set' + m_val[mode] + 'PBand:CH' + str(int(ch)) + ':' + str(val))

    def get_P(self, ch, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        return self.m_query('Get' + m_val[mode] + 'PBand:CH' + str(int(ch)))[0]

    def set_I(self, ch, val, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        self.m_write('Set' + m_val[mode] + 'IBand:CH' + str(int(ch)) + ':' + str(val))

    def get_I(self, ch, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        return self.m_query('Get' + m_val[mode] + 'IBand:CH' + str(int(ch)))[0]

    def set_D(self, ch, val, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        self.m_write('Set' + m_val[mode] + 'DBand:CH' + str(int(ch)) + ':' + str(val))

    def get_D(self, ch, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        return self.m_query('Get' + m_val[mode] + 'DBand:CH' + str(int(ch)))[0]

    def set_sv(self, ch, val):
        self.m_write('SetSV:CH' + str(int(ch)) + ':' + str(val))

    def set_sv_high_lim(self, ch, val):
        self.m_write('SetSVHighLimit:CH' + str(int(ch)) + ':' + str(val))

    def set_sv_low_lim(self, ch, val):
        self.m_write('SetSVLowLimit:CH' + str(int(ch)) + ':' + str(val))

    def get_sv_high_lim(self, ch):
        return self.m_query('GetSVHighLimit:CH' + str(int(ch)))[0]

    def get_sv_low_lim(self, ch):
        return self.m_query('GetSVLowLimit:CH' + str(int(ch)))[0]

    def set_units(self, ch, unit='C'):
        self.m_write('SetUnit:CH' + str(int(ch)) + ':' + unit)

    def get_units(self, ch):
        units = ['C', 'F']
        val = self.m_query('GetUnit:CH' + str(int(ch)))[0]
        return units[val]

    def set_thermometer(self, ch, therm='KL'):
        self.m_write('SetTherm:CH' + str(int(ch)) + ':' + therm)

    def get_thermometer(self, ch):
        val = self.m_query('GetTherm:CH' + str(int(ch)))[0]
        types = ['KH', 'KL', 'JH', 'JL', 'EH', 'EL', 'TH', 'TL', 'B', 'R', 'S', 'N', 'C', 'G', 'LH', 'LL', 'UH', 'UL',
                 'Plat', 'JPt100H', 'JPt100L', 'DPt100H', 'DPt100L']
        return types[val]

    def run(self, ch, state=False):
        if state == False:
            val = 'STOP'
        elif state == True:
            val = 'START'
        self.m_write('Run:CH' + str(int(ch)) + ':' + val)

    def set_con_interval(self, ch, val, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        self.m_write('Set' + m_val[mode] + 'ConTime:CH' + str(int(ch)) + ':' + str(val))

    def get_con_interval(self, ch, mode='heating'):
        m_val = {'heating': 'H', 'cooling': 'C'}
        return self.m_query('Get' + m_val[mode] + 'ConTime:CH' + str(int(ch)))[0]

    def set_PID(self, ch, sv, P=10, I=0, D=0, mode='heating'):
        self.run(ch)
        self.set_mode(ch, mode)
        if mode == 'both':
            self.set_control_type(ch, 'PID-PID', mode)
        else:
            self.set_control_type(ch, 'PID', mode)
        self.set_P(ch, P, mode)
        self.set_I(ch, I, mode)
        self.set_D(ch, D, mode)
        self.set_sv(ch, sv)

    def get_temp(self, ch):
        temps = []
        if ch == 0:
            temps = []
            for i in range(1, 8):
                temps.append(self.m_query('ReadTemp:CH' + str(int(i)))[0])
            return temps
        else:
            return self.m_query('ReadTemp:CH' + str(int(ch)))[0]

    def close(self):
        self.s_handle.close()

    def start_all(self):
        if self.config == None:
            for ch in range(1, 8):
                if self.get_temp(ch) == 3100.00:
                    pass
                else:
                    self.run(ch, state=True)
        else:
            for ch in self.CH_configs:
                CH_props = self.CH_configs[ch]
                if CH_props['RUN'] == True:
                    if self.get_temp(ch) == 3100.00:
                        pass
                    else:
                        self.run(ch, state=CH_props['RUN'])
        print('______________________')
        print('Heaters On')
        print('______________________')

    def stop_all(self):
        if self.config == None:
            for ch in range(1, 8):
                self.run(ch, state=False)
        else:
            for ch in self.CH_configs:
                self.run(ch, state=False)
        print('______________________')
        print('Heaters Off')
        print('______________________')

    def set_ramp_up_rate(self, ch, val, time_scale='MIN'):
        if val < 0 and val > 9999:
            raise Exception('Value must be between 0 (OFF) and 9999 C/F')
        else:
            pass
        if time_scale != 'MIN' and time_scale != 'SEC' and time_scale != 'HOUR':
            raise Exception('Time scale must be SEC, MIN (default), HOUR')
        else:
            pass
        self.m_write('SetRampTimeUnit:CH%d:%s' % (int(ch), time_scale))
        self.m_write('SetRampUpRate:CH%d:%d' % (int(ch), int(val)))
        return

    def set_ramp_down_rate(self, ch, val=0, time_scale='MIN'):
        if val < 0 and val > 9999:
            raise Exception('Value must be between 0 (OFF) and 9999 C/F')
        if time_scale != 'MIN' and time_scale != 'SEC' and time_scale != 'HOUR':
            raise Exception('Time scale must be SEC, MIN (default), HOUR')
        self.m_write('SetRampTimeUnit:CH%d:%s' % (int(ch), time_scale))
        self.m_write('SetRampDownRate:CH%d:%d' % (int(ch), int(val)))
        return

    def get_ramp_down_rate(self, ch):
        return self.m_query('GetRampDownRate:CH%d' % (int(ch)))[0]

    def get_ramp_up_rate(self, ch):
        return self.m_query('GetRampUpRate:CH%d' % (int(ch)))[0]

    def get_ramp_time_units(self, ch):
        units = self.m_query('GetRampTimeUnit:CH%d' % (int(ch)))[0]
        if units == 0:
            return 'SEC'
        elif units == 1:
            return 'MIN'
        elif units == 2:
            return 'HOUR'
        else:
            return

    def set_ramp_time_units(self, ch, time_scale='MIN'):
        if time_scale != 'MIN' and time_scale != 'SEC' and time_scale != 'HOUR':
            raise Exception('Time scale must be SEC, MIN (default), HOUR')
        self.m_write('SetRampTimeUnit:CH%d:%s' % (int(ch), time_scale))
        return