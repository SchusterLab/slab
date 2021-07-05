'''
Agilent AG850/TV551 Twistorr pump controller driver.

Author: Andrew Oriani

'''

import slab
import numpy as np
import time
from slab.instruments import SerialInstrument
import yaml

class AG850(SerialInstrument):
    def __init__(self, name="", address='COM8', enabled=True, timeout=.5, config_path=None):
        self.query_sleep = .1
        self.timeout = timeout
        self.recv_length = 1024
        self.s_handle = SerialInstrument.__init__(self, name, address, enabled, timeout, baudrate=9600,
                                                  query_sleep=self.query_sleep)
        self.config_path=config_path
        if config_path == None:
            pass
        else:
            try:
                self.load_config(config_path=self.config)
            except:
                print('Unable to load Turbo Controller configuration')

    def load_config(self, config_path=None):
        self.config = config_path
        if config_path == None:
            print('No config present')
            self.config_path = config_path
        else:
            with open(config_path, 'r') as f:
                self.P_config = yaml.full_load(f)['TURBO']
            self.set_cooling_mode(self.P_config['COOLING'])
            self.low_speed_mode(self.P_config['LOW_SPEED'])
            self.set_point_type(self.P_config['SP_TYPE'])
            self.set_soft_start(self.P_config['SOFT_START'])
            self.set_point_val(self.P_config['SP_VAL'])
            self.set_point_hysteresis(self.P_config['SP_HYST'])
            self.set_high_speed(self.P_config['HIGH_SPEED_VAL'])
            self.set_load_gas(self.P_config['GAS_TYPE'])
            self.set_low_speed(self.P_config['LOW_SPEED_VAL'])
            print('______________________')
            print('Turbo Pump Parameters Set')
            print('______________________')

    def cmd_gen(self, window, cmd, dtype='numeric', WR='write'):

        start_byte = '\x02'
        addr = '\x80'
        ETX = '\x03'

        if WR == 'read':
            wr_byte = bytes([48])
        elif WR == 'write':
            wr_byte = bytes([49])

        win = []
        for val in window:
            win.append(val.encode())

        cmd_vals = [start_byte, addr, win[0], win[1], win[2], bytes(wr_byte)]
        if WR == 'write':
            if dtype == 'numeric':
                cmd = str(cmd).rjust(6, '0')
            elif dtype == 'bool':

                if len(str(cmd)) > 1 or cmd > 2:
                    raise Exception('Boolean must 0 or 1')
                else:
                    cmd = str(cmd)
            for val in cmd:
                cmd_vals.append(val.encode())

            cmd_vals.append('\x03')

            int_array = []
            for vals in cmd_vals:
                int_array.append(ord(vals))

            checksum = 0
            cmd_str = bytes(int_array[1::])
            for el in cmd_str:
                checksum ^= el

            crc_val = hex(checksum).split('x')[1].capitalize().encode().decode()

            for val in crc_val:
                if val.isalpha():
                    val = val.capitalize()
                else:
                    pass
                int_array.append(ord(val))
            cmd_str = bytes(int_array)
            return cmd_str

        elif WR == 'read':
            cmd_vals.append('\x03')
            int_array = []
            for vals in cmd_vals:
                int_array.append(ord(vals))

            checksum = 0
            cmd_str = bytes(int_array[1::])
            for el in cmd_str:
                checksum ^= el

            crc_val = hex(checksum).split('x')[1].capitalize().encode().decode()

            for val in crc_val:
                if val.isalpha():
                    val.capitalize()
                else:
                    pass
                int_array.append(ord(val))
            cmd_str = bytes(int_array)
            return cmd_str
        else:
            raise Exception('Must be read or write')

    def c_write(self, cmd):
        win, dtype, val, wr = self.cmds_table(cmd)
        self.write(self.cmd_gen(win, val, dtype, wr))
        time.sleep(self.query_sleep)
        val = self.ser.read(self.recv_length)
        b_string = list(val)
        ack = b_string[2]
        if ack == 6:
            pass
        elif ack == 51:
            raise Exception('Data Type Error')
        elif ack == 21:
            raise Exception('Command has failed to process by controller, check controller')
        elif ack == 53:
            raise Exception('Invalid command window')
        elif ack == 54:
            raise Exception('Input datatype error, check datatype input')
        elif ack == 55:
            raise Exception('Input value out of range, reference manual')
        elif ack == 56:
            raise Exception('Input currently disabled, stop pump before inputting value')
        else:
            raise Exception('Unknown error, check controller and connection')
        return

    def c_query(self, cmd):
        win, dtype, val, wr = self.cmds_table(cmd)
        if wr == 'read':
            self.write(self.cmd_gen(win, val, dtype, wr))
            time.sleep(self.query_sleep)
            val = self.ser.read(self.recv_length)
            byte_list = list(val)
            stop_byte = byte_list.index(3)
            start_byte = byte_list.index(128)
            win_bytes = bytes(byte_list[start_byte + 1:stop_byte]).decode()[0:3]
            if win_bytes == win:
                if dtype == 'numeric':
                    data_bytes = bytes(byte_list[start_byte + 1:stop_byte]).decode()[3::].lstrip('0')
                    if data_bytes == '':
                        data_bytes = 0
                    else:
                        pass
                    return float(data_bytes)
                elif dtype == 'bool':
                    stop_byte = byte_list.index(3)
                    start_byte = byte_list.index(128)
                    data_bytes = bytes(byte_list[start_byte + 1:stop_byte]).decode()[4::]
                    return int(data_bytes)
            else:
                raise Exception('Command Error, improper returned values')
        else:
            raise Exception('Improper query command')

    def cmds_table(self, keyword):
        cmd_table = {'SetState': {'dtype': 'bool', 'window': '000', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'ConMode': {'dtype': 'bool', 'window': '008', 'val': {'serial': 0, 'remote': 1, 'front': 2},
                                 'WR': 'write'},
                     'SoftStart': {'dtype': 'bool', 'window': '100', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'CoolingType': {'dtype': 'bool', 'window': '106', 'val': {'air': 0, 'water': 1}, 'WR': 'write'},
                     'GetCurr': {'dtype': 'numeric', 'window': '200', 'val': None, 'WR': 'read'},
                     'GetPow': {'dtype': 'numeric', 'window': '202', 'val': None, 'WR': 'read'},
                     'GetFreq': {'dtype': 'numeric', 'window': '203', 'val': None, 'WR': 'read'},
                     'GetTemp': {'dtype': 'numeric', 'window': '222', 'val': None, 'WR': 'read'},
                     'GetStat': {'dtype': 'numeric', 'window': '205', 'val': None, 'WR': 'read'},
                     'GetError': {'dtype': 'bool', 'window': '206', 'val': None, 'WR': 'read'},
                     'GetContTemp': {'dtype': 'numeric', 'window': '211', 'val': None, 'WR': 'read'},
                     'ExtOut': {'dtype': 'bool', 'window': '122', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'ExtOut2': {'dtype': 'bool', 'window': '145', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'SetPointType': {'dtype': 'numeric', 'window': '101', 'val': {'power': 1, 'freq': 0},
                                      'WR': 'write'},
                     'GetSetPointType': {'dtype': 'bool', 'window': '101', 'val': None, 'WR': 'read'},
                     'SetPointVal': {'dtype': 'numeric', 'window': '102', 'val': None, 'WR': 'write'},
                     'GetSetPointVal': {'dtype': 'numeric', 'window': '102', 'val': None, 'WR': 'read'},
                     'SetPointLevel': {'dtype': 'bool', 'window': '104', 'val': {'low': 1, 'high': 0}, 'WR': 'write'},
                     'GetSetPointLevel': {'dtype': 'bool', 'window': '104', 'val': None, 'WR': 'read'},
                     'SetCoolingMode': {'dtype': 'bool', 'window': '106', 'val': {'water': 1, 'air': 0}, 'WR': 'write'},
                     'GetCoolingMode': {'dtype': 'bool', 'window': '106', 'val': None, 'WR': 'read'},
                     'SetLowSpeed': {'dtype': 'numeric', 'window': '117', 'val': None, 'WR': 'write'},
                     'SetHighSpeed': {'dtype': 'numeric', 'window': '120', 'val': None, 'WR': 'write'},
                     'GetLowSpeed': {'dtype': 'numeric', 'window': '117', 'val': None, 'WR': 'read'},
                     'GetHighSpeed': {'dtype': 'numeric', 'window': '120', 'val': None, 'WR': 'read'},
                     'SetPointHyst': {'dtype': 'numeric', 'window': '105', 'val': None, 'WR': 'write'},
                     'GetSetPointHyst': {'dtype': 'numeric', 'window': '105', 'val': None, 'WR': 'read'},
                     'GasLoadType': {'dtype': 'bool', 'window': '157', 'val': {'N2': 1, 'Ar': 0}, 'WR': 'write'},
                     'GetGasLoadType': {'dtype': 'bool', 'window': '157', 'val': None, 'WR': 'read'},
                     'GetMaxSpeed': {'dtype': 'numeric', 'window': '121', 'val': None, 'WR': 'read'},
                     'SetMaxSpeed': {'dtype': 'numeric', 'window': '121', 'val': None, 'WR': 'write'},
                     'LowSpeedSet': {'dtype': 'bool', 'window': '001', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'GetCycleTime': {'dtype': 'numeric', 'window': '300', 'val': None, 'WR': 'read'},
                     'GetPumpLife': {'dtype': 'numeric', 'window': '302', 'val': None, 'WR': 'read'},
                     'GetCycleNumber':{'dtype': 'numeric', 'window': '301', 'val': None, 'WR': 'read'},
                     'SetFan':{'dtype': 'bool', 'window': '144', 'val': {'on': 1, 'off': 0}, 'WR': 'write'},
                     'SetFanMode':{'dtype': 'numeric', 'window': '143', 'val': {'on':0, 'auto':1, 'serial':2}, 'WR': 'write'}
                     }

        cmd = keyword.split(':')[0]

        win = cmd_table[cmd]['window']
        dtype = cmd_table[cmd]['dtype']
        if len(keyword.split(':')) == 1:
            val = None
            cmd_table[cmd]['WR'] = 'read'
        else:
            val_in = keyword.split(':')[1]
            if dtype == 'bool':
                val = cmd_table[cmd]['val'][val_in]
            elif dtype == 'numeric':
                if cmd_table[cmd]['val'] == None:
                    val = val_in
                else:
                    val = cmd_table[cmd]['val'][val_in]

        wr = cmd_table[cmd]['WR']

        return win, dtype, val, wr

    def set_fan(self, state=True):
        if state == True:
            cmd = 'SetFan:on'
        elif state == False:
            cmd = 'SetFan:off'
        else:
            raise Exception('Invalid input')
        self.c_write(cmd)
        return

    def get_power(self):
        return self.c_query('GetPow')

    def get_freq(self):
        return self.c_query('GetFreq')

    def get_temp(self):
        return self.c_query('GetTemp')

    def low_speed_mode(self, state=False):
        current_state = int(self.c_query('GetStat'))
        if state == True:
            if current_state != 0:
                print('Pump must be in stop condition to proceed')
                return
            else:
                cmd = 'LowSpeedSet:on'
        elif state == False:
            cmd = 'LowSpeedSet:off'
        else:
            raise Exception('Invalid input')
        self.c_write(cmd)
        return

    def set_soft_start(self, state):
        current_state = int(self.c_query('GetStat'))
        if current_state != 0:
            print('Pump must be in stop condition to proceed')
            return

        if state == True:
            cmd = 'SoftStart:on'
        elif state == False:
            cmd = 'SoftStart:off'
        else:
            raise Exception('Invalid input')
        self.c_write(cmd)
        return

    def start(self):
        self.set_fan(0)
        status = self.c_query('GetStat')
        if status == 0:
            self.c_write('SetState:on')
            self.set_fan(1)
        else:
            pass
        status = self.get_status()
        return status

    def stop(self):
        status = self.c_query('GetStat')
        if status != 0:
            self.set_fan(0)
            self.c_write('SetState:off')
        else:
            pass
        status = self.get_status()
        return status

    def get_status(self):
        status = int(self.c_query('GetStat'))
        status_vals = ['stop', 'interlock', 'starting', 'auto-tune', 'braking', 'normal', 'fail', 'leak check']
        return status_vals[status]

    def set_control_mode(self, mode):
        self.c_write('ConMode:' + mode)

    def get_soft_start(self):
        return self.c_query('SoftStart')

    def get_current(self):
        return self.c_query('GetCurr')

    def set_vent(self, state):
        if state == True:
            cmd = 'ExtOut:on'
        elif state == False:
            cmd = 'ExtOut:off'
        else:
            raise Exception('Invalid input')
        self.c_write(cmd)
        return

    def set_purge(self, state):
        if state == True:
            cmd = 'ExtOut2:on'
        elif state == False:
            cmd = 'ExtOut2:off'
        else:
            raise Exception('Invalid input')
        self.c_write(cmd)
        return

    def set_low_speed(self, speed=350):
        self.c_write('SetLowSpeed:%d' % (int(speed)))
        return

    def set_high_speed(self, speed=825):
        self.c_write('SetHighSpeed:%d' % (int(speed)))
        return

    def get_low_speed(self):
        return self.c_query('GetLowSpeed')

    def get_high_speed(self):
        return self.c_query('GetHighSpeed')

    def set_point_val(self, val):
        self.c_write('SetPointVal:%d' % (int(val)))
        return

    def get_set_point_val(self):
        return self.c_query('GetSetPointVal')

    def set_point_type(self, set_type='freq'):
        if set_type != 'freq' and set_type != 'power':
            raise Exception('Invalid mode input, must be freq or power')
        self.c_write('SetPointType:%s' % (set_type))
        return

    def get_set_point_type(self):
        set_type = self.c_query('GetSetPointType')
        if set_type == 0:
            return 'freq'
        elif set_type == 1:
            return 'power'
        else:
            return

    def set_cooling_mode(self, mode='water'):
        if mode != 'air' and mode != 'water':
            raise Exception('Invalid mode input, must be air or water')
        self.c_write('SetCoolingMode:%s' % (mode))
        return

    def get_cooling_mode(self):
        mode = self.c_query('GetCoolingMode')
        if mode == 0:
            return 'air'
        elif mode == 1:
            return 'water'
        else:
            return

    def set_point_out_level(self, level='high'):
        if level != 'high' and level != 'low':
            raise Exception('Invalid output level, must be high or low')
        self.c_write('SetPointLevel:%s' % (level))
        return

    def get_set_point_out_level(self):
        level = self.c_query('GetSetPointLevel')
        if level == 0:
            return 'high'
        elif level == 1:
            return 'low'
        else:
            return

    def set_point_hysteresis(self, val=2):
        if val < 0 and val > 100:
            raise Exception('Value must be in \% threshold from 0-100')
        self.c_write('SetPointHyst:%d' % (int(val)))
        return

    def get_set_point_hysteresis(self):
        return self.c_query('GetSetPointHyst')

    def set_load_gas(self, gas='N2'):
        if gas != 'N2' and gas != 'Ar':
            raise Exception('Load gas must be N2 or Ar')
        self.c_write('GasLoadType:%s' % (gas))
        return

    def get_load_gas(self):
        gas = self.c_query('GetGasLoadType')
        if gas == 0:
            return 'Ar'
        elif gas == 1:
            return 'N2'
        else:
            return

    def set_fan_mode(self, mode='serial'):
        if mode!='serial' and mode!='auto' and mode!='on':
            raise Exception('INVALID INPUT: must be on, auto, or serial')
        else:
            self.c_write('SetFanMode:%s'%(mode))
            return
