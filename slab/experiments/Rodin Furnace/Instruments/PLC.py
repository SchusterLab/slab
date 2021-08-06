"""
This is a driver for a PLC designed to be used ot control the small Nb cavity treatment furnace. It is an ardbox
(Industrial Shields) PLC based off of an arduino leonardo that provides 8 relay channels and 8 isolated logic inputs.

Author: Andrew Oriani

"""

import slab
import numpy as np
import time
from slab.instruments import SerialInstrument

class FurnacePLC(SerialInstrument):
    def __init__(self, name="", address='COM20', enabled=True, timeout=.02):
        self.query_sleep = .005
        self.timeout = timeout
        self.recv_length = 32
        self.term_char = '\r\n'
        SerialInstrument.__init__(self, name, address, enabled, timeout, self.recv_length, baudrate=115200,
                                                  query_sleep=self.query_sleep)

        self.valve_state=self.get_all()
        self.pulse_state=self.get_pulse_state()
        self.pulse_time=self.set_pulse_time(1)


    def set_valve(self, ch, state=False):
        if state == True:
            val = "OPEN"
        elif state == False:
            val = "CLOSE"
        else:
            raise Exception("State must be boolean")

        res = self.v_query('SET:%s:%d' % (val, int(ch)))
        if res=='':
            return [int(ch-1), int(self.valve_state[ch-1])]
        else:
            pass
        val = res.split(":")
        if val[0] == "ERR":
            raise Exception("ERROR: type=%s" % (val[1]))
        elif val[0] == "INTLK":
            if val[1] == "OVERPRESSURE":
                print('INTERLOCK: Unable to open gate, chamber pressure to high')
            elif val[1] == "VENT":
                print('INTERLOCK: Unable to vent, turbo running')
            else:
                pass
            return -1
        else:
            self.valve_state[ch-1]=int(state)
            return [int(val[0]), int(val[1])]

    def get_state(self, ch):
        if ch==0:
            return self.get_all()
        else:
            pass
        res = self.v_query('GET:%d' % (int(ch)), bit_len=5)
        if res=='':
            return [int(ch), int(self.valve_state[ch-1])]
        else:
            pass
        val = res.split(":")
        if int(val[1])!=self.valve_state[int(val[0])-1]:
            self.valve_state[int(val[0])-1]=int(val[1])
        else:
            pass
        return [int(val[0]), int(val[1])]

    def get_pulse_state(self):
        res = self.v_query('PULSE:STATE', bit_len=1024)
        if res=='':
            return self.pulse_state
        else:
            pass
        state = res.split(':')[1]
        self.pulse_state=state
        return bool(int(state))

    def get_all(self):
        res = self.v_query('GET:0', bit_len=21)
        if res=='':
            return self.valve_state
        else:
            pass
        vals = res.split(":")[1][1:-1].split(',')
        self.valve_state=list(map(int, vals))
        return self.valve_state

    def set_pulse_time(self, time=1):
        res=self.v_query("PULSE:TIME:%d" % (int(time)), bit_len=7+len(str(time)))
        if res=='':
            return self.pulse_time
        else:
            pass
        self.pulse_time=res
        return res

    def pulse(self, ch):
        res = self.v_query("SET:PULSE:%d" % (int(ch)), bit_len=13)
        if res=='':
            return "HOLD"
        else:
            pass
        self.pulse_state=True
        return res

    def v_query(self, cmd, bit_len=1024):
        self.ser.write(str(cmd+'\n').encode())
        time.sleep(self.query_sleep)
        mes=self.ser.read(bit_len)
        return mes.decode().split(self.term_char)[0]

    def reset(self):
        res=self.v_query('RESET:')
        if res.split(':')[0]=='ERR':
            self.write('RESET:')