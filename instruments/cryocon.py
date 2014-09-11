# -*- coding: utf-8 -*-
"""
Created on Wed Sep 03 02:46:30 2014

@author: Phil
"""

from slab.instruments import SocketInstrument

class Cryocon(SocketInstrument):
    default_port = 5000

    def __init__(self, name="Cryocon", address=None, enabled=True):
        SocketInstrument.__init__(self, name, address, enabled=enabled, timeout=10, recv_length=2 ** 20)
        self.query_sleep = 0.05

    def get_id(self):
        return "Cryocon 18I"

    def get_query_sleep(self):
        return self.query_sleep

    def get_temp(self,ch='A'):
        return float(self.query('INPUT? '+ch.upper()))
    
    def set_sensor_type(self,ch,sensor_type=0):
        self.write("INPUT %s:SENSOR %d" %(ch,sensor_type) )
                

##### Manage internal data logging features
    def reset_log(self):
        self.write('DLOG:RESET')
        
    def clear_log(self):
        self.write('DLOG:CLEAR')
        
    def set_logging(self,state):
        if state:
            self.write('DLOG:RUN ON')
        else:
            self.write('DLOG:RUN OFF')

    def get_log(self):
        s = self.query("DLOG:READ?")

        dates=[]
        times=[]
        temps=[]        
        for line in s:
            sl = line.split(',')
            dates.append(sl[0])
            times.append(sl[1])
            temps.append(sl[2:])
        return (dates,times,temps)
            
  
      