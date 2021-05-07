# -*- coding: utf-8 -*-
"""
Created on Sun Dec 02 22:58:15 2012

@author: Dave
"""

from slab import *
from slab.instruments import InstrumentManager
from numpy import *
import time,datetime
from matplotlib.pyplot import show,plot

class eHe:
    
    biasCh=1
    trapCh=2
    
    def __init__(self,expt_path=None,instrument_manager=None):
        if instrument_manager is None:
            instrument_manager=InstrumentManager()
        self.im=instrument_manager
        self.expt_path=expt_path
        
        self.filament=self.im['filament']
        self.srs=self.im['SRS']
        self.na=self.im['NWA']
        self.fridge=self.im['FRIDGE']
        
    def dump_electrons(self, biasDumpV,trapDumpV,wait_time=0):
        self.srs.set_volt(biasCh,biasDumpV)
        self.srs.set_volt(trapCh,trapDumpV)
        time.sleep(wait_time)
        
    def load_electrons(self,biasV,trapV,amplitude,offset,frequency,pulse_length):
        self.srs.set_volt(biasCh,biasV)
        self.srs.set_volt(trapCh,trapV)
        self.filament.setup_driver(amplitude,offset,frequency,pulse_length)
    
    def save_system_state(self,state):
        na.clear_averages()
        na.trigger_single()
        temperature=fridge.get_mc_temperature()
        
    #    #tic()
    #    biasV=srs.get_volt(channel=BiasCh)
    #    trapV=srs.get_volt(channel=TrapCh)
    #    #print "SRS: ", toc()
    #    
    #    #tic()    
    #    temperature=fridge.get_mc_temperature()
    #    #print "Fridge: ", toc()
        
        state={'tpts':t,'biasV':biasV,'trapV':trapV,'temperature':temperature}    
        fname= "%04d_%.3f_%s_%4f_%4f_%4f.csv" % (count,t,prefix,
                                                 biasV,trapV,temperature)
    
        na.averaging_complete()
        #na.set_format('slog')
        if fname is not None:
            na.save_file(os.path.join(datapath,fname))
        data=na.read_data()
        
        #data=take_one(na,os.path.join(datapath,fname))
        addpt(datapath,prefix,data,state)
        time.sleep(delay)