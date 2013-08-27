# -*- coding: utf-8 -*-
"""
Agilent AWG81180A (awg.py)
==========================
:Author: David Schuster
"""

from numpy import *
from slab.instruments import *
import time

class ArbitraryWaveformGenerator(Instrument):
    def __init__(self,name,address='',enabled=True):
        pass
    
    def get_channels(self):
        return self.channels
        
    def get_channel(self,index):
        return self.channels[index]
    
class AWGChannel:
    def __init__(self,name,awg,index):
        self.awg=awg
        self.name=name
        self.index=index
        
        self.write=self.awg.write
        self.read=self.awg.read
        self.query=self.awg.query
    
    def set_name(self,name):
        self.name=name
        
    def get_name(self):
        return name
        
"""
AWGChannel properties
        channelName;			%
        amplitude;              %	       
        analogHigh;             %	
        analogLow;              %	
        DACResolution;          %	
        delay;                 	%
        enabled;				%
        lowpassFilterFrequency;	%
        marker1High;			%	
        marker1Level;			%
        marker1Low;				%
        marker1Offset;			%
        marker1Amplitude;		%
        marker2High;			%
        marker2Level;			%
        marker2Low;				%
        marker2Offset;			%
        marker2Amplitude;		%
        name;					%
        outputWaveformName;		%
        offset;					%
        skew;					%	
	  phase; 					%
"""