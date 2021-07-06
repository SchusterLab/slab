# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:33:58 2018

@author: Josie Meyer (jcmeyer@stanford.edu)

This file exists solely for integration between the Keysight code I wrote in
the slab/instruments/keysight folder and the instrumentmanager class. It
essentially aliases the KeysightChassis class.
"""
import keysight.KeysightLib as key

class keysightM3xxxA(key.KeysightChassis):
    
    def __init__(self, name, address):
        self._name = name
        self._address = address
        
        self.