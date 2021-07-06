# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:01:14 2018

@author: Josie Meyer (jcmeyer@stanford.edu)

Allows existing pulse sequence code to be played out on the Keysight M32xxA
modules. This is the 'dumb' implementation that does not take advantage of the
full capability of the HVI and only uses it for synching. Moreover, individual
channels are all activated simultaneously; they may be manually muted if
necessary.

"""

import KeysightLib as key
import keysightSD1 as SD1
import json

class KeysightPulseSequence():
    '''Class representing a pulse sequence as customized for a Keysight M32xxA
        module.'''
    
    def __init__(self):
        '''Initializes the pulse sequence object.'''
        self._dict = {}
        
    def addWaveform(self, chassis_number, slot_number, channel_number,
                    waveform_array):
        '''Adds one channel's waveform to the pulse sequence.
        Params:
            chassis_number: The number of the chassis where the desired channel
                is housed.
            slot_number: The number of the slot where the desired channel is
                housed.
            channel_number: The channel number of the desired channel
            waveform: The array representing the waveform'''
        self._dict[key.Tools.serializeChannel(chassis_number, 
                    slot_number, channel_number)] = waveform_array
    
    def writeToFile(self, filename):
        '''Writes the pulse sequence to a file that can be opened.'''
        with open(filename, 'w') as f:
            json.dump(self._dict, f)
            
    @staticmethod
    def loadFromFile(chassis, filename):
        '''Loads the pulse sequence from a file into the AWG's.
        Params:
            chassis: The chassis object where the waveforms are to be loaded and
                queued.
            filename: The filename (and path) where the json file containing the
                experiment is stored.
        '''
        pulse_sequence = KeysightPulseSequence()
        with open(filename) as f:
            pulse_sequence._dict = json.load(f)
        pulse_sequence.loadToExperiment(chassis)
        
    def loadToExperiment(self, chassis):
        '''Loads a pulse sequence object directly into the AWG's.
        Params:
            chassis: The chassis object where the waveforms are to be loaded and
                queued.'''
        for serialized_channel in self._dict:
            (chassis_number, slot_number,
                 channel_number) = key.Tools.deserializeChannel(serialized_channel)
            channel = chassis.getChannel(slot_number, channel_number)
            waveform = key.Waveform(self._dict[serialized_channel])
            waveform.loadToModule(channel.getModule())
            waveform.queue(channel, trigger_mode = SD1.SD_TriggerModes.SWHVITRIG)
            