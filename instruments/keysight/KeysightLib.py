# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 2018

@author: Josephine Meyer (jcmeyer@stanford.edu)

Schuster lab Python libraries for interfacing with Keysight modules. Use this
library code instead of the manufacturer-provided code whenever possible
because the manufacturer-provided libraries are full of bugs and gaps in
documentation, lack comments, and appear to be ported directly from C.
"""

'''
Note on naming conventions:
    Chassis are named by their number, i.e. 1, 2, etc.
    Modules are named by two digits. The first corresponds to the chassis,
       the second to the slot (where 0 stands for 10).
       i.e. "14" is chassis 1 slot 4, "20" is chassis 2 slot 10, etc.
    Individual ports are named by three digits, or two digits, one letter.
       The first digit is the chassis, the second is the slot, the third is the
       channel. T is the trigger port, C is the clock port.
       i.e. "132" is chassis 1 slot 3 channel 2, "20C" is chassis 2 slot 10 clock
'''


import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import numpy as np
import keysightSD1 as SD1
from abc import ABCMeta, abstractmethod

# MODULE CONSTANTS
PRODUCT = ""

NUM_CHANNELS = 4
INPUT_IMPEDANCE = 1 #50 ohm
DC_COUPLING = 0 #we are DC coupled
INF_CYCLES = 0 #code for infinite cycles

'''------------------------------------------------------------------------'''
#Short auxiliary classes

class ModuleType:
    '''Identifies the types of Keysight modules that can be installed'''
    INPUT = 1 #M31xxA
    OUTPUT = 2 #M32xxA
        
class InvalidChannelNumberException(Exception):
    '''Exception to use for invalid channel number.'''
    def __init__(self, message):
        self.message = message

class ChannelType:
    '''Identifies the type of each channel'''
    IN = 1
    OUT = 2
    CLK = 3
    TRIG = 4

class DigitalTriggerConfig:
    '''Digital trigger configurations'''
    TRIGGER = 1 #used as trigger
    IN = 1 #digital input, should be same as TRIGGER
    OUT = 0 #digital output


'''-------------------------------------------------------------------------'''

class KeysightChassis:
    '''Class used to represent a Keysight chassis.'''
    
    def __init__(self, chassis_number, modules = {}):
        '''Initializes the chassis
        Params:
            chassis_number: the model of the chassis
            modules: a dictionary of form {slot number:ModuleType}'''
        self._chassis_number = chassis_number
        self._modules = {}
        for slot_number in modules:
            if (modules[slot_number] == ModuleType.INPUT):
                self._modules[slot_number]=KeysightModuleIn(self, slot_number)
            elif modules[slot_number] == ModuleType.OUTPUT:
                self._modules[slot_number]=KeysightModuleOut(self, slot_number)
        
    def getChassisNumber(self):
        '''Returns the number of the chassis'''
        return self._chassis_number
    
    def modulesDict(self):
        '''Returns a dictonary of {slot number:module}'''
        return self._modules
    
    def modules(self):
        '''Returns a list of modules over which to iterate.'''
        return self._modules.values()
    
    def getModule(self, slot_number):
        '''Returns the module corresponding to a specific slot.'''
        return self._modules[slot_number]

    def channelsDict(self):
        '''Returns a dict of {identifier: channel object}'''
        channels_dict = {}
        for module in self.modules():
            for channel in module.channels():
                channels_dict[channel.getIdentifier()] = channel
        return channels_dict
    
    def channels(self):
        '''Returns a list of channel objects that can be iterated over
        in logical order.'''
        channels_list = []
        for module in self.modules():
            for channel in module.channels():
                channels_list.append(channel)
        return channels_list
    
    def inputChannels(self):
        '''Returns a list of input channel objects that can be iterated over
        in logical order.'''
        channels_list = []
        for module in self.modules():
            if type(module) == KeysightModuleIn:
                for channel in module.channels():
                    channels_list.append(channel)
        return channels_list
    
    def outputChannels(self):
        '''Returns a list of output channel objects that can be iterated over
        in logical order.'''
        channels_list = []
        for module in self.modules():
            if type(module) == KeysightModuleOut:
                for channel in module.channels():
                    channels_list.append(channel)
        return channels_list
    
    def getChannel(self, slot_number, channel_number):
        '''Returns the specified channel if it exists.
        Params:
            slot_number: The slot at which the particular channel is found,
               or "C" for clock channel, or "T" for trigger channel
            channel_number: The channel number on the particular module
        Returns: The channel object corresponding to the specified channel'''
        return self._modules[slot_number].getChannel(channel_number)
    
    def getChannelByID(self, identifier):
        '''Returns the specified channel if it exists.
        Params:
            identifier: The 3 character string identifier that uniquely defines
            the channel, as specified above
        Returns: The channel object corresponding to the specified channel'''
        if identifier[0] != str(self._chassis_number):
            raise ValueError("called wrong chassis")
        else:
            slot_number = int(identifier[1])
            channel_number = int(identifier[2])
            if slot_number == 0:
                slot_number = 10
            return self.getChannel(slot_number, channel_number)


'''-----------------------------------------------------------------------'''

class KeysightModule:
    '''Class that corresponds to a single Keysight module slot of type 
       M31xxA or M32xxA. Could extend to M33xxA without too much effort.
       Abstract base class. Do not instantiate directly; instead instantiate
       daughter classes specific to the module type.'''
    
    NUM_CHANNELS = 4 #the number of numbered channels on each module
    
    __metaclass__ = ABCMeta
    
    def __init__(self, chassis, slot_number):
        '''Initializes the Keysight module.
        Params:
            chassis: The chassis where the module is located
            slot_number: The slot on the chassis where the module is located
            module_type: A code indicating whether the module is used for input
               or output.
        '''
        self._chassis = chassis
        self._slot_number = slot_number
        self._channels = {}
            
        #add trigger and clock channels
        self._channels["C"] = ClkChannel(self)
        self._channels["T"] = TrigChannel(self)
        
        
    def chassis(self):
        '''Returns the chassis where the module can be found.'''
        return self._chassis
        
    def getSlotNumber(self):
        '''Returns the slot number within the chassis where the module can
           be found.'''
        return self._slot_number
    
    def getIdentifier(self):
        '''Returns a string indicating the unique name of the module,
           as described above under naming conventions.'''
        return (str(self._chassis.getChassisNumber()) + 
            str(self._slot_number % 10))
    
    def getChannel(self, channel_number):
        '''Returns a specific channel.
        Params:
            channel_number: the number of the channel on the module'''
        return self._channels[channel_number]
    
    def channelsDict(self):
        '''Returns a dictionary of {channel number: channel object}'''
        return self._channels
    
    def channels(self):
        '''Returns a list of channel objects that can be iterated over'''
        return self._channels.values()
    
    @staticmethod
    def _channelsToMask(*channels):
        '''Converts a list of channels to a mask to feed into native functions.
        No need to call directly.
        Params:
            Channels: Any number of channel numbers in same module
        Returns: The mask'''
        mask = 0
        for c in channels:
            if isinstance(c, KeysightChannel): #if you pass in channel object
                mask += 2 * c.getChannelNumber()
            else: #if you pass in integers
                mask += 2 * c
        return mask     
    

'''-------------------------------------------------------------------------'''

class KeysightModuleIn(KeysightModule, SD1.SD_AIN):
    '''Class that corresponds to a single Keysight module slot of type 
       M31xxA. Also inherits methods from SD1.SD_AIN, but these should
       not have to be called directly by users of this library.'''
    
    
    def __init__(self, chassis, slot_number):
        '''Initializes the Keysight module.
        Params:
            chassis: The chassis where the module is located
            slot_number: The slot on the chassis where the module is located
            module_type: A code indicating whether the module is used for input
               or output.
        '''
        SD1.SD_AIN.__init__(self)
        KeysightModule.__init__(self, chassis, slot_number)
        
        for i in range(1, NUM_CHANNELS):
            self._channels[i] = InputChannel(self, i)
        
        module_in_ID = self.openWithSlot(PRODUCT, 
                        self._chassis.getChassisNumber(), self._slot_number)

        if module_in_ID < 0:
            print("Error opening module IN - error code:", module_in_ID, 
                      "Slot:", slot_number)
        else:
            print("===== MODULE IN =====")
            print("ID:\t\t", module_in_ID)
            print("Product name:\t", self.getProductName())
            print("Serial number:\t", self.getSerialNumber())
            print("Chassis:\t", self.getChassis())
            print("Slot:\t\t", self.getSlot())
            print()

    
    def __str__(self):
        '''Returns a string describing the module'''
        return ("Chassis: " + str(self._chassis.getChassisNumber()) + "   Slot: " +
                str(self._slot_number) + "    Module Type: M31xxA")
        
    def triggerChannels(self, *channels):
        '''Triggers multiple channels on same module simultaneously
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error messages'''
        mask = KeysightModule._channelsToMask(channels)
        return self.DAQtriggerMultiple(mask)
    
    def startChannels(self, *channels):
        '''Starts multiple channels simultaneously on the same module. Can also
        be used as alternative to calling start() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.DAQstartMultiple(mask)
    
    def stopChannels(self, *channels):
        '''Stop multiple channels simultaneously on the same module. Can also
        be used as alternative to calling stop() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.DAQstopMultiple(mask)
    
    def pauseChannels(self, *channels):
        '''Pauses multiple channels simultaneously on the same module. Can also
        be used as alternative to calling pause() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.DAQpauseMultiple(mask)
    
    def resumeChannels(self, *channels):
        '''Resumes multiple channels simultaneously on the same module after
        pause. Can also be used as alternative to calling resume() on
        individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.DAQstartMultiple(mask)
    
    def close(self):
        '''Closes module and releases any resources still in use.
        Returns: any error messages'''
        return SD1.SD_AIN.close(self)
    

        
    
'''-------------------------------------------------------------------------'''

class KeysightModuleOut(KeysightModule, SD1.SD_AOU):
    '''Class that corresponds to a single Keysight module slot of type 
       M32xxA. Also inherits methods from SD1.SD_AOU, but these should
       not have to be called directly by users of the library'''
    
    
    def __init__(self, chassis, slot_number):
        '''Initializes the Keysight module.
        Params:
            chassis: The chassis where the module is located
            slot_number: The slot on the chassis where the module is located
            module_type: A code indicating whether the module is used for input
               or output.
        '''
        SD1.SD_AOU.__init__(self)
        KeysightModule.__init__(self, chassis, slot_number)
        

        module_out_ID = self.openWithSlot(PRODUCT, chassis.getChassisNumber(),
                                          slot_number)

        if module_out_ID < 0:
            print("Error opening module OUT - error coode:", module_out_ID)
        else:
            print("===== MODULE OUT =====")
            print("Module opened:", module_out_ID)
            print("Module name:", self.getProductName())
            print("slot:", self.getSlot())
            print("Chassis:", self.getChassis())
            print()
            self.triggerIOconfig(1)
        
        for i in range(1, NUM_CHANNELS):
            self._channels[i] = OutputChannel(self, i)
    
    def __str__(self):
        '''Returns a string describing the module'''
        return ("Chassis: " + str(self.chassis.getChassisNumber()) + "   Slot: " +
                str(self._slot_number) + "    Module Type: M32xxA")
        
    def loadWaveform(self, waveform):
        '''Loads a waveform into the module. Equivalent to calling loadToAWG
        on the waveform object itself.
        Params:
            waveform: the waveform object
        Returns: any errors'''
        wave = SD1.SD_Wave()
        wave.newFromArrayDouble(0, waveform.getBaseArray())
        return self.waveformLoad(wave, waveform.getWaveformNumber(), 0)
    
    def close(self):
        '''Closes module and releases any resources still in use.
        Returns: any error messages'''
        return SD1.SD_AOU.close(self)
    
    def triggerChannels(self, *channels):
        '''Triggers multiple channels on same module simultaneously
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error messages'''
        mask = KeysightModule._channelsToMask(channels)
        return self.AWGtriggerMultiple(mask)
    
    def startChannels(self, *channels):
        '''Starts multiple channels simultaneously on the same module. Can also
        be used as alternative to calling start() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.AWGstartMultiple(mask)
    
    def stopChannels(self, *channels):
        '''Stop multiple channels simultaneously on the same module. Can also
        be used as alternative to calling stop() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.AWGstopMultiple(mask)
    
    def pauseChannels(self, *channels):
        '''Pauses multiple channels simultaneously on the same module. Can also
        be used as alternative to calling pause() on individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.AWGpauseMultiple(mask)
    
    def resumeChannels(self, *channels):
        '''Resumes multiple channels simultaneously on the same module after
        pause. Can also be used as alternative to calling resume() on
        individual channel.
        Params:
            channels: any number of either channel objects or integers
               corresponding to channel numbers
        Returns: any error message'''
        mask = KeysightModule._channelsToMask(channels)
        return self.AWGstartMultiple(mask)
    
    def clearAll(self):
        '''Deletes all waveforms within the module, even those queued on a 
        particular channel.
        Returns: any errors.'''
        return self.waveformFlush()
        

'''-------------------------------------------------------------------------'''

class KeysightChannel():
    '''Abstract base class representing one channel on a Keysight module. Don't
       instantiate this class directly; instead instantiate a daughter class
       representing the specific functionality of the channel.'''
       
    __metaclass__ = ABCMeta
       
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannel object.
        Params:
            module: The module object where the channel is located
            channel_number: #1-4 to correspond to the standard I/O channels,
                  or "T" for I/O trigger or "C" for clock in/out'''  
        self._module = module
        self._chassis = self._module.chassis()
        if (((type(channel_number) == str) and 
             (channel_number == "C" or channel_number == "T")) 
                or (type(channel_number) == int and channel_number > 0 
                    and channel_number <= KeysightModule.NUM_CHANNELS)):
            self._channel_number = channel_number
        else:
            raise InvalidChannelNumberException("Invalid Channel Number")          
    
    def chassis(self):
        '''Returns the chassis object where the channel is located.'''
        return self._chassis
    
    def getModule(self):
        '''Returns the module where the channel is located.'''
        return self._module
    
    def getChannelNumber(self):
        '''Returns the channel number.
        Note: the string "T" indicates the trigger in/out channel
        Note: the string "C" indicates the clock out channel.'''
        return self._channel_number
    
    def getIdentifier(self):
        '''Returns the unique identifier of the channel, according to naming
        scheme discussed in KeysightChassis documentation.'''
        return (str(self._chassis.getChassisNumber()) + 
                str(self._module.getSlotNumber() % 10) + str(self.getChannelNumber()))
    
    @abstractmethod
    def getChannelType(self):
        '''Returns the type of channel according to the codes in ChannelType
        class. Each subclass should implement this method.'''
        pass
    
    def __str__(self):
        '''Returns a string describing the channel. Can be overridden
        by underlying methods.'''
        return ("Chassis: " + str(self._chassis.getChassisNumber()) +
                "    Channel: " + str(self.getChannelNumber()) + "    Type: " 
                + str(self.getChannelType()))
    

'''-------------------------------------------------------------------------'''

class InputChannel(KeysightChannel):
    '''Class representing a Keysight standard input channel.'''
    
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannelIn object.
        Params:
            module: the module where the channel is located
            channel_number: the number of the channel
        '''
        KeysightChannel.__init__(self, module, channel_number)
    
    def configure(self, points_per_cycle = 2000, cycles = 1, full_scale = 100,
                  delay = 0, trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
                  trigger_extern_source = 
                     SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                  trigger_extern_behavior = 
                     SD1.SD_TriggerBehaviors.TRIGGER_RISE, prescaler = 1):
        '''Configures channel for data acquisition.
        Params:
            points_per_cycle: points to acquire per DAQ trigger cycle
            cycles: number of cycles to acquire data, or 0 for unlimited
            full_scale: full scale of output data
            delay: delay after trigger/command after which to begin taking data
            trigger_mode: indicates how the DAQ process should be initiated
               (i.e. automatically or at receipt of trigger). Default is
               externally triggered
            trigger_extern_source: if externally triggered, the channel where
               the trigger will be received. Default is the trigger port on
               same module, but can also be triggered via PXI
            trigger_extern_behavior: if externally triggered, determines when
               trigger is detected based on input signal. Default is when
               signal rises. If using digital trigger, want TRIGGER_HIGH
            prescaler (int): Reduces effective sampling by factor 1/prescaler
        Returns:
            A tuple of error messages from the component operations that can
            be analyzed for diagnostic purposes
            '''
        err0 = self._module.channelInputConfig(self._channel_number, 
                            full_scale, INPUT_IMPEDANCE, DC_COUPLING)
        err1 = self._module.DAQconfig(self._channel_number, points_per_cycle, 
                            cycles, delay, trigger_mode)
        err2 = 0
        if (trigger_mode == SD1.SD_TriggerModes.EXTTRIG):
           err2 = self._module.DAQtriggerExternalConfig(self._channel_number,
                    SD1.SD_TriggerExternalSources.TRIGGER_EXTERN)
        err3 = self._module.channelPrescalerConfig(self._channel_number,
                                                   prescaler)
        return (err0, err1, err2, err3)
    
    def getChannelType(self):
        '''Returns the channel type.'''
        return ChannelType.IN
    
    def __str__(self):
        '''Returns a string describing the channel.'''
        return KeysightChannel.__str__(self)
    
    def flush(self):
        '''Flushes the data acquired previously.
        Returns: any error message'''
        return self._module.DAQflush(self._channel_number)
    
    def start(self):
        '''Starts the data acquisition.
        Returns: any error message'''
        return self._module.DAQstart(self._channel_number)
    
    def stop(self):
        '''Stops the data acquisition.
        Returns: any error message'''
        return self._module.DAQstop(self._channel_number)
    
    def resume(self):
        '''Resumes data acquisition.
        Returns: any error message'''
        return self._module.DAQresume(self._channel_number)
    
    def pause(self):
        '''Pauses the data acquisition, to be resumed later.
        Returns: any error message.'''
        return self._module.DAQpause(self._channel_number)
    
    def read(self, num_points, timeout=0):
        '''Reads the data acquired.
        Params:
            num_points: The number of data points to collect
            timeout: time in 10s of ns to wait for data, or 0 for infinite
        Returns: The data as an array, or any errors'''
        return self._module.DAQread(self._channel_number, num_points, timeout)
    
    def trigger(self):
        '''Triggers data acquisition if in PXI mode.
        Returns: any error messages'''
        return self._module.DAQtrigger(self._channel_number)
        

'''-------------------------------------------------------------------------'''

class OutputChannel(KeysightChannel):
    '''Class representing a Keysight standard output channel.'''
    
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannelOut object.
        Params:
            module: the module where the channel is located
            channel_number: the number of the channel
        '''

        KeysightChannel.__init__(self, module, channel_number)
        
    def configure(self, channel_amplitude = 0.4, trigger_extern_source = 
                     SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                  trigger_extern_behavior = 
                     SD1.SD_TriggerBehaviors.TRIGGER_RISE):
        '''Flushes and configures the AWG.
        Params:
            channel_amplitude: the output max amplitude of the channel
            trigger_extern_source: if externally triggered, the channel where
               the trigger will be received. Default is the trigger port on
               same module, but can also be triggered via PXI
            trigger_extern_behavior: if externally triggered, determines when
               trigger is detected based on input signal. Default is when
               signal rises. If using digital trigger, want TRIGGER_HIGH
        Returns:
            A tuple of error messages from the component operations that can
            be analyzed for diagnostic purposes
        '''
        err0 = self.flush()
        err1 = self._module.channelAmplitude(self._channel_number, 
                                             channel_amplitude)
        err2 = self._module.channelWaveShape(self._channel_number,
                                             SD1.SD_Waveshapes.AOU_AWG)
        err3 = self._module.AWGtriggerExternalConfig(self._channel_number,
                        trigger_extern_source, trigger_extern_behavior)
        return (err0, err1, err2, err3)
        
    def getChannelType(self):
        '''Returns the channel type.'''
        return ChannelType.OUT
    
    def __str__(self):
        '''Returns a string describing the channel.'''
        return KeysightChannel.__str__(self)
    
    def flush(self):
        '''Flushes the channel of any pre-existing queued waveforms.
        Returns:
            Any error messages'''
        return self._module.AWGflush(self._channel_number)
    
    def queue(self, waveform, trigger_mode = 0, start_delay = 0, 
              cycles = 1):
        '''Adds a pre-loaded waveform to the queue. Will return error if not
        loaded to module.
        Params:
            waveform: The waveform object, or the waveform number
            trigger_mode: Indicates whether AWG will wait for trigger. Default
            is immediate playback after previous waveform or start signal
            start_delay: Delay after trigger/previous waveform in 10s of ns
            cycles: Number of cycles to play, or KeysightLib.INF_CYCLES for
            infinite.
        Returns: any errors'''
        
        if type(waveform) == Waveform: 
            #if we pass in the actual waveform object
            waveform = waveform.getWaveformNumber()
        return self._module.AWGqueueWaveform(self._channel_number,
                        waveform, trigger_mode,
                        start_delay, cycles, 0)
    
    def start(self):
        '''Starts the waveform queue.
        Returns: any error message'''
        return self._module.AWGstart(self._channel_number)
    
    def stop(self):
        '''Stops the AWG.
        Returns: any error message'''
        return self._module.AWGstop(self._channel_number)
    
    def resume(self):
        '''Resumes the AWG after pause.
        Returns: any error message'''
        return self._module.AWGresume(self._channel_number)
    
    def pause(self):
        '''Pauses the AWG, to be resumed later.
        Returns: any error message.'''
        return self._module.AWGpause(self._channel_number)
    
    def trigger(self):
        '''Triggers data acquisition if in PXI mode.
        Returns: any error messages'''
        return self._module.AWGtrigger(self._channel_number)
    
    def reset(self):
        '''Resets the channel to start of queue.
        Returns: any error messages'''
        return self._module.AWGreset(self._channel_number)
    
    def isRunning(self):
        '''Checks whether the channel is currently running.
        Returns: 1/0 for running/not running, negative number for error'''
        return self._module.AWGisRunning(self._channel_number)

    
'''-------------------------------------------------------------------------'''

class ClkChannel(KeysightChannel):
    '''Class representing a Keysight clock in/out channel.'''
    
    def __init__(self, module):
        '''Initializes the KeysightChannelClk object.
        Params:
            module: the module where the channel is located'''
        KeysightChannel.__init__(self, module, "C")
        
    def getChannelType(self):
        '''Returns the channel type.'''
        return ChannelType.CLK
    
    def __str__(self):
        '''Returns a string describing the channel.'''
        return KeysightChannel.__str__(self)
    
'''-------------------------------------------------------------------------'''

class TrigChannel(KeysightChannel):
    '''Class representing a Keysight trigger in/out channel.'''
    
    def __init__(self, module, direction = DigitalTriggerConfig.TRIGGER):
        '''Initializes the KeysightChannelTrig object.
        Params:
            module: the module where the channel is located
            direction: whether the channel is used for triggering, input,
               or output'''
        KeysightChannel.__init__(self, module, "T")
        
    def getChannelType(self):
        '''Returns the channel type.'''
        return ChannelType.TRIG
    
    def configure(self, direction):
        '''Sets whether the trigger channel should be an input (for triggering
        or data acquisition) or function as a digital output. Direction is
        given by DigitalTriggerConfig'''
        return self._module.triggerIOconfig(direction)
    
    def read(self):
        '''Reads the trigger channel.
        Returns:
            0 or 1 for off/on
            Negative number for error'''
        return self._module.triggerIOread()
    
    def __str__(self):
        '''Returns a string describing the channel.'''
        return KeysightChannel.__str__(self)
    
    def write(self, value):
        '''Writes 0 or 1 to the trigger channel.
        Params:
            Value: int (0 or 1) or boolean corresponding to value to write
        Returns: Any errors'''
        return self._module.triggerIOwrite(int(value))
    
'''-------------------------------------------------------------------------'''

class Waveform():
    '''Class representing a waveform, built off numpy. Use this instead of the
    native SD_Wave class for greater functionality and cleaner implementation.'''
    
    
    def __init__(self, arr = [], waveform_number = None):
        '''Initializes the waveform object.
        Params:
            arr: A list or numpy array of floats/decimals (-1 to 1)
               corrresponding to the waveform shape. If empty or
               omitted, generates a waveform of length 0, which will
               not play
            waveform_number: A unique identifier assigned to the waveform.
               User is responsible for preventing naming collisions.
               If None, program will auto-assign waveform numbers that prevent
               naming collisions automatically if all are assigned this way.
        '''
        if (waveform_number is None):
            Waveform._waveform_number_counter += 1
            self._waveform_number = Waveform._waveform_number_counter
        else:
            self._waveform_number = waveform_number
        self._arr = np.array(arr) #converts to numpy array if list passed in
    
    @classmethod #factory method
    def generate(cls, form, num_points, waveform_number = None,
                 scale = 1, sigma = 10, duty_cycle = 1,
                 append_zero = True):
        '''Generates the desired waveform.
        Params:
            form: the type of waveform, as a string
                "sin": sin wave
                "half_sin": first half-period of the sin wave
                "gauss": a gaussian
                "sq": a square wave
                "tri": a triangular wave
                "ramp": a ramp
            num_points: the number of points to include in waveform
            waveform_number: the unique number identifying the waveform,
               or "None" to have waveform number auto-assigned to prevent
               naming conflicts
            scale: an overall scale factor between -1 and 1
            sigma: for Gaussian, standard deviation in # data points
            duty_cycle: duty cycle for square wave
            append_zero: if true, forces last element to be 0 for ramp
        Returns:
            The waveform as a np array
        '''
        if form == "sin":
            arr = cls.generateSin(num_points, scale)
        elif form == "half_sin":
            arr = cls.halfSin(num_points, scale)
        elif form == "gauss":
            arr = cls.generateGauss(num_points, scale, sigma)
        elif form == "sq":
            arr = cls.generateSquare(num_points, scale, duty_cycle)
        elif form == "ramp":
            arr = cls.generateRamp(num_points, scale, append_zero)
        elif form == "tri":
            arr = cls.generateTriangle(num_points, scale)
        else:
            raise ValueError("Invalid waveform input")
            
        return Waveform(arr, waveform_number)
            
    def setWaveformNumber(self, waveform_number):
        '''Sets the waveform number.
        Params:
            waveform_number: the new waveform number
            
        WARNING: Calling this method if you have opted to set waveform numbers
        automatically can lead to naming collisions and unexpected behavior'''
        self._waveform_number = waveform_number
    
    def getWaveformNumber(self):
        '''Returns: the waveform number of the selected waveform'''
        return self._waveform_number
    
    def loadToAWG(self, module):
        '''Loads the wave onto an AWG module.
        Params:
            module: The module to which to load the waveform.
        Returns: any errors'''
        if not isinstance(module, OutputChannel):
            raise ValueError("Cannot load to non-output module")
        wave = SD1.SD_Wave()
        wave.newFromArrayDouble(0, self.arr)
        return module.waveformLoad(wave, self._waveform_number, 0)
    
    def getBaseArray(self):
        '''Returns the base array underlying the waveform object'''
        return self._arr
    
    @staticmethod
    def generateSin(num_points, scale):
        '''Generates one period of a sin wave.
        Params:
            num_points: the number of points in the waveform
            scale: the maximum value of the signal. - sign inverts
        Returns: a numpy array representing the waveform
        '''
        arr = []
        d_theta = 2 * np.pi / (num_points)
        for i in range(num_points-1):
            arr.append(scale * np.sin(d_theta * (1+i)))
        arr.append(0)
        return np.array(arr)
    
    @staticmethod
    def halfSin(numPoints, scale):
        '''Generates the first half-period of a sin wave.
        Params:
            num_points: the number of points in the waveform
            scale: the extremal value of the signal. - sign inverts
        Returns: a numpy array representing the waveform
            '''
        arr = []
        d_theta = np.pi / numPoints
        for i in range(numPoints-1):
            arr.append(scale * np.sin(d_theta * (1+i)))
        arr.append(0)
        return np.array(arr)
            
    @staticmethod
    def gaussian(x, mu, sigma):
        '''Function for a generic Gaussian.
        Params:
            x: independent variable
            mu: mean
            sigma: standard deviation 
        Returns: Gaussian evaluated at x
        '''
        return np.exp(-((x-mu)/(np.sqrt(2) * sigma))**2)
    
    @staticmethod
    def generateGauss(num_points, scale, sigma):
        '''Generates a Gaussian waveform, with final 0 at end to rezero signal.
        Params:
            num_points: the number of points in the waveform
            scale: the peak value of the signal. - sign inverts
            sigma: the standard deviation in points
        Returns: a numpy array representing the waveform
        '''
        arr = []
        mu = num_points / 2
        for i in range(num_points-1):
            arr.append(scale * Waveform.gaussian(i, mu, sigma))
        arr.append(0)
        return np.array(arr)
    
    @staticmethod
    def generateSquare(num_points, scale, duty_cycle, append_zero = True):
        '''Generates a square waveform.
        Params:
            num_points: the number of points in the waveform
            scale: the maximum value of the signal
            duty_cycle: the duty cycle of the signal.
            append_zero: whether to force a zero at end
        Returns: an array corresponding to the waveform, as closely matching
        duty cycle as allowed for given number of points.
        '''
        n = round(num_points * duty_cycle)
        if (append_zero and n == num_points): #ensures at least 1 zero at end!
            n -= 1
        arr = np.empty(n)
        arr.fill(scale)
        return np.concatenate((arr, np.zeros(num_points - n)))
    
    @staticmethod
    def generateRamp(num_points, scale, append_zero = True):
        '''Generates a rising ramp.
        Params:
            num_points: the number of points in the waveform
            scale: the maximum value of the signal
            append_zero: whether to place a final zero at end to rezero signal
        Returns: an array corresponding to the waveform
        '''
        if append_zero:
            return np.concatenate((np.linspace(0, scale, num = num_points)[1:], 
                                   np.array([0])))
        return np.linspace(0, scale, num = num_points + 1)[1:]
    
    @staticmethod
    def generateTriangle(numPoints, scale):
        ''' Generates a triangular wave pulse.
        Params:
            num_points: the number of points in the waveform
            scale: the maximum value of the signal
        Returns: an array corresponding to the waveform
        '''
        if (numPoints % 2 == 1): #numPoints is odd
            return np.concatenate((
                    np.linspace(0, scale, num = (numPoints - 1) // 2, endpoint=False),
                    np.linspace(scale, 0, num = (numPoints + 1) // 2)))
        elif (numPoints % 2 == 0): #numPoints is even
            return Waveform.generateTriangle(numPoints + 1, scale)[1:]

    #class variable
    _waveform_number_counter = 0
    #incremented every time a new waveform is generated to avoid naming collision
    
    
        
    
        
    
    