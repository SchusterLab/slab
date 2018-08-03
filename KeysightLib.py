# -*- coding: utf-8 -*-
"""
Created on July 26, 2018

@author: Josephine Meyer (jcmeyer@stanford.edu)

Library classes for Python interface with Keysight M31xxA and M32xxA modules.
Use instead of the native keysightSD1 classes for better functionality and
compatibility with our existing code base.
"""

import keysightSD1 as SD1
import numpy as np


'''Short auxiliary classes and constants'''

class KeysightConstants:
    '''Useful general constants'''
    NUM_CHANNELS = 4 #number of modules
    PRODUCT = ""
    MASK_ALL = 30 #2**1 + 2**2 + 2**3 + 2**4 -- mask for all 4 channels
    INPUT_IMPEDANCE = 1 #50 ohm
    DC_COUPLING = 0 #we are DC coupled
    INFINITY = 0 #Blame Keysight for this one!

class ModuleType:
    '''Integer code representing the module type of a given module.'''
    INPUT = 1
    OUTPUT = 0
    M3102A = INPUT #convenient aliases
    M3201A = OUTPUT
    M3202A = OUTPUT
    
class ChannelNumber:
    '''Generates effective channel numbers for the trigger and clock channels
    to enable them to be treated on par with numbered channels.'''
    TRIG = 101
    TRIGGER = TRIG
    CLK = 102
    CLOCK = CLK
    
class TriggerIODirection:
    '''Trigger channel in or out'''
    IN = 1
    OUT = 0
    
class Tools:
    '''Useful static methods used in implementation. Should be no reason for
    end user to call these methods directly.'''
    
    @staticmethod
    def _channelsToMask(*channels):
        '''Converts list of channels to a mask accepted by several native
            methods.
        Params:
            *channels: Any number of channel numbers from same module
        Returns: A mask representing the channels. User should have no need
            to call this method directly.'''
        mask= 0
        for c in channels:
            mask += 2**c
        return mask

"-----------------------------------------------------------------------------"

class KeysightChassis:
    '''Class representing a Keysight chassis.'''
    
    def __init__(self, chassis_number, modules_dict = {}):
        '''Initializes the Keysight Chassis object.
        Params:
            chassis_number: The chassis number of the chassis we are creating
            modules_dict: A dictionary representing the modules formatted as
               {slot number: ModuleType}
        '''
        self._chassis_number = chassis_number
        self._modules = {}
        
        #initialize modules
        for module_number in modules_dict:
            if modules_dict[module_number] == ModuleType.INPUT:
                self._modules[module_number] = KeysightModuleIn(
                        self, module_number)
            elif modules_dict[module_number] == ModuleType.OUTPUT:
                self._modules[module_number] = KeysightModuleOut(
                        self, module_number)
                
    def chassisNumber(self):
        '''Returns the chassis number of the Keysight Chassis object'''
        return self._chassis_number
    
    def modulesDict(self):
        '''Returns a dictionary mapping the slot number of each active slot
            to its corresponding module object. {slot number: KeysightModule}
        '''
        return self._modules
    
    def modulesList(self):
        '''Returns a list of the active modules that is iterable. Note that the
        modules may not be in order.'''
        return self._modules.values()
    
    def getModule(self, slot_number):
        '''Gets the module object corresponding to any module connected to the
            chassis.
        Params:
            slot_number: The slot number of the module to retrieve
        Returns: The module object corresponding to the given slot'''
        return self._modules[slot_number]
    
    def getChannel(self, slot_number, channel_number):
        '''Gets a channel associated with any module connected to the chassis.
        Params:
            slot_number: The slot number containing the module containing the
                channel.
            channel_number: The channel number to retrieve. Should be an
                integer 1-4, or ChannelNumber.TRIG/ChannelNumber.CLK
        Returns: The channel object corresponding to the given channel'''
        return self._modules[slot_number].getChannel(channel_number)
    
    def closeAll(self):
        '''Closes all modules associated with the chassis.'''
        for module in self.modulesList():
            module.close()
            
    def clearAll(self):
        '''Clears all modules associated with the chassis.'''
        for module in self.modulesList():
            module.clearAll()
            
    def __str__(self):
        '''Returns a string representation of the chassis and its modules.'''
        return ("Keysight chassis. Chassis number: " +
                str(self._chassis_number) + "\n Modules:" + str(self._modules))

"----------------------------------------------------------------------------"

class KeysightModule:
    '''Abstract base class representing a Keysight module (M31xxA or M32xxA). 
    DO NOT INSTANTIATE THIS CLASS DIRECTLY. Instead, instantiate one of the
    daughter classes KeysightModuleIn or KeysightModuleOut.'''
    
    def __init__(self, chassis, slot_number):
        '''Initializes the KeysightModule object.
        Params:
            chassis: The chassis object corresponding to the chassis where
                the module is housed.
            slot_number: The number of the slot where the module is housed'''
        self._chassis = chassis
        self._slot_number = slot_number
        self._channels = {}
        
        #configure channels common to all module types
        self._channels[ChannelNumber.CLK] = KeysightChannelClk(self)
        self._channels[ChannelNumber.TRIG] = KeysightChannelTrig(self)
        
    def chassis(self):
        '''Returns the chassis object where the module is housed'''
        return self._chassis
    
    def slotNumber(self):
        '''Returns the slot number where the module is housed'''
        return self._slot_number
    
    def channelsDict(self):
        '''Returns a dictionary mapping the channel number of each channel
            to its corresponding channel object. 
            {channel number: KeysightChannel}
            Note: Trig and Clk channels assigned channel number through
                ChannelNumber class constants'''
        return self._channels
    
    def channelsList(self):
        '''Returns a list of the channels in the module. Note: channels are not
        guaranteed to be returned in logical order.'''
        return self._channels.values()
    
    def getChannel(self, channel_number):
        '''Returns a specific channel within the module.
        Params:
            channel_number: The channel number of the desired channel. Use
            ChannelNumber class for Clk and Trg channel numbers.
        Returns: The desired channel'''
        return self._channels[channel_number]
    


"----------------------------------------------------------------------------"

class KeysightModuleIn(KeysightModule, SD1.SD_AIN):
    '''Class representing a Keysight input module (M31xxA). Inherits from
    the KeysightModule parent class as well as the native SD_AIN class for
    implementation purposes. However, there should be no reason for user 
    to directly call methods from SD_AIN.'''
    
    def __init__(self, chassis, slot_number):
        '''Initializes the ModuleIn object.
        Params:
            chassis: The chassis object where the module is housed
            slot_number: The slot where the module is housed
        '''
        
        
        SD1.SD_AIN.__init__(self)
        module_in_ID = self.openWithSlot(KeysightConstants.PRODUCT,
                                         chassis.chassisNumber(), slot_number)
        
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
        
        KeysightModule.__init__(self, chassis, slot_number)
        
        #initialize channels
        for i in range(1, 1 + KeysightConstants.NUM_CHANNELS):
            self._channels[i] = KeysightChannelIn(self, i)
            
    def getModuleType(self):
        '''Returns a constant corresponding to the module type as given in 
        class ModuleType.'''
        return ModuleType.INPUT
    
    def clearAll(self):
        '''Clears the data acquisition buffers on all channels.'''
        return self.DAQflushMultiple(KeysightConstants.MASK_ALL)
    
    def startChannels(self, *channels):
        '''Starts channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.DAQstartMultiple(Tools._channelsToMask(channels))
    
    def stopChannels(self, *channels):
        '''Stops channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.DAQstopMultiple(Tools._channelsToMask(channels))
    
    def pauseChannels(self, *channels):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.DAQpauseMultiple(Tools._channelsToMask(channels))

    def resumeChannels(self, *channels):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling resume() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.DAQresumeMultiple(Tools._channelsToMask(channels))
    
    def triggerChannels(self, *channels):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.DAQresumeMultiple(Tools._channelsToMask(channels))
        
    
    def startAll(self):
        '''Starts ALL channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Returns: Any error message'''
        return self.DAQstartMultiple(KeysightConstants.MASK_ALL)
    
    def stopAll(self):
        '''Stops ALL channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Returns: Any error message'''
        return self.DAQstopMultiple(KeysightConstants.MASK_ALL)
    
    def pauseAll(self):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Returns: Any error message'''
        return self.DAQpauseMultiple(KeysightConstants.MASK_ALL)

    def resumeAll(self):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling resume() on individual channels.
        Returns: Any error message'''
        return self.DAQresumeMultiple(KeysightConstants.MASK_ALL)
    
    def triggerAll(self):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Returns: Any error message'''
        return self.DAQtriggerMultiple(KeysightConstants.MASK_ALL)
    
    def close(self):
        '''Closes the current module.
        Returns: any error messages.'''
        self.clearAll()
        return SD1.SD_AIN.close(self)
    
    def __str__(self):
        '''Returns a string representation of the module.'''
        return ("Module in. Chassis = " + str(self.chassis().chassisNumber()) +
                ", Slot = " + str(self.slotNumber()))
    
"-----------------------------------------------------------------------------"

class KeysightModuleOut(KeysightModule, SD1.SD_AOU):
    '''Class representing a Keysight output module (M32xxA). Inherits from
    the KeysightModule parent class as well as the native SD_AOU class for
    implementation purposes. However, there should be no reason for user 
    to directly call methods from SD_AOU.'''
    
    def __init__(self, chassis, slot_number):
        '''Initializes the ModuleIn object.
        Params:
            chassis: The chassis object where the module is housed
            slot_number: The slot where the module is housed
        '''
        SD1.SD_AOU.__init__(self)
        module_out_ID = self.openWithSlot(KeysightConstants.PRODUCT, 
                                          chassis.chassisNumber(), slot_number)

        if module_out_ID < 0:
            print("Error opening module OUT - error coode:", module_out_ID)
        else:
            print("===== MODULE OUT =====")
            print("Module opened:", module_out_ID)
            print("Module name:", self.getProductName())
            print("slot:", self.getSlot())
            print("Chassis:", self.getChassis())
            print()
            
        KeysightModule.__init__(self, chassis, slot_number)
        
        #initialize channels
        for i in range(1, 1 + KeysightConstants.NUM_CHANNELS):
            self._channels[i] = KeysightChannelOut(self, i)
            
    def getModuleType(self):
        '''Returns a constant corresponding to the module type as given in 
        class ModuleType.'''
        return ModuleType.OUTPUT
    
    def startChannels(self, *channels):
        '''Starts channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.AWGstartMultiple(Tools._channelsToMask(channels))
    
    def stopChannels(self, *channels):
        '''Stops channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.AWGstopMultiple(Tools._channelsToMask(channels))
    
    def pauseChannels(self, *channels):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.AWGpauseMultiple(Tools._channelsToMask(channels))

    def resumeChannels(self, *channels):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        
        return self.AWGresumeMultiple(Tools._channelsToMask(channels))
    
    def triggerChannels(self, *channels):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        Returns: Any error message'''
        return self.AWGtriggerMultiple(Tools._channelsToMask(channels))
    
    def startAll(self):
        '''Starts ALL channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Returns: Any error message'''
        return self.AWGstartMultiple(KeysightConstants.MASK_ALL)
    
    def stopAll(self):
        '''Stops ALL channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Returns: Any error message'''
        return self.AWGstopMultiple(KeysightConstants.MASK_ALL)
    
    def pauseAll(self):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Returns: Any error message'''
        return self.AWGpauseMultiple(KeysightConstants.MASK_ALL)

    def resumeAll(self):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Returns: Any error message'''
        return self.AWGresumeMultiple(KeysightConstants.MASK_ALL)
    
    def triggerAll(self):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Returns: Any error message'''
        return self.AWGtriggerMultiple(KeysightConstants.MASK_ALL)
    
    def close(self):
        '''Closes the current module.
        Returns: any error messages.'''
        self.clearAll()
        return SD1.SD_AOU.close(self)
    
    def loadWaveform(self, waveform):
        '''Loads a waveform into memory on the specific module.
        Params:
            waveform: The waveform object to load
        Returns: a tuple of error messages'''
        wave = SD1.SD_Wave()
        err1 = wave.newFromArrayDouble(SD1.SD_WaveformTypes.WAVE_ANALOG,
                                waveform.getBaseArray())
        err2 = self.waveformLoad(wave, waveform.getWaveformNumber())
        return (err1, err2)
    
    def clearAll(self):
        '''Stops all AWG's, deletes waveforms from memory, and flushes queues
        of every channel.'''
        self.stopAll()
        return self.waveformFlush()
    
    def __str__(self):
        '''Returns a string representation of the module.'''
        return ("Module out. Chassis = " + str(self.chassis().chassisNumber()) +
                ", Slot = " + str(self.slotNumber()))

"-----------------------------------------------------------------------------"

class KeysightChannel:
    '''Class representing a single channel. Abstract base class. DO NOT
        INSTANTIATE THIS CLASS DIRECTLY. Instead use one of the daughter
        classes below.'''
    
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannel object.
        Params:
            module: The module where the channel is housed
            channel_number: The channel number, including effective channel
                numbers in ChannelNumber class for CLK and TRIG channels.'''
        self._module = module
        self._channel_number = channel_number
        
    def module(self):
        '''Returns the module where the channel is housed'''
        return self._module
    
    def channelNumber(self):
        '''Returns the channel number'''
        return self._channel_number
    
    def chassis(self):
        '''Returns the chassis where the channel is located'''
        return self._module.chassis()
    
"-----------------------------------------------------------------------------"

class KeysightChannelIn(KeysightChannel):
    '''Class representing a single Keysight input channel, as would be found
    on an M31xxA module.'''
    
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannelIn object.
        Params:
            module: The module object where the channel is housed
            channel_number: The channel number.'''
        KeysightChannel.__init__(self, module, channel_number)
        
    def configure(self, full_scale = 1, prescaler = 1, points_per_cycle = 1000,
                  cycles = 1, delay = 0,
                  trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
                  digital_trigger_behavior = 
                  SD1.SD_TriggerBehaviors.TRIGGER_RISE):
        '''Configures the Keysight input module.
        Params:
            full_scale: The full scale of the input signal, in volts
            prescaler: Optional setting that only selects 1 out of every n
                data points to keep. Useful for limiting file size.
            points_per_cycle: Number of data points to acquire each cycle
            cycles: Number of data acquisition cycles
            delay: Number of samples delayed after trigger
                (or negative to advance)
            trigger_mode: Where the module should look to find the trigger.
                Can be AUTOTRIG for no need for trigger; EXTTRIG for external
                trigger; SWHVITRIG for trigger by software/HVI
        Returns: a tuple of any error messages
        '''
            
        err1 = self._module.channelInputConfig(self._channel_number,
                    full_scale, KeysightConstants.INPUT_IMPEDANCE, 
                    KeysightConstants.DC_COUPLING)
        err2 = self._module.channelPrescalerConfig(self._channel_number,
                    prescaler)
        err3 = self._module.DAQconfig(self._channel_number, points_per_cycle,
                    cycles, delay, trigger_mode)
        err3 = self._module.DAQdigitalTriggerConfig(self._channel_number, 0,
                    digital_trigger_behavior)
        return (err1, err2, err3)
    
    def readData(self, data_points, timeout = KeysightConstants.INFINITY):
        '''Reads data from the channel buffer.
        Params:
            data_points: The number of data points to acquire
            timeout: The timeout in ms for which to wait for data, or
                KeysightConstants.INFINITY for no timeout'''
        return self._module.DAQread(self._channel_number, data_points, timeout)
    
    def start(self):
        '''Starts the channel.
        Returns: any error messages'''
        return self._module.DAQstart(self._channel_number)
    
    def stop(self):
        '''Stops the channel.
        Returns: any error messages'''
        return self._module.DAQstop(self._channel_number)
    
    def pause(self):
        '''Pauses the channel.
        Returns: any error messages'''
        return self._module.DAQpause(self._channel_number)

    def resume(self):
        '''Resumes the channel after a pause.
        Returns: any error messages'''
        return self._module.DAQresume(self._channel_number)
    
    def flush(self):
        '''Flushes the data acquisition buffer.
        Returns: any error messages'''
        return self._module.DAQflush(self._channel_number)
    
    def trigger(self):
        '''Triggers data acquisition.
        Returns: any error messages'''
        return self._module.DAQtrigger(self._channel_number)
    
    def __str__(self):
        '''Returns a string representing the channel.'''
        return ("Channel in. Chassis = " +
                str(self._module.chassis().chassisNumber()) +
                ", Slot = " + str(self._module.slotNumber()) + ", Channel = " +
                str(self._channel_number))

"-----------------------------------------------------------------------------"   

class KeysightChannelOut(KeysightChannel):
    '''Class representing a single Keysight output channel, as would be found
    on an M32xxA module.'''
    
    def __init__(self, module, channel_number):
        '''Initializes the KeysightChannelOut object.
        Params:
            module: The module object where the channel is housed
            channel_number: The channel number.'''
        KeysightChannel.__init__(self, module, channel_number)
        module.channelWaveShape(channel_number, SD1.SD_Waveshapes.AOU_AWG)
        #sets to AWG mode rather than preset waveforms
        
    def configure(self, amplitude = 1, offset_voltage = 0, trigger_behavior =
                  SD1.SD_TriggerBehaviors.TRIGGER_RISE):
        '''Configures the output channel.
        Params:
            amplitude: The amplitude of the signal when 1 is given in the
                waveform. (i.e. 0.5 for a sin wave that is 1.0Vpp)
            offset_voltage: The offset voltage for the channel
            trigger_behavior: How channel should interpret trigger signal
        Returns: A tuple of error messages'''
        amplitude /= 2 #TODO: This line fixes a bug but is janky. Would be nice
        #to get to the root of the problem.
        err1 = self._module.channelAmplitude(self._channel_number, amplitude)
        err2 = self._module.channelOffset(self._channel_number, offset_voltage)
        err3 = self._module.AWGtriggerExternalConfig(self._channel_number,
                    0, trigger_behavior)
        return (err1, err2, err3)
    
    def start(self):
        '''Starts the channel.
        Returns: any error messages'''
        return self._module.AWGstart(self._channel_number)
    
    def stop(self):
        '''Stops the channel.
        Returns: any error messages'''
        return self._module.AWGstop(self._channel_number)
    
    def pause(self):
        '''Pauses the channel.
        Returns: any error messages'''
        return self._module.AWGpause(self._channel_number)

    def resume(self):
        '''Resumes the channel after a pause.
        Returns: any error messages'''
        return self._module.AWGresume(self._channel_number)
    
    def flush(self):
        '''Flushes the data acquisition buffer.
        Returns: any error messages'''
        return self._module.AWGflush(self._channel_number)
    
    def trigger(self):
        '''Triggers data acquisition.
        Returns: any error messages'''
        return self._module.AWGtrigger(self._channel_number)
    
    def resetQueue(self):
        '''Resets the channel's queue from beginning.
        Returns: any error messages'''
        return self._module.AWGreset(self._channel_number)
    
    def isRunning(self):
        '''Returns whether the channel is currently running (1) or is stopped
            (0) or a negative number for error.'''
        return self._module.AWGisRunning(self._channel_number)
    
    def queue(self, waveform_number, trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
              delay = 0, cycles = 1, prescaler = 1):
        '''Queues the desired waveform in the queue for the specified channel.
        Waveform must already have been loaded into module.
        Params:
            waveform_number: The number of the waveform to queue
            trigger_mode: Indicates how waveform should be triggered
            delay: Delay between trigger and waveform start, in 10s of ns
            cycles: Number of cycles that waveform should play, or
                negative value for infinite
            prescalar: An integer that dictates that only 1 out of n ticks of
                clock plays next value in waveform; reduces frequency.
        Returns: Any error messages'''
        return self._module.AWGqueueWaveform(self._channel_number,
                    waveform_number, trigger_mode, delay, cycles, prescaler)
        
    def __str__(self):
        '''Returns a string representing the channel.'''
        return ("Channel out. Chassis = " +
                str(self._module.chassis().chassisNumber()) +
                ", Slot = " + str(self._module.slotNumber()) + ", Channel = " +
                str(self._channel_number))

        
"-----------------------------------------------------------------------------"

class KeysightChannelClk(KeysightChannel):
    '''Class representing a single Keysight clock channel.'''
    
    def __init__(self, module):
        '''Initializes the KeysightChannelClk object.
        Params:
            module: The module object where the channel is housed'''
        KeysightChannel.__init__(self, module, ChannelNumber.CLK)
        
    def __str__(self):
        '''Returns a string representing the channel.'''
        return ("Channel. Chassis = " +
                str(self._module.chassis().chassisNumber()) +
                ", Slot = " + str(self._module.slotNumber()) +
                ", Channel = Clk")


"-----------------------------------------------------------------------------"

class KeysightChannelTrig(KeysightChannel):
    '''Class representing a single Keysight digital I/O trigger channel.'''
    
    def __init__(self, module, mode = TriggerIODirection.IN):
        '''Initializes the KeysightChannelTrig object.
        Params:
            module: The module object where the channel is housed
            mode: Whether the trigger channel is used for input or output,
                given by constants in class TriggerIODirection'''
        KeysightChannel.__init__(self, module, ChannelNumber.TRIG)
        module.triggerIOconfig(mode)
        
    def setDirection(self, mode):
        '''Sets whether the channel is configured for input (triggering/reading)
        or output.
        Params:
            mode: Whether the trigger channel is used for input or output,
            given by constants in class TriggerIODirection'''
        self._module.triggerIOconfig(mode)
        
    def write(self, value):
        '''Writes 0 or 1 to trigger in TriggerIODirection.OUT mode.
        Params:
            value: 0 or 1 to write to trigger channel
        Returns: any error messages'''
        return self._module.triggerIOwrite(value)
    
    def read(self):
        '''Reads whether trigger is 0 or 1.
        Returns: 0 or 1 corresponding to digital signal on trigger, or any
        error codes (negative).'''
        return self._module.triggerIOread()
    
    def __str__(self):
        '''Returns a string representing the channel.'''
        return ("Channel. Chassis = " +
                str(self._module.chassis().chassisNumber()) +
                ", Slot = " + str(self._module.slotNumber()) +
                ", Channel = Trig I/O")


"-----------------------------------------------------------------------------"

class Waveform:
    '''Class representing a waveform. Implemented in numpy to easily integrate
    with rest of code base. Use this instead of the native SD_Wave class for
    compatibility and simplicity.'''
    
    _waveform_number_counter = 0 #used to assign unique waveform numbers
    
    def __init__(self, arr = [], waveform_number = None, append_zero = False):
        '''Initializes the waveform object.
        Params:
            arr: A list or numpy array of data points making up the waveform
            waveform_number: The unique number to be associated with the
               waveform as an identifier. If None, waveform_number will be
               assigned automatically in a way that prevents naming collisions.
               If any are assigned manually, user is responsible for
               consequences of naming collisions.
            append_zero: Appends a zero at the end of the waveform if none
               present, to prevent nonzero signal from continuing after
               waveform stops.
        Note: when loaded into memory, all waveforms will have zeros appended
        to end until length of base array is at least 5. Thus, putting in a
        single value to output a constant signal will give unexpected results.
        '''
        if append_zero and arr[len(arr)-1] != 0:
            arr.append(0)
        self._arr = np.array(arr)
        self.setWaveformNumber(waveform_number)
        
    def setWaveformNumber(self, waveform_number = None):
        '''Sets a waveform's waveform number.
        Params:
            waveform_number: The unique number to be associated with the
               waveform as an identifier. If None, waveform_number will be
               assigned automatically in a way that prevents naming collisions.
               If any are assigned manually, user is responsible for
               consequences of naming collisions.'''
               
        if waveform_number is None:
            Waveform._waveform_number_counter += 1
            self._waveform_number = Waveform._waveform_number_counter
        else:
            self._waveform_number = waveform_number
            
    def getWaveformNumber(self):
        '''Returns the waveform's waveform number.'''
        return self._waveform_number
    
    def getBaseArray(self):
        '''Returns the base numpy array underlying the waveform object.'''
        return self._arr
    
    def loadToModule(self, module):
        '''Loads a waveform into memory on the specific module.
        Params:
            module: The module to which to save the waveform
        Returns: a tuple of error messages'''
        return module.loadWaveform(self)
    
    def queue(self, channel, trigger_mode = SD1.SD_TriggerModes.EXTTRIG,
              delay = 0, cycles = 1, prescaler = 1):
        '''Queues the waveform for a specific channel, provided it has already
        been loaded into the corresponding module's memory.
        Params:
            channel: The otuput channel object to which to queue the waveform
            trigger_mode: Indicates how waveform should be triggered
            delay: Delay between trigger and waveform start, in 10s of ns
            cycles: Number of cycles that waveform should play, or
                negative value for infinite
            prescalar: An integer that dictates that only 1 out of n ticks of
                clock plays next value in waveform; reduces frequency.
        Returns: Any error messages'''
        return channel.queue(self.getWaveformNumber(), trigger_mode, delay,
                             cycles, prescaler)
        
    def __str__(self):
        '''Returns a string representation of the waveform'''
        return ("Waveform_number: " + str(self._waveform_number) + "\n" +
                str(self.getBaseArray()))