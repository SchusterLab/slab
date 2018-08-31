
# -*- coding: utf-8 -*-
"""
Created on July 26, 2018

Version: 1.1.3 updated 8/31/18
Changes since 1.0:
    Implemented HVI class to represent a Keysight chassis functioning as
        an HVI.
    Added __str__ methods to each class to aid debugging
    Added close() methods to several classes to ensure modules are closed
        before they go out of scope.
    Added serialize methods to serialize channel numbers (useful for 
        saving experiments to disk)
    Added mute() methods to mute unwanted channels without having to redo an
        HVI
    Added static Tools.decodeError and Tools.isError methods
    Fixed a couple of minor bugs and added documentation throughout.
    Added FileDecodingTools class
    Made several additions to HVI class
Changes since 1.1.2:
    Added support for exception handling throughout
    Several bug fixes
    

@author: Josephine Meyer (jcmeyer@stanford.edu)

Library classes for Python interface with Keysight M31xxA and M32xxA modules.
Use instead of the native keysightSD1 classes for better functionality and
compatibility with our existing code base.

See native keysightSD1 code for definitions of error messages and some 
enumerated types. 
"""

import keysightSD1 as SD1
import numpy as np
import ast

'''Short auxiliary classes and constants'''

class KeysightConstants:
    '''Useful general constants'''
    NUM_CHANNELS = 4 #number of modules
    PRODUCT = ""
    MASK_ALL = 30 #2**1 + 2**2 + 2**3 + 2**4 -- mask for all 4 channels
    INPUT_IMPEDANCE = 1 #50 ohm
    DC_COUPLING = 0 #we are DC coupled
    INFINITY = 0 #Blame Keysight for this one!
    MODULE_CONFIG_SUFFIX = ".keycfg" #suffix for hardware config file

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
    TRIG = 9
    TRIGGER = TRIG
    CLK = 0
    CLOCK = CLK
    
class TriggerIODirection:
    '''Trigger channel in or out'''
    IN = 1
    OUT = 0
    
class HVIStatus:
    '''Status of an HVI'''
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2

class Tools:
    '''Useful static methods used in implementation and debugging.'''
    
    @staticmethod
    def channelsToMask(*channels):
        '''Converts list of channels to a mask accepted by several native
            methods in SD1.
        Params:
            *channels: Any number of channel numbers from same module
        Returns: A mask representing the channels. User should have no need
            to call this method directly.'''
        mask = 0
        for c in channels:
            mask += 2**c
        return mask
    
    @staticmethod
    def serializeChannel(chassis_number, slot_number, channel_number):
        '''Generates serial representation of channel's location within the
            chassis assembly. Note: to serialize a channel object already
            created, simply call serialize() on the channel, as this value
            is stored when the object is created. The primary reason you would
            want to do this is to store the channel as a unique dict key.
        Params:
            chassis_number: The chassis number where the channel is located
            slot_number: The slot number where the channel is located
            channel_number: The channel number where the channel is located,
                using codes in ChannelNumber class to encode clk and trig
                channels
        Returns: the serial representation of the channel
        '''
        return (str(chassis_number) + str(slot_number % 10) +
                str(channel_number))
        
    @staticmethod
    def deserializeChannel(serial):
        '''Returns channel characteristics from serial representation.
            Undoes the Tools.serializeChannel() method
        Params: The serial representation of the channel
        Returns: a tuple of (chassis_number, slot_number, channel_number)'''
        chassis_number = int(serial[0])
        slot_number = int(serial[1])
        if slot_number == 0:
            slot_number = 10
        channel_number = int(serial[2])
        return (chassis_number, slot_number, channel_number)
    
    @staticmethod
    def isError(error):
        '''Returns whether an integer describing an error message is an error.
        Use to decode native error messages.
        Params:
            error: The error code
        Returns: whether the error code represents an error'''
        return error < 0
    
    @staticmethod
    def decodeError(error):
        '''Returns a string providing a descriptive error message for any
        native error code, based on manufacturer provided error codes. Errors
        are negative values; a nonnegative value returns 'No error.'
        
        Params:
            error: The error code, or a tuple of error codes
        Returns: a descriptive string describing the error code'''
        if error >= 0:
            return "No error"
        elif error not in Tools.error_defs:
            return "Error code not in dictionary: " + str(error)
        else:
            return Tools.error_defs[error] 
    
    
    #Error definitions as given in native SD1 code
    error_defs = {
            -8000: "Opening module",        -8001: "Closing module",
            -8002: "Opening HVI",           -8003: "Closing HVI",
            -8004: "Module not opened",     -8005: "Module not opened by user",
            -8006: "Module already opened", -8007: "HVI not opened",
            -8008: "Invalid ObjectID",      -8009: "Invalid ModuleID",
            -8010: "Invalid module user name",
            -8011: "Invalid HVI",           -8012: "Invalid object",
            -8013: "Invalid channel number",-8014: "Bus doesn't exist",
            -8015: "Any input assigned to the bit map does not exist",
            -8016: "Input size does not fit on this bus",
            -8017: "Input data does not fit on this bus",
            -8018: "Invalid value",         -8019: "Creating waveform",
            -8020: "Invalid parameters",    -8021: "AWG function failed",
            -8022: "Invalid DAQ functionality",
            -8023: "DAQ buffer pool is already running",
            -8024: "Unknown error",         -8025: "Invalid parameter",
            -8026: "Module not found",      -8027: "Driver resource busy",
            -8028: "Driver resource not ready",
            -8029: "Cannot allocate buffer in driver",
            -8030: "Cannot allocate buffer",-8031: "Resource not ready",
            -8032: "Hardware error",        -8033: "Invalid operation",
            -8034: "No compiled code in the module",
            -8035: "Firmware verification failed",
            -8036: "Compatibility error",   -8037: "Invalid type",
            -8038: "Demo module",           -8039: "Invalid buffer",
            -8040: "Invalid index",         -8041: "Invalid histogram number",
            -8042: "Invalid number of bins",-8043: "Invalid mask",
            -8044: "Invalid waveform",      -8045: "Invalid strobe",
            -8046: "Invalid strobe value",  -8047: "Invalid debouncing",
            -8048: "Invalid prescaler",     -8049: "Invalid port",
            -8050: "Invalid direction",     -8051: "Invalid mode",
            -8052: "Invalid frequency",     -8053: "Invalid impedance",
            -8054: "Invalid gain",          -8055: "Invalid full scale",
            -8056: "Invalid file",          -8057: "Invalid slot",
            -8058: "Invalid product name",  -8059: "Invalid serial number",
            -8060: "Invalid start",         -8061: "Invalid end",
            -8062: "Invalid number of cycles",
            -8063: "Invalid number of modules on HVI",
            -8064: "DAQ P2P is already running",
            }


class KeysightError(Exception):
    '''Exception thrown caused by error within this library or native code.
    Automatically translates native error codes to message.'''
    
    def __init__(self, msg="", code=None):
        '''Initializes the exception.
        Params:
            msg: The message to be printed by the exception.
            code: Any native error codes to decode and append to the message,
            or None if there is no native error code.'''
        if code is not None:
            msg += (": " + Tools.decodeError(code))
        Exception.__init__(self, msg)
        

'''-------------------------------------------------------------------------'''

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
    
    def close(self):
        '''Clears and closes all modules associated with the chassis.'''
        for module in self.modulesList():
            module.close()
            
    def clearAll(self):
        '''Clears all modules associated with the chassis.'''
        for module in self.modulesList():
            module.clearAll()
            
    def __str__(self):
        '''Returns a string representation of the chassis and its modules.'''
        return ("Keysight chassis. Chassis number: " +
                str(self._chassis_number) + "\n    Modules:" + 
                str(self._modules))
                
    def save(self, filename):
        '''Saves the hardware configuration information (chassis and slot
        numbers) to file.
        Params:
            filename: The name and path at which to store the file.
        Returns: Whether the save was successful. Prints any error message.'''
        if not filename.endswith(KeysightConstants.MODULE_CONFIG_SUFFIX):
            filename.append(KeysightConstants.MODULE_CONFIG_SUFFIX)
        lines = []
        lines.append("Chassis Number: " + str(self._chassis_number))
        input_modules = []
        output_modules = []
        for module in self.modulesList():
            if module.getModuleType() == ModuleType.INPUT:
                input_modules.append(module.slotNumber())    
            elif module.getModuleType() == ModuleType.OUTPUT:
                input_modules.append(module.slotNumber())
        lines.append("Input Modules: " + str(input_modules))
        lines.append("Output Modules: " + str(output_modules))
        try:
            file = open(filename, 'w')
            for line in lines:
                file.write(line)
            return True
        except Exception as e:
            print("Error saving file: " + str(e))
            return False
        finally:
            file.close()
            
    @staticmethod
    def fromFile(filename):
        '''Generates a chassis object from a saved configuration file of type
        .keycfg describing the hardware configuration of the modules.
        Params:
            filename: The name and path at which the .keycfg file is stored
        Returns: a KeysightChassis object, or None if an error'''

        chassis, modules = FileDecodingTools._readHardwareConfigFile(filename)
        return KeysightChassis(chassis, modules)
    
"----------------------------------------------------------------------------"

class HVI(KeysightChassis, SD1.SD_HVI):
    '''Class represents a chassis onto which an HVI is loaded. Inherits from
    both the KeysightChassis class (since the HVI controls all modules) and
    the native SD_HVI class for implementation purposes. There should be no
    reason to call SD_HVI methods directly in implementation.
    '''
    
    def __init__(self, HVI_filename, chassis_number, modules_dict={}):
        '''Initializes the underlying Keysight Chassis object and loads
            the HVI.
        Params:
            HVI_filename: The filename (including path) where the compiled
                HVI code is stored.
            chassis_number: The chassis number of the chassis we are creating
            modules_dict: A dictionary representing the modules formatted as
               {slot number: ModuleType}.
        '''
        SD1.SD_HVI.__init__(self)
        KeysightChassis.__init__(self, chassis_number, modules_dict)
        
        #TODO: remove
        ch = self.getChannel(6, 1)
        print(ch)
        waveform = Waveform(arr = np.arange(0, 1, 101), append_zero = True,
                            waveform_number = 1000)
        waveform.loadToModule(ch.module())
        waveform.queue(ch, trigger_mode=SD1.SD_TriggerModes.AUTOTRIG,
                       delay = 0, cycles = 0)
        ch.start()
        
        print(HVI_filename)
        
        err0 = self.open(HVI_filename)
        if err0 < 0:
            raise KeysightError("Error opening HVI file", err0)
            
        num_errors = self.compile()
        if num_errors > 0:
            msg = ""
            for i in range(num_errors):
                msg += Tools.decodeError(self.compilationErrorMessage(i))
                msg += "\n"
            raise KeysightError("Error(s) compiling HVI file:\n" + msg) 
            
        err1 = self.load()     
        if err1 < 0:
            raise KeysightError("Error loading HVI to modules", err1)
            
        self._is_running = HVIStatus.STOPPED
        self._filename = HVI_filename
        self._nicknames = self._generateModuleNicknamesDict()
        
    def close(self):
        '''Closes the HVI.'''
        err = SD1.SD_HVI.close(self)
        if err < 0:
            raise KeysightError("Error closing HVI", err)
        KeysightChassis.close(self)
        
    def start(self):
        '''Starts the HVI.'''
        err = SD1.SD_HVI.start(self)
        if err < 0:
            raise KeysightError("Error starting HVI", err)
        self._is_running = HVIStatus.RUNNING
    
    def pause(self):
        '''Pauses the operation of the HVI.'''
        err = SD1.SD_HVI.pause(self)
        if err < 0:
            raise KeysightError("Error pausing HVI", err)
        self._is_running = HVIStatus.PAUSED
    
    def resume(self):
        '''Resumes the operation of the HVI.'''
        err = SD1.SD_HVI.resume(self)
        if err < 0:
            raise KeysightError("Error resuming HVI", err)
        self._is_running = HVIStatus.RUNNING
    
    def stop(self):
        '''Stops the operation of the HVI.'''
        err = SD1.SD_HVI.stop(self)
        if err < 0:
            raise KeysightError("Error stopping HVI", err)
        self._is_running = HVIStatus.STOPPED
    
    def getStatus(self):
        '''Returns the status of the HVI using codes defined in Status class.
        Note that status will continue to read HVIStatus.RUNNING until stop()
        is explicitly called, even if the HVI has finished executing.'''
        return self._is_running
    
    def reset(self):
        '''Resets the HVI to beginning.'''
        err = SD1.SD_HVI.reset(self)
        if err < 0:
            raise KeysightError("Error resetting HVI", err)
    
    def __str__(self):
        '''Returns a string representation of the chassis functioning as an
            HVI.'''
        return (KeysightChassis.__str__(self) + "\n   HVI: " + self._filename)
    
    @staticmethod #factory method #overrides method in KeysightChassis
    def fromFile(HVI_filename, hardware_config_filename):
        '''Constructs an HVI object from a saved hardware configuration file
            and the HVI file.
        Params:
            HVI_filename: The HVI file name and path
            hardware_config_filename: The file name and path where the .keycfg
                file indicating the hardware configuration is stored.
        Returns: An HVI object.
        '''
        chassis, modules = FileDecodingTools._readHardwareConfigFile(
                hardware_config_filename)
        if chassis is None:
            raise KeysightError("Cannot open file")
        return HVI(HVI_filename, chassis, modules)
    
    #internal methods used for implementation
    
    def _generateModuleNicknamesDict(self):
        '''Generates a dictionary that holds the slot numbers as keys and the 
        nicknames as values'''
        nicknames_dict = {}
        num_modules = self.getNumberOfModules()
        if num_modules < 0: #native method returned an error
            raise KeysightError("Error getting number of modules from HVI", 
                                num_modules)
        
        for i in range(num_modules):
            nickname = self.getModuleName(i)
            if not isinstance(nickname, str):
                raise KeysightError("Error getting module nickname from HVI",
                                    nickname)
                
            module_base = self.getModuleByIndex(i)
            if not isinstance(module_base, SD1.SD_Module):
                raise KeysightError("Error getting module from nickname", 
                                    module_base)
                
            slot = module_base.getSlot()
            if slot < 0:
                raise KeysightError("Error getting slot", slot)
                
            nicknames_dict[slot] = nickname
        return nicknames_dict
            
        

"----------------------------------------------------------------------------"

class KeysightModule(SD1.SD_Module):
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
    
    def moduleNumber(self):
        '''Returns the slot number where the module is housed. Alias for
        slotNumber().'''
        return self.slotNumber()
    
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
    
    def readRegister(self, register_number):
        '''Reads the (int) value from a register associated with the module.
        Params:
            register_number: The number of the register to be read
        Returns: the value on the register.'''
        err, value = self.readRegisterByNumber(register_number)
        if err < 0:
            raise KeysightError("Error reading register " + 
                                str(register_number), err)
        return value
    
    def writeRegister(self, register_number, value):
        '''Writes an (int) value to a register associated with the module.
        Params:
            register_number: The number of the register to be written to
            value: The value to write to the register
        '''
        err = self.writeRegisterByNumber(register_number, value)
        if err < 0:
            raise KeysightError("Error writign register " + str(register_number),
                                err)
    
    def readPXI(self, PXI_number):
        '''Reads the PXI trigger (binary value).
        Params:
            PXI_number: The number of the PXI port to read.
        Returns: A boolean value corresponding to the value on the PXI port.'''
        value = self.PXItriggerRead()
        if value < 0: #there is an error
            raise KeysightError("Error reading PXI " + str(PXI_number), value)
        return not value #not a bug; native method gives inverted logic
        
    def writePXI(self, PXI_number, value):
        '''Writes a boolean value to a PXI trigger.
        Params:
            PXI_number: The number of the PXI port.
            Value: a boolean value to write to the trigger'''
        err = self.PXItriggerWrite(PXI_number, int(not value)) #not a bug
        if err < 0:
            raise KeysightError("Error writing value to PXI " + 
                                str(PXI_number), err)
    
    def readHVIConstant(self, constant_name, constant_type = float):
        '''Reads the value of an HVI constant associated with this module.
        Params:
            constant_name: The name of the constant (string)
            constant_type: int or float, corresponding to the type of constant
        Returns: The value of the constant'''
        chassis = self._chassis
        if not isinstance(chassis, HVI):
            raise KeysightError("No HVI loaded")
        if constant_type == int:
            err, value = chassis.readIntegerConstantWithUserName(
                    chassis._nicknames[self._slot_number])
        elif constant_type == float:
            err, value = chassis.readDoubleConstantWithUserName(
                    chassis._nicknames[self._slot_number])
        else:
            raise ValueError("constant_type must be int or float")
        
        if err < 0:
            raise KeysightError("Error reading constant", err)
        return value
    
    def writeHVIConstant(self, constant_name, value, constant_type = float,
                         unit = ""):
        chassis = self._chassis
        '''Writes a new value to an HVI constant associated with this module.
        Params:
            constant_name: The name of the constant (string) as defined in file
            value: The value of the constant
            constant_type: int or float, corresponding to the type of constant
                declared in the HVI file. Will convert "value" to this type
            unit: Optional parameter for declaring units for constants of type
                "float."
        '''
        if not isinstance(chassis, HVI):
            raise KeysightError("No HVI loaded")
        if constant_type == int:
            err = chassis.writeIntegerConstantWithUserName(
                    chassis._nicknames[self._slot_number], int(value))
        elif constant_type == float:
            err = chassis.writeDoubleConstantWithUserName(
                    chassis._nicknames[self._slot_number], float(value), unit)
        else:
            raise ValueError("constant_type must be int or float")
        
        if err < 0:
            raise KeysightError("Error writing constant", err)
    
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
            raise KeysightError("Could not open module:\n Chassis "
                + str(chassis) + "\n Slot " + str(slot_number), module_in_ID)
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
        
        self.flushAll()
            
    def getModuleType(self):
        '''Returns a constant corresponding to the module type as given in 
        class ModuleType.'''
        return ModuleType.INPUT
    
    def clearAll(self):
        '''Clears the data acquisition buffers on all channels.'''
        for channel in self.channelsList():
            if isinstance(channel, KeysightChannelIn) or isinstance(
                    channel, KeysightChannelOut):
                channel.clear()
    
    def startChannels(self, *channels):
        '''Starts channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.DAQstartMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error starting channels " + str(channels), err)
    
    def stopChannels(self, *channels):
        '''Stops channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.DAQstopMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error stopping channels " + str(channels), err)
    
    def pauseChannels(self, *channels):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.DAQpauseMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error pausing channels " + str(channels), err)

    def resumeChannels(self, *channels):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling resume() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.DAQresumeMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error resuming channels " + str(channels), err)
    
    def triggerChannels(self, *channels):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.DAQtriggerMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error triggering channels " + str(channels), 
                                err)

    def startAll(self):
        '''Starts ALL channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        '''
        err = self.DAQstartMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error starting all channels", err)
    
    def stopAll(self):
        '''Stops ALL channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        '''
        err = self.DAQstopMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error stopping all channels", err)
    
    def pauseAll(self):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        '''
        err = self.DAQpauseMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error pausing all channels", err)

    def resumeAll(self):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling resume() on individual channels.
        '''
        err = self.DAQresumeMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error resuming all channels", err)
    
    def triggerAll(self):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        '''
        err = self.DAQtriggerMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error triggering all channels", err)
            
    def flushAll(self):
        '''Flushes the data acquisition pool. Alternative to calling flush()
        on individual channels.'''
        err = self.DAQflushMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error flushing all channels", err)
    
    def close(self):
        '''Closes the current module.'''
        self.clearAll()
        err = SD1.SD_AIN.close(self)
        if err < 0:
            raise KeysightError("Error closing module", err)
    
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
            raise KeysightError("Could not open module:\n Chassis "
                + str(chassis) + "\n Slot " + str(slot_number), module_out_ID)
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
            
        err = self.waveformFlush()
        if err < 0:
            raise KeysightError("Error flushing waveforms", err)
            
    def getModuleType(self):
        '''Returns a constant corresponding to the module type as given in 
        class ModuleType.'''
        return ModuleType.OUTPUT
    
    def startChannels(self, *channels):
        '''Starts channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.AWGstartMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error starting channels " + str(channels), err)
    
    def stopChannels(self, *channels):
        '''Stops channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.AWGstopMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error stopping channels " + str(channels), err)
    
    def pauseChannels(self, *channels):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.AWGpauseMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error pausing channels " + str(channels), err)

    def resumeChannels(self, *channels):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.AWGresumeMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error resuming channels " + str(channels), err)
    
    def triggerChannels(self, *channels):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.
        Params:
            *channels: Any number of channel numbers on the same module
        '''
        err = self.AWGtriggerMultiple(Tools.channelsToMask(channels))
        if err < 0:
            raise KeysightError("Error triggering channels " + str(channels),
                                err)
    
    def startAll(self):
        '''Starts ALL channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        '''
        err = self.AWGstartMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error starting all channels", err)
    
    def stopAll(self):
        '''Stops ALL channels on a given module simultaneously. Alternative to
        calling stop() on individual channels.
        '''
        err = self.AWGstopMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error stopping all channels", err)
    
    def pauseAll(self):
        '''Pauses channels on a given module simultaneously. Alternative to
        calling pause() on individual channels.
        '''
        err = self.AWGpauseMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error pausing all channels", err)

    def resumeAll(self):
        '''Resumes channels on a given module simultaneously. Alternative to
        calling start() on individual channels.
        '''
        err = self.AWGresumeMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error resuming all channels", err)
    
    def triggerAll(self):
        '''Triggers channels on a given module simultaneously. Alternative to
        calling trigger() on individual channels.'''
        err = self.AWGtriggerMultiple(KeysightConstants.MASK_ALL)
        if err < 0:
            raise KeysightError("Error triggering all channels", err)
    
    def close(self):
        '''Closes the current module..'''
        self.clearAll()
        err = SD1.SD_AOU.close(self)
        if err < 0:
            raise KeysightError("Err closing module", err)
    
    def loadWaveform(self, waveform):
        '''Loads a waveform into memory on the specific module.
        Params:
            waveform: The waveform object to load
        '''
        wave = SD1.SD_Wave()
        err0 = wave.newFromArrayDouble(SD1.SD_WaveformTypes.WAVE_ANALOG,
                                waveform.getBaseArray())
        if err0 < 0:
            raise KeysightError("Error creating native waveform object: ", err0)
        err1 = self.waveformLoad(wave, waveform.getWaveformNumber())
        if err1 < 0:
            raise KeysightError("Error loading waveform to module: ", err1)
    
    def clearAll(self):
        '''Stops all AWG's, deletes waveforms from memory, and flushes queues
        of every channel.'''
        self.stopAll()
        err = self.waveformFlush()
        if err < 0:
            raise KeysightError("Error flushing waveforms from module", err)
    
    def __str__(self):
        '''Returns a string representation of the module.'''
        return ("Module out. Chassis = " + str(self.chassis().chassisNumber()) 
                + ", Slot = " + str(self.slotNumber()))

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
        self._serial = Tools.serializeChannel(module.chassis().chassisNumber(),
                                              module.moduleNumber(),
                                              channel_number)
        
    def module(self):
        '''Returns the module where the channel is housed'''
        return self._module
    
    def channelNumber(self):
        '''Returns the channel number'''
        return self._channel_number
    
    def chassis(self):
        '''Returns the chassis where the channel is located'''
        return self._module.chassis()
    
    def serialize(self):
        '''Returns serial representation of the channel's location within
            the Keysight chassis assembly. Used to store channel as dict key'''
        return self._serial
    
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
                  trigger_mode = SD1.SD_TriggerModes.SWHVITRIG,
                  digital_trigger_behavior = 
                  SD1.SD_TriggerBehaviors.TRIGGER_RISE,
                  use_buffering = True,
                  buffer_size = None,
                  buffer_time_out = KeysightConstants.INFINITY):
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
            buffer_size: The number of data points that should be stored in
                buffer before transmission to PC. If None, defaults to value
                in points_per_cycle.
            buffer_time_out: Maximum time allotted to fill buffer before
                buffer will be returned, even if not full. Default is INFINITY
                (will not release data until buffer is full).
        '''
        buffer_size = buffer_size or points_per_cycle
        
        err0 = self._module.channelInputConfig(self._channel_number,
                    full_scale, KeysightConstants.INPUT_IMPEDANCE, 
                    KeysightConstants.DC_COUPLING)
        if err0 < 0:
            raise KeysightError("Error configuring channel", err0)
            
        err1 = self._module.channelPrescalerConfig(self._channel_number,
                    prescaler)
        if err1 < 0:
            raise KeysightError("Error setting prescaler", err1)
            
        err2 = self._module.DAQconfig(self._channel_number, points_per_cycle,
                    cycles, delay, trigger_mode)
        if err2 < 0:
            raise KeysightError("Error configuring channel", err2)
            
        err3 = self._module.DAQdigitalTriggerConfig(self._channel_number, 0,
                    digital_trigger_behavior)
        if err3 < 0:
            raise KeysightError("Error configuring trigger", err3)
            
        err4 = self._module.DAQbufferPoolConfig(self._channel_number,
                            buffer_size, buffer_time_out)
        if err4 < 0:
            raise KeysightError("Error configuring buffer pool", err4)
    
    def readData(self, data_points, timeout = KeysightConstants.INFINITY):
        '''Reads arbitrary length of data from the digitizer. Useful for
        testing purposes. Normally, for experiments would want to call
        readDataBuffered().
        Params:
            data_points: The number of data points to acquire
            timeout: The timeout in ms for which to wait for data, or
                KeysightConstants.INFINITY for no timeout
        Returns: The data as a numpy array'''
        data = self._module.DAQread(self._channel_number, data_points, timeout)
        if (not data) or isinstance(data, int):
            raise KeysightError("Error acquiring data", data)
        return data
    
    def readDataBuffered(self):
        '''Reads one buffer of data.
        Returns: The data as a numpy array, or an error message that can
        be read by Tools.decodeError(). "Quietly" does not throw an exception
        because we don't want to interrupt experiment.'''
        return self._module.DAQbufferGet(self._channel_number)
    
    def start(self):
        '''Starts the channel.'''
        err = self._module.DAQstart(self._channel_number)
        if err < 0:
            raise KeysightError("Error starting channel", err)
    
    def stop(self):
        '''Stops the channel.'''
        err = self._module.DAQstop(self._channel_number)
        if err < 0:
            raise KeysightError("Error stopping channel", err)
    
    def pause(self):
        '''Pauses the channel.'''
        err = self._module.DAQpause(self._channel_number)
        if err < 0:
            raise KeysightError("Error pausing channel", err)

    def resume(self):
        '''Resumes the channel after a pause.'''
        err = self._module.DAQresume(self._channel_number)
        if err < 0:
            raise KeysightError("Error resuming channel", err)
    
    def clear(self):
        '''Flushes the data acquisition buffer and releases buffer pool.'''
        self.flush()
        err = self._module.DAQbufferPoolRelease(self._channel_number)
        if err < 0:
            raise KeysightError("Error releasing buffer pool", err)
            
    def flush(self):
        '''Flushes the data acquisition queue.'''
        err = self._module.DAQflush(self._channel_number)
        if err < 0:
            raise KeysightError("Error flushing data", err)
    
    def trigger(self):
        '''Triggers data acquisition.'''
        err = self._module.DAQtrigger(self._channel_number)
        if err < 0:
            raise KeysightError("Error triggering channel", err)
    
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
        err = module.channelWaveShape(channel_number, SD1.SD_Waveshapes.AOU_AWG)
        if err < 0:
            raise KeysightError("Cannot set to AWG mode", err)
        #sets to AWG mode rather than preset waveforms
        self._amplitude = None
        self._muted = False
        
    def configure(self, amplitude = 1, offset_voltage = 0, trigger_behavior =
                  SD1.SD_TriggerBehaviors.TRIGGER_RISE, trigger_source = 
                  SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                  repeat_queue = True):
        '''Configures the output channel.
        Params:
            amplitude: The amplitude of the signal when 1 is given in the
                waveform. (i.e. 0.5 for a sin wave that is 1.0Vpp)
            offset_voltage: The offset voltage for the channel
            trigger_behavior: How channel should interpret trigger signal
            trigger_source: Where channel should look for trigger (external
                port or PXI port). If PXI, input:
                SD1.SD_TriggerExternalSources.TRIGGER_PXI + trigger_number
            repeat_queue: Whether to repeat the queue after each run.
        '''
        self.setAmplitude(amplitude)
        
        err1 = self._module.channelOffset(self._channel_number, offset_voltage)
        if err1 < 0:
            raise KeysightError("Error setting offset voltage", err1)
            
        err2 = self._module.AWGtriggerExternalConfig(self._channel_number,
                    trigger_source, trigger_behavior)
        if err2 < 0:
            raise KeysightError("Error configuring trigger", err2)
            
        err3 = self._module.AWGqueueConfig(self._channel_number, 
                                           int(repeat_queue))
        if err3 < 0:
            raise KeysightError("Error configuring queue", err3)
    
    def start(self):
        '''Starts the channel.'''
        err = self._module.AWGstart(self._channel_number)
        if err < 0:
            raise KeysightError("Error starting channel", err)
    
    def stop(self):
        '''Stops the channel.'''
        err = self._module.AWGstop(self._channel_number)
        if err < 0:
            raise KeysightError("Error stopping channel", err)
    
    def pause(self):
        '''Pauses the channel.'''
        err = self._module.AWGpause(self._channel_number)
        if err < 0:
            raise KeysightError("Error pausing channel", err)

    def resume(self):
        '''Resumes the channel after a pause.'''
        err = self._module.AWGresume(self._channel_number)
        if err < 0:
            raise KeysightError("Error resuming channel", err)
    
    def clear(self):
        '''Clears the stored waveforms.'''
        err = self._module.AWGflush(self._channel_number)
        if err < 0:
            raise KeysightError("Error clearing channel", err)
    
    def trigger(self):
        '''Triggers data acquisition.'''
        err = self._module.AWGtrigger(self._channel_number)
        if err < 0:
            raise KeysightError("Error triggering channel", err)
    
    def resetQueue(self):
        '''Resets the channel's queue from beginning.'''
        err = self._module.AWGreset(self._channel_number)
        if err < 0:
            raise KeysightError("Error resetting queue", err)
    
    def isRunning(self):
        '''Returns whether the channel is currently running.'''
        is_running = self._module.AWGisRunning(self._channel_number)
        if is_running < 0: #if native method returned error
            raise KeysightError("Error querying status of AWG",
                                is_running)
        return bool(is_running)
    
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
            prescaler: An integer that dictates that only 1 out of n ticks of
                clock plays next value in waveform; reduces frequency.'''
        err = self._module.AWGqueueWaveform(self._channel_number,
                    waveform_number, trigger_mode, delay, cycles, prescaler)
        if err < 0:
            raise KeysightError("Error queueing waveform", err)
        
    def __str__(self):
        '''Returns a string representing the channel.'''
        return ("Channel out. Chassis = " +
                str(self._module.chassis().chassisNumber()) +
                ", Slot = " + str(self._module.slotNumber()) + ", Channel = " +
                str(self._channel_number))
        
    def setAmplitude(self, amplitude):
        '''Sets the amplitude of the channel.'''
        amplitude /= 2 #TODO: This line fixes a bug but is janky. Would be nice
        #to get to the root of the problem.
        self._amplitude = amplitude
        if not self._muted:
            err = self._module.channelAmplitude(
                    self._channel_number, amplitude)
            if err < 0:
                raise KeysightError("Error setting amplitude", err)
                
    def mute(self):
        '''Mutes the channel by setting amplitude to 0.'''
        err = self._module.channelAmplitude(self._channel_number, 0)
        if err < 0:
            raise KeysightError("Error setting amplitude to mute", err)
        self._muted = True
    
    def unmute(self):
        '''Unmutes the channel. Channel returns to saved amplitude from before
            when mute() was called. If channel not currently muted, has no
            effect.'''
        if self._muted:
            err = self._module.channelAmplitude(self._channel_number,
                                             self._amplitude)
            if err < 0:
                raise KeysightError("Error setting amplitude to unmute", err)
            self._muted = False
        
    def loadWaveform(self, waveform):
        '''Loads a waveform to the channel's module. Useful alias to the
        loadWaveform() method on an output module. User is responsible for
        avoiding naming collisions between different channels on same module.
        Params:
            waveform: The waveform object to load onto the module'''
        self._module.loadWaveform(waveform)
        
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
        self.setDirection(mode)
        
    def setDirection(self, mode):
        '''Sets whether the channel is configured for input (triggering/reading)
        or output.
        Params:
            mode: Whether the trigger channel is used for input or output,
            given by constants in class TriggerIODirection'''
        err = self._module.triggerIOconfig(mode)
        if err < 0:
            raise KeysightError("Error setting trigger IO mode", err)
        
    def write(self, value):
        '''Writes boolean value to trigger in TriggerIODirection.OUT mode.
        Params:
            value: True or False (on/off) to write to trigger channel'''
        err = self._module.triggerIOwrite(value)
        if err < 0:
            raise KeysightError("Error writing to trigger", err)
    
    def read(self):
        '''Reads whether trigger is on or off and returns whether it is on.'''
        value = self._module.triggerIOread()
        if value < 0: #native method returned error instead of value
            raise KeysightError("Error reading trigger", value)
        return bool(value)
    
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
            waveform_number: The unique number to be associated
               with the waveform as an identifier. If None, waveform_number
               will be assigned automatically in a way that prevents naming
               collisions. If any are assigned manually, user is responsible
               for consequences of naming collisions.
            append_zero: Appends a zero at the end of the waveform if none
               present, to prevent nonzero signal from continuing after
               waveform stops.
        Note: when loaded into memory, all waveforms will have zeros appended
        to end until length of base array is at least 5. Thus, putting in a
        single value to output a constant signal will give unexpected results.
        '''
        if append_zero and arr[len(arr)-1] != 0:
            np.append(arr, [0])
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
            module: The module to which to save the waveform'''
        module.loadWaveform(self)
    
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
                clock plays next value in waveform; reduces frequency.'''
        channel.queue(self.getWaveformNumber(), trigger_mode, delay,
                             cycles, prescaler)
        
    def __str__(self):
        '''Returns a string representation of the waveform'''
        return ("Waveform_number: " + str(self._waveform_number) + "\n" +
                str(self.getBaseArray()))
        
        
"----------------------------------------------------------------------------"
'''Internal helper class with useful methods for implementation. Not intended
to be exported.'''
class FileDecodingTools:
    @staticmethod #internal helper method
    def _isCommentOrWhitespace(line):
        '''Helper method that indicates whether a line in a file is a blank
        line (containing only whitespace characters) or commented out (starting
        with #), in which case it should be ignored.'''
        stripped = line.strip()
        return stripped == "" or stripped[0] == "#"
    
    @staticmethod #internal helper method
    def _splitByColon(line):
        '''Splits a line of text. Delimiter is a colon : followed by any
        amount of whitespace.
        Params:
            line: The line to split
        Returns: A list of the two substrings separated by the colons'''
        substrings = []
        for string in line.split(":", 2):
            substrings.append(string.strip())
        return substrings
    
    @staticmethod #internal helper method
    def _readHardwareConfigFile(filename):
        '''Obtains the chassis number and module information from a Keysight
            hardware config file .keycfd.
        Params:
            filename: The name and path of the relevant file
        Returns:
            chassis_number: The number of the chassis
            modules_dict: A dictionary of {Slot number: module type} to
                feed into chassis/HVI object constructor.
            If an error, both are None.'''
                
        if not filename.endswith(KeysightConstants.MODULE_CONFIG_SUFFIX):
            filename.append(KeysightConstants.MODULE_CONFIG_SUFFIX)
        try:
            file = open(filename, 'r')
            lines = file.readlines()
        except Exception as e:
            print("Error opening file: " + str(e))
            return None, None
        finally:
            file.close()
            
        chassis_number = None
        input_modules = None
        output_modules = None
        for line in lines:
            if not FileDecodingTools._isCommentOrWhitespace(line):
                fragments = FileDecodingTools._splitByColon(line)
                header = fragments[0].lower()
                if header == "chassis number":
                    chassis_number = int(fragments[1])
                elif header == "input modules":
                    input_modules = ast.literal_eval(fragments[1])
                elif header == "output modules":
                    output_modules = ast.literal_eval(fragments[1])
        modules_dict = {}
        for module_number in input_modules:
            modules_dict[module_number] = ModuleType.INPUT
        for module_number in output_modules:
            modules_dict[module_number] = ModuleType.OUTPUT
        return chassis_number, modules_dict