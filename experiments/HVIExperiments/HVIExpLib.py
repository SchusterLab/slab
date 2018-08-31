# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:06:24 2018

@author: Josie Meyer (jcmeyer@stanford.edu)

This class is the parent class for HVI enabled single qubit experiments.
Because the HVI paradigm is a very different paradigm than our previous
experiments, this class replicates much of the functionality of, and borrows
some code from, the QubitPulseSequenceExperiment, but with a different
underlying architecture.
"""

#import error fix
import sys
sys.path.append(r'C:\_Lib\python\slab\instruments\keysight')


from slab.experiment import Experiment
from slab.instruments.keysight import KeysightLib as key
from threading import Thread
from queue import Empty, Queue
import numpy as np
import os
import time
import json
import ast


'''-------------------------------------------------------------------------'''

class ControlModuleConstants:    
    '''This class outputs a number of constants that should be used as a standard
across the experiments. One 'control module' is responsible for talking to the
computer and performing iteration, ensuring data locality. This control module
has designated registers that should not be used for other purposes, to ensure
continuity across experiments.

Registers are numbered 0-15.

General philosophy:
    Small number registers: Constants written by computer
    High numbers: Loop variables
    Register 15: Reserved for flag communicating status with computer. Could
        create a mask that allows individual bits to be read, but for most
        simple experiments all we care about is a boolean flag about whether
        HVI is running or not.'''

    '''Number of the control module. Could override if needed, but better not
    to unless a compelling reason!'''
    CONTROL_MODULE = 6

    class Registers: #auxiliary class within ControlModuleConstants
        '''Parameters that control iteration in the HVI. Think:
        for SWEEPS_LOOP_VAR in range(NUM_SWEEPS):
            for PARAM_LOOP_VAR in range(PARAM_START, PARAM_STOP, PARAM_STEP):
                #do something'''
        NUM_SWEEPS = 0 #indicates number of sweeps; set by computer
        PARAM_START = 1 #starting value of parameter to vary
        PARAM_STOP = 2 #value at which parameter stops
        PARAM_STEP = 3 #step value
        #loop variables that vary during experiment
        PARAM_LOOP_VAR = 13
        SWEEPS_LOOP_VAR = 14
    
        '''Flag used for communicating experiment status with computer'''
        STATUS = 15 #used to communicate whether experiment is done
        
        
'''-------------------------------------------------------------------------'''

class HVIEnabledQubitExperiment(Experiment):
    def __init__(self, config_file=None,
                 liveplot_enabled = True, 
                 control_module = ControlModuleConstants.CONTROL_MODULE,
                 save_in_register = False,
                 HVI_query_pause = 0.5,
                 **kwargs):
        '''Initializes the experiment.
        Params:
            prefix: The prefix to append to data file names
            config_file: The file path with config information from the experiment
            module_config_kile: The .keycfg file that lists the module config
                of the chassis.
            liveplot_enabled: Whether to enable liveplot for the experiment
            HVI_file: The file path where the HVI is stored
            control_module: The module responsible for controlling the overall
                execution of the experiment, by controlling the decision making
                process within loops and communicating with the HVI
            save_in_register: Whether to save data in local register for
                fast analysis post-data collection. Typically set to False to
                conserve RAM and increase speed. If you wish to use the
                postProcess feature, then set to true.
            HVI_query_pause: Time (in seconds) to pause between queries of
                whether the HVI is finished. Adjust as necessary to balance
                CPU time with latency.
            kwargs: Passed to the experiment class
        '''
        
        Experiment.__init__(self, '', '', config_file, liveplot_enabled, 
                            **kwargs)
        self._pulse_sequence = self.buildHVIPulseSequence()
        self._HVI = self.initializeExperiment()
        self._control_module = self._HVI.getModule(control_module)
        self._data = {} #maps channels to listen to the data they produce
        self._HVI_query_pause = HVI_query_pause
        self._save_in_register = save_in_register
        self._dispatcher = Dispatcher(self.channelsListening(),
                                      save_in_register)
    
    def addChannelToListen(self, channel):
        '''Adds a channel to which the experiment will listen for data.
        Params:
            channel: The channel object from which to listen'''
        self._data[channel.serialize()] = []
        
    def channelsListening(self):
        '''Returns a list of the channels to which the experiment is listening
        for data.
        '''
        channels_listening = []
        for serialized_channel in self._data:
            _, module, channel = key.Tools.deserializeChannel(
                    serialized_channel)
            channels_listening.append(self._HVI.getChannel(module, channel))
        return channels_listening
    
    def buildHVIPulseSequence(self):
        '''Builds the underlying pulse sequence object.
        This method should be overridden in subclasses.
        Returns: A pre-initialized HVIPulseSequence object (or a subclass
        better suited for experiment).
        '''
        #fill in here using the pulse_sequence.addWaveform and
        #pulse_sequence.setNumberOfTrials and setIterationParameters methods.
        #Customize in subclasses
        return None
    
    def addWorkers(self, *workers):
        '''Adds list of constructed workers to the experiment. Use in
        subclasses. Call after constructing workers and after calling __init__
        method of this (parent) class.
        
        Params:
            workers: Any number of Worker objects that have been created.'''
        for worker in workers:
            self._dispatcher.addWorker(worker)
        
    def initializeExperiment(self):
        '''Initializes the experiment. Constructs chassis and HVI and loads
        pulse sequence. Will probably be extended in subclasses
        Returns: the HVI object'''
        HVI = self._pulse_sequence.buildHVI()
        if HVI is None:
            print("Error constructing HVI")
        return HVI
    
    def go(self):
        '''Starts the experiment. May be extended/overridden in subclasses if
        necessary. However, to the extent possible, to preserve workflow,
        only override self.pre_run() and self.post_run() if at all possible.'''
        self.pre_run()
        self.takeData()
        self.waitForHVIToComplete()
        self.stopDataHandling()
        self.post_run()
        
    def takeData(self):
        '''Begins taking data.'''
        self.writeHVIDoneFlag(False)
        self._dispatcher.takeData()
        self._HVI.start()
        
    def waitForHVIToComplete(self):
        '''Waits until the signal is received from the HVI that all data has
        been taken.'''
        while not self.readHVIDoneFlag():
            time.sleep(self._HVI_query_pause)
        
    def stopDataHandling(self):
        '''Stops data handling. If we are saving the data in local register,
        also waits until all the data is ready.'''
        self._dispatcher.stopAllWhenReady()
        if self._save_in_register:
            for handler in self._dispatcher._data_handlers:
                handler.join()
    
    def post_run(self):
        '''Method used to perform post-processing on the data. Override in
        subclasses if this functionality is desired. In order to be useful,
        you probably want to have enabled save_in_register in the init method.
        Mimics function of existing experiment class.
        '''
        pass
    
    def pre_run(self):
        '''Method called before the HVI is started and data is taken. Override
        in subclasses if this functionality is desired. Mimics function of
        existing experiment class.'''
        pass
    
    def getChannel(self, module_number, channel_number):
        '''Gets a channel by module and channel number. The primary reason you
        would want to call this method is in post-processing, to retrieve a
        particular channel (perhaps to get its data via self.getData()).
        
        Params:
            module_number: The module number where the channel is located
            channel_number: The channel number of the desired channel
        Returns: The desired channel'''
        return self._HVI.getChannel(module_number, channel_number)
    
    def getData(self, channel):
        '''Gets the saved data for a particular channel.
        Params:
            channel: The channel object associated with the data.
        Returns: The data, or an empty list if the data was not collected
        (either because of an error, or because save_in_register disabled.'''
        return self._data[channel.serialize()]
    
    def deleteData(self, channel):
        '''Deletes all the data associated with a channel in RAM. Does not
        affect saved data.
        Params:
            The channel whose data is to be deleted.
        Returns:
            Whether data was deleted (False if data already deleted or never
            saved).'''
        try:
            data = self._data[channel.serialize()]
        except KeyError:
            data = None
        if not data:
            return False
        data.clear()
        return True

    def writeHVIDoneFlag(self, value):
        '''Writes the signal register regarding whether HVI is done. Typically
        will only be used to zero the port.
        Params:
            value: The value to write to the signal flag. Typically False.'''
        self._control_module.writeRegister(
                ControlModuleConstants.Registers.STATUS, bool(value))
    
    def readHVIDoneFlag(self):
        '''Reads the signal register regarding whether HVI is done.
        Returns: The value on the port (True = on, False = off).
        '''
        return self._control_module.readRegister(
                ControlModuleConstants.Registers.STATUS)
'''-------------------------------------------------------------------------'''

class Stoppable():
    '''Interface class that offers simple way to control threads and loops. Classes
    implementing this interface must read self._stop_flag and self._kill_flag
    periodically. They must halt execution as soon as practical (usually within
    one loop of a constantly running loop) when self._kill_flag becomes True.
    They must halt execution once their current pipeline is empty upon
    self._stop_flag becoming True.'''
    
    def __init__(self):
        '''Initializes the underlying flags'''
        self._stop_flag = False
        self._kill_flag = False
        
    def stopWhenReady(self):
        '''Indicates that the program is ready to be stopped when data is done
        being processed. '''
        self._stop_flag = True
        
    def kill(self):
        '''Kills the underlying thread as soon as it is safe to do so.
        May cause loss of data. Only use in emergencies, or when the
        data would be discarded anyway.'''
        self._kill_flag = True

'''-------------------------------------------------------------------------'''

class Dispatcher():
    '''Dispatcher class for handling data in real time. Stores the data
    internally in Experiment class or directs Worker classes (see below)
    to enable multithreaded real time data processing..''' 

    def __init__(self, channels = [], save_in_register = False):
        '''Initializes the Data Handler class.
        Params:
            channels: A list of the channel objects that the DataHandler is
                listening to.
            save_in_register: Whether to save data in local register for
                fast access after the experiment. If data is not explicitly
                needed immediately after the experiment, set this parameter
                to False to save time and RAM.'''
        
        self._save_in_register = save_in_register
        self._data_handlers = {}
        self._workers = []
        for channel in channels:
            self._data_handlers[channel.serialize()] = DataHandler(
                    save_in_register)
        self._started = False
    
    def addWorker(self, worker):
        '''Adds a worker to the dispatcher.'''
        self._workers.append(worker)
        for queue in worker.queues():
            self._data_handlers[queue.channel().serialize()].addQueue(queue)
            
    def takeData(self):
        '''Starts all the data handler and worker threads'''
        self._started = True
        for data_handler in self._data_handlers:
            data_handler.start()
        for worker in self._workers:
            worker.start()
            
    def stopAllWhenReady(self):
        '''Stops all the Stoppable data handlers and workers that the
        dispatcher controls when they are done processing data. No loss of
        data.'''
        for data_handler in self._data_handlers:
            data_handler.stopWhenReady()
        for worker in self._workers:
            if isinstance(worker, Stoppable):
                worker.stopWhenReady()
    
    def killDataHandlersAndStop(self):
        '''Kills all the data handlers so they do not accept any more data,
        then stops Stoppable workers when they are done processing the data
        queued so by the data handlers. May lose data that hasn't been handled
        yet.'''
        for data_handler in self._data_handlers:
            data_handler.kill()
        for worker in self._workers:
            if isinstance(worker, Stoppable):
                worker.stopWhenReady()
            
    def killAll(self):
        '''Kills every data handler and killable worker as fast as possible.
        Bye bye data.'''
        for data_handler in self._data_handlers:
            data_handler.kill()
        for worker in self._workers:
            if isinstance(worker, Stoppable):
                worker.kill()
                
    def hasStarted(self):
        '''Returns whether the Dispatcher has started taking data.'''
        return self._started
                
    def isHandlingData(self):
        '''Returns whether any of the data handlers are still processing data.
        Note: it is still possible that saving, plotting, etc. are going on in
        separate threads, but we no longer are actively taking data from the
        Keysight module.'''
        if not self._started:
            return False
        for handler in self._data_handlers:
            if not handler.isDone():
                return True
        return False
'''-------------------------------------------------------------------------'''

class DataHandler(Thread, Stoppable):
    
    def __init__(self, channel, save_in_register = False, experiment = None):
        '''Initializes the Data Handler.
        Params:
            channel: The KeysightChannelIn object from which the DataHandler
                is receiving data.
            save_in_register: Whether to save the data in register for faster
                processing later, at expense of RAM capacity and speed now.
            experiment: A reference back to the experiment. Only necessary
                if save_in_register is enabled.'''
                
        self._channel = channel
        self._queues = []
        self._save_in_register = save_in_register
        self._experiment = experiment
        
    def addQueue(self, queue):
        '''Adds a ChannelQueue to the handler as a destination for data.'''
        if self._channel is not queue.channel():
            raise QueueError('''Queue appended to data handler for different 
                             thread.''')
        self._queues.append(queue)
        
    def start(self):
        '''Starts the Data Handler. Has no effect if we are neither saving
            the data nor have it assigned to any workers, because there is no
            use running unnecessary threads that waste CPU time. 
        Extends the Thread.start() method.
        Returns: Whether the thread was indeed started.'''
        if self._save_in_register or self._queues:
            Thread.start(self)
            return True
        else:
            return False
        
    def run(self):
        '''Loop that runs in the background while data is being collected.
        Overrides Thread.run(). Exits when kill flag is set to True, or
        if stop flag is set to True and there is no more data.'''
        while not self._kill_flag:
            data = self._channel.readDataBuffered()
            if DataHandler._isError(data):
                if self._stop_flag or self._kill_flag:
                    break
            else:
                for queue in self.queues():
                    queue.put(data)
                if self._save_in_register:
                    self._experiment._data[self._channel.serialize()].append(
                            data)
        
    def isDone(self):
        '''Returns whether the data handler is done processing data after kill()
        or stopWhenNoMoreData() is called. Useful
        after stopWhenNoMoreData() to verify processing has actually stopped.
        '''
        return not self.is_alive()
    
    @staticmethod #private helper method
    def _isError(data):
        '''Checks whether the output of readDataBuffered is an error or
            represents valid data.'''
        return type(data) != np.ndarray or len(data) == 0
    
    
'''-------------------------------------------------------------------------'''

class Worker(Thread):
    '''Base class for workers, which are autonomous threads that
    handle data in real time.'''
    
    def __init__(self, num_channels = 1):
        '''Initializes the Worker.
        Params:
            num_channels: The number of channels that the worker will process.
        '''
        Thread.__init__(self)
        self._queues = []
        self._num_channels = num_channels
        
    def getNumberOfChannels(self):
        '''Returns the number of channels that the worker requires.'''
        return self._num_channels
        
    def queues(self):
        '''Returns the list of queues from which the Worker reads data. Used
        to set up the Dispatcher.'''
        return self._queues
    
    def start(self):
        '''Launches the worker. May need to extend (with prior instructions)
        in subclasses.'''
        Thread.start(self)
        
    def run(self):
        '''Code called when Worker is activated by start() method. Override
        in subclasses.'''
        pass
        
    def assignChannels(self, *channels):
        '''Assigns the channels in sequential order from which the worker will
            be taking data.
        Params: 
            channels: Any number of channels (corresponding to self._num_channels,
                itself determined by the subclass) in sequential order that the
                worker will read data from. A channel may be assigned to
                multiple workers, which will receive independent queues of data,
                but the data will not be stored in RAM twice.
        Throws: QueueError if wrong number of channels are assigned.
            '''
        if len(channels) != self._num_channels:
            raise QueueError("""Wrong Number of channels assigned to worker of
                             type """ + str(type(self)))
        for channel in channels:
            self._queues.append(ChannelQueue(channel))
            
'''-------------------------------------------------------------------------'''

class ChannelQueue(Queue):
    '''Thread-safe queue for storing of data retrieved from Keysight modules.
    Inherits all methods from queue.Queue, including put() and get().
    Only difference is that, for convenience, ChannelQueue also stores its own
    channel.'''
    
    def __init__(self, channel):
        '''Initializes the queue and extends init method of Queue.
        Params:
            channel: The channel object form which the queue will obtain data.
        '''
        Queue.__init__(self)
        self._channel = channel
        
    def channel(self):
        '''Returns the channel object from which the queue receives data.'''
        return self._channel
    
'''-------------------------------------------------------------------------'''

class QueueError(RuntimeError):
    '''Exception thrown when queues have not been assigned correctly.'''
    pass #no extension necessary beyond renaming      
    
    
'''-------------------------------------------------------------------------'''

class Saver(Worker, Stoppable):
    '''Helper class that saves data to disk as a background thread without
        bogging down the rest of the computer.'''
        
    DATA_WAIT_TIMEOUT = 0.5
    '''Timeout after stopWhenReady() called and no more data available before
        quitting (sec)''' #FEEL FREE TO EDIT AS NEEDED
        
    def __init__(self, channel, filepath = None, prefix = None):
        '''Initializes the Saver object.
        
        Params:
            channel: The channel object from which the Saver will get and save
                data.
            filepath: The path where the data is to be saved. If None, defaults
                to the directory containing the code initially launched.
            prefix: The prefix of the file where the data is saved. Each trial
                will be saved as XXX#####.npy where ###### is the order in
                which the trial ran.'''
        if filepath is None:
            filepath = os.path.dirname(os.path.realpath(__file__))
        if prefix is None:
            prefix = "data" + str(channel.channelNumber())
        Worker.__init__(self, num_channels = 1)
        Stoppable.__init__(self)
        self.assignChannels(channel)
        self._filepath = filepath
        self._prefix = prefix
        
    def run(self):
        '''Runs in the background, checking for data and saving it. Overrides
        the Worker.run() method.'''
        count = 0
        while not self._kill_flag:
            timeout = Saver.DATA_WAIT_TIMEOUT if self._stop_flag else (
                    key.KeysightConstants.INFINITY)
            try:
                data = self._queues[0].get(timeout=timeout)
            except(Empty): #Queue is empty and timeout has elapsed
                break
            np.save(self._filepath + self._prefix + str(count).zfill(5), data)
            count += 1
            
'''-------------------------------------------------------------------------'''

class Plotter1Var(Worker):
    pass #TODO: Implement this class

class Plotter2Var(Worker):
    pass #TODO: Implement this class
    
'''-------------------------------------------------------------------------'''

class HVIPulseSequence():
    '''Class that handles the control information for a pulse sequence
    experiment using the Keysight HVI.'''
    
    FILE_SUFFIX = ".hvips"
    KEYSIGHT_CONFIG_SUFFIX = ".keycfg"
    
    def __init__(self, HVI_file = None, module_config_file = None):
        '''Initializes the empty pulse sequence object.
        Params:
            HVI_file: The file path and name of the HVI file.
            module_config_file: The file path and name of the Keysight module
                config file .keycfg.
            chassis_number: The number of the chassis'''
        self._HVI_file = HVI_file
        self._waveforms = {}
        self._registers = {}
        self._params = {}
        self._keycfg_file = module_config_file
        
    def setHVIFile(self, filename):
        '''Sets the HVI file that is to be used to execute the pulse sequence.
        Params:
            filename: The file path and name of the HVI file.'''
        self._HVI_file = filename
        
        
    def addWaveform(self, waveform, waveform_number = None, modules = [1]):
        '''Adds a waveform to the pulse sequence that will be played
        Params:
            waveform: The waveform data, either as a KeysightLib.Waveform
                object or as a numpy array.
            waveform_number: The number of the waveform, which must be
                set by the user to avoid naming collisions. Not used in case
                of waveform objects, which already have waveform number. If
                a numpy array is loaded and waveform_number is None, waveform
                number assigned automatically according to documentation in
                KeysightLib.Waveform
            modules: A list of the modules to which the waveform should be
                loaded.'''
        if not isinstance(waveform, key.Waveform): #need to make an object
            waveform = key.Waveform(waveform, waveform_number)
        self._waveforms[waveform.getWaveformNumber()] = (waveform, modules)
    
    def setNumberOfSweeps(self, num_sweeps):
        '''Sets the number of sweeps for which the experiment will run.
        Params:
            num_trials: The number of trials for which the experiment will run
        '''
        self._registers[(ControlModuleConstants.CONTROL_MODULE,
                    ControlModuleConstants.Registers.NUM_SWEEPS)] = num_sweeps
        
    def setIterationParameters(self, start, stop, step):
        '''Sets values of any experiment parameters over which to iterate,
        i.e. those that are changed from run to run. Essentially implements
        a for loop in HVI.
        Params:
            start: The starting value of the parameter
            stop: The end value of the parameter. Stops before this value,
                not upon reaching it, in the same manner as Python range()
                function in a for loop
            step: How much the parameter is incremented each step. Can be
                negative.
            '''
        self._registers[(ControlModuleConstants.CONTROL_MODULE,
                         ControlModuleConstants.Registers.PARAM_START)] = start
        self._registers[(ControlModuleConstants.CONTROL_MODULE,
                         ControlModuleConstants.Registers.PARAM_STOP)] = stop
        self._registers[(ControlModuleConstants.CONTROL_MODULE,
                         ControlModuleConstants.Registers.PARAM_STEP)] = step
        
    def setRegister(self, module_number, register_number, value):
        '''Sets the value on an arbitrary register. Choose registers not used for
            iteration parameters; typically on a module other than the main
            control module so as not to interfere with iteration parameters.
        Params:
            module_number: The module number where the port is located.
            port_number: The number of the port whose value is to be set.
            value: The value to be written to the port.
        '''
        self._registers[(module_number, register_number)] = int(value)
        
    def setPXIPort(self, module_number, port_number, value):
        '''Sets the boolean value of an arbitrary PXI port. Note that PXI port
        0 on the main control module is reserved for experiment implementation!
        Params:
            module_number: The module number where the port is located.
            port_number: The number of the port whose value is to be set.
            value: The value to be written to the port.'''
        self._PXI_ports[(module_number, port_number)] = int(bool(value))
        
    def save(self, filename):
        '''Saves the pulse sequence object (serialized) as .hvips file. Will
            overwrite existing file if present.
        Params:
            filename: The file path and name where the file is to be saved.
        Returns: Whether the save was successful. Prints any error messages.
        '''
        if not filename.endswith(HVIPulseSequence.FILE_SUFFIX):
            filename += HVIPulseSequence.FILE_SUFFIX
        try:
            with open(filename, 'w') as file:
                file.write(self.serialize())
            return True
        except Exception as e:
            print("Error writing file: " + str(e))
            return False
       
    def serialize(self):
        '''Serializes the pulse sequence object. Useful for saving or
        transmitting the object.
        Returns: a string representing the serialized object.
        '''
        lines = [] #speed process by not creating/destroying strings
        lines.append("$HVI File: ")
        lines.append("File: " + self._HVI_file)
        lines.append("$Keysight Config File: ")
        lines.append("File: " + self._keycfg_file)
        lines.append("$Waveforms: ")
        for waveform_number in self._waveforms:
            lines.append("Number: " + str(waveform_number))
            lines.append("Array: " + str(
                self._waveforms[waveform_number][0].getBaseArray().toList()))
            lines.append("Modules: "+ str(self._waveforms[waveform_number][1]))
        lines.append("$Registers: ")
        for module_number, register in self._registers:
            lines.append("Module: " + str(module_number))
            lines.append("Port: " + str(register))
            lines.append("Value: " + str(self._HVI_ports[
                    (module_number, register)]))
        for module_number, port_number in self._PXI_ports:
            lines.append("Module: " + str(module_number))
            lines.append("Port: " + str(port_number))
            lines.append("Value: " + str(self._HVI_ports
                                         [(module_number, port_number)]))
        lines.append("$Params:")
        lines.append("Params: " + json.dumps(self._params))
        return '\n'.join(lines) #concatenate all the lines separated by newline
    
    @staticmethod #factory method
    def deserialize(serialized_sequence):
        '''Deserializes the string representation of the pulse sequence object.
        Params:
            serialized_sequence: The serial (string) representation of the
                pulse sequence object.
        Returns: The deserialized pulse sequence object.
        May throw: ValueError if not a valid serial representation.
        '''
        chunks = HVIPulseSequence._chunk(serialized_sequence)
        pulse_sequence = HVIPulseSequence(HVI_file = chunks["HVI File"][1])
        pulse_sequence._parse_waveform_chunk(chunks["Waveforms"])
        HVIPulseSequence._parse_ports_chunk(chunks["Registers"], 
                                          pulse_sequence._registers)
        HVIPulseSequence._parse_ports_chunk(chunks["PXI Ports"],
                                            pulse_sequence._PXI_ports)
        pulse_sequence._params = json.loads(chunks["Params"][1])
        return pulse_sequence
            
    @staticmethod #factory method
    def load(self, filename, buildHVI = False):
        '''Loads the pulse sequence object from file.
        Params:
            filename: The name and path from which the file is to be loaded
            buildHVI: If true, returns the fully initialized and prepared
                HVI (requires chassis to be connected).
                If false, returns the only the pulse sequence
        Returns: Value specified by buildHVI, or None if unable to load (prints
                 error)'''
        if not filename.endswith(HVIPulseSequence.FILE_SUFFIX):
            filename += HVIPulseSequence.FILE_SUFFIX
        try:
            with open(filename, 'r') as file:
                pulse_seq = HVIPulseSequence.deserialize(file.read())
                if buildHVI:
                    return pulse_seq.buildHVI()
                else:
                    return pulse_seq
        except Exception as e:
            print("Unable to load file: ", str(e))
            return None
        
    def buildHVI(self):
        '''Builds the HVI by loading the HVI file and all waveforms and 
        parameters. Must be connected to modules for this function to work,
        or an IOError will be thrown from HVI object constructor.
        Returns:
            The HVI object, or None if HVI cannot be initialized
        '''
        chassis_number, modules = key.FileDecodingTools._readHardwareConfigFile(
                self._keycfg_file)
        HVI = key.HVI(self._HVI_file, chassis_number, modules)
        for waveform, modules in self._waveforms:
            for module_number in modules:
                waveform.loadToModule(HVI.getModule(module_number))
        for module, register in self._registers:
            HVI.getModule(module).writeRegister(register, 
                         self._HVI_ports[(module, register)])
        for module, port in self._PXI_ports:
            HVI.getModule(module).writePXI(port, 
                               self._PXI_ports[(module, port)])
        return HVI
            
            
    #internal helper methods
    @staticmethod
    def _chunk(text):
        '''Parses the serialized pulse sequence object into chunks. Removes
            comments (starting with #) and blank lines.
        Params:
            text: The text to be broken down into chunks, which are delimited
                by lines beginning with the '$' character.' Should be a single
                string.
        Returns: a dictionary consisting of the text after the '$' of each
            chunk header as the key, and a list of succeeding lines (strings)
            as value, themselves split at the colon.
        Throws: ValueError if given an improperly formatted string.'''
        
        most_recent_key = None
        chunks = {}
        for original_line in text.splitLines():
            line = original_line.strip()
            if not key.FileDecodingTools._isCommentOrWhitespace(line):
                if line[0] == "$": #denotes beginning of chunk
                    most_recent_key = line[1:].strip()
                    chunks[most_recent_key] = []
                else:
                    try:
                        chunks[most_recent_key].append(
                                key.FileDecodingTools._splitByColon(line))
                    except KeyError: #most_recent_key is still None
                        raise ValueError("Invalid serial: missing '$'")
        return chunks
    
    def _parse_waveform_chunk(self, chunk):
        '''Parses the chunk of the serialized representation that encodes 
        information on waveforms.
        Params:
            chunk: The waveform data after calling _chunk["Waveforms"]. Should
                be an array of length-2 arrays.
        May raise: ValueError for invalid chunk'''
        number = None
        array = None
        modules = None
        for line in chunk:
            header = line[0].lower()
            if header == "number":
                if number is not None: #avoids problematic first iteration
                    self.addWaveform(key.Waveform(array, number, modules))
                number = int(line[1])
            elif header == "array":
                array = np.array(ast.literal_eval(line[1]))
            elif header == "modules":
                modules = ast.literal_eval(line[1])
            else:
                raise ValueError('''Invalid serial representation.\nInvalid 
                                 header in waveform subsection: ''' + header)
        self.addWaveform(key.Waveform(array, number, modules)) #last waveform
        
    @staticmethod
    def _parse_ports_chunk(chunk, destination):
        '''Parses the chunk of the serialized representation that encodes
        information on values to write to HVI or PXI ports.
        Params:
            chunk: The ports data after calling _chunk. Should be an
                array of length-2 arrays.
            destination: The dictionary into which to write values. i.e.
                self._PXI_ports or self._HVI_ports
        May raise: ValueError for invalid chunk'''
        module = None
        port = None
        value = None
        for line in chunk:
            header = line[0].lower()
            if header == "module":
                if module is not None: #avoids problematic first iteration
                    destination[(module, port)] = value
                module = int(line[1])
            elif header == "port":
                port = int(line[1])
            elif header == "value":
                value = int(line[1])
            else:
                raise ValueError('''Invalid serial representation.\nInvalid
                                 header in HVI ports subsection: ''' + header)
        destination[(module, port)] = value #last port
        
        
'''-------------------------------------------------------------------------'''


    