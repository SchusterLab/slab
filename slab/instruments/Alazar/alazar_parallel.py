from __future__ import division

import ctypes as C
import numpy as np
import warnings
from multiprocessing import cpu_count
warnings.simplefilter('ignore')

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

records_per_buffer = 10
records_per_acquisition = 0x7fffffff
bytes_per_sample = 1
samples_per_record = 2**14
buffers_per_merge = 100
buffer_count = cpu_count()
iterations = 10000
timeout = 1000
samples_per_ms = 1000000
samples_per_second = samples_per_ms * 1e3
seconds_per_record = 17e-6
seconds_per_buffer = seconds_per_record * records_per_buffer
bytes_per_record = bytes_per_sample * samples_per_record
bytes_per_buffer = bytes_per_record * records_per_buffer
samples_per_buffer = samples_per_record * records_per_buffer
seconds_per_plot = 1


class AlazarException(Exception):
    pass

class Bunch(object):
    def __init__(self, d):
        self.__dict__.update(d)

def validate_config_parameters(config, trigger_time):
     c = Bunch(config)

def average_worker(ignored, buf, buf_ready_event, buf_post_event, avg_buffer, buffers_merged):
    arr = np.frombuffer(buf.get_obj(), U8)
    avg_buffer_arr = np.frombuffer(avg_buffer.get_obj(), C.c_longdouble)
    sum_buffer = np.zeros(bytes_per_buffer, np.uint32)
    my_buffers_completed = 0
    for _ in range(iterations):
        buf_ready_event.wait()

        buf_ready_event.clear()
        sum_buffer += arr
        buf_post_event.set()
        my_buffers_completed += 1

        if my_buffers_completed % buffers_per_merge == 0:
            with avg_buffer.get_lock():
                n = buffers_merged.value
                avg_buffer_arr *= (n - 1.) / n
                avg_buffer_arr += sum_buffer / (n*buffers_per_merge)
                buffers_merged.value += 1
            sum_buffer.fill(0)

def average_reshape_worker(reshape_fn, buf, buf_ready_event, buf_post_event, avg_buffer, buffers_merged):
    arr = np.frombuffer(buf.get_obj(), U8)
    avg_buffer_arr = np.frombuffer(avg_buffer.get_obj(), C.c_longdouble)
    sum_buffer = np.zeros(len(avg_buffer_arr), np.uint32)
    my_buffers_completed = 0
    for _ in range(iterations):
        buf_ready_event.wait()

        buf_ready_event.clear()
        reshape_fn(sum_buffer, arr)
        buf_post_event.set()
        my_buffers_completed += 1

        if my_buffers_completed % buffers_per_merge == 0:
            with avg_buffer.get_lock():
                n = buffers_merged.value
                avg_buffer_arr *= (n - 1.) / n
                avg_buffer_arr += sum_buffer / (n*buffers_per_merge)
                buffers_merged.value += 1
            sum_buffer.fill(0)

def reshape_collapse_records(res, arr):
    samples_per_record = len(res)
    records_per_buffer = int(len(arr) / samples_per_record)
    for i in range(records_per_buffer):
        res += arr[i*samples_per_record:(i+1)*samples_per_record]

def single_shot_worker(proc_fun, buf, buf_ready_event, buf_post_event, result_buffer, buffers_merged):
    arr = np.frombuffer(buf.get_obj(), U8)
    results = []
    my_buffers_completed = 0
    for _ in range(iterations):
        buf_ready_event.wait()
        buf_ready_event.clear()
        results.append(proc_fun(arr))
        buf_post_event.set()
        my_buffers_completed += 1

    with result_buffer.get_lock():
        n = buffers_merged.value
        result_buffer[n*iterations:(n+1)*iterations] = np.array(results)
        buffers_merged.value += 1

from slab.instruments import AlazarConfig
class AlazarParallelConfig(AlazarConfig):
    buffers_per_merge = 100
    
    @property
    def merges_per_acquisition(self):
        return self.merges

def acquire_avg_data_parallel(worker, proc_fun, result_shape, plot=True):
    """
    :param worker: Function which the subordinate threads execute as their target
    :param proc_fun: Function used by the subordinate threads to process their buffers
    :param result_shape: Shape of the buffer which is the result of the entire acquisition.
    """
    from multiprocessing import Process, Array, Value, Event
    from slab.instruments import Alazar, AlazarConfig
    from slab.plotting import ScriptPlotter
    import time

    acquire_buffer_time = samples_per_buffer / (samples_per_second)
    print 'Acquire buffer time %.2e' % acquire_buffer_time
    print 'Inter-buffer time %.2e' % seconds_per_buffer
    print 'Duty Cycle', acquire_buffer_time / seconds_per_buffer

    # Configure card
    config={'clock_edge': 'rising', 'clock_source': 'reference',  
            'trigger_coupling': 'DC',  'trigger_operation': 'or', 
            'trigger_source2': 'disabled','trigger_level2': 1.0, 'trigger_edge2': 'rising', 
            'trigger_level1': 0.1, 'trigger_edge1': 'rising', 'trigger_source1': 'external', 
            'ch1_enabled': True,  'ch1_coupling': 'AC', 'ch1_range': 1, 'ch1_filter': False, 
            'ch2_enabled': False, 'ch2_coupling': 'AC','ch2_range': 1, 'ch2_filter': False,            
            'bufferCount': buffer_count,'recordsPerBuffer': records_per_buffer, 'trigger_delay': 0, 'timeout': timeout,
            'samplesPerRecord':samples_per_record, 'recordsPerAcquisition':records_per_acquisition, 'sample_rate': samples_per_ms}

    card = Alazar(AlazarConfig(config))
    card.configure()
    Az = card.Az
    handle = card.handle

    def assert_az(retCode, buffers_acquired, name):
        Az.AlazarErrorToText.restype = C.c_char_p
        if retCode != 512:
            raise AlazarException("::".join(map(str, 
                (name, buffers_acquired, retCode, Az.AlazarErrorToText(U32(retCode))))))    
    
    try:
        # Initialize buffers
        buffers = [ Array(U8, bytes_per_buffer) for _ in range(buffer_count) ]
        for b in buffers:
            ret = Az.AlazarPostAsyncBuffer(handle, b.get_obj(), U32(bytes_per_buffer))
            assert_az(ret, 0, 'Initial Post Buffer')
        avg_buffer = Array(C.c_longdouble, result_shape)
        
        # Initialize threads  
        bufs_merged = Value(U32, 1)
        buf_ready_events = [ Event() for _ in range(buffer_count) ]
        buf_post_events = [ Event() for _ in range(buffer_count) ]
        workers = [ Process(target=worker, args=(proc_fun, b, bre, bpe, avg_buffer, bufs_merged)) 
                    for b, bre, bpe in zip(buffers, buf_ready_events, buf_post_events) ]
        
        for w in workers:
            w.start()
        time.sleep(1)
        
        import atexit
        atexit.register(lambda: [w.terminate() for w in workers])


        # Initialize things used during capture
        if plot:
            plotter = ScriptPlotter()
            plotter.init_plot('Data', rank=1, accum=False)
        buffers_acquired, buffers_completed, plot_count = 0, 0, 0
        start_time = time.time()
        

        # Begin capture
        ret = Az.AlazarStartCapture(handle)
        assert_az(ret, 0, "Start Capture")
        unready_count = 0
        while buffers_completed < (iterations * buffer_count):
            
            # Post all completed buffers
            while buf_post_events[buffers_completed % buffer_count].is_set():
                buf_post_events[buffers_completed % buffer_count].clear()
                buf = buffers[buffers_completed % buffer_count]
                with buf.get_lock():
                    ret = Az.AlazarPostAsyncBuffer(handle, buf.get_obj(), U32(bytes_per_buffer))
                    assert_az(ret, buffers_acquired, 'Post Buffer')
                buffers_completed += 1        
            
            # Current buffer rotates in a ring
            buf_idx = buffers_acquired % buffer_count
            buf = buffers[buf_idx]
            
            # Pull data to buffer
            with buf.get_lock():
                ret = Az.AlazarWaitAsyncBufferComplete(handle, buf.get_obj(), U32(timeout))
                if ret == 573: 
                    unready_count += 1
                    continue # BufferNotReady, go back and try to post some buffers.
                else:
                    assert_az(ret, buffers_acquired, 'Wait Buffer Complete')
                
            buffers_acquired += 1
            
            # Tell worker thread to begin processing 
            buf_ready_events[buf_idx].set()
            
            # If a second has elapsed, replot the avg_buffer
            if (time.time() - start_time) / seconds_per_plot > plot_count:
                if plot:
                        with avg_buffer.get_lock():
                            plotter.msg(buffers_acquired, bufs_merged.value)
                            plotter.plot(np.frombuffer(avg_buffer.get_obj()), 'Data')
                        plot_count += 1
                else:                
                    print buffers_acquired, buffers_completed
                    plot_count += 1
    finally:
        Az.AlazarAbortAsyncRead(handle)
        final_time = time.time()
        print 'Unready Count', unready_count
        total_time = final_time - start_time
        print 'Total time', total_time
        if buffers_completed:
            actual_time_per_buffer = total_time / buffers_completed
            print 'Time per buffer %.2e' %  actual_time_per_buffer
            errf = lambda a, b: abs(a - b)/min(a, b)
            print 'Perceived overhead %.1f%%' % (errf(actual_time_per_buffer, seconds_per_buffer) * 100)
        else:
            print 'No buffers completed'

    return np.frombuffer(avg_buffer.get_obj())


class AlazarConstants():
    clock_source = {"internal" : U32(1),
                    "reference": U32(7),
                    "60 MHz" : U32(4),
                    "1 GHz" : U32(5)}

    sample_rate = {1000: U32(1),
                   2000: U32(2),
                   5000: U32(4),
                   10000: U32(8),
                   20000: U32(10),
                   50000: U32(12),
                   100000: U32(14),
                   200000: U32(16),
                   500000: U32(18),
                   1000000: U32(20),
                   2000000: U32(24),
                   5000000: U32(26),
                   10000000: U32(28),
                   20000000: U32(30),
                   50000000: U32(34),
                   100000000: U32(36),
                   250000000: U32(43),
                   500000000: U32(48),
                   1000000000: U32(53),
                   'external': U32(64),
                   'reference': U32(1000000000)}

    _trigger_source = {"ch_a": U32(0),
                      "ch_b": U32(1),
                      "external": U32(2),
                      "disabled": U32(3)}
    trigger_source1 = _trigger_source
    trigger_source2 = _trigger_source
    trigger_operation = {"or": U32(2),
                         "and": U32(3),
                         "xor": U32(4),
                         "and not": U32(5)}

    _coupling = {"ac": U32(1), "dc": U32(2)}
    trigger_coupling = _coupling
    input_coupling = _coupling

    trigger_single_op = U32(0)
    trigger_engine_1 = U32(0)
    trigger_engine_2 = U32(1)
    _edge = {"rising": U32(1), "falling":U32(2)}
    trigger_edge = _edge
    clock_edge = _edge

    # Channel Constants
    channel = {"CH_A": U8(1), "CH_B": U8(2)}
    _input_range = {0.04: U32(2), 0.1: U32(5), 0.2: U32(6),
                    0.4: U32(6), 1: U32(10), 2: U32(11), 4: U32(12)}
    ch1_range = _input_range
    ch2_range = _input_range
    #input_filter = {False: U32(0), True: U32(1)}

    #ApiSuccess = 512

class AlazarConfig:
    samplesPerRecord = 2**12
    recordsPerBuffer = 10
    bufferCount = cpu_count()
    buffersPerAcquisition = 0x7fffffff
    clock_source = 'external'
    clock_edge = 'rising'
    sample_rate = 1000000
    trigger_source1 = 'external'
    trigger_edge1 = 'rising'
    trigger_level1 = .1
    trigger_source2 = 'disabled'
    trigger_edge2 = 'rising'
    trigger_level2 = .1
    trigger_operation = 'and'
    trigger_coupling = 'DC'
    trigger_delay = 0
    timeout = 1000
    ch1_enabled = True
    ch1_coupling = 'DC'
    ch1_range = 1
    ch1_filter = False
    ch2_enabled = False
    ch2_coupling = 'DC'
    ch2_range = 1
    ch2_filter = False

    @property
    def recordsPerAcquisition(self):
        return self.recordsPerBuffer * self.buffersPerAcquisition

    @property
    def samplesPerBuffer(self):
        return self.samplesPerRecord * self.recordsPerBuffer
    
    def __init__(self, config_dict=None, **kwargs):
        config_dict = config_dict if config_dict else kwargs
        assert all([k in self.__dict__ for k in config_dict.keys()])
        self.__dict__.update(config_dict)

    def normalize(self):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                self.__dict__[k] = v.lower()

    def validate(self):
        self.normalize()
        #assert self.clock_source in ('external', 'interal', 'reference')
        #assert all(v in ('rising', 'falling') for v in (self.clock_edge, self.trigger_edge1, self.trigger_edge2))
        #assert all(v in ('external', 'interal', 'disabled') for v in (self.trigger_source1, self.trigger_source2))
        #assert self.trigger_operation in ('and', 'or')

        for k, v =x
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                potentials = getattr(AlazarConstants, k).keys()
                if v not in potentials:
                    raise ValueError(k + ' must be one of ' + ", ".join(potentials) + " not " + v)

            if isinstance(v, (int, float)):
                lower, upper = getattr(AlazarConstants, k)
                if not lower < v < upper:
                    raise ValueError(k + ' must be between %f and %f' % (lower, upper))

        if self.samplesPerRecord < 256 or (self.samplesPerRecord % 64) != 0:
            lower, upper = (max(256, self.samplesPerRecord-(self.samplesPerRecord % 64)),
                            max(256, self.samplesPerRecord+64-(self.samplesPerRecord % 64)))
            raise ValueError("Invalid samplesPerRecord, frames will not align properly. Try %d or %d" % (lower, upper))

class Alazar():
    def __init__(self, config=None, handle=None):
        self.Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
        self.config = config if config is not None else AlazarConfig()
        self.handle = handle if handle is not None else self.get_handle()

        if not self.handle:
            raise RuntimeError("Board could not be found")
            
    def close(self):
        del self.Az

    def get_handle(self):
        return self.Az.AlazarGetBoardBySystemID(U32(1), U32(1))
        
    def configure(self, config=None):
        if config is not None:
            self.config = config

        self.configure_clock()
        self.configure_trigger()
        self.configure_inputs()
        #self.configure_buffers()
           
    def configure_clock(self, source=None, rate=None, edge=None):
        """
        :param source: 'internal' to use internal clock with rate specified by
                       rate parameter. '60 MHz' for external clock with rate <= 60 MHz.
                       '1 GHz' for external clock with rate <= 1 GHz.
                       'reference' --> see Alazar documentation for AlazarSetCaptureClock
        :param rate: Rate (in KHz) for the internal clock, ignored for external
                     This will be rounded down to closest value specified by AlazarSetCaptureClock
                     documentation
        :param edge: 'rising' or 'falling'
        """
        if source is not None:
            self.config.clock_source = source
        if rate is not None:
            self.config.clock_rate = rate
        if edge is not None:
            self.config.edge = edge

        #convert clock config

        if self.config.clock_source == "internal":
            source = AlazarConstants.clock_source["internal"]
            decimation=U32(0)
            for (rate, value) in AlazarConstants.sample_rate:
                if rate >= self.config.sample_rate:
                    if rate > self.config.sample_rate:
                        print "Warning: sample_rate not found. Using first smaller value", rate, "Khz"
                        self.config.sample_rate = rate
                    sample_rate = value
                    break
        elif self.config.clock_source == "external":
            sample_rate = AlazarConstants.sample_rate_external
            decimation=U32(0)
            if self.config.sample_rate < 60000:
                source = AlazarConstants.clock_source["60 MHz"]
            elif self.config.sample_rate < 1000000:
                source = AlazarConstants.clock_source["1 GHz"]
            else:
                raise ValueError("Not supported (yet?)")
        elif self.config.clock_source == "reference":
            source = AlazarConstants.clock_source["reference"]
            sample_rate=U32(1000000000)
            decimation=int(1e9/(self.config.sample_rate*1e3))
            if (decimation != 1) and (decimation != 2) and (decimation != 1) and (decimation %10 != 0):
                print "Warning: sample_rate must be 1Gs/s / 1,2,4 or a multiple of 10. Using 1Gs/s."
                decimation=1
            decimation=U32(decimation)
        else:
            raise ValueError("reference signal not implemented yet")
        ret = self.Az.AlazarSetCaptureClock(self.handle, source, sample_rate, AlazarConstants.clock_edge[self.config.clock_edge], decimation)
        if DEBUGALAZAR: print "ClockConfig:", ret_to_str(ret, self.Az)

    def configure_trigger(self,source=None, source2=None, edge=None, edge2=None,
                 level=None, level2=None, operation=None, coupling=None, timeout=None,delay=0):
        """
        Can set up to two trigger operations to be performed

        :param source: Where the first trigger engine should take its input.
                       'CH_A' for channel A, 'CH_B' for channel B, "external" for external source,
                       'disabled' to disable trigger engine
        :param edge: 'rising' or 'falling'
        :param level: integer in interval [-100, 100], i.e. a percent of the input range
                      at which to trigger a capture
        :param coupling: 'AC' or 'DC'
        :param operation: How to combine two enabled triggers to generate capture events
                          'or' to trigger on either engine, 'and' to trigger only when both go high,
                          'xor', 'and not' offered as well.
        :param timeout: How long to wait for a trigger before giving up (milliseconds)
        """
        if source is not None: self.config.trigger_source1 = source
        if source2 is not None: self.config.trigger_source2 = source2
        if edge is not None: self.config.trigger_edge1 = edge
        if edge2 is not None: self.config.trigger_edge2 = edge2
        if level is not None: self.config.trigger_level1 = level
        if level2 is not None: self.config.trigger_level2 = level2
        
        if not (self.config.trigger_level1 >= -100 and self.config.trigger_level1 < 100):
            raise ValueError("Level must be value in [-100,100]")
        if not (self.config.trigger_level2 >= -100 and self.config.trigger_level2 < 100):
            raise ValueError("Level must be value in [-100,100]")
        if operation is not None: self.config.trigger_operation = operation
        if coupling is not None: self.config.trigger_coupling= coupling
        if timeout is not None: self.config.timeout= timeout
        if delay is not None: self.config.trigger_delay = delay
        
        
        if source2 == "disabled":
            op = AlazarConstants.single_op
        else:
            op = AlazarConstants.trigger_operation[self.config.trigger_operation]

        ret = self.Az.AlazarSetTriggerOperation(self.handle, op,
                AlazarConstants.trigger_engine_1, AlazarConstants.trigger_source[self.config.trigger_source1], AlazarConstants.trigger_edge[self.config.trigger_edge1], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)),
                AlazarConstants.trigger_engine_2, AlazarConstants.trigger_source[self.config.trigger_source2], AlazarConstants.trigger_edge[self.config.trigger_edge2], U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)))
        if DEBUGALAZAR: print "Set Trigger:", ret_to_str(ret, self.Az)
        if self.config.trigger_source1 == "external":
            ret = self.Az.AlazarSetExternalTrigger(self.handle, AlazarConstants.trigger_ext_coupling[self.config.trigger_coupling], U32(0))
            if DEBUGALAZAR: print "Set External Trigger:", ret_to_str(ret, self.Az)
            
        
        self.triggerDelay_samples = int(self.config.trigger_delay * self.config.sample_rate*1e3 + 0.5)
        ret = self.Az.AlazarSetTriggerDelay(self.handle, U32(self.triggerDelay_samples))
        if DEBUGALAZAR: print "Set Trigger Delay:", ret_to_str(ret, self.Az)
        
        
    def configure_inputs(self, enabled1=None, coupling1=None, range1=None, filter1=None, enabled2=None, coupling2=None, range2=None, filter2=None):
        """
        :param channel: 'CH_A' or 'CH_B'. Create two InputConfig classes for both
        :param coupling: 'AC' or 'DC'
        :param input_range: Input range in volts. rounds down to the closest value
                            provided by AlazarInputControl
        :param filter_above_20MHz: if True, enable the 20MHz BW filter
        """
        
        if enabled1 is not None: self.config.ch1_enabled = enabled1
        if coupling1 is not None: self.config.ch1_coupling= coupling1
        if range1 is not None: self.config.ch1_range = range1
        if filter1 is not None: self.config.ch1_filter

        if enabled2 is not None: self.config.ch2_enabled = enabled2
        if coupling2 is not None: self.config.ch2_coupling= coupling2
        if range2 is not None: self.config.ch2_range = range2
        if filter2 is not None: self.config.ch2_filter
        
        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch1_range <= voltage:
                if self.config.ch1_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch1_range = voltage
                ch1_range_value=value
                break
        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch2_range <= voltage:
                if self.config.ch2_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch2_range = voltage
                ch2_range_value=value
                break

        if self.config.ch1_enabled:  
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_A"], AlazarConstants.input_coupling[self.config.ch1_coupling], ch1_range_value, U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_A"], AlazarConstants.input_filter[self.config.ch1_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)
        if self.config.ch2_enabled:  
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_B"], AlazarConstants.input_coupling[self.config.ch2_coupling], ch2_range_value, U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_B"], AlazarConstants.input_filter[self.config.ch2_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)


    def configure_buffers(self,samplesPerRecord=None,recordsPerBuffer=None,recordsPerAcquisition=None,bufferCount=None):
        if samplesPerRecord is not None: self.config.samplesPerRecord=samplesPerRecord
        if recordsPerBuffer is not None: self.config.recordsPerBuffer=recordsPerBuffer
        if recordsPerAcquisition is not None: self.config.recordsPerAcquisition=recordsPerAcquisition
        if bufferCount is not None: self.config.bufferCount = bufferCount
        
        self.config.channelCount=0
        channel=0        #Create channel flag
        if self.config.ch1_enabled: 
            channel= channel | 1
            self.config.channelCount+=1
        if self.config.ch2_enabled: 
            channel= channel | 2
            self.config.channelCount+=1
        
        pretriggers=C.c_long(0) #no pretriggering support for now
        flags = U32 (513) #ADMA flags, should update to be more general
        
        ret = self.Az.AlazarSetRecordSize(self.handle,U32(0),U32(self.config.samplesPerRecord))        
        if DEBUGALAZAR: print "Set Record Size:", ret_to_str(ret,self.Az)
        
        ret = self.Az.AlazarBeforeAsyncRead(self.handle,U32(channel),pretriggers,
                                       U32(self.config.samplesPerRecord), 
                                       U32(self.config.recordsPerBuffer),
                                       U32(self.config.recordsPerAcquisition),
                                       flags)
        if DEBUGALAZAR: print "Before Read:", ret_to_str(ret,self.Az)

        self.config.bytesPerBuffer=(self.config.samplesPerRecord * self.config.recordsPerBuffer * self.config.channelCount)        
        self.bufs=[]
        buftype=U8 * self.config.bytesPerBuffer
        for i in range(self.config.bufferCount):
            self.bufs.append(buftype())            
            for j in range(self.config.bytesPerBuffer):
                self.bufs[i][j]=U8(0)
            #ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[i],U32(self.config.bytesPerBuffer))
        if DEBUGALAZAR: print "Posted buffers: ", ret_to_str(ret,self.Az)
        self.arrs=[np.ctypeslib.as_array(b) for b in self.bufs]

def acquire_average_data(self, n_records, n_averages):
    buffer_count = cpu_count()
    

def identity(x):
    return x

if __name__ == "__main__":
    #import cProfile
    acquire_avg_data_parallel(average_reshape_worker, reshape_collapse_records, samples_per_record, plot=True)
    print 'done'
