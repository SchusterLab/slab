from __future__ import division

import ctypes as C
import numpy as np
import warnings
from multiprocessing import cpu_count, Process
warnings.simplefilter('ignore')

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

DEBUGALAZAR = False


def ret_to_str(retCode, Az):
    Az.AlazarErrorToText.restype = C.c_char_p
    return Az.AlazarErrorToText(U32(retCode))


class AlazarException(Exception):
    pass


class AlazarConstants():
    clock_source = {"internal": U32(1),
                    "reference": U32(7),
                    "60 MHz": U32(4),
                    "1 GHz": U32(5)}

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
    _edge = {"rising": U32(1), "falling": U32(2)}
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
    iterations = 10000
    timeout = 500

    expected_trigger_time = None
    seconds_per_plot = 1

    bytesPerSample = bytes_per_sample = 1
    samplesPerRecord = samples_per_record = 2 ** 12

    @property
    def bytes_per_record(self):
        return self.bytes_per_sample * self.samples_per_record

    @property
    def bytes_per_buffer(self):
        return self.bytes_per_record * self.records_per_buffer

    @property
    def samples_per_buffer(self):
        return self.samples_per_record * self.records_per_buffer

    recordsPerBuffer = records_per_buffer = 10
    recordsPerAcquisition = records_per_acquisition = 0x7fffffff

    @property
    def buffers_per_acquisition(self):
        assert (self.records_per_acquisition % self.records_per_buffer) == 0
        return int(self.records_per_acquisition / self.records_per_buffer)

    buffers_per_merge = 100
    bufferCount = buffer_count = cpu_count()

    @property
    def buffers_per_worker(self):
        assert (self.records_per_acquisition % self.records_per_buffer) == 0
        return self.buffers_per_acquisition / self.buffer_count

    clock_source = 'external'
    clock_edge = 'rising'
    sample_rate = samples_per_ms = 1000000
    samples_per_second = samples_per_ms * 1000

    @property
    def seconds_per_record(self):
        return self.samples_per_record / self.samples_per_second

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

        for k, v in self.__dict__.items():
            if isinstance(v, str):
                potentials = getattr(AlazarConstants, k).keys()
                if v not in potentials:
                    raise ValueError(k + ' must be one of ' + ", ".join(potentials) + " not " + v)

            if isinstance(v, (int, float)):
                lower, upper = getattr(AlazarConstants, k)
                if not lower < v < upper:
                    raise ValueError(k + ' must be between %f and %f' % (lower, upper))

        if self.records_per_acquisition != 0x7fffffff and (self.records_per_acquisition % self.records_per_buffer) != 0:
            raise ValueError("Invalid recordsPerAcquisition, needs to be a multiple of recordsPerBuffer")

        if self.samplesPerRecord < 256 or (self.samplesPerRecord % 64) != 0:
            lower, upper = (max(256, self.samplesPerRecord - (self.samplesPerRecord % 64)),
                            max(256, self.samplesPerRecord + 64 - (self.samplesPerRecord % 64)))
            raise ValueError("Invalid samplesPerRecord, frames will not align properly. Try %d or %d" % (lower, upper))

        if self.expected_trigger_time and self.expected_trigger_time < self.seconds_per_record:
            raise ValueError('More than one trigger occurs during record acquisition.\n' +
                             'Consider longer trigger time, fewer samples, or higher rate')


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
            decimation = U32(0)
            for (rate, value) in AlazarConstants.sample_rate:
                if rate >= self.config.sample_rate:
                    if rate > self.config.sample_rate:
                        print "Warning: sample_rate not found. Using first smaller value", rate, "Khz"
                        self.config.sample_rate = rate
                    sample_rate = value
                    break
        elif self.config.clock_source == "external":
            sample_rate = AlazarConstants.sample_rate_external
            decimation = U32(0)
            if self.config.sample_rate < 60000:
                source = AlazarConstants.clock_source["60 MHz"]
            elif self.config.sample_rate < 1000000:
                source = AlazarConstants.clock_source["1 GHz"]
            else:
                raise ValueError("Not supported (yet?)")
        elif self.config.clock_source == "reference":
            source = AlazarConstants.clock_source["reference"]
            sample_rate = U32(1000000000)
            decimation = int(1e9 / (self.config.sample_rate * 1e3))
            if (decimation != 1) and (decimation != 2) and (decimation != 1) and (decimation % 10 != 0):
                print "Warning: sample_rate must be 1Gs/s / 1,2,4 or a multiple of 10. Using 1Gs/s."
                decimation = 1
            decimation = U32(decimation)
        else:
            raise ValueError("reference signal not implemented yet")
        ret = self.Az.AlazarSetCaptureClock(self.handle, source, sample_rate,
                                            AlazarConstants.clock_edge[self.config.clock_edge], decimation)
        if DEBUGALAZAR: print "ClockConfig:", ret_to_str(ret, self.Az)

    def configure_trigger(self, source=None, source2=None, edge=None, edge2=None,
                          level=None, level2=None, operation=None, coupling=None, timeout=None, delay=0):
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
        if coupling is not None: self.config.trigger_coupling = coupling
        if timeout is not None: self.config.timeout = timeout
        if delay is not None: self.config.trigger_delay = delay

        if source2 == "disabled":
            op = AlazarConstants.single_op
        else:
            op = AlazarConstants.trigger_operation[self.config.trigger_operation]

        ret = self.Az.AlazarSetTriggerOperation(self.handle, op,
                                                AlazarConstants.trigger_engine_1,
                                                AlazarConstants.trigger_source[self.config.trigger_source1],
                                                AlazarConstants.trigger_edge[self.config.trigger_edge1],
                                                U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)),
                                                AlazarConstants.trigger_engine_2,
                                                AlazarConstants.trigger_source[self.config.trigger_source2],
                                                AlazarConstants.trigger_edge[self.config.trigger_edge2],
                                                U32(int_linterp(self.config.trigger_level1, -100, 100, 0, 255)))
        if DEBUGALAZAR: print "Set Trigger:", ret_to_str(ret, self.Az)
        if self.config.trigger_source1 == "external":
            ret = self.Az.AlazarSetExternalTrigger(self.handle,
                                                   AlazarConstants.trigger_ext_coupling[self.config.trigger_coupling],
                                                   U32(0))
            if DEBUGALAZAR: print "Set External Trigger:", ret_to_str(ret, self.Az)

        self.triggerDelay_samples = int(self.config.trigger_delay * self.config.sample_rate * 1e3 + 0.5)
        ret = self.Az.AlazarSetTriggerDelay(self.handle, U32(self.triggerDelay_samples))
        if DEBUGALAZAR: print "Set Trigger Delay:", ret_to_str(ret, self.Az)

    def configure_inputs(self, enabled1=None, coupling1=None, range1=None, filter1=None, enabled2=None, coupling2=None,
                         range2=None, filter2=None):
        """
        :param channel: 'CH_A' or 'CH_B'. Create two InputConfig classes for both
        :param coupling: 'AC' or 'DC'
        :param input_range: Input range in volts. rounds down to the closest value
                            provided by AlazarInputControl
        :param filter_above_20MHz: if True, enable the 20MHz BW filter
        """

        if enabled1 is not None: self.config.ch1_enabled = enabled1
        if coupling1 is not None: self.config.ch1_coupling = coupling1
        if range1 is not None: self.config.ch1_range = range1
        if filter1 is not None: self.config.ch1_filter

        if enabled2 is not None: self.config.ch2_enabled = enabled2
        if coupling2 is not None: self.config.ch2_coupling = coupling2
        if range2 is not None: self.config.ch2_range = range2
        if filter2 is not None: self.config.ch2_filter

        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch1_range <= voltage:
                if self.config.ch1_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch1_range = voltage
                ch1_range_value = value
                break
        for (voltage, value) in AlazarConstants.input_range:
            if self.config.ch2_range <= voltage:
                if self.config.ch2_range < voltage:
                    if DEBUGALAZAR: print "Warning: input range not found, using closest value,", voltage, "Volts"
                self.config.ch2_range = voltage
                ch2_range_value = value
                break

        if self.config.ch1_enabled:
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_A"],
                                             AlazarConstants.input_coupling[self.config.ch1_coupling], ch1_range_value,
                                             U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_A"],
                                           AlazarConstants.input_filter[self.config.ch1_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)
        if self.config.ch2_enabled:
            ret = self.Az.AlazarInputControl(self.handle, AlazarConstants.channel["CH_B"],
                                             AlazarConstants.input_coupling[self.config.ch2_coupling], ch2_range_value,
                                             U32(2))
            if DEBUGALAZAR: print "Input Control CH1:", ret_to_str(ret, self.Az)
            ret = self.Az.AlazarSetBWLimit(self.handle, AlazarConstants.channel["CH_B"],
                                           AlazarConstants.input_filter[self.config.ch2_filter])
            if DEBUGALAZAR: print "Set BW Limit:", ret_to_str(ret, self.Az)

    def configure_buffers(self, samplesPerRecord=None, recordsPerBuffer=None, recordsPerAcquisition=None,
                          bufferCount=None):
        if samplesPerRecord is not None: self.config.samplesPerRecord = samplesPerRecord
        if recordsPerBuffer is not None: self.config.recordsPerBuffer = recordsPerBuffer
        if recordsPerAcquisition is not None: self.config.recordsPerAcquisition = recordsPerAcquisition
        if bufferCount is not None: self.config.bufferCount = bufferCount

        self.config.channelCount = 0
        channel = 0        #Create channel flag
        if self.config.ch1_enabled:
            channel = channel | 1
            self.config.channelCount += 1
        if self.config.ch2_enabled:
            channel = channel | 2
            self.config.channelCount += 1

        pretriggers = C.c_long(0) #no pretriggering support for now
        flags = U32(513) #ADMA flags, should update to be more general

        ret = self.Az.AlazarSetRecordSize(self.handle, U32(0), U32(self.config.samplesPerRecord))
        if DEBUGALAZAR: print "Set Record Size:", ret_to_str(ret, self.Az)

        ret = self.Az.AlazarBeforeAsyncRead(self.handle, U32(channel), pretriggers,
                                            U32(self.config.samplesPerRecord),
                                            U32(self.config.recordsPerBuffer),
                                            U32(self.config.recordsPerAcquisition),
                                            flags)
        if DEBUGALAZAR: print "Before Read:", ret_to_str(ret, self.Az)

        self.config.bytesPerBuffer = (
        self.config.samplesPerRecord * self.config.recordsPerBuffer * self.config.channelCount)
        self.bufs = []
        buftype = U8 * self.config.bytesPerBuffer
        for i in range(self.config.bufferCount):
            self.bufs.append(buftype())
            for j in range(self.config.bytesPerBuffer):
                self.bufs[i][j] = U8(0)
                #ret = self.Az.AlazarPostAsyncBuffer(self.handle,self.bufs[i],U32(self.config.bytesPerBuffer))
        if DEBUGALAZAR: print "Posted buffers: ", ret_to_str(ret, self.Az)
        self.arrs = [np.ctypeslib.as_array(b) for b in self.bufs]

    def assert_az(self, retCode, buffers_acquired, name):
        self.Az.AlazarErrorToText.restype = C.c_char_p
        if retCode != 512:
            raise AlazarException("::".join(map(str, (name, buffers_acquired,
                                                      retCode, self.Az.AlazarErrorToText(U32(retCode))))))

    def __getattr__(self, item):
        if item in self.config:
            return getattr(self.config, item)
        else:
            raise AttributeError

    def acquire_parallel(self, worker_cls, worker_args, result_shape, plot=False):
        """
        :param worker: Function which the subordinate threads execute as their target
        :param proc_fun: Function used by the subordinate threads to process their buffers
        :param result_shape: Shape of the buffer which is the result of the entire acquisition.
        """
        from multiprocessing import Array, Value, Event
        from slab.plotting import ScriptPlotter
        import time

        acquire_buffer_time = self.samples_per_buffer / (self.samples_per_second)
        print 'Acquire buffer time %.2e' % acquire_buffer_time
        print 'Inter-buffer time %.2e' % self.seconds_per_buffer
        print 'Duty Cycle', acquire_buffer_time / self.seconds_per_buffer

        try:
            # Initialize buffers
            buffers = [Array(U8, self.bytes_per_buffer) for _ in range(self.buffer_count)]
            for b in buffers:
                ret = self.Az.AlazarPostAsyncBuffer(self.handle, b.get_obj(), U32(self.bytes_per_buffer))
                self.assert_az(ret, 0, 'Initial Post Buffer')

            res_buffer = Array(C.c_longdouble, result_shape)

            # Initialize threads
            bufs_merged = Value(U32, 1)
            buf_ready_events = [Event() for _ in range(self.buffer_count)]
            buf_post_events = [Event() for _ in range(self.buffer_count)]
            workers = [worker_cls(*(worker_args + (self.config, b, bre, bpe, res_buffer, bufs_merged)))
                       for b, bre, bpe in zip(buffers, buf_ready_events, buf_post_events)]

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
            ret = self.Az.AlazarStartCapture(self.handle)
            self.assert_az(ret, 0, "Start Capture")
            unready_count = 0
            while buffers_completed < self.buffers_per_acquisition:

                # Post all completed buffers
                while buf_post_events[buffers_completed % self.buffer_count].is_set():
                    buf_post_events[buffers_completed % self.buffer_count].clear()
                    buf = buffers[buffers_completed % self.buffer_count]
                    with buf.get_lock():
                        ret = self.Az.AlazarPostAsyncBuffer(self.handle, buf.get_obj(), U32(self.bytes_per_buffer))
                        self.assert_az(ret, buffers_acquired, 'Post Buffer')
                    buffers_completed += 1

                    # Current buffer rotates in a ring
                buf_idx = buffers_acquired % self.buffer_count
                buf = buffers[buf_idx]

                # Pull data to buffer
                with buf.get_lock():
                    ret = self.Az.AlazarWaitAsyncBufferComplete(self.handle, buf.get_obj(), U32(self.timeout))
                    if ret == 573:
                        unready_count += 1
                        continue # BufferNotReady, go back and try to post some buffers.
                    else:
                        self.assert_az(ret, buffers_acquired, 'Wait Buffer Complete')

                buffers_acquired += 1

                # Tell worker thread to begin processing
                buf_ready_events[buf_idx].set()

                # If a second has elapsed, replot the avg_buffer
                if (time.time() - start_time) / self.seconds_per_plot > plot_count:
                    if plot:
                        with res_buffer.get_lock():
                            plotter.msg(buffers_acquired, buffers_completed, bufs_merged.value)
                            plotter.plot(np.frombuffer(res_buffer.get_obj()), 'Data')
                        plot_count += 1
                    else:
                        print buffers_acquired, buffers_completed
                        plot_count += 1
        finally:
            self.Az.AlazarAbortAsyncRead(self.handle)
            if buffers_completed:
                final_time = time.time()
                print 'Unready Count', unready_count
                total_time = final_time - start_time
                print 'Total time', total_time
                actual_time_per_buffer = total_time / buffers_completed
                print 'Time per buffer %.2e' % actual_time_per_buffer
                errf = lambda a, b: abs(a - b) / min(a, b)
                print 'Perceived overhead %.1f%%' % (errf(actual_time_per_buffer, seconds_per_buffer) * 100)
            else:
                print 'No buffers completed'

        res = np.frombuffer(res_buffer.get_obj())
        return res

    def acquire_average_parallel(self, averages, **kwargs):
        return self.acquire_parallel(AverageWorker, (), self.samples_per_buffer)

    def acquire_homodyne_single_shot(self, frequency):
        sin_arr = np.sin(frequency * np.linspace(0, self.seconds_per_record, self.samples_per_record))
        cos_arr = np.cos(frequency * np.linspace(0, self.seconds_per_record, self.samples_per_record))
        records_per_buffer = self.records_per_buffer
        proc_fun = lambda buf_arr: [(np.sum(sin_arr * arr), np.sum(cos_arr * arr)) for arr in np.split(buf_arr, records_per_buffer)]
        self.acquire_parallel(SingleShotWorker, (proc_fun,), (self.records_per_acquisition, 2))


class Worker(Process):
    def __init__(self, az_config, buf, buf_ready_event, buf_post_event, res_buffer, buffers_merged):
        self.az_config = az_config
        self.buf = buf
        self.buf_ready_event = buf_ready_event
        self.buf_post_event = buf_post_event
        self.res_buffer = res_buffer
        self.buffers_merged = buffers_merged

class AverageWorker(Worker):
    def run(self):
        arr = np.frombuffer(self.buf.get_obj(), U8)
        avg_buffer_arr = np.frombuffer(self.res_buffer.get_obj(), C.c_longdouble)
        sum_buffer = np.zeros(self.az_config.bytes_per_buffer, np.uint32)
        my_buffers_completed = 0
        for _ in range(self.az_config.buffers_per_worker):
            self.buf_ready_event.wait()
            self.buf_ready_event.clear()
            sum_buffer += arr
            self.buf_post_event.set()
            my_buffers_completed += 1

            if my_buffers_completed % self.az_config.buffers_per_merge == 0:
                with self.avg_buffer.get_lock():
                    n = self.buffers_merged.value
                    avg_buffer_arr *= (n - 1.) / n
                    avg_buffer_arr += sum_buffer / (n * self.az_config.buffers_per_merge)
                    self.buffers_merged.value += 1
                sum_buffer.fill(0)

class SingleShotWorker(Worker):
    def __init__(self, proc_fun, *args):
        self.proc_fun = proc_fun
        Worker.__init__(self, *args)

    def run(self):
        arr = np.frombuffer(self.buf.get_obj(), U8)
        results = []
        my_buffers_completed = 0
        iterations = self.az_config.buffers_per_worker
        for _ in range(iterations):
            self.buf_ready_event.wait()
            self.buf_ready_event.clear()
            results.extend(self.proc_fun(arr))
            self.buf_post_event.set()
            my_buffers_completed += 1

        with self.res_buffer.get_lock():
            n = self.buffers_merged.value
            print iterations, len(results), n, len(self.res_buffer)
            self.res_buffer[(n - 1) * iterations:n * iterations] = np.array(results)
            self.buffers_merged.value += 1



# class AverageReshapeWorker(Worker):
#     def __init__(self, reshape_fn, *args):
#         self.reshape_fn = reshape_fn
#         Worker.__init__(self, *args)
#
#     def run(self):
#         arr = np.frombuffer(self.buf.get_obj(), U8)
#         avg_buffer_arr = np.frombuffer(avg_buffer.get_obj(), C.c_longdouble)
#         sum_buffer = np.zeros(len(avg_buffer_arr), np.uint32)
#         my_buffers_completed = 0
#         for _ in range(iterations):
#             buf_ready_event.wait()
#
#             buf_ready_event.clear()
#             reshape_fn(sum_buffer, arr)
#             buf_post_event.set()
#             my_buffers_completed += 1
#
#             if my_buffers_completed % buffers_per_merge == 0:
#                 with avg_buffer.get_lock():
#                     n = buffers_merged.value
#                     avg_buffer_arr *= (n - 1.) / n
#                     avg_buffer_arr += sum_buffer / (n * buffers_per_merge)
#                     buffers_merged.value += 1
#                 sum_buffer.fill(0)


# def reshape_collapse_records(res, arr):
#     samples_per_record = len(res)
#     records_per_buffer = int(len(arr) / samples_per_record)
#     for i in range(records_per_buffer):
#         res += arr[i * samples_per_record:(i + 1) * samples_per_record]


def profile_single_shot(*args):
    import cProfile
    cProfile.runctx('single_shot_worker(*args)', globals(), locals(), 'worker_stats')


def main():
    print 'starting'
    res = acquire_avg_data_parallel(profile_single_shot, sum_array, buffer_count*iterations, plot=True)
    print 'returning', res

if __name__ == "__main__":
    import cProfile, pstats
    #cProfile.run('main()', 'alazar_stats')
    main()
    s = pstats.Stats('worker_stats')
    s.sort_stats('cumulative').print_stats(20)
    
    print 'done'
