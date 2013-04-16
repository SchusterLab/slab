from __future__ import division

import ctypes as C
import numpy as np
import warnings
warnings.simplefilter('ignore')

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

records_per_buffer = 10
records_per_acquisition = 0x7fffffff
bytes_per_sample = 1
samples_per_record = 2**14
buffers_per_merge = 1000
buffer_count = 8
iterations = 100000
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
     samplesPerBuffer = c.samplesPerRecord * c.recordsPerBuffer
     duty_cycle = samplesPerBuffer / (c.sample_rate * 1e3)
     if duty_cycle > .7:
         raise ValueError('Card is acquiring for %.1f of the trigger time' % duty_cycle)


def average_worker(reshape_fn, buf, buf_ready_event, buf_post_event, avg_buffer, buffers_merged):
    arr = np.frombuffer(buf.get_obj(), U8)
    avg_buffer_arr = np.frombuffer(avg_buffer.get_obj(), C.c_longdouble)
    sum_buffer = np.zeros(bytes_per_buffer, np.uint32)
    my_buffers_completed = 0
    for _ in range(iterations):
        buf_ready_event.wait()

        buf_ready_event.clear()
        sum_buffer += reshape_fn(arr)
        buf_post_event.set()
        my_buffers_completed += 1

        if my_buffers_completed % buffers_per_merge == 0:
            with avg_buffer.get_lock():
                n = buffers_merged.value
                avg_buffer_arr *= (n - 1.) / n
                avg_buffer_arr += sum_buffer / (n*buffers_per_merge)
                buffers_merged.value += 1
            sum_buffer.fill(0)

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
        result_buffer[n*iterations:(n+1)*iterations] = array(results)
        buffers_merged.value += 1


def acquire_avg_data_parallel(worker, proc_fun, result_shape):
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
            'trigger_level1': 0.6, 'trigger_edge1': 'rising', 'trigger_source1': 'external', 
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
        avg_buffer = Array(C.c_longdouble, bytes_per_buffer)
        
        # Initialize threads  
        bufs_merged = Value(U32, 1)
        buf_ready_events = [ Event() for _ in range(buffer_count) ]
        buf_post_events = [ Event() for _ in range(buffer_count) ]
        workers = [ Process(target=worker, args=(proc_fun, b, bre, bpe, avg_buffer, bufs_merged)) 
                    for b, bre, bpe in zip(buffers, buf_ready_events, buf_post_events) ]
        
        for w in workers:
            w.start()
        time.sleep(1)

        # Initialize things used during capture
        plotter = ScriptPlotter()
        buffers_acquired, buffers_completed, plot_count = 0, 0, 0
        start_time = time.time()
        plotter.init_plot('Data', rank=1, accum=False)

        # Begin capture
        ret = Az.AlazarStartCapture(handle)
        assert_az(ret, 0, "Start Capture")
        unready_count = 0
        while buffers_acquired < (iterations * buffer_count):
            
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
                with avg_buffer.get_lock():
                    plotter.msg(buffers_acquired, bufs_merged.value)
                    plotter.plot(np.frombuffer(avg_buffer.get_obj()), 'Data')
                plot_count += 1
    finally:
        Az.AlazarAbortAsyncRead(handle)
        final_time = time.time()
        print 'Unready Count', unready_count
        total_time = final_time - start_time
        print 'Total time', total_time
        actual_time_per_buffer = total_time / buffers_completed
        print 'Time per buffer %.2e' %  actual_time_per_buffer
        errf = lambda a, b: abs(a - b)/min(a, b)
        print 'Perceived overhead %.1f%%' % (errf(actual_time_per_buffer, seconds_per_buffer) * 100)

    return np.frombuffer(avg_buffer.get_obj())


def identity(x):
    return x

if __name__ == "__main__":
    #import cProfile
    acquire_avg_data_parallel(average_worker, identity, (records_per_buffer,))
    print 'done'
