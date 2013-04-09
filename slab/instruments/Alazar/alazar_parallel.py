from __future__ import division

import ctypes as C
import numpy as np
import warnings
warnings.simplefilter('ignore')

# Plan of attack :
# post buffers in main thread, distribute as arguments to the worker threads

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

recordsPerBuffer = 100
recordsPerAcquisition = 0x7fffffff
samplesPerRecord = bytesPerRecord = 1024
bytesPerBuffer = bytesPerRecord * recordsPerBuffer
buffersPerMerge = 10
buffers_per_merge = 10
bufferCount = 8
iterations = 10000
timeout = 1000

class AlazarException(Exception):
    pass

def worker(buf, buf_ready_event, buf_post_event, avg_buffer, buffers_merged):
    arr = np.frombuffer(buf.get_obj(), U8)
    avg_buffer_arr = np.frombuffer(avg_buffer.get_obj(), C.c_longdouble)
    sum_buffer = np.zeros(bytesPerBuffer, np.uint32)
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

def acquire_avg_data_parallel():
    from multiprocessing import Process, Array, Value, Event
    from slab.instruments import Alazar, AlazarConfig
    from slab.plotting import ScriptPlotter
    import time

    # Configure card
    config={'clock_edge': 'rising', 'clock_source': 'reference',  
            'trigger_coupling': 'DC',  'trigger_operation': 'or', 
            'trigger_source2': 'disabled','trigger_level2': 1.0, 'trigger_edge2': 'rising', 
            'trigger_level1': 0.6, 'trigger_edge1': 'rising', 'trigger_source1': 'external', 
            'ch1_enabled': True,  'ch1_coupling': 'AC', 'ch1_range': 1, 'ch1_filter': False, 
            'ch2_enabled': False, 'ch2_coupling': 'AC','ch2_range': 1, 'ch2_filter': False,            
            'bufferCount': bufferCount,'recordsPerBuffer': recordsPerBuffer, 'trigger_delay': 0, 'timeout': timeout,
            'samplesPerRecord':samplesPerRecord,'recordsPerAcquisition':recordsPerAcquisition, 'sample_rate': 1000000}
    card = Alazar(AlazarConfig(config))
    card.configure()
    Az = card.Az
    handle = card.handle

    def assert_az(retCode, buffers_acquired, name):
        Az.AlazarErrorToText.restype = C.c_char_p
        if retCode != 512:
            raise AlazarException("::".join(map(str, 
                (name, buffers_acquired, retCode, Az.AlazarErrorToText(U32(retCode))))))    
    
    # Initialize buffers
    buffers = [ Array(U8, bytesPerBuffer) for _ in range(bufferCount) ]
    for b in buffers:
        ret = Az.AlazarPostAsyncBuffer(handle, b.get_obj(), U32(bytesPerBuffer))
        assert_az(ret, 0, 'Initial Post Buffer')
    avg_buffer = Array(C.c_longdouble, bytesPerBuffer)
    
    # Initialize threads  
    bufs_merged = Value(U32, 1)
    buf_ready_events = [ Event() for _ in range(bufferCount) ]
    buf_post_events = [ Event() for _ in range(bufferCount) ]
    workers = [ Process(target=worker, args=(b, bre, bpe, avg_buffer, bufs_merged)) 
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
    while buffers_acquired < (iterations * bufferCount):
        
        # Post all completed buffers
        while buf_post_events[buffers_completed % bufferCount].is_set():
            buf_post_events[buffers_completed % bufferCount].clear()
            buf = buffers[buffers_completed % bufferCount]
            with buf.get_lock():
                ret = Az.AlazarPostAsyncBuffer(handle, buf.get_obj(), U32(bytesPerBuffer))
                assert_az(ret, buffers_acquired, 'Post Buffer')
            buffers_completed += 1        
        
        # Current buffer rotates in a ring
        buf_idx = buffers_acquired % bufferCount
        buf = buffers[buf_idx]
        
        # Pull data to buffer
        with buf.get_lock():
            ret = Az.AlazarWaitAsyncBufferComplete(handle, buf.get_obj(), U32(timeout))
            if ret == 573: 
                continue # BufferNotReady, go back and try to post some buffers.
            else:
                assert_az(ret, buffers_acquired, 'Wait Buffer Complete')
            
        buffers_acquired += 1
        
        # Tell worker thread to begin processing 
        buf_ready_events[buf_idx].set()
        
        # If a second has elapsed, replot the avg_buffer
        if time.time() - start_time > plot_count:
            with avg_buffer.get_lock():
                plotter.msg(buffers_acquired, bufs_merged.value)
                plotter.plot(np.frombuffer(avg_buffer.get_obj()), 'Data')
            plot_count += 1
    
    return np.frombuffer(avg_buffer.get_obj())

if __name__ == "__main__":
    import cProfile
    #cProfile.run('acquire_avg_data_parallel()')
    acquire_avg_data_parallel()
    print 'done'
