from __future__ import division
from multiprocessing import Process, Lock, Array, Value, Condition, Event
from multiprocessing.managers import BaseManager
import ctypes as C
import numpy as np
import time
from slab.plotting import ScriptPlotter
import warnings
from slab.instruments import Alazar, AlazarConfig
warnings.simplefilter('ignore')

# Plan of attack :
# post buffers in main thread, distribute as arguments to the worker threads

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

use_alazar = True
recordsPerBuffer = 100
bytesPerRecord = None
recordsPerAcquisition = 0x7fffffff
samplesPerRecord = 1024
bytesPerBuffer = 10000
buffersPerMerge = 10
buffers_per_merge = 10
bufferCount = 8
iterations = 1000
timeout = 1000


def ret_to_str(retCode, Az):
    Az.AlazarErrorToText.restype = C.c_char_p
    return Az.AlazarErrorToText(U32(retCode))

# Define thread actions
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
            sum_buffer.fill(0)

def acquire_avg_data_parallel():
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
    
    # Initialize buffers
    buffers = [ Array(U8, bytesPerBuffer) for _ in range(bufferCount) ]    
    for b in buffers:
        ret = Az.AlazarPostAsyncBuffer(handle, b.get_obj(), U32(bytesPerBuffer))
        print "InitPost", ret_to_str(ret, Az)
    avg_buffer = Array(C.c_longdouble, bytesPerBuffer)
    
    # Initialize threads  
    bufs_merged = Value(U32, 1)
    buf_ready_events = [ Event() for _ in range(bufferCount) ]
    buf_post_events = [ Event() for _ in range(bufferCount) ]
    workers = [ Process(target=worker, args=(b, bre, bpe, avg_buffer, bufs_merged)) 
                for b, bre, bpe in zip(buffers, buf_ready_events, buf_post_events) ]
    
    # Begin capture
    ret = Az.AlazarStartCapture(handle)    
    print "StartCapture", ret_to_str(ret, Az)
    for w in workers:
        w.start()
    
    plotter = ScriptPlotter()
    buffers_acquired, buffers_completed, plot_count = 0, 0, 0
    start_time = time.time()
    for _ in range(iterations * bufferCount):
        buf_idx = buffers_acquired % bufferCount
        buf = buffers[buf_idx]
        with buf.get_lock():
            ret = Az.AlazarWaitAsyncBufferComplete(handle, buf.get_obj(), U32(timeout))
            print "WaitComplete", ret_to_str(ret, Az)
        buffers_acquired += 1
        # Tell worker thread to begin processing 
        buf_ready_events[buf_idx].set()
        
        # Post completed buffers
        while buf_post_events[buffers_completed % bufferCount].is_set():
            buf_post_events[buffers_completed % bufferCount].clear()
            buf = buffers[buffers_completed % bufferCount]
            with buf.get_lock():
                ret = Az.AlazarPostAsyncBuffer(handle, buf.get_obj(), U32(bytesPerBuffer))
                print "PostBuffer", ret_to_str(ret, Az)
            buffers_completed += 1
        
        # If a second has elapsed, replot the avg_buffer
        if time.time() - start_time > plot_count:
            with avg_buffer.get_lock():
                plotter.plot(np.frombuffer(avg_buffer.get_obj()), 'Data')
            plot_count += 1
"""
class MpBarrier:
    '''
    Taken from https://bitbucket.org/rickshin/multiprocessing-extras/src/751b8521f0d3/mp-extra.py
    MpBarrier class implements a process barrier in python. 
    It is allocated with the number of processes to expect and
    causes processes to wait until all processes ready to 
    move forward.
    '''
    def __init__(self, num):
        self.lock = Lock()
        self.num = num
        self.cur_num = Value('i', 0)
        self.condition = Condition()

    def wait(self):

        skip_wait=False

        self.lock.acquire()
        self.cur_num.value += 1
        if (self.cur_num.value >= self.num):
            skip_wait=True
            self.cur_num.value = 0
            with self.condition:
                self.condition.notify_all()
        self.lock.release()

        if not skip_wait:
            with self.condition:
                self.condition.wait()

class AzDll:
    def __init__(self):
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
        self.Az = card.Az
        #self.handle = card.handle
        
    def __getattr__(self, method):
        print method
        return lambda *args: getattr(self.Az, method)(*args)

class AzManager(BaseManager):
    pass

AzManager.register('Az', AzDll)

def worker(Az, buffers_ready, az_lock, avg_data, acq_count):
    #Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
    buf_data = (U8 * bytesPerBuffer)()
    arr_data = np.ctypeslib.as_array(buf_data)
    sum_data = np.zeros(bytesPerBuffer)
    with az_lock:
        handle = Az.AlazarGetBoardBySystemID(U32(1), U32(1))
        ret = Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
        print 'initial post', ret_to_str(ret, Az)
    buffers_ready.wait()
    for _ in range(iterations):
        sum_data.fill(0)
        for i in range(buffersPerMerge):
            if use_alazar:                
                with az_lock:
                    ret = Az.AlazarWaitAsyncBufferComplete(handle, buf_data, U32(timeout))
                    print "WaitComplete", ret_to_str(ret, Az)
                sum_data += arr_data
                if i is not 0:
                    with az_lock:
                        ret = Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
                        print "PostBuffer", ret_to_str(ret, Az)
            else:
                arr_data = np.sin(np.linspace(0, 2*np.pi, bytesPerBuffer))
                arr_data += np.random.normal(0, 2, bytesPerBuffer)
        with avg_data.get_lock():
            avg_data_v = np.frombuffer(avg_data.get_obj())
            acq_count_v = acq_count.value
            avg_data_v *= (acq_count_v - 1) / acq_count_v
            avg_data_v += arr_data / acq_count_v
            acq_count.value += 1

def acquire_avg_data_parallel_old():

    if use_alazar:
        az_mgr = AzManager()
        az_mgr.start()
        Az = az_mgr.Az()
        handle = Az.AlazarGetBoardBySystemID(U32(1), U32(1))
    
    az_lock = Lock()
    avg_data = Array(C.c_longdouble, bytesPerBuffer)
    acq_count = Value(U32, 1)
    buffers_ready = MpBarrier(bufferCount)

    buffers, workers = [], []
    print bufferCount
    for _ in range(bufferCount):
        print '.',
        # buffers.append((C.c_uint8 * bytesPerBuffer)())
        workers.append(Process(target=worker, args=(Az, buffers_ready, az_lock, avg_data, acq_count)))
        # Az.AlazarPostAsyncBuffer(handle, buffers[-1], U32(bytesPerBuffer))
        
    if use_alazar:
        #Az.AlazarBeforeAsyncRead(handle,U32(1), C.c_long(0),
        #                         U32(samplesPerRecord), 
        #                         U32(recordsPerBuffer),
        #                         U32(recordsPerAcquisition),
        #                         U32(513))
        Az.AlazarStartCapture(handle)
    for w in workers:
        w.start()
        print w
    with ScriptPlotter() as plotter:
        plotter.init_plot("Data", accum=False)
        while acq_count.value < bufferCount * iterations:
            time.sleep(.5)
            avg_data.acquire()
            plotter.plot(np.frombuffer(avg_data.get_obj()), "Data")
            plotter.msg(acq_count.value)
            avg_data.release()
    for w in workers:
        w.join()
    return np.array(avg_data)
"""

if __name__ == "__main__":
    try:
        data = acquire_avg_data_parallel()
    except:
        #Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
        #Az.AlazarAbortAsyncRead(handle)
        raise
    print 'done'
