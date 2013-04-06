from __future__ import division
from multiprocessing import Process, Lock, Array, Value
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

recordsPerBuffer = None
bytesPerRecord = None
bytesPerBuffer = 10000
buffersPerMerge = 10
bufferCount = 8
iterations = 1000
handle = None
timeout = None

use_alazar = True
if use_alazar:
    config={'clock_edge': 'rising', 'clock_source': 'reference',  
        'trigger_coupling': 'DC',  'trigger_operation': 'or', 
        'trigger_source2': 'disabled','trigger_level2': 1.0, 'trigger_edge2': 'rising', 
        'trigger_level1': 0.6, 'trigger_edge1': 'rising', 'trigger_source1': 'external', 
        'ch1_enabled': True,  'ch1_coupling': 'AC', 'ch1_range': 1, 'ch1_filter': False, 
        'ch2_enabled': False, 'ch2_coupling': 'AC','ch2_range': 1, 'ch2_filter': False,            
        'bufferCount': 20,'recordsPerBuffer': 100, 'trigger_delay': 0, 'timeout': 1000,
        'samplesPerRecord':1024,'recordsPerAcquisition': 0x7fffffff, 'sample_rate': 1000000}
    card = Alazar(AlazarConfig(config))
    card.configure()
    Az = card.Az
    handle = card.handle

    #Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')

def worker(az_lock, avg_data, acq_count):
    #Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')
    #buf_data = (C.c_uint8 * bytesPerBuffer)()
    #arr_data = np.ctypeslib.as_array(buf_data)
    sum_data = np.zeros(bytesPerBuffer)
#    with az_lock:
#        Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
    for _ in range(iterations):
        sum_data.fill(0)
        for i in range(buffersPerMerge):
            if use_alazar:                
                with az_lock:
                    Az.AlazarWaitAsyncBufferComplete(handle, buf_data, U32(timeout))
                sum_data += arr_data
                if i is not 0:
                    with az_lock:
                        Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
            else:
                arr_data = np.sin(np.linspace(0, 2*np.pi, bytesPerBuffer))
                arr_data += np.random.normal(0, 2, bytesPerBuffer)
        avg_data.acquire()
        avg_data_v = np.frombuffer(avg_data.get_obj())
        acq_count_v = acq_count.value
        avg_data_v *= (acq_count_v - 1) / acq_count_v
        avg_data_v += arr_data / acq_count_v
        acq_count.value += 1
        avg_data.release()

def acquire_avg_data_parallel():
    az_lock = Lock()
    avg_data = Array(C.c_longdouble, bytesPerBuffer)
    acq_count = Value(U32, 1)

    workers = []
    for _ in range(bufferCount):
        workers.append(Process(target=worker, args=(az_lock, avg_data, acq_count)))
        
    if use_alazar:
       # self.Az.AlazarBeforeAsyncRead(self.handle,U32(channel),pretriggers,
       #                                U32(self.config.samplesPerRecord), 
       #                                U32(self.config.recordsPerBuffer),
       #                                U32(self.config.recordsPerAcquisition),
       #                                flags)
        Az.AlazarStartCapture(handle)
    for w in workers:
        w.start()
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

if __name__ == "__main__":
    try:
        data = acquire_avg_data_parallel()
    except:
        Az.AlazarAbortAsyncRead
    print 'done'
