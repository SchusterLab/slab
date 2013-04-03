from __future__ import division
from multiprocessing import Process, Lock, Array, Value
import ctypes as C
import numpy as np
import time
from slab.plotting import ScriptPlotter
import warnings
warnings.simplefilter('ignore')
#import matplotlib.pyplot as plt

use_alazar = False
if use_alazar:
    Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')

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

def worker(az_lock, avg_data, acq_count):
    buf_data = (C.c_uint8 * bytesPerBuffer)()
    arr_data = np.ctypeslib.as_array(buf_data)
    sum_data = np.zeros(bytesPerBuffer)
    for _ in range(iterations):
        sum_data.fill(0)
        for i in range(buffersPerMerge):
            az_lock.acquire()
            if use_alazar:
                if i is not 0:
                    Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
                Az.AlazarWaitAsyncBufferComplete(handle, buf_data, U32(timeout))
            else:
                arr_data = np.sin(np.linspace(0, 2*np.pi, bytesPerBuffer))
                arr_data += np.random.normal(0, 2, bytesPerBuffer)
            az_lock.release()
            sum_data += arr_data
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
    data = acquire_avg_data_parallel()
